//! Verified file download for the pinned embedding model.
//!
//! Ported from Claudinio Code's `download.rs` minus its network-activity
//! accounting. The invariant is the same: a corrupt or truncated download can
//! never become the cache. Bytes stream into a `.part` file while a sha256 is
//! computed incrementally, size and hash are both checked, and only then is
//! the file renamed into place.
//!
//! The `.part` file belongs to one download alone. Several servers start at
//! once on a fresh machine — one per editor, all sharing this cache — and
//! with a shared temporary name the second would truncate the file the first
//! was still writing; the first would then verify its *stream*, rename the
//! other's half-written file into place and load it.

use std::path::Path;

pub const DEFAULT_RETRIES: usize = 3;

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(concat!("claudinio-code-intel/", env!("CARGO_PKG_VERSION")))
        .build()
        .expect("reqwest client")
}

/// Download `url` to `dest`, verifying length and sha256 before committing.
pub async fn download_verified(
    url: &str,
    dest: &Path,
    label: &str,
    sha256_hex: &str,
    expected_len: u64,
) -> Result<(), String> {
    use futures::StreamExt;
    use sha2::Digest;

    let response = client()
        .get(url)
        .send()
        .await
        .map_err(|e| format!("download {label}: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("download {label} failed: HTTP {status}"));
    }

    let part_path = part_path_for(dest);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
        remove_stale_parts(parent);
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&part_path)
        .map_err(|e| format!("create {}: {e}", part_path.display()))?;
    let mut hasher = sha2::Sha256::new();
    let mut written: u64 = 0;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let step = chunk
            .map_err(|e| format!("read {label}: {e}"))
            .and_then(|chunk| {
                std::io::Write::write_all(&mut file, &chunk).map_err(|e| format!("write {label}: {e}"))?;
                Ok(chunk)
            });
        match step {
            Ok(chunk) => {
                hasher.update(&chunk);
                written += chunk.len() as u64;
            }
            Err(e) => {
                drop(file);
                let _ = std::fs::remove_file(&part_path);
                return Err(e);
            }
        }
    }
    drop(file);

    let digest = format!("{:x}", hasher.finalize());
    if written != expected_len || digest != sha256_hex {
        let _ = std::fs::remove_file(&part_path);
        return Err(format!(
            "verification failed for {label}: got {written} bytes sha256 {digest}, \
             expected {expected_len} bytes sha256 {sha256_hex}"
        ));
    }
    if let Err(e) = std::fs::rename(&part_path, dest) {
        let _ = std::fs::remove_file(&part_path);
        // Another download of the same pinned file got there first (and on
        // Windows a file someone has open cannot be replaced): theirs passed
        // the same checks, so the cache is whole either way.
        if !dest.exists() {
            return Err(format!("finalize {label}: {e}"));
        }
    }
    Ok(())
}

/// A temporary name no other download can share: the full file name (the
/// graph `x.onnx` and its weights `x.onnx_data` must not collide either),
/// this process, and a counter for downloads within it.
fn part_path_for(dest: &Path) -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = dest
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "download".into());
    dest.with_file_name(format!("{name}.{}.{n}.part", std::process::id()))
}

/// A download that died leaves its `.part` behind, and with per-download
/// names nothing would ever overwrite it. A day is far longer than any
/// download here takes.
fn remove_stale_parts(dir: &Path) {
    const STALE: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_part = path.extension().is_some_and(|e| e == "part");
        let old = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > STALE);
        if is_part && old {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// `download_verified` with exponential backoff. A hash mismatch is retried
/// like any other failure: the usual cause is a truncated body from a flaky
/// connection, not a genuinely changed artifact.
pub async fn download_verified_with_retries(
    url: &str,
    dest: &Path,
    label: &str,
    sha256_hex: &str,
    expected_len: u64,
    retries: usize,
) -> Result<(), String> {
    let mut last_error = String::new();
    for attempt in 0..retries.max(1) {
        if attempt > 0 {
            tokio::time::sleep(std::time::Duration::from_secs(2 << (attempt - 1))).await;
        }
        match download_verified(url, dest, label, sha256_hex, expected_len).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                eprintln!("[download] {label} attempt {} failed: {e}", attempt + 1);
                last_error = e;
            }
        }
    }
    Err(last_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;
    use std::io::{Read, Write};

    /// Serves `body` to every connection, a slice at a time, so that two
    /// downloads of it overlap for certain.
    fn serve_slowly(body: std::sync::Arc<Vec<u8>>, chunk: usize, pause_ms: u64) -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let body = body.clone();
                std::thread::spawn(move || {
                    let mut buf = [0u8; 4096];
                    let mut req = Vec::new();
                    while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                        match stream.read(&mut buf) {
                            Ok(0) | Err(_) => return,
                            Ok(n) => req.extend_from_slice(&buf[..n]),
                        }
                    }
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    if stream.write_all(head.as_bytes()).is_err() {
                        return;
                    }
                    for part in body.chunks(chunk) {
                        if stream.write_all(part).is_err() {
                            return;
                        }
                        let _ = stream.flush();
                        std::thread::sleep(std::time::Duration::from_millis(pause_ms));
                    }
                });
            }
        });
        port
    }

    /// Two servers starting together download the same model file into the
    /// same cache. Whenever either of them is told the file is ready, the
    /// file on disk must be the whole, verified file — it is about to be
    /// loaded.
    #[test]
    fn overlapping_downloads_of_one_file_never_expose_a_partial_one() {
        let len = 2_000_000usize;
        let mut state = 12_345u32;
        let body: Vec<u8> = (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect();
        let sha = format!("{:x}", sha2::Sha256::digest(&body));
        let body = std::sync::Arc::new(body);
        // 50 slices, 20 ms apart: one download takes about a second.
        let port = serve_slowly(body.clone(), 40_000, 20);
        let url = format!("http://127.0.0.1:{port}/model_q4.onnx_data");
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("model_q4.onnx_data");

        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let whole = |who: &str| {
                let on_disk = std::fs::read(&dest).unwrap_or_default();
                assert!(on_disk == *body, "{who} was handed a file of {} bytes that is not the verified one", on_disk.len());
            };
            let first = async {
                download_verified(&url, &dest, "first", &sha, len as u64).await.expect("first download");
                whole("the first download");
            };
            let second = async {
                // Starts when the first is about two thirds through, so that
                // when the first finishes the second has written only the
                // head of the file: with a shared temporary file that leaves
                // a hole in the middle of what the first renames into place.
                tokio::time::sleep(std::time::Duration::from_millis(650)).await;
                download_verified(&url, &dest, "second", &sha, len as u64).await.expect("second download");
                whole("the second download");
            };
            futures::future::join(first, second).await;
        });
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".part"))
            .collect();
        assert!(leftovers.is_empty(), "temporary files left behind: {leftovers:?}");
    }

    /// The graph and its weights differ only by a suffix; their temporary
    /// files must not be the same file.
    #[test]
    fn temporary_names_are_per_file_and_per_download() {
        let a = part_path_for(Path::new("/cache/model_q4.onnx"));
        let b = part_path_for(Path::new("/cache/model_q4.onnx_data"));
        let again = part_path_for(Path::new("/cache/model_q4.onnx"));
        assert_ne!(a, b);
        assert_ne!(a, again);
        for p in [&a, &b, &again] {
            assert_eq!(p.extension().unwrap(), "part");
            assert_eq!(p.parent(), Some(Path::new("/cache")));
        }
    }

    #[test]
    fn a_wrong_hash_or_length_leaves_nothing_behind() {
        let body = std::sync::Arc::new(vec![7u8; 50_000]);
        let port = serve_slowly(body.clone(), 50_000, 0);
        let url = format!("http://127.0.0.1:{port}/f");
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("f.bin");
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let wrong_hash = rt.block_on(download_verified(&url, &dest, "f", &"0".repeat(64), 50_000));
        assert!(wrong_hash.is_err());
        let sha = format!("{:x}", sha2::Sha256::digest(&body[..]));
        let wrong_len = rt.block_on(download_verified(&url, &dest, "f", &sha, 49_999));
        assert!(wrong_len.is_err());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0, "no file, no .part");
        rt.block_on(download_verified(&url, &dest, "f", &sha, 50_000)).expect("the right pins download");
        assert_eq!(std::fs::read(&dest).unwrap(), *body);
    }
}
