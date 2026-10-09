//! Verified file download for the pinned embedding model.
//!
//! Ported from Claudinio Code's `download.rs`; its network-activity accounting
//! is left to the host through a [`DownloadObserver`]. The invariant is the same: a corrupt or truncated download can
//! never become the cache. Bytes stream into a `.part` file while a sha256 is
//! computed incrementally, size and hash are both checked, and only then is
//! the file renamed into place.

use std::path::Path;

pub const DEFAULT_RETRIES: usize = 3;

/// Watches downloads for the host application. Called once per file with its
/// label; the callback it returns receives each chunk's byte count and is
/// dropped when that file's download ends, whichever way it ends. Claudinio
/// Code puts its network-activity guard inside that callback.
pub type DownloadObserver =
    std::sync::Arc<dyn Fn(&str) -> Box<dyn Fn(u64) + Send + Sync> + Send + Sync>;

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
    observer: Option<&DownloadObserver>,
) -> Result<(), String> {
    use futures::StreamExt;
    use sha2::Digest;

    // Begun before the request so the host sees the connection, not only the
    // body; dropped on every exit path below.
    let on_bytes = observer.map(|o| o(label));
    let response = client()
        .get(url)
        .send()
        .await
        .map_err(|e| format!("download {label}: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("download {label} failed: HTTP {status}"));
    }

    let part_path = dest.with_extension("part");
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let mut file = std::fs::File::create(&part_path)
        .map_err(|e| format!("create {}: {e}", part_path.display()))?;
    let mut hasher = sha2::Sha256::new();
    let mut written: u64 = 0;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("read {label}: {e}"))?;
        std::io::Write::write_all(&mut file, &chunk).map_err(|e| format!("write {label}: {e}"))?;
        hasher.update(&chunk);
        written += chunk.len() as u64;
        if let Some(cb) = &on_bytes {
            cb(chunk.len() as u64);
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
    std::fs::rename(&part_path, dest).map_err(|e| format!("finalize {label}: {e}"))?;
    Ok(())
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
    observer: Option<&DownloadObserver>,
    retries: usize,
) -> Result<(), String> {
    let mut last_error = String::new();
    for attempt in 0..retries.max(1) {
        if attempt > 0 {
            tokio::time::sleep(std::time::Duration::from_secs(2 << (attempt - 1))).await;
        }
        match download_verified(url, dest, label, sha256_hex, expected_len, observer).await {
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
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Serves `body` once over plain HTTP on 127.0.0.1 and returns its URL.
    async fn serve_once(body: &'static [u8]) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut req = [0u8; 1024];
            let _ = sock.read(&mut req).await;
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            sock.write_all(head.as_bytes()).await.unwrap();
            sock.write_all(body).await.unwrap();
        });
        format!("http://{addr}/model.bin")
    }

    /// The host app shows every download in its network-activity indicator, so
    /// the observer must hear about each file once, by its label, receive every
    /// byte that was written, and be released when the download is over.
    #[tokio::test]
    async fn observer_sees_the_download_begin_carry_every_byte_and_end() {
        use sha2::Digest;
        const BODY: &[u8] = b"pinned embedding model bytes";
        let sha = format!("{:x}", sha2::Sha256::digest(BODY));
        let url = serve_once(BODY).await;
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("model.bin");

        let labels = Arc::new(Mutex::new(Vec::<String>::new()));
        let bytes = Arc::new(AtomicU64::new(0));
        let released = Arc::new(AtomicUsize::new(0));
        struct OnDrop(Arc<AtomicUsize>);
        impl Drop for OnDrop {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let observer: DownloadObserver = {
            let (labels, bytes, released) = (labels.clone(), bytes.clone(), released.clone());
            Arc::new(move |label: &str| {
                labels.lock().unwrap().push(label.to_string());
                let bytes = bytes.clone();
                let guard = OnDrop(released.clone());
                Box::new(move |n: u64| {
                    let _ = &guard;
                    bytes.fetch_add(n, Ordering::SeqCst);
                })
            })
        };

        download_verified(
            &url,
            &dest,
            "model.bin",
            &sha,
            BODY.len() as u64,
            Some(&observer),
        )
        .await
        .unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), BODY);
        assert_eq!(*labels.lock().unwrap(), vec!["model.bin".to_string()]);
        assert_eq!(bytes.load(Ordering::SeqCst), BODY.len() as u64);
        assert_eq!(
            released.load(Ordering::SeqCst),
            1,
            "the per-file callback must be dropped when the download ends"
        );
    }
}
