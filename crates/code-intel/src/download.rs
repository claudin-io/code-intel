//! Verified file download for the pinned embedding model.
//!
//! Ported from Claudinio Code's `download.rs` minus its network-activity
//! accounting. The invariant is the same: a corrupt or truncated download can
//! never become the cache. Bytes stream into a `.part` file while a sha256 is
//! computed incrementally, size and hash are both checked, and only then is
//! the file renamed into place.

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
