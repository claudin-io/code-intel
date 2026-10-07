//! The real EmbeddingGemma 2 weights, downloaded from Hugging Face.
//!
//! Ignored by default — it fetches ~470 MB — and the one test that can say
//! whether the model actually retrieves: `gemma2_synthetic.rs` proves the
//! plumbing against a stand-in, this proves the plumbing is connected to the
//! right things (prompt prefixes, placeholder tokens, pixel and mel layout).
//! A channel-order bug or a wrong prefix passes every other test and fails
//! here.
//!
//!     cargo test -p claudinio-code-intel --test gemma2_e2e -- --ignored --nocapture
//!
//! `CODE_INTEL_E2E_MODELS` names the models directory to download into and
//! reuse (default: under `target/`). The scores and timings it prints are
//! what `GEMMA2_MIN_COSINE_CANDIDATE` and `MEDIA_MIN_COSINE` in `src/db.rs`
//! should be calibrated from.
#![cfg(feature = "embeddings")]

use claudinio_code_intel::embeddings::{self, ModelChoice};
use claudinio_code_intel::gemma2;
use claudinio_code_intel::media::{MediaKind, MediaNeeds};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn models_root() -> PathBuf {
    std::env::var_os("CODE_INTEL_E2E_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_TARGET_TMPDIR")).join("e2e-models"))
}

/// A shape on a white background: `disc` for a filled circle, otherwise a
/// filled square.
fn shape_png(path: &Path, rgb: [u8; 3], disc: bool) {
    let size = 256i32;
    let mut img =
        image::RgbImage::from_pixel(size as u32, size as u32, image::Rgb([255, 255, 255]));
    for y in 0..size {
        for x in 0..size {
            let (dx, dy) = (x - size / 2, y - size / 2);
            let inside = if disc {
                dx * dx + dy * dy <= 90 * 90
            } else {
                dx.abs() <= 80 && dy.abs() <= 80
            };
            if inside {
                img.put_pixel(x as u32, y as u32, image::Rgb(rgb));
            }
        }
    }
    img.save(path).unwrap();
}

fn wav(path: &Path, samples: &[i16]) {
    let rate = 16_000u32;
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + samples.len() as u32 * 2).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(samples.len() as u32 * 2).to_le_bytes());
    for s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

#[test]
#[ignore = "downloads the real EmbeddingGemma 2 weights (~470 MB) from Hugging Face"]
fn real_model_ranks_code_images_and_audio() {
    let root = models_root();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let needs = MediaNeeds {
        images: true,
        audio: true,
    };
    let started = Instant::now();
    // `Gemma2`, not `Auto`: a silent fallback to MiniLM must fail this test.
    let shared = rt
        .block_on(embeddings::ensure_and_load(
            &root,
            ModelChoice::Gemma2,
            needs,
        ))
        .expect("EmbeddingGemma 2 downloads and loads");
    eprintln!(
        "download + load: {:.1}s into {}",
        started.elapsed().as_secs_f32(),
        root.display()
    );
    {
        let guard = shared.lock().unwrap();
        assert_eq!(guard.model_id(), gemma2::MODEL_ID);
        assert_eq!(guard.media_support(), needs, "both encoders load");
    }

    // ── code ────────────────────────────────────────────────────────────
    let docs = [
        "function_item: refresh_token_if_stale | fn refresh_token_if_stale(session: &mut Session) { if session.expires_at < now() { session.token = renew(session.refresh_token); } }",
        "function_item: quicksort | fn quicksort<T: Ord>(v: &mut [T]) { if v.len() <= 1 { return; } let p = partition(v); quicksort(&mut v[..p]); quicksort(&mut v[p + 1..]); }",
        "function_item: retry_failed_upload | async fn retry_failed_upload(job: &UploadJob) { let mut delay = BASE; while let Err(e) = upload(job).await { sleep(delay).await; delay *= 2; } }",
        "doc_section: Deploying | Run the deploy script after merging to main. It performs a blue-green swap and waits for the health check before switching traffic.",
    ];
    let queries = [
        "where do we handle expired login sessions",
        "sort a slice in place",
        "retry an upload with exponential backoff",
        "how do I ship a release to production",
    ];
    let t = Instant::now();
    let doc_vecs = shared
        .lock()
        .unwrap()
        .encode(&docs)
        .expect("encode documents");
    eprintln!(
        "{} code chunks: {:.2}s",
        docs.len(),
        t.elapsed().as_secs_f32()
    );
    assert_eq!(doc_vecs[0].len(), gemma2::EMBED_DIM);
    eprintln!("\ncode  (rows: queries, columns: documents)");
    for (qi, q) in queries.iter().enumerate() {
        let qv = shared
            .lock()
            .unwrap()
            .encode_query(q)
            .expect("encode query");
        let scores: Vec<f32> = doc_vecs.iter().map(|d| dot(&qv, d)).collect();
        eprintln!("  {scores:.3?}  {q}");
        let best = scores.iter().cloned().fold(f32::MIN, f32::max);
        assert_eq!(
            scores[qi], best,
            "query {qi} ({q}) must rank its own document first: {scores:?}"
        );
    }

    // ── images ──────────────────────────────────────────────────────────
    let dir = tempfile::tempdir().unwrap();
    let images = [
        ("red-disc.png", [220u8, 20, 20], true, "a red circle"),
        ("blue-square.png", [20, 40, 220], false, "a blue square"),
        ("green-disc.png", [20, 170, 40], true, "a green circle"),
    ];
    let mut image_vecs = Vec::new();
    for (name, rgb, disc, _) in &images {
        let path = dir.path().join(name);
        shape_png(&path, *rgb, *disc);
        let t = Instant::now();
        let v =
            embeddings::embed_media_file(&shared, &path, MediaKind::Image).expect("embed image");
        eprintln!("{name}: {:.2}s", t.elapsed().as_secs_f32());
        assert!((dot(&v, &v) - 1.0).abs() < 1e-3);
        image_vecs.push(v);
    }
    eprintln!("\nimages  (rows: queries, columns: images)");
    for (qi, (_, _, _, q)) in images.iter().enumerate() {
        let qv = shared
            .lock()
            .unwrap()
            .encode_media_query(q)
            .unwrap()
            .expect("media query");
        let scores: Vec<f32> = image_vecs.iter().map(|d| dot(&qv, d)).collect();
        eprintln!("  {scores:.3?}  {q}");
        let best = scores.iter().cloned().fold(f32::MIN, f32::max);
        // Colour and shape are about the simplest things a picture can
        // show. Getting these wrong means the pixels are reaching the
        // encoder scrambled (channel order, patch layout, scaling).
        assert_eq!(
            scores[qi], best,
            "query {qi} ({q}) must rank its own image first: {scores:?}"
        );
    }

    // ── audio ───────────────────────────────────────────────────────────
    let tone: Vec<i16> = (0..32_000)
        .map(|i| {
            ((2.0 * std::f64::consts::PI * 880.0 * i as f64 / 16_000.0).sin() * 12_000.0) as i16
        })
        .collect();
    let mut state = 0x2545_F491u32;
    let noise: Vec<i16> = (0..32_000)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 16) as i16) / 4
        })
        .collect();
    let clips = [
        ("tone.wav", &tone, "a steady high-pitched beep"),
        ("noise.wav", &noise, "static white noise hiss"),
    ];
    let mut audio_vecs = Vec::new();
    for (name, samples, _) in &clips {
        let path = dir.path().join(name);
        wav(&path, samples);
        let t = Instant::now();
        let v =
            embeddings::embed_media_file(&shared, &path, MediaKind::Audio).expect("embed audio");
        eprintln!("{name} (2 s): {:.2}s", t.elapsed().as_secs_f32());
        assert!((dot(&v, &v) - 1.0).abs() < 1e-3);
        audio_vecs.push(v);
    }
    assert!(
        dot(&audio_vecs[0], &audio_vecs[1]) < 0.98,
        "a tone and noise must not embed alike"
    );
    // Printed, not asserted: synthetic sounds are a weaker probe than
    // coloured shapes, and a wrong ranking here is a prompt to listen to
    // real clips, not proof of a bug.
    eprintln!("\naudio  (rows: queries, columns: clips)");
    for (_, _, q) in &clips {
        let qv = shared
            .lock()
            .unwrap()
            .encode_media_query(q)
            .unwrap()
            .expect("media query");
        let scores: Vec<f32> = audio_vecs.iter().map(|d| dot(&qv, d)).collect();
        eprintln!("  {scores:.3?}  {q}");
    }
}
