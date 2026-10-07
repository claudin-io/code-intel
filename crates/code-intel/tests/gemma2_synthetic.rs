//! The EmbeddingGemma 2 code path, end to end, through real ONNX Runtime —
//! against a stand-in model.
//!
//! `fixtures/gemma2-synthetic/` holds three tiny graphs and a tokenizer with
//! the real model's contract (file names, inputs, outputs, dtypes, dynamic
//! shapes, external weight files, placeholder tokens). Everything between a
//! file on disk and a vector in the index runs for real here: session
//! loading, zero-row soft-token inputs, batching and padding, image and audio
//! decoding, the two-phase media encode, the index bookkeeping around a model
//! change. What these tests cannot tell is whether the real weights retrieve
//! well; `gemma2_e2e.rs` is for that.
#![cfg(feature = "embeddings")]

use claudinio_code_intel::db::IndexDb;
use claudinio_code_intel::embeddings::{
    self, CodeEmbedder, MINILM_MODEL_ID, ModelChoice, SharedEmbedder,
};
use claudinio_code_intel::gemma2::{self, Gemma2Embedder, QueryTask};
use claudinio_code_intel::indexer;
use claudinio_code_intel::media::{MediaKind, MediaNeeds};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const TEXT: [&str; 3] = ["tokenizer.json", "model_q4.onnx", "model_q4.onnx_data"];
const VISION: [&str; 2] = ["vision_encoder_q4.onnx", "vision_encoder_q4.onnx_data"];
const AUDIO: [&str; 2] = ["audio_encoder_q4.onnx", "audio_encoder_q4.onnx_data"];

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma2-synthetic")
}

/// A model directory holding just the named files, as a partial download
/// would leave it.
fn model_dir(files: &[&str]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for name in files {
        std::fs::copy(fixture_dir().join(name), dir.path().join(name)).unwrap();
    }
    dir
}

fn all_files() -> Vec<&'static str> {
    TEXT.iter()
        .chain(VISION.iter())
        .chain(AUDIO.iter())
        .copied()
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn assert_unit(v: &[f32]) {
    assert_eq!(v.len(), gemma2::EMBED_DIM);
    assert!(
        (dot(v, v) - 1.0).abs() < 1e-4,
        "not unit length: {}",
        dot(v, v)
    );
}

fn write_png(path: &Path, w: u32, h: u32, rgb: [u8; 3]) {
    // A flat colour with one contrasting block, so two images differ in
    // content and not only in size.
    let mut img = image::RgbImage::from_pixel(w, h, image::Rgb(rgb));
    for y in 0..h / 3 {
        for x in 0..w / 3 {
            img.put_pixel(x, y, image::Rgb([255 - rgb[0], 255 - rgb[1], 255 - rgb[2]]));
        }
    }
    img.save(path).unwrap();
}

fn write_wav(path: &Path, hz: f64, seconds: f64) {
    let rate = 22_050u32;
    let n = (rate as f64 * seconds) as usize;
    let mut pcm: Vec<u8> = Vec::with_capacity(n * 2);
    for i in 0..n {
        let s =
            ((2.0 * std::f64::consts::PI * hz * i as f64 / rate as f64).sin() * 10_000.0) as i16;
        pcm.extend_from_slice(&s.to_le_bytes());
    }
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&pcm);
    std::fs::write(path, bytes).unwrap();
}

/// A text-only download is a complete, working embedder: no encoder file is
/// needed to load it or to embed text.
#[test]
fn text_model_alone_loads_and_embeds() {
    let dir = model_dir(&TEXT);
    let mut e = Gemma2Embedder::load(dir.path()).expect("text model loads without any encoder");
    assert_eq!(e.media_support(), MediaNeeds::NONE);

    let docs = [
        "fn refresh_token_if_stale(session: &mut Session) {}",
        "struct FileWatcher",
        "a considerably longer piece of text, so that the two shorter ones above are padded when batched with it",
    ];
    let batched = e.encode_documents(&docs).unwrap();
    assert_eq!(batched.len(), 3);
    for v in &batched {
        assert_unit(v);
    }
    assert!(
        dot(&batched[0], &batched[1]) < 0.999,
        "different texts, different vectors"
    );

    // Padding must not leak into the result: a text embedded in a batch with
    // longer ones equals the same text embedded alone.
    for (i, doc) in docs.iter().enumerate() {
        let alone = e.encode_documents(&[doc]).unwrap().pop().unwrap();
        assert!(
            dot(&alone, &batched[i]) > 0.9999,
            "doc {i} differs between batch and single: {}",
            dot(&alone, &batched[i])
        );
    }

    // Queries and documents carry different task prefixes, and so do the two
    // kinds of query — the same words must not embed identically.
    let doc = e
        .encode_documents(&["retry failed uploads"])
        .unwrap()
        .pop()
        .unwrap();
    let code_q = e
        .encode_query("retry failed uploads", QueryTask::Code)
        .unwrap();
    let media_q = e
        .encode_query("retry failed uploads", QueryTask::Media)
        .unwrap();
    assert_unit(&code_q);
    assert!(dot(&doc, &code_q) < 0.9999);
    assert!(dot(&code_q, &media_q) < 0.9999);

    assert!(e.encode_documents(&[]).unwrap().is_empty());
}

#[test]
fn a_missing_or_incomplete_model_is_an_error_at_load() {
    let empty = tempfile::tempdir().unwrap();
    assert!(Gemma2Embedder::load(empty.path()).is_err());
    // The graph without its weights file is not a model.
    let partial = model_dir(&["tokenizer.json", "model_q4.onnx"]);
    assert!(Gemma2Embedder::load(partial.path()).is_err());
}

/// Encoders are loaded per kind, on request, and each makes exactly its own
/// kind of file embeddable.
#[test]
fn media_encoders_load_independently_and_embed_files() {
    let files: Vec<&str> = TEXT.iter().chain(VISION.iter()).copied().collect();
    let dir = model_dir(&files);
    let media = tempfile::tempdir().unwrap();
    let red = media.path().join("red.png");
    let blue = media.path().join("blue.png");
    let tone = media.path().join("tone.wav");
    write_png(&red, 120, 80, [220, 30, 30]);
    write_png(&blue, 64, 200, [20, 40, 230]);
    write_wav(&tone, 440.0, 1.2);

    let mut e = CodeEmbedder::load_gemma2(dir.path()).unwrap();
    assert_eq!(e.model_id(), gemma2::MODEL_ID);
    assert!(
        e.embed_media_file(&red, MediaKind::Image).is_err(),
        "no encoder loaded yet"
    );

    let shared: SharedEmbedder = Arc::new(Mutex::new(e));
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    // The vision files are already in the directory, so nothing is fetched.
    rt.block_on(embeddings::extend_media(
        &shared,
        MediaNeeds {
            images: true,
            audio: false,
        },
    ));
    let support = shared.lock().unwrap().media_support();
    assert_eq!(
        support,
        MediaNeeds {
            images: true,
            audio: false
        }
    );

    let v_red = embeddings::embed_media_file(&shared, &red, MediaKind::Image).unwrap();
    let v_blue = embeddings::embed_media_file(&shared, &blue, MediaKind::Image).unwrap();
    assert_unit(&v_red);
    assert_unit(&v_blue);
    assert!(
        dot(&v_red, &v_blue) < 0.9999,
        "different images, different vectors"
    );
    let again = embeddings::embed_media_file(&shared, &red, MediaKind::Image).unwrap();
    assert!(
        dot(&v_red, &again) > 0.9999,
        "embedding a file is deterministic"
    );
    // The lock-holding variant the watcher uses gives the same answer.
    let held = shared
        .lock()
        .unwrap()
        .embed_media_file(&red, MediaKind::Image)
        .unwrap();
    assert!(dot(&v_red, &held) > 0.9999);

    assert!(
        embeddings::embed_media_file(&shared, &tone, MediaKind::Audio).is_err(),
        "no audio encoder"
    );
    assert!(
        embeddings::embed_media_file(&shared, &tone, MediaKind::Image).is_err(),
        "a WAV is not an image"
    );

    // Text still works with an encoder loaded, and a media query is offered.
    let mut guard = shared.lock().unwrap();
    assert_unit(&guard.encode(&["fn main() {}"]).unwrap()[0]);
    assert_unit(
        &guard
            .encode_media_query("a red picture")
            .unwrap()
            .expect("gemma2 has a media space"),
    );
}

#[test]
fn audio_encoder_embeds_clips() {
    let files: Vec<&str> = TEXT.iter().chain(AUDIO.iter()).copied().collect();
    let dir = model_dir(&files);
    let media = tempfile::tempdir().unwrap();
    let low = media.path().join("low.wav");
    let high = media.path().join("high.wav");
    write_wav(&low, 220.0, 0.8);
    write_wav(&high, 3000.0, 2.5);

    let e = Gemma2Embedder::load(dir.path()).unwrap();
    assert!(
        e.load_encoder(MediaKind::Image).is_err(),
        "vision files were not downloaded"
    );
    e.load_encoder(MediaKind::Audio).unwrap();
    assert_eq!(
        e.media_support(),
        MediaNeeds {
            images: false,
            audio: true
        }
    );
    let mut e = e;
    let soft_low = gemma2::encode_media_file(&e.encoders(), &low, MediaKind::Audio).unwrap();
    let soft_high = gemma2::encode_media_file(&e.encoders(), &high, MediaKind::Audio).unwrap();
    // 25 soft tokens per second of audio, 8 values each in the stand-in.
    assert_eq!(soft_low.len() / 8, 20);
    assert!(
        (soft_high.len() as i64 / 8 - 62).abs() <= 1,
        "{}",
        soft_high.len() / 8
    );
    let v_low = e.embed_soft_tokens(MediaKind::Audio, &soft_low).unwrap();
    let v_high = e.embed_soft_tokens(MediaKind::Audio, &soft_high).unwrap();
    assert_unit(&v_low);
    assert!(dot(&v_low, &v_high) < 0.9999);
    // Soft tokens that do not divide into rows are refused, not truncated.
    assert!(
        e.embed_soft_tokens(MediaKind::Audio, &soft_low[..soft_low.len() - 3])
            .is_err()
    );
}

fn workspace() -> tempfile::TempDir {
    let ws = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(ws.path().join("src")).unwrap();
    std::fs::create_dir_all(ws.path().join("assets")).unwrap();
    std::fs::write(
        ws.path().join("src").join("upload.rs"),
        "/// Retries a failed upload with exponential backoff until the server accepts it.\n\
         pub fn retry_failed_upload(attempts: u32) -> bool {\n    let mut delay = 100;\n    for _ in 0..attempts {\n        delay *= 2;\n    }\n    delay > 0\n}\n",
    )
    .unwrap();
    write_png(
        &ws.path().join("assets").join("app-logo.png"),
        96,
        96,
        [240, 200, 20],
    );
    write_wav(&ws.path().join("assets").join("error-beep.wav"), 880.0, 0.6);
    ws
}

fn shared_with(dir: &Path, needs: MediaNeeds) -> SharedEmbedder {
    // The encoder files are already in `dir`, so `extend_media` only loads.
    let embedder = CodeEmbedder::load_gemma2(dir).unwrap();
    let shared: SharedEmbedder = Arc::new(Mutex::new(embedder));
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(embeddings::extend_media(&shared, needs));
    assert_eq!(shared.lock().unwrap().media_support(), needs);
    shared
}

/// The whole indexing path: code and media scanned, everything embedded, the
/// two kinds of result kept apart, and a model change re-embedding the index.
#[test]
fn a_workspace_with_media_is_indexed_and_searchable() {
    let dir = model_dir(&all_files());
    let ws = workspace();
    let root = ws.path().to_string_lossy().to_string();
    let db = IndexDb::open(&ws.path().join("index.db")).unwrap();

    indexer::scan_workspace(&db, &root, None, None, None).unwrap();
    assert_eq!(db.media_file_counts().unwrap(), (1, 1));
    assert_eq!(db.media_embedding_count().unwrap(), 0);
    // Before any model has run, media is already findable by name.
    let by_name = db.search_media("app logo", None, 3).unwrap();
    assert_eq!(by_name.len(), 1);
    assert_eq!(by_name[0].kind, "image");
    assert_eq!(by_name[0].match_type, "lexical");
    assert!(by_name[0].file_path.ends_with("app-logo.png"));

    let shared = shared_with(
        dir.path(),
        MediaNeeds {
            images: true,
            audio: true,
        },
    );
    let (_, vectors) = indexer::generate_all_embeddings(&db, &shared, None, &root).unwrap();
    assert!(
        vectors >= 3,
        "code chunk(s) plus two media files, got {vectors}"
    );
    assert_eq!(db.media_embedding_count().unwrap(), 2);
    assert_eq!(db.embedding_pending_files().unwrap(), 0);
    assert_eq!(db.embedding_model().as_deref(), Some(gemma2::MODEL_ID));

    // A query vector equal to the logo's own vector is the best possible
    // content match: it must come back, as a content hit, ahead of the beep.
    let logo = ws.path().join("assets").join("app-logo.png");
    let logo_vec = embeddings::embed_media_file(&shared, &logo, MediaKind::Image).unwrap();
    let hits = db.search_media("zzz", Some(&logo_vec), 3).unwrap();
    assert!(!hits.is_empty());
    assert!(hits[0].file_path.ends_with("app-logo.png"));
    assert_eq!(hits[0].match_type, "semantic");
    // Name and content agreeing is the strongest evidence.
    let both = db.search_media("app logo", Some(&logo_vec), 3).unwrap();
    assert_eq!(both[0].match_type, "hybrid");
    assert!(both[0].score > hits[0].score);

    // Code search is not polluted by media rows, with or without a vector.
    let q = shared
        .lock()
        .unwrap()
        .encode_query("retry a failed upload")
        .unwrap();
    for results in [
        db.search_hybrid("retry failed upload", Some(&q), 10)
            .unwrap(),
        db.search_hybrid("app logo error beep", None, 10).unwrap(),
        db.search_hybrid("app logo", Some(&logo_vec), 10).unwrap(),
    ] {
        assert!(
            results
                .iter()
                .all(|r| r.kind != "image" && r.kind != "audio"),
            "{results:?}"
        );
    }
    assert!(
        db.search_hybrid("retry failed upload", Some(&q), 10)
            .unwrap()
            .iter()
            .any(|r| r.name == "retry_failed_upload")
    );

    // Re-running is a no-op: nothing pending, nothing re-embedded.
    let (_, again) = indexer::generate_all_embeddings(&db, &shared, None, &root).unwrap();
    assert_eq!(again, 0);

    // The model changes under the index (here: the fallback to MiniLM).
    // Vectors of the old model are gone and every file is queued again.
    let reset = db
        .reconcile_embedding_model(MINILM_MODEL_ID, embeddings::MINILM_DIM, MediaNeeds::NONE)
        .unwrap();
    assert!(reset.model_changed);
    assert_eq!(db.index_stats().unwrap().2, 0);
    assert!(db.embedding_pending_files().unwrap() >= 3);
    assert_eq!(db.embedding_model().as_deref(), Some(MINILM_MODEL_ID));
}

/// "Download the encoder only when the project has that kind of file", over
/// time: an index built without encoders gives media no content vector and
/// does not keep it pending; once an encoder is there, those files — and only
/// those — are embedded.
#[test]
fn media_gets_content_vectors_once_an_encoder_arrives() {
    let dir = model_dir(&all_files());
    let ws = workspace();
    let root = ws.path().to_string_lossy().to_string();
    let db = IndexDb::open(&ws.path().join("index.db")).unwrap();
    indexer::scan_workspace(&db, &root, None, None, None).unwrap();

    let text_only = shared_with(dir.path(), MediaNeeds::NONE);
    let (_, text_vectors) = indexer::generate_all_embeddings(&db, &text_only, None, &root).unwrap();
    assert!(text_vectors >= 1);
    assert_eq!(db.media_embedding_count().unwrap(), 0);
    assert_eq!(
        db.embedding_pending_files().unwrap(),
        0,
        "media without an encoder is not left pending"
    );
    let code_vectors = db.index_stats().unwrap().2;

    let with_vision = shared_with(
        dir.path(),
        MediaNeeds {
            images: true,
            audio: false,
        },
    );
    let (_, added) = indexer::generate_all_embeddings(&db, &with_vision, None, &root).unwrap();
    assert_eq!(
        added, 1,
        "exactly the image is embedded; no code is re-embedded"
    );
    assert_eq!(db.media_embedding_count().unwrap(), 1);
    assert_eq!(db.index_stats().unwrap().2, code_vectors + 1);
    assert_eq!(db.embedding_pending_files().unwrap(), 0);

    // An image that changes on disk is re-registered and re-embedded.
    let logo = ws.path().join("assets").join("app-logo.png");
    write_png(&logo, 300, 100, [10, 10, 10]);
    {
        let mut guard = with_vision.lock().unwrap();
        indexer::reindex_file(&db, &logo.to_string_lossy(), Some(&mut guard), Some(&root)).unwrap();
    }
    assert_eq!(db.media_embedding_count().unwrap(), 1);
    assert_eq!(db.embedding_pending_files().unwrap(), 0);

    // And one that is deleted leaves the index.
    std::fs::remove_file(&logo).unwrap();
    indexer::reindex_file(&db, &logo.to_string_lossy(), None, Some(&root)).unwrap();
    assert_eq!(db.media_file_counts().unwrap(), (0, 1));
    assert_eq!(db.media_embedding_count().unwrap(), 0);
}

/// A file with an image extension that is not an image must not stall or
/// poison the pass: it stays findable by name and is not retried forever.
#[test]
fn an_undecodable_media_file_is_skipped_once() {
    let dir = model_dir(&all_files());
    let ws = tempfile::tempdir().unwrap();
    std::fs::write(ws.path().join("broken-banner.png"), b"this is not a png").unwrap();
    let root = ws.path().to_string_lossy().to_string();
    let db = IndexDb::open(&ws.path().join("index.db")).unwrap();
    indexer::scan_workspace(&db, &root, None, None, None).unwrap();

    let shared = shared_with(
        dir.path(),
        MediaNeeds {
            images: true,
            audio: true,
        },
    );
    let (_, vectors) = indexer::generate_all_embeddings(&db, &shared, None, &root).unwrap();
    assert_eq!(vectors, 0);
    assert_eq!(db.embedding_pending_files().unwrap(), 0);
    assert_eq!(db.search_media("broken banner", None, 3).unwrap().len(), 1);
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// A models root (`<cache>/models`) with the stand-in model where the real
/// one would be downloaded to.
fn models_root(files: &[&str]) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join(gemma2::CACHE_DIRNAME);
    std::fs::create_dir_all(&dir).unwrap();
    for name in files {
        std::fs::copy(fixture_dir().join(name), dir.join(name)).unwrap();
    }
    root
}

/// What the server calls at start-up. With the files already in the cache
/// nothing is downloaded; the encoders loaded are exactly the ones the
/// workspace's media asks for.
#[test]
fn ensure_and_load_picks_gemma2_and_only_the_encoders_needed() {
    let root = models_root(&all_files());
    let rt = runtime();
    for needs in [
        MediaNeeds::NONE,
        MediaNeeds {
            images: true,
            audio: false,
        },
        MediaNeeds {
            images: false,
            audio: true,
        },
    ] {
        let shared = rt
            .block_on(embeddings::ensure_and_load(
                root.path(),
                ModelChoice::Auto,
                needs,
            ))
            .unwrap();
        let guard = shared.lock().unwrap();
        assert_eq!(guard.model_id(), gemma2::MODEL_ID);
        assert_eq!(guard.media_support(), needs);
    }

    // A second workspace with audio joins a process that started without it.
    let shared = rt
        .block_on(embeddings::ensure_and_load(
            root.path(),
            ModelChoice::Auto,
            MediaNeeds {
                images: true,
                audio: false,
            },
        ))
        .unwrap();
    rt.block_on(embeddings::extend_media(
        &shared,
        MediaNeeds {
            images: false,
            audio: true,
        },
    ));
    assert_eq!(
        shared.lock().unwrap().media_support(),
        MediaNeeds {
            images: true,
            audio: true
        }
    );
}

/// A text model this runtime cannot load must not end semantic search: under
/// the default choice the loader moves on to MiniLM, and only an explicit
/// request for EmbeddingGemma 2 stops there. Nothing is downloaded here —
/// every file "exists", they are just not models — so what MiniLM's turn
/// leaves behind is its self-heal: the unusable directory removed.
#[test]
fn an_unloadable_gemma2_falls_back_to_minilm_unless_forced() {
    let root = tempfile::tempdir().unwrap();
    let gemma_dir = root.path().join(gemma2::CACHE_DIRNAME);
    std::fs::create_dir_all(&gemma_dir).unwrap();
    for (_, local, _, _) in gemma2::component_files(None) {
        std::fs::write(gemma_dir.join(local), b"not a model").unwrap();
    }
    let minilm_dir = embeddings::minilm_dir(root.path());
    let plant_minilm = || {
        std::fs::create_dir_all(&minilm_dir).unwrap();
        for (_, local, _, _) in embeddings::required_model_files() {
            std::fs::write(minilm_dir.join(local), b"not a model").unwrap();
        }
    };
    let rt = runtime();

    plant_minilm();
    let forced = rt.block_on(embeddings::ensure_and_load(
        root.path(),
        ModelChoice::Gemma2,
        MediaNeeds::NONE,
    ));
    assert!(forced.is_err());
    assert!(
        minilm_dir.exists(),
        "an explicit EmbeddingGemma 2 request never touches MiniLM"
    );

    let auto = rt.block_on(embeddings::ensure_and_load(
        root.path(),
        ModelChoice::Auto,
        MediaNeeds::NONE,
    ));
    assert!(auto.is_err(), "neither planted model is real");
    assert!(
        !minilm_dir.exists(),
        "MiniLM was tried after EmbeddingGemma 2 failed"
    );
    assert!(
        gemma_dir.exists(),
        "hash-checked EmbeddingGemma 2 files are not deleted on a load error"
    );
}
