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

use claudinio_code_intel::db::{IndexDb, VectorSet};
use claudinio_code_intel::embeddings::{
    self, CodeEmbedder, MINILM_MODEL_ID, ModelChoice, ModelProfile, SharedEmbedder,
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
    let e = Gemma2Embedder::load(dir.path()).expect("text model loads without any encoder");
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

    // The text model changes under the index (here: back to MiniLM). The
    // code vectors are gone and the code queued again; the media vectors,
    // which are not MiniLM's business, stay.
    let reset = db
        .reconcile_embedding_model(&ModelProfile::text_only(
            MINILM_MODEL_ID,
            embeddings::MINILM_DIM,
        ))
        .unwrap();
    assert!(reset.model_changed && !reset.media_model_changed);
    assert_eq!(db.index_stats().unwrap().2, 2, "the two media vectors");
    assert_eq!(db.media_embedding_count().unwrap(), 2);
    assert!(db.embedding_pending_files().unwrap() >= 1);
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

fn minilm_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/minilm-synthetic")
}

/// A models root (`<cache>/models`) with the stand-in MiniLM in place and the
/// named stand-in EmbeddingGemma 2 files where the real ones would be
/// downloaded to. Nothing in these tests may reach for a file that is not
/// here: that would be a download of the real model.
fn models_root(gemma2_files: &[&str]) -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    let minilm = embeddings::minilm_dir(root.path());
    std::fs::create_dir_all(&minilm).unwrap();
    for name in ["model_quantized.onnx", "tokenizer.json"] {
        std::fs::copy(minilm_fixture_dir().join(name), minilm.join(name)).unwrap();
    }
    if !gemma2_files.is_empty() {
        let dir = root.path().join(gemma2::CACHE_DIRNAME);
        std::fs::create_dir_all(&dir).unwrap();
        for name in gemma2_files {
            std::fs::copy(fixture_dir().join(name), dir.join(name)).unwrap();
        }
    }
    root
}

const IMAGES: MediaNeeds = MediaNeeds {
    images: true,
    audio: false,
};
const AUDIO_ONLY: MediaNeeds = MediaNeeds {
    images: false,
    audio: true,
};
const BOTH: MediaNeeds = MediaNeeds {
    images: true,
    audio: true,
};

/// What the server calls at start-up, under the default choice. Text is
/// MiniLM whatever the workspace holds. EmbeddingGemma 2 is not so much as
/// looked for until a workspace has media, and then it comes with exactly the
/// encoders that media asks for.
#[test]
fn by_default_text_is_minilm_and_gemma2_joins_only_for_media() {
    let rt = runtime();

    // A text-only project, with no EmbeddingGemma 2 anywhere: nothing is
    // missed and nothing is fetched.
    let bare = models_root(&[]);
    let shared = rt
        .block_on(embeddings::ensure_and_load(
            bare.path(),
            ModelChoice::Auto,
            MediaNeeds::NONE,
        ))
        .unwrap();
    {
        let mut guard = shared.lock().unwrap();
        assert_eq!(guard.model_id(), MINILM_MODEL_ID);
        assert_eq!(guard.embedding_dim(), embeddings::MINILM_DIM);
        assert_eq!(guard.media_model(), None);
        assert_eq!(guard.media_support(), MediaNeeds::NONE);
        assert_eq!(guard.encode_media_query("a red picture").unwrap(), None);
        assert_eq!(
            guard.profile(),
            ModelProfile::text_only(MINILM_MODEL_ID, embeddings::MINILM_DIM)
        );
    }
    assert!(
        !bare.path().join(gemma2::CACHE_DIRNAME).exists(),
        "a text-only workspace must not start a download"
    );

    let root = models_root(&all_files());
    for needs in [IMAGES, AUDIO_ONLY] {
        let shared = rt
            .block_on(embeddings::ensure_and_load(
                root.path(),
                ModelChoice::Auto,
                needs,
            ))
            .unwrap();
        let mut guard = shared.lock().unwrap();
        assert_eq!(guard.model_id(), MINILM_MODEL_ID, "text stays on MiniLM");
        assert_eq!(guard.embedding_dim(), embeddings::MINILM_DIM);
        assert_eq!(
            guard.media_model(),
            Some((gemma2::MODEL_ID, gemma2::EMBED_DIM))
        );
        assert_eq!(guard.media_support(), needs);
        // Two models, two spaces: a query is embedded once for each.
        assert_eq!(
            guard.encode_query("retry upload").unwrap().len(),
            embeddings::MINILM_DIM
        );
        assert_eq!(
            guard
                .encode_media_query("a red picture")
                .unwrap()
                .expect("the media engine answers media queries")
                .len(),
            gemma2::EMBED_DIM
        );
    }

    // A process that started on a text-only workspace, joined by one with
    // images and then one with audio.
    let shared = rt
        .block_on(embeddings::ensure_and_load(
            root.path(),
            ModelChoice::Auto,
            MediaNeeds::NONE,
        ))
        .unwrap();
    assert_eq!(shared.lock().unwrap().media_model(), None);
    rt.block_on(embeddings::extend_media(&shared, IMAGES));
    assert_eq!(shared.lock().unwrap().media_support(), IMAGES);
    rt.block_on(embeddings::extend_media(&shared, AUDIO_ONLY));
    assert_eq!(shared.lock().unwrap().media_support(), BOTH);
    assert_eq!(shared.lock().unwrap().model_id(), MINILM_MODEL_ID);
}

/// The two explicit choices. MiniLM by name is MiniLM and nothing else, media
/// or not; EmbeddingGemma 2 by name is that model for text as well.
#[test]
fn an_explicit_model_choice_is_taken_literally() {
    let root = models_root(&all_files());
    let rt = runtime();

    let minilm = rt
        .block_on(embeddings::ensure_and_load(
            root.path(),
            ModelChoice::MiniLm,
            BOTH,
        ))
        .unwrap();
    rt.block_on(embeddings::extend_media(&minilm, BOTH));
    {
        let guard = minilm.lock().unwrap();
        assert_eq!(guard.model_id(), MINILM_MODEL_ID);
        assert_eq!(guard.media_model(), None);
        assert_eq!(guard.media_support(), MediaNeeds::NONE);
    }

    for needs in [MediaNeeds::NONE, IMAGES, AUDIO_ONLY] {
        let shared = rt
            .block_on(embeddings::ensure_and_load(
                root.path(),
                ModelChoice::Gemma2,
                needs,
            ))
            .unwrap();
        let guard = shared.lock().unwrap();
        assert_eq!(guard.model_id(), gemma2::MODEL_ID);
        assert_eq!(guard.embedding_dim(), gemma2::EMBED_DIM);
        assert_eq!(guard.media_support(), needs);
        assert_eq!(
            guard.profile(),
            ModelProfile::single(gemma2::MODEL_ID, gemma2::EMBED_DIM, needs)
        );
    }
}

/// An EmbeddingGemma 2 this runtime cannot load costs content search over
/// media and nothing else: under the default choice text search comes up on
/// MiniLM regardless, and only an explicit request for EmbeddingGemma 2
/// fails. Nothing is downloaded here — every file "exists", they are just
/// not models.
#[test]
fn an_unloadable_gemma2_costs_media_search_only_unless_forced() {
    let root = models_root(&[]);
    let gemma_dir = root.path().join(gemma2::CACHE_DIRNAME);
    std::fs::create_dir_all(&gemma_dir).unwrap();
    for kind in [None, Some(MediaKind::Image), Some(MediaKind::Audio)] {
        for (_, local, _, _) in gemma2::component_files(kind) {
            std::fs::write(gemma_dir.join(local), b"not a model").unwrap();
        }
    }
    let rt = runtime();

    let forced = rt.block_on(embeddings::ensure_and_load(
        root.path(),
        ModelChoice::Gemma2,
        BOTH,
    ));
    assert!(
        forced.is_err(),
        "an explicit choice is never swapped for MiniLM"
    );

    let shared = rt
        .block_on(embeddings::ensure_and_load(
            root.path(),
            ModelChoice::Auto,
            BOTH,
        ))
        .expect("text search does not depend on EmbeddingGemma 2");
    {
        let mut guard = shared.lock().unwrap();
        assert_eq!(guard.model_id(), MINILM_MODEL_ID);
        assert_eq!(guard.media_model(), None);
        assert_eq!(guard.media_support(), MediaNeeds::NONE);
        assert_eq!(
            guard.encode_query("retry upload").unwrap().len(),
            embeddings::MINILM_DIM
        );
    }
    // The failure is remembered: opening another workspace with media must
    // not load the same unusable files again (it would, here, find them
    // replaced by a working model).
    for name in all_files() {
        std::fs::copy(fixture_dir().join(name), gemma_dir.join(name)).unwrap();
    }
    rt.block_on(embeddings::extend_media(&shared, BOTH));
    assert_eq!(shared.lock().unwrap().media_model(), None);
    assert!(
        gemma_dir.join("model_q4.onnx").exists(),
        "hash-checked EmbeddingGemma 2 files are not deleted on a load error"
    );

    // A workspace whose media cannot get content vectors is still indexed:
    // code by MiniLM, media by name.
    let ws = workspace();
    let ws_root = ws.path().to_string_lossy().to_string();
    let db = IndexDb::open(&ws.path().join("index.db")).unwrap();
    indexer::scan_workspace(&db, &ws_root, None, None, None).unwrap();
    let (_, vectors) = indexer::generate_all_embeddings(&db, &shared, None, &ws_root).unwrap();
    assert!(vectors >= 1);
    assert_eq!(db.media_embedding_count().unwrap(), 0);
    assert_eq!(db.embedding_pending_files().unwrap(), 0);
    assert_eq!(db.search_media("app logo", None, 3).unwrap().len(), 1);
}

/// The default pairing, end to end: one workspace, two models, one index.
/// Code gets MiniLM vectors, media gets EmbeddingGemma 2 vectors, each kind
/// of search compares like with like, and re-opening re-embeds nothing.
#[test]
fn the_default_pairing_indexes_code_and_media_side_by_side() {
    let models = models_root(&all_files());
    let rt = runtime();
    let ws = workspace();
    let root = ws.path().to_string_lossy().to_string();
    let db = IndexDb::open(&ws.path().join("index.db")).unwrap();
    indexer::scan_workspace(&db, &root, None, None, None).unwrap();

    let shared = rt
        .block_on(embeddings::ensure_and_load(
            models.path(),
            ModelChoice::Auto,
            BOTH,
        ))
        .unwrap();
    let (_, vectors) = indexer::generate_all_embeddings(&db, &shared, None, &root).unwrap();
    assert!(vectors >= 3, "code plus two media files, got {vectors}");
    assert_eq!(db.media_embedding_count().unwrap(), 2);
    assert_eq!(db.embedding_pending_files().unwrap(), 0);
    assert_eq!(db.embedding_model().as_deref(), Some(MINILM_MODEL_ID));
    assert_eq!(
        db.media_embedding_model().as_deref(),
        Some(gemma2::MODEL_ID)
    );

    // Code search: a MiniLM query against MiniLM vectors. The 768-d media
    // vectors in the same table are not candidates and not an error.
    let q = shared
        .lock()
        .unwrap()
        .encode_query("retry failed upload")
        .unwrap();
    assert_eq!(q.len(), embeddings::MINILM_DIM);
    let code = db.search_hybrid("retry failed upload", Some(&q), 10).unwrap();
    assert_eq!(code[0].name, "retry_failed_upload");
    assert_eq!(
        code[0].match_type, "hybrid",
        "the vector leg took part: {code:?}"
    );
    assert!(code.iter().all(|r| r.kind != "image" && r.kind != "audio"));

    // Media search: the logo's own vector is the best possible content
    // match, and it is found by content.
    let logo = ws.path().join("assets").join("app-logo.png");
    let logo_vec = embeddings::embed_media_file(&shared, &logo, MediaKind::Image).unwrap();
    assert_eq!(logo_vec.len(), gemma2::EMBED_DIM);
    let hits = db.search_media("zzz", Some(&logo_vec), 3).unwrap();
    assert!(hits[0].file_path.ends_with("app-logo.png"));
    assert_eq!(hits[0].match_type, "semantic");
    // A MiniLM vector handed to media search by mistake matches nothing by
    // content rather than everything by accident.
    let wrong = db.search_media("zzz", Some(&q), 3).unwrap();
    assert!(wrong.is_empty(), "{wrong:?}");

    // Re-opening — same process or the next one — is a no-op.
    let (_, again) = indexer::generate_all_embeddings(&db, &shared, None, &root).unwrap();
    assert_eq!(again, 0);
    let reopened = rt
        .block_on(embeddings::ensure_and_load(
            models.path(),
            ModelChoice::Auto,
            BOTH,
        ))
        .unwrap();
    let (_, again) = indexer::generate_all_embeddings(&db, &reopened, None, &root).unwrap();
    assert_eq!(again, 0);

    // Re-opened as a text-only session (no media engine loaded): nothing is
    // re-embedded and the media vectors are still there for next time.
    let text_only = rt
        .block_on(embeddings::ensure_and_load(
            models.path(),
            ModelChoice::MiniLm,
            BOTH,
        ))
        .unwrap();
    let (_, again) = indexer::generate_all_embeddings(&db, &text_only, None, &root).unwrap();
    assert_eq!(again, 0);
    assert_eq!(db.media_embedding_count().unwrap(), 2);

    // Switching text to EmbeddingGemma 2 re-embeds the code and only the
    // code: the media vectors already are that model's.
    let all_gemma = rt
        .block_on(embeddings::ensure_and_load(
            models.path(),
            ModelChoice::Gemma2,
            BOTH,
        ))
        .unwrap();
    let before = db.index_stats().unwrap().2 - 2;
    let (_, redone) = indexer::generate_all_embeddings(&db, &all_gemma, None, &root).unwrap();
    assert_eq!(redone, before, "every code chunk, no media file");
    assert_eq!(db.media_embedding_count().unwrap(), 2);
    assert_eq!(db.embedding_model().as_deref(), Some(gemma2::MODEL_ID));
}

/// The order the server works in: text first, on a model that is there in
/// seconds; media when its model has arrived, which on a first run is
/// minutes later. The pass that ran without the media model must not leave
/// media pending, and the pass after it must embed the media and only that.
#[test]
fn media_is_embedded_by_a_later_pass_once_its_model_arrives() {
    let models = models_root(&all_files());
    let rt = runtime();
    let ws = workspace();
    let root = ws.path().to_string_lossy().to_string();
    let db = IndexDb::open(&ws.path().join("index.db")).unwrap();
    indexer::scan_workspace(&db, &root, None, None, None).unwrap();

    let shared = rt
        .block_on(embeddings::ensure_and_load(
            models.path(),
            ModelChoice::Auto,
            MediaNeeds::NONE,
        ))
        .unwrap();
    let (_, text) = indexer::generate_all_embeddings(&db, &shared, None, &root).unwrap();
    assert!(text >= 1);
    assert_eq!(db.media_embedding_count().unwrap(), 0);
    assert_eq!(db.embedding_pending_files().unwrap(), 0);
    assert_eq!(db.media_embedding_model(), None);

    rt.block_on(embeddings::extend_media(&shared, BOTH));
    let (_, media) = indexer::generate_all_embeddings(&db, &shared, None, &root).unwrap();
    assert_eq!(media, 2, "the picture and the clip; no code again");
    assert_eq!(db.media_embedding_count().unwrap(), 2);
    assert_eq!(db.index_stats().unwrap().2, text + 2);
    assert_eq!(db.embedding_pending_files().unwrap(), 0);
}

/// The text upgrade: MiniLM has written the index; EmbeddingGemma 2, once it
/// is there, embeds the same chunks again beside MiniLM's without touching
/// them — a file at a time, so that a pass which never finished is resumed
/// rather than started over, and a file edited since is redone alone.
#[test]
fn the_upgrade_set_is_filled_beside_the_primary_one_and_kept_current() {
    let models = models_root(&TEXT);
    let rt = runtime();
    let ws = workspace();
    std::fs::write(
        ws.path().join("src").join("session.rs"),
        "/// Signs the user out when the stored session has expired.\n\
         pub fn expire_stale_session(age_minutes: u32) -> bool {\n    age_minutes > 30\n}\n",
    )
    .unwrap();
    let root = ws.path().to_string_lossy().to_string();
    let db = IndexDb::open(&ws.path().join("index.db")).unwrap();
    indexer::scan_workspace(&db, &root, None, None, None).unwrap();

    let shared = rt
        .block_on(embeddings::ensure_and_load(
            models.path(),
            ModelChoice::Auto,
            MediaNeeds::NONE,
        ))
        .unwrap();
    let (_, text) = indexer::generate_all_embeddings(&db, &shared, None, &root).unwrap();
    assert!(text >= 2, "both source files");

    // Before EmbeddingGemma 2 is in the process there is nothing to upgrade
    // with, and saying so is all that happens.
    assert_eq!(shared.lock().unwrap().upgrade_model(), None);
    assert!(indexer::generate_upgrade_embeddings(&db, &shared, None, &root, indexer::UpgradeTurn::default()).is_err());
    assert!(embeddings::encode_upgrade_query(&shared, "expired session").is_err());
    assert_eq!(db.upgrade_state().unwrap().vectors, 0);

    assert!(rt.block_on(embeddings::ensure_text_upgrade(&shared)));
    assert_eq!(
        shared.lock().unwrap().upgrade_model(),
        Some((gemma2::MODEL_ID, gemma2::EMBED_DIM))
    );
    assert_eq!(
        shared.lock().unwrap().model_id(),
        MINILM_MODEL_ID,
        "the text model itself is still MiniLM"
    );

    // A turn that is over before it starts does nothing, and says how much
    // there is still to do.
    let out_of_time = indexer::generate_upgrade_embeddings(
        &db,
        &shared,
        None,
        &root,
        indexer::UpgradeTurn {
            budget: Some(std::time::Duration::ZERO),
            after: None,
        },
    )
    .unwrap();
    assert_eq!(
        (out_of_time.files, out_of_time.remaining, out_of_time.complete),
        (0, 2, false)
    );
    assert!(!db.upgrade_state().unwrap().usable(gemma2::MODEL_ID));

    let seen: Mutex<Vec<(i64, i64)>> = Mutex::new(Vec::new());
    let sink = |p: indexer::IndexProgress| seen.lock().unwrap().push((p.files_indexed, p.total_files));
    let pass =
        indexer::generate_upgrade_embeddings(&db, &shared, Some(&sink), &root, indexer::UpgradeTurn::default())
            .unwrap();
    assert_eq!(*seen.lock().unwrap(), [(1, 2), (2, 2)], "progress counts the workspace's text files");
    assert!(pass.complete);
    assert_eq!((pass.left, pass.remaining), (0, 0));
    assert_eq!(pass.vectors, text, "every chunk MiniLM embedded, and no media");
    let state = db.upgrade_state().unwrap();
    assert_eq!(state.vectors, text);
    assert_eq!(state.pending_files, 0);
    assert!(state.usable(gemma2::MODEL_ID));
    assert_eq!(db.index_stats().unwrap().2, text, "MiniLM's vectors are all still there");
    assert_eq!(db.embedding_model().as_deref(), Some(MINILM_MODEL_ID));

    // A query for each set comes from that set's model.
    let query = embeddings::encode_upgrade_query(&shared, "expired session").unwrap();
    assert_unit(&query);
    assert_eq!(
        embeddings::encode_query(&shared, "expired session").unwrap().len(),
        embeddings::MINILM_DIM
    );
    let hits = db
        .search_hybrid_in(VectorSet::Upgrade, "expire stale session", Some(&query), 5)
        .unwrap();
    assert!(hits.iter().any(|h| h.file_path.ends_with("session.rs")));

    // Nothing changed: a second pass has nothing to do.
    let again = indexer::generate_upgrade_embeddings(&db, &shared, None, &root, indexer::UpgradeTurn::default()).unwrap();
    assert_eq!((again.files, again.vectors, again.complete), (0, 0, true));

    // One file is edited. The watcher re-embeds it with MiniLM at once; the
    // upgrade set is one file behind — still the one searched — and the next
    // pass embeds that file and no other.
    let session = ws.path().join("src").join("session.rs");
    std::fs::write(
        &session,
        "/// Signs the user out when the stored session has expired.\n\
         pub fn expire_stale_session(age_minutes: u32) -> bool {\n    age_minutes > 45\n}\n\n\
         /// Extends a session that is about to expire.\n\
         pub fn renew_session(age_minutes: u32) -> u32 {\n    age_minutes / 2\n}\n",
    )
    .unwrap();
    indexer::reindex_file(
        &db,
        &session.to_string_lossy(),
        Some(&mut *shared.lock().unwrap()),
        Some(&root),
    )
    .unwrap();
    let state = db.upgrade_state().unwrap();
    assert_eq!(state.pending_files, 1);
    assert!(state.usable(gemma2::MODEL_ID));
    let behind = state.vectors;
    assert!(behind < text, "the edited file's old vectors are gone");

    let pass = indexer::generate_upgrade_embeddings(&db, &shared, None, &root, indexer::UpgradeTurn::default()).unwrap();
    assert_eq!(pass.files, 1);
    assert!(pass.complete);
    let state = db.upgrade_state().unwrap();
    assert_eq!(state.pending_files, 0);
    assert_eq!(state.vectors, behind + pass.vectors);
    assert_eq!(
        state.vectors,
        db.index_stats().unwrap().2,
        "both sets hold the same chunks again"
    );
}

/// Only the default choice has an upgrade: MiniLM asked for by name stays
/// MiniLM, and EmbeddingGemma 2 as the text model has nothing to upgrade to.
#[test]
fn an_explicit_model_choice_has_no_upgrade() {
    let models = models_root(&TEXT);
    let rt = runtime();
    for choice in [ModelChoice::MiniLm, ModelChoice::Gemma2] {
        let shared = rt
            .block_on(embeddings::ensure_and_load(
                models.path(),
                choice,
                MediaNeeds::NONE,
            ))
            .unwrap();
        assert!(!rt.block_on(embeddings::ensure_text_upgrade(&shared)), "{choice:?}");
        assert_eq!(shared.lock().unwrap().upgrade_model(), None, "{choice:?}");
        assert!(embeddings::encode_upgrade_documents(&shared, &["fn main() {}"]).is_err());
    }
}

/// An EmbeddingGemma 2 that cannot be loaded costs the upgrade and nothing
/// else — and is not tried again for every workspace.
#[test]
fn an_unloadable_gemma2_leaves_text_on_minilm_without_an_upgrade() {
    let models = models_root(&[]);
    let dir = models.path().join(gemma2::CACHE_DIRNAME);
    std::fs::create_dir_all(&dir).unwrap();
    for (_, local, _, _) in gemma2::component_files(None) {
        std::fs::write(dir.join(local), b"not a model").unwrap();
    }
    let rt = runtime();
    let shared = rt
        .block_on(embeddings::ensure_and_load(
            models.path(),
            ModelChoice::Auto,
            MediaNeeds::NONE,
        ))
        .unwrap();
    assert!(!rt.block_on(embeddings::ensure_text_upgrade(&shared)));
    assert!(!rt.block_on(embeddings::ensure_text_upgrade(&shared)));
    assert_eq!(shared.lock().unwrap().upgrade_model(), None);
    assert_eq!(
        embeddings::encode_query(&shared, "expired session").unwrap().len(),
        embeddings::MINILM_DIM
    );
}

/// One turn takes up where the last one stopped, so a file that cannot be
/// finished is not where every turn begins.
#[test]
fn an_upgrade_turn_starts_after_the_file_the_last_one_ended_on() {
    let (db, shared, ws, root) = upgrade_ready_index(&[
        ("a.rs", "/// First.\npub fn first_thing() -> u32 { 1 }\n"),
        ("b.rs", "/// Second.\npub fn second_thing() -> u32 { 2 }\n"),
        ("c.rs", "/// Third.\npub fn third_thing() -> u32 { 3 }\n"),
    ]);
    let ids: Vec<i64> = ["a.rs", "b.rs", "c.rs"]
        .iter()
        .map(|name| {
            let path = ws.path().join("src").join(name);
            db.file_by_path(&path.to_string_lossy()).unwrap().unwrap().id
        })
        .collect();
    let mut ordered = ids.clone();
    ordered.sort_unstable();

    let turn = indexer::UpgradeTurn {
        budget: None,
        after: Some(ordered[0]),
    };
    let pass = indexer::generate_upgrade_embeddings(&db, &shared, None, &root, turn).unwrap();
    assert_eq!((pass.files, pass.remaining), (2, 0));
    assert_eq!(pass.last_file, Some(ordered[2]));
    assert!(!pass.complete, "the first file is still to do");
    assert_eq!(db.upgrade_state().unwrap().pending_files, 1);

    // The round after it starts from the top and finds only that one.
    let pass =
        indexer::generate_upgrade_embeddings(&db, &shared, None, &root, indexer::UpgradeTurn::default()).unwrap();
    assert_eq!((pass.files, pass.last_file, pass.complete), (1, Some(ordered[0]), true));
}

/// A file edited and then put back as it was has its old hash and none of
/// its old vectors: the upgrade set must see it as a file to embed.
#[test]
fn a_file_edited_and_put_back_is_embedded_again() {
    let original = "/// Signs the user out when the stored session has expired.\n\
                    pub fn expire_stale_session(age_minutes: u32) -> bool {\n    age_minutes > 30\n}\n";
    let (db, shared, ws, root) = upgrade_ready_index(&[("session.rs", original)]);
    let pass =
        indexer::generate_upgrade_embeddings(&db, &shared, None, &root, indexer::UpgradeTurn::default()).unwrap();
    assert!(pass.complete);
    let vectors = db.upgrade_state().unwrap().vectors;
    assert!(vectors >= 1);

    let path = ws.path().join("src").join("session.rs");
    let reindex = |content: &str| {
        std::fs::write(&path, content).unwrap();
        indexer::reindex_file(
            &db,
            &path.to_string_lossy(),
            Some(&mut *shared.lock().unwrap()),
            Some(&root),
        )
        .unwrap();
    };
    reindex("/// Something else entirely.\npub fn unrelated() -> u32 { 7 }\n");
    reindex(original);

    let state = db.upgrade_state().unwrap();
    assert_eq!(state.vectors, 0, "the old symbols took their vectors with them");
    assert_eq!(state.pending_files, 1, "the same hash as before is not the same as done");
    let pass =
        indexer::generate_upgrade_embeddings(&db, &shared, None, &root, indexer::UpgradeTurn::default()).unwrap();
    assert_eq!((pass.files, pass.complete), (1, true));
    assert_eq!(db.upgrade_state().unwrap().vectors, vectors);
}

/// The upgrade runs with the watcher free to take edits in — here, the same
/// file over and over, in both directions, while its chunks are at the
/// model. Whatever the interleaving, what is left once the dust has settled
/// is exactly what a fresh index of the final content holds: no file marked
/// done that is not, and no vector of an older text.
#[test]
fn a_file_that_keeps_changing_under_the_upgrade_ends_up_right() {
    let body = |tag: &str| -> String {
        (0..40)
            .map(|i| {
                format!(
                    "/// Handles step {i} of the {tag} pipeline.\npub fn {tag}_step_{i}(input: u32) -> u32 {{\n    input + {i}\n}}\n\n"
                )
            })
            .collect()
    };
    let (first, second) = (body("ingest"), body("export"));
    let (db, shared, ws, root) = upgrade_ready_index(&[
        ("pipeline.rs", first.as_str()),
        ("steady.rs", "/// Never edited.\npub fn steady_state() -> u32 { 4 }\n"),
    ]);
    let path = ws.path().join("src").join("pipeline.rs");

    std::thread::scope(|scope| {
        let upgrade = scope.spawn(|| {
            // Rounds, as the worker runs them, for as long as edits come in.
            let mut rounds = 0;
            loop {
                let pass =
                    indexer::generate_upgrade_embeddings(&db, &shared, None, &root, indexer::UpgradeTurn::default())
                        .unwrap();
                rounds += 1;
                if (pass.left == 0 && rounds >= 4) || rounds >= 40 {
                    return;
                }
            }
        });
        for round in 0..6 {
            std::thread::sleep(std::time::Duration::from_millis(40 + 25 * round));
            let content = if round % 2 == 0 { &second } else { &first };
            // What the watcher does: under the indexing semaphore.
            let editing = loop {
                match claudinio_code_intel::INDEX_SEMAPHORE.try_acquire() {
                    Ok(permit) => break permit,
                    Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
                }
            };
            std::fs::write(&path, content).unwrap();
            indexer::reindex_file(
                &db,
                &path.to_string_lossy(),
                Some(&mut *shared.lock().unwrap()),
                Some(&root),
            )
            .unwrap();
            drop(editing);
        }
        upgrade.join().unwrap();
    });
    // The last edit put the first content back.
    let last =
        indexer::generate_upgrade_embeddings(&db, &shared, None, &root, indexer::UpgradeTurn::default()).unwrap();
    assert!(last.complete);
    let state = db.upgrade_state().unwrap();
    assert_eq!(state.pending_files, 0);
    assert_eq!(state.vectors, db.index_stats().unwrap().2, "one upgrade vector per chunk, no more");

    let (fresh_db, _, _fresh_ws, fresh_root) = upgrade_ready_index(&[
        ("pipeline.rs", first.as_str()),
        ("steady.rs", "/// Never edited.\npub fn steady_state() -> u32 { 4 }\n"),
    ]);
    indexer::generate_upgrade_embeddings(&fresh_db, &shared, None, &fresh_root, indexer::UpgradeTurn::default())
        .unwrap();
    assert_eq!(fresh_db.upgrade_state().unwrap().vectors, state.vectors);
    // Vector for vector: every chunk of the final content has, in the index
    // that was edited under the upgrade, the vector a fresh index gives it.
    let vectors = |db: &IndexDb| -> Vec<(String, i64, Vec<f32>)> {
        let mut rows: Vec<(String, i64, Vec<f32>)> = db
            .load_all_embeddings_in(VectorSet::Upgrade)
            .unwrap()
            .into_iter()
            .map(|(symbol, start, _, vector)| (symbol.name, start, vector))
            .collect();
        rows.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        rows
    };
    let (stressed, fresh) = (vectors(&db), vectors(&fresh_db));
    assert!(fresh.len() >= 41, "forty steps and the steady file");
    assert_eq!(stressed.len(), fresh.len());
    for (s, f) in stressed.iter().zip(&fresh) {
        assert_eq!((&s.0, s.1), (&f.0, f.1));
        assert!(dot(&s.2, &f.2) > 0.9999, "{} holds a vector of another text", s.0);
    }
}

/// An index of the given `src/` files, embedded by MiniLM, with the upgrade
/// model loaded and the upgrade set still empty.
fn upgrade_ready_index(files: &[(&str, &str)]) -> (IndexDb, SharedEmbedder, tempfile::TempDir, String) {
    let models = models_root(&TEXT);
    let rt = runtime();
    let ws = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(ws.path().join("src")).unwrap();
    for (name, content) in files {
        std::fs::write(ws.path().join("src").join(name), content).unwrap();
    }
    let root = ws.path().to_string_lossy().to_string();
    let db = IndexDb::open(&ws.path().join("index.db")).unwrap();
    indexer::scan_workspace(&db, &root, None, None, None).unwrap();
    let shared = rt
        .block_on(embeddings::ensure_and_load(
            models.path(),
            ModelChoice::Auto,
            MediaNeeds::NONE,
        ))
        .unwrap();
    indexer::generate_all_embeddings(&db, &shared, None, &root).unwrap();
    assert!(rt.block_on(embeddings::ensure_text_upgrade(&shared)));
    (db, shared, ws, root)
}
