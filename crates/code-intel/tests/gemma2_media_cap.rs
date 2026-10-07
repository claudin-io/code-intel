//! The per-workspace cap on media content vectors. Alone in its own test
//! binary because it sets `CODE_INTEL_MEDIA_MAX` for the whole process.
#![cfg(feature = "embeddings")]

use claudinio_code_intel::db::IndexDb;
use claudinio_code_intel::embeddings::{self, CodeEmbedder, SharedEmbedder};
use claudinio_code_intel::indexer;
use claudinio_code_intel::media::MediaNeeds;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// With a cap of one: one image gets a vector, the others stay findable by
/// name and are not left pending — and when the embedded one is deleted, its
/// slot goes to the next image instead of staying empty forever.
#[test]
fn the_cap_is_respected_and_a_freed_slot_is_reused() {
    // SAFETY: the only test in this binary, so nothing else reads the
    // environment concurrently.
    unsafe { std::env::set_var("CODE_INTEL_MEDIA_MAX", "1") };

    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma2-synthetic");
    let shared: SharedEmbedder = Arc::new(Mutex::new(CodeEmbedder::load_gemma2(&model).unwrap()));
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(embeddings::extend_media(
            &shared,
            MediaNeeds {
                images: true,
                audio: false,
            },
        ));

    let ws = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(ws.path().join("assets")).unwrap();
    for (name, rgb) in [
        ("avatar.png", [200u8, 10, 10]),
        ("badge.png", [10, 200, 10]),
        ("cover.png", [10, 10, 200]),
    ] {
        image::RgbImage::from_pixel(96, 96, image::Rgb(rgb))
            .save(ws.path().join("assets").join(name))
            .unwrap();
    }
    let root = ws.path().to_string_lossy().to_string();
    let dbdir = tempfile::tempdir().unwrap();
    let db = IndexDb::open(&dbdir.path().join("index.db")).unwrap();

    indexer::scan_workspace(&db, &root, None, None, None).unwrap();
    indexer::generate_all_embeddings(&db, &shared, None, &root).unwrap();
    assert_eq!(db.media_embedding_count().unwrap(), 1, "the cap");
    assert_eq!(
        db.embedding_pending_files().unwrap(),
        0,
        "the rest are done, by name only"
    );
    assert_eq!(
        db.search_media("badge", None, 5).unwrap().len(),
        1,
        "and still findable"
    );

    // Opening the workspace again changes nothing while the slot is taken.
    indexer::scan_workspace(&db, &root, None, None, None).unwrap();
    indexer::generate_all_embeddings(&db, &shared, None, &root).unwrap();
    assert_eq!(db.media_embedding_count().unwrap(), 1);

    std::fs::remove_file(ws.path().join("assets/avatar.png")).unwrap();
    indexer::scan_workspace(&db, &root, None, None, None).unwrap();
    assert_eq!(
        db.media_embedding_count().unwrap(),
        0,
        "avatar.png took its vector with it"
    );
    let (_, added) = indexer::generate_all_embeddings(&db, &shared, None, &root).unwrap();
    assert_eq!(added, 1, "the freed slot goes to the next image");
    assert_eq!(db.media_embedding_count().unwrap(), 1);
    assert_eq!(db.embedding_pending_files().unwrap(), 0);
}
