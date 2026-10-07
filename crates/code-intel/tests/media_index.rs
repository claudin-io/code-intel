//! Media in the index, without any embedding model: what every build does,
//! including the ones that will never have a vision or audio encoder.

use claudinio_code_intel::db::IndexDb;
use claudinio_code_intel::embeddings::MINILM_DIM;
use claudinio_code_intel::indexer;
use claudinio_code_intel::media::{MediaKind, MediaNeeds};
use std::path::Path;

fn workspace_with_logos(n: usize) -> tempfile::TempDir {
    let ws = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(ws.path().join("src")).unwrap();
    std::fs::create_dir_all(ws.path().join("assets")).unwrap();
    std::fs::write(
        ws.path().join("src").join("logo.rs"),
        "/// Draws the logo at the requested size, scaled for the display density.\n\
         pub fn render_logo(size: u32) -> u32 { size * 2 }\n\n\
         /// Rendered logos by size, so a redraw does not rasterize again.\n\
         pub struct LogoCache { entries: u32 }\n",
    )
    .unwrap();
    for i in 0..n {
        // Not decodable — nothing here decodes; the index goes by name.
        std::fs::write(ws.path().join(format!("assets/logo-{i:02}.png")), b"png").unwrap();
    }
    std::fs::write(ws.path().join("assets").join("startup-chime.wav"), b"wav").unwrap();
    ws
}

fn open(ws: &tempfile::TempDir) -> (tempfile::TempDir, IndexDb, String) {
    let dbdir = tempfile::tempdir().unwrap();
    let db = IndexDb::open(&dbdir.path().join("index.db")).unwrap();
    let root = ws.path().to_string_lossy().to_string();
    indexer::scan_workspace(&db, &root, None, None, None).unwrap();
    (dbdir, db, root)
}

/// Thirty `logo-NN.png` must not turn "logo" into thirty pictures and no
/// function: the symbol tools answer with code, media has its own list.
#[test]
fn media_never_crowds_code_out_of_the_symbol_tools() {
    let ws = workspace_with_logos(30);
    let (_dbdir, db, _root) = open(&ws);
    assert_eq!(db.media_file_counts().unwrap(), (30, 1));

    let hits = db.search_symbols("logo", 20).unwrap();
    assert!(hits.iter().any(|h| h.name == "render_logo"), "{hits:?}");
    assert!(
        hits.iter().all(|h| h.kind != "image" && h.kind != "audio"),
        "{hits:?}"
    );
    assert!(db.search_symbols("png", 20).unwrap().is_empty());
    assert!(
        db.lookup_symbols_exact("logo-00.png", 20)
            .unwrap()
            .is_empty()
    );

    let code = db.search_hybrid("logo", None, 15).unwrap();
    assert!(!code.is_empty());
    assert!(
        code.iter().all(|r| r.kind != "image" && r.kind != "audio"),
        "{code:?}"
    );

    // The pictures are there for the search that is about pictures.
    let media = db.search_media("logo", None, 3).unwrap();
    assert_eq!(media.len(), 3);
    assert!(
        media
            .iter()
            .all(|m| m.kind == "image" && m.match_type == "lexical")
    );
    let chime = db.search_media("startup chime", None, 3).unwrap();
    assert_eq!(chime.len(), 1);
    assert_eq!(chime[0].kind, "audio");
    // One incidental word shared with a long question is not a match.
    assert!(
        db.search_media(
            "where is the startup sequence of the server configured",
            None,
            3
        )
        .unwrap()
        .is_empty()
    );
}

/// A rescan follows the filesystem: unchanged media is left alone, deleted
/// media leaves the index, and with no encoder anywhere nothing stays
/// "pending" once a (text-only) embedding pass has run.
#[test]
fn rescans_follow_media_on_disk() {
    let ws = workspace_with_logos(3);
    let (_dbdir, db, root) = open(&ws);
    let logo = ws.path().join("assets").join("logo-00.png");
    let logo_str = logo.to_string_lossy().to_string();
    let before = db.file_by_path(&logo_str).unwrap().expect("indexed");
    assert_eq!(before.language.as_deref(), Some("image"));

    assert!(
        !indexer::index_media_file(&db, &logo_str, MediaKind::Image).unwrap(),
        "unchanged"
    );
    indexer::scan_workspace(&db, &root, None, None, None).unwrap();
    assert_eq!(db.file_by_path(&logo_str).unwrap().unwrap().id, before.id);

    std::fs::remove_file(&logo).unwrap();
    indexer::scan_workspace(&db, &root, None, None, None).unwrap();
    assert!(db.file_by_path(&logo_str).unwrap().is_none());
    assert_eq!(db.media_file_counts().unwrap(), (2, 1));
    assert!(
        db.search_media("logo 00", None, 5)
            .unwrap()
            .iter()
            .all(|m| !m.file_path.ends_with("logo-00.png"))
    );

    // The watcher's path for a deleted and for a new media file, no embedder.
    let chime = ws.path().join("assets").join("startup-chime.wav");
    std::fs::remove_file(&chime).unwrap();
    indexer::reindex_file(&db, &chime.to_string_lossy(), None, Some(&root)).unwrap();
    assert_eq!(db.media_file_counts().unwrap(), (2, 0));
    let added = ws.path().join("assets").join("banner.webp");
    std::fs::write(&added, b"webp").unwrap();
    indexer::reindex_file(&db, &added.to_string_lossy(), None, Some(&root)).unwrap();
    assert_eq!(db.media_file_counts().unwrap(), (3, 0));
    assert_eq!(db.search_media("banner", None, 3).unwrap().len(), 1);
}

/// An image re-exported at the same dimensions keeps its byte size, and the
/// save can land in the same second as the last one. It is still a change.
#[test]
fn a_same_size_edit_is_noticed() {
    let ws = tempfile::tempdir().unwrap();
    let tile = ws.path().join("tile.bmp");
    std::fs::write(&tile, [1u8; 64]).unwrap();
    let stamp =
        std::time::SystemTime::UNIX_EPOCH + std::time::Duration::new(1_800_000_000, 100_000_000);
    std::fs::File::options()
        .write(true)
        .open(&tile)
        .unwrap()
        .set_modified(stamp)
        .unwrap();

    let dbdir = tempfile::tempdir().unwrap();
    let db = IndexDb::open(&dbdir.path().join("index.db")).unwrap();
    let path = tile.to_string_lossy().to_string();
    assert!(indexer::index_media_file(&db, &path, MediaKind::Image).unwrap());
    assert!(!indexer::index_media_file(&db, &path, MediaKind::Image).unwrap());

    // Same length, same second, 300 ms later.
    std::fs::write(&tile, [2u8; 64]).unwrap();
    let later = stamp + std::time::Duration::from_millis(300);
    std::fs::File::options()
        .write(true)
        .open(&tile)
        .unwrap()
        .set_modified(later)
        .unwrap();
    let kept = std::fs::metadata(&tile).unwrap().modified().unwrap();
    if kept == stamp {
        eprintln!("filesystem keeps whole seconds only; nothing to tell the two saves apart");
        return;
    }
    assert!(
        indexer::index_media_file(&db, &path, MediaKind::Image).unwrap(),
        "the re-save must be re-registered"
    );
}

/// An index holding one embedded file. `written_by` is the model that built
/// it, recorded the way the embedding pass records it — before the first
/// vector lands; `None` is an index from before models were recorded.
fn db_with_vector(dim: usize, written_by: Option<&str>) -> (tempfile::TempDir, IndexDb) {
    let dbdir = tempfile::tempdir().unwrap();
    let db = IndexDb::open(&dbdir.path().join("index.db")).unwrap();
    if let Some(model) = written_by {
        db.reconcile_embedding_model(model, dim, MediaNeeds::NONE)
            .unwrap();
    }
    let fid = db.upsert_file("/ws/a.rs", "rust", "h1", 0, 10).unwrap();
    let sid = db
        .insert_symbol(fid, "foo", "function_item", None, 1, 0, 2, 0, None)
        .unwrap();
    db.insert_chunk(
        sid,
        0,
        1,
        2,
        "function_item: foo | fn foo() {}",
        "foo",
        "a",
        "fn foo",
    )
    .unwrap();
    db.upsert_embedding(sid, 0, 1, 2, &vec![0.05f32; dim])
        .unwrap();
    db.set_embed_hash(fid, "h1").unwrap();
    (dbdir, db)
}

/// An index never mixes two models' vectors, whichever way the change is
/// noticed: by the recorded id, by an index from before ids were recorded
/// (it can only hold MiniLM), or by vectors of the wrong size written by a
/// binary that records nothing.
#[test]
fn a_model_change_re_embeds_instead_of_mixing_vectors() {
    const G2: &str = "embeddinggemma-2-q4-768";
    const MINILM: &str = "all-MiniLM-L6-v2";

    // Same model, same size: nothing is touched.
    let (_d, db) = db_with_vector(MINILM_DIM, None);
    assert_eq!(
        db.embedding_model().as_deref(),
        Some(MINILM),
        "an unrecorded index is MiniLM's"
    );
    let reset = db
        .reconcile_embedding_model(MINILM, MINILM_DIM, MediaNeeds::NONE)
        .unwrap();
    assert!(!reset.model_changed);
    assert_eq!(db.index_stats().unwrap().2, 1);
    assert_eq!(db.embedding_pending_files().unwrap(), 0);

    // The upgrade: a pre-existing MiniLM index opened by EmbeddingGemma 2.
    let reset = db
        .reconcile_embedding_model(G2, 768, MediaNeeds::NONE)
        .unwrap();
    assert!(reset.model_changed);
    assert_eq!(db.index_stats().unwrap().2, 0, "old vectors are gone");
    assert_eq!(
        db.embedding_pending_files().unwrap(),
        1,
        "and the file is queued, not forgotten"
    );
    assert_eq!(db.embedding_model().as_deref(), Some(G2));
    // Once is enough.
    assert!(
        !db.reconcile_embedding_model(G2, 768, MediaNeeds::NONE)
            .unwrap()
            .model_changed
    );

    // The fallback, the other way round.
    let (_d, db) = db_with_vector(768, Some(G2));
    assert!(
        !db.reconcile_embedding_model(G2, 768, MediaNeeds::NONE)
            .unwrap()
            .model_changed
    );
    assert_eq!(db.index_stats().unwrap().2, 1);
    assert!(
        db.reconcile_embedding_model(MINILM, MINILM_DIM, MediaNeeds::NONE)
            .unwrap()
            .model_changed
    );
    assert_eq!(db.embedding_pending_files().unwrap(), 1);

    // An older binary wrote a 384-d vector into an index recorded as
    // EmbeddingGemma 2's. The recorded id still matches; the size does not.
    let (_d, db) = db_with_vector(768, Some(G2));
    let fid = db.upsert_file("/ws/b.rs", "rust", "h2", 0, 10).unwrap();
    let sid = db
        .insert_symbol(fid, "bar", "function_item", None, 1, 0, 2, 0, None)
        .unwrap();
    db.upsert_embedding(sid, 0, 1, 2, &vec![0.05f32; MINILM_DIM])
        .unwrap();
    db.set_embed_hash(fid, "h2").unwrap();
    let reset = db
        .reconcile_embedding_model(G2, 768, MediaNeeds::NONE)
        .unwrap();
    assert!(
        reset.model_changed,
        "vectors of another size are another model's"
    );
    assert_eq!(db.index_stats().unwrap().2, 0);
    assert_eq!(db.embedding_pending_files().unwrap(), 2);
}

/// `Path` keeps this file honest about what it claims to test: no feature
/// gate, so it runs in the candle build too.
#[test]
fn this_suite_needs_no_embedding_backend() {
    assert!(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/media_index.rs")
            .exists()
    );
}
