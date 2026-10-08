//! Claudinio Code's code intelligence, as a library.
//!
//! `parser` turns source into symbols with tree-sitter (77 grammars), `db`
//! keeps them in SQLite next to an FTS5 table and embedding rows, `embeddings`
//! runs the embedding model in-process, and `IndexDb::search_hybrid` fuses
//! BM25 and vector ranks with reciprocal rank fusion. Code is never sent
//! anywhere to be indexed.
//!
//! Text is embedded with all-MiniLM-L6-v2 in every build: search works
//! minutes after a workspace is opened. The ONNX Runtime build also has
//! EmbeddingGemma 2 (`gemma2`), slower by a factor of thirty and somewhat
//! better. It embeds a workspace's images and audio, in one space with the
//! queries that describe them, and it embeds the text a second time in the
//! background — the index's upgrade set (`db::VectorSet`), which search moves
//! to once it is complete. It can also be made the text model outright
//! (`embeddings::ModelChoice`). `media` lists a workspace's images and audio;
//! `media_prep` decodes them for the encoders.
//!
//! Extracted from `claudin-io/claudinio-code` `src-tauri/src/code_intel/`; the
//! only change is that progress is reported through [`indexer::ProgressSink`]
//! instead of Tauri events, and the file watcher takes its embedder from a
//! [`watcher::EmbedderSource`] instead of the app state.

pub mod db;
pub mod download;
pub mod embeddings;
pub mod fallback;
#[cfg(feature = "embeddings")]
pub mod gemma2;
pub mod indexer;
pub mod media;
#[cfg(feature = "embeddings")]
pub mod media_prep;
pub mod parser;
pub mod text;
pub mod thread_priority;
pub mod watcher;

/// Only one workspace indexes at a time: parallel scans + embedding runs peg
/// every core and hammer slow/network drives.
pub static INDEX_SEMAPHORE: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);

/// Only one workspace at a time has its upgrade set filled in
/// (`indexer::generate_upgrade_embeddings`): that pass is the heaviest thing
/// this crate does, by far, and two of them would halve each other. It is
/// deliberately not `INDEX_SEMAPHORE` — holding that for an hour would keep
/// the watcher, and the first index of every other workspace, waiting.
pub static UPGRADE_SEMAPHORE: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);
