//! Claudinio Code's code intelligence, as a library.
//!
//! `parser` turns source into symbols with tree-sitter (77 grammars), `db`
//! keeps them in SQLite next to an FTS5 table and embedding rows, `embeddings`
//! runs the embedding model in-process, and `IndexDb::search_hybrid` fuses
//! BM25 and vector ranks with reciprocal rank fusion. Code is never sent
//! anywhere to be indexed.
//!
//! Text is embedded with all-MiniLM-L6-v2 in every build. In the ONNX Runtime
//! build, a workspace with images or audio also gets EmbeddingGemma 2
//! (`gemma2`), which places those in one space with the queries that
//! describe them; it can be made the text model as well
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
