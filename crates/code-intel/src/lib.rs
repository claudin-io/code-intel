//! Claudinio Code's code intelligence, as a library.
//!
//! `parser` turns source into symbols with tree-sitter (77 grammars), `db`
//! keeps them in SQLite next to an FTS5 table and embedding rows, `embeddings`
//! runs the embedding model in-process, and `IndexDb::search_hybrid` fuses
//! BM25 and vector ranks with reciprocal rank fusion. Code is never sent
//! anywhere to be indexed.
//!
//! The embedding model is EmbeddingGemma 2 in the ONNX Runtime build
//! (`gemma2`: text, plus images and audio when the workspace has any) and
//! all-MiniLM-L6-v2 everywhere else — and wherever the larger model cannot be
//! downloaded or loaded. `media` lists a workspace's images and audio;
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
