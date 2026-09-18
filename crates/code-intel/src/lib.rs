//! Claudinio Code's code intelligence, as a library.
//!
//! `parser` turns source into symbols with tree-sitter (77 grammars), `db`
//! keeps them in SQLite next to an FTS5 table and embedding rows, `embeddings`
//! runs all-MiniLM-L6-v2 in-process, and `IndexDb::search_hybrid` fuses BM25
//! and vector ranks with reciprocal rank fusion. Code is never sent anywhere
//! to be indexed.
//!
//! Extracted from `claudin-io/claudinio-code` `src-tauri/src/code_intel/`; the
//! only change is that progress is reported through [`indexer::ProgressSink`]
//! instead of Tauri events, and the file watcher takes its embedder from a
//! [`watcher::EmbedderSource`] instead of the app state.

pub mod db;
pub mod download;
pub mod embeddings;
pub mod fallback;
pub mod indexer;
pub mod parser;
pub mod text;
pub mod thread_priority;
pub mod watcher;

/// Only one workspace indexes at a time: parallel scans + embedding runs peg
/// every core and hammer slow/network drives.
pub static INDEX_SEMAPHORE: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);
