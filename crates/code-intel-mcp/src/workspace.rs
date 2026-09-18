//! One indexed workspace: its SQLite index, the shared embedder, the file
//! watcher that keeps the index fresh, and the phase the pipeline is in.
//!
//! The pipeline is the one `open_workspace` runs in Claudinio Code
//! (`src-tauri/src/commands/code_intel.rs`), minus the UI: scan first so
//! lexical search works within seconds, load the model in parallel, embed in
//! the background, then watch. Every step is best-effort — a workspace with
//! no embedding model still answers every tool, just lexically.

use claudinio_code_intel::db::IndexDb;
use claudinio_code_intel::embeddings::{self, SharedEmbedder};
use claudinio_code_intel::indexer::{self, IndexProgress};
use claudinio_code_intel::watcher::{FileWatcher, WatchEvent};
use claudinio_code_intel::{INDEX_SEMAPHORE, thread_priority};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::task::spawn_blocking;

/// Where the pipeline is. Reported verbatim by `index_status` so an agent can
/// decide whether to wait or fall back to grep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// tree-sitter scan in progress; searches fail fast with the progress.
    Indexing,
    /// Symbols are searchable; embeddings still being generated.
    Embedding,
    /// Symbols searchable, embedding model unavailable (download/load failed
    /// or disabled) — every search is lexical-only.
    LexicalOnly,
    /// Everything is up: hybrid search and live watching.
    Ready,
    /// The scan itself failed; the workspace answers nothing useful.
    Failed,
}

pub struct Workspace {
    pub root: PathBuf,
    pub db_path: PathBuf,
    pub db: Arc<IndexDb>,
    pub phase: Mutex<Phase>,
    pub error: Mutex<Option<String>>,
    /// `Some` while the scan runs, `None` once symbols are queryable — the
    /// same convention the agent tools in Claudinio Code key off.
    pub progress: Arc<Mutex<Option<IndexProgress>>>,
    pub embed_progress: Mutex<Option<IndexProgress>>,
    pub embedder: Arc<Mutex<Option<SharedEmbedder>>>,
    pub watcher_warning: Mutex<Option<String>>,
    watcher: Mutex<Option<FileWatcher>>,
}

/// Machine-local cache root. Never inside the workspace: SQLite in WAL mode
/// is unsupported over network filesystems, and one model per project is a
/// 23 MB download per repo.
pub fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("claudinio-code-intel")
}

pub fn index_db_path(cache_dir: &Path, workspace_root: &Path) -> PathBuf {
    let stem: String = workspace_root
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "workspace".into())
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .take(40)
        .collect();
    let hash = xxhash_rust::xxh3::xxh3_64(workspace_root.to_string_lossy().as_bytes());
    cache_dir
        .join("indexes")
        .join(format!("{stem}-{hash:016x}.db"))
}

pub fn model_dir(cache_dir: &Path) -> PathBuf {
    cache_dir
        .join("models")
        .join(embeddings::model_cache_dirname())
}

impl Workspace {
    pub fn open(root: PathBuf, cache_dir: &Path) -> Result<Arc<Self>, String> {
        let db_path = index_db_path(cache_dir, &root);
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
        }
        let db = Arc::new(IndexDb::open(&db_path)?);
        Ok(Arc::new(Self {
            root: root.clone(),
            db_path,
            db,
            phase: Mutex::new(Phase::Indexing),
            error: Mutex::new(None),
            progress: Arc::new(Mutex::new(Some(IndexProgress {
                status: "indexing".into(),
                files_indexed: 0,
                symbols_indexed: 0,
                total_files: 0,
                workspace: root.to_string_lossy().to_string(),
            }))),
            embed_progress: Mutex::new(None),
            embedder: Arc::new(Mutex::new(None)),
            watcher_warning: Mutex::new(None),
            watcher: Mutex::new(None),
        }))
    }

    pub fn phase(&self) -> Phase {
        *self.phase.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn set_phase(&self, p: Phase) {
        *self.phase.lock().unwrap_or_else(|e| e.into_inner()) = p;
    }

    /// True once the tree-sitter scan has finished at least once for this
    /// process — lexical tools may answer.
    pub fn symbols_ready(&self) -> bool {
        self.progress
            .lock()
            .map(|p| p.is_none())
            .unwrap_or(false)
    }

    pub fn current_embedder(&self) -> Option<SharedEmbedder> {
        self.embedder.lock().ok().and_then(|g| g.clone())
    }

    /// Scan → model → embed → watch. Runs to completion in the background;
    /// tools read `phase`/`progress` meanwhile.
    pub async fn run_pipeline(self: Arc<Self>, cache_dir: PathBuf, want_embeddings: bool) {
        let root_str = self.root.to_string_lossy().to_string();

        // Model download in parallel with the scan: both are best-effort and
        // independent, and the scan is what unblocks the first search.
        let model_dir = model_dir(&cache_dir);
        let download = if want_embeddings {
            let md = model_dir.clone();
            Some(tokio::spawn(async move {
                embeddings::ensure_model_downloaded(&md).await
            }))
        } else {
            None
        };

        let permit = match INDEX_SEMAPHORE.acquire().await {
            Ok(p) => p,
            Err(e) => {
                self.fail(format!("index semaphore: {e}"));
                return;
            }
        };

        let scan = {
            let ws = self.clone();
            let root = root_str.clone();
            spawn_blocking(move || {
                let _prio = thread_priority::BackgroundPriority::begin();
                let shared = ws.progress.clone();
                let sink = |p: IndexProgress| {
                    tracing::debug!(files = p.files_indexed, total = p.total_files, "scan");
                };
                indexer::scan_workspace(ws.db.as_ref(), &root, Some(&sink), None, Some(&shared))
            })
        };

        let scanned = match scan.await {
            Ok(Ok(counts)) => counts,
            Ok(Err(e)) => {
                self.fail(format!("scan failed: {e}"));
                return;
            }
            Err(e) => {
                self.fail(format!("scan task panicked: {e}"));
                return;
            }
        };
        tracing::info!(files = scanned.0, symbols = scanned.1, root = %root_str, "scan complete");
        if let Ok(mut p) = self.progress.lock() {
            *p = None;
        }

        // The watcher goes up before the (slow) embedding pass so edits made
        // during it are not lost. It reads the embedder lazily per batch.
        self.start_watcher();

        let embedder: Option<SharedEmbedder> = if want_embeddings {
            if let Some(dl) = download {
                match dl.await {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => tracing::warn!("embedding model download failed: {e}"),
                    Err(e) => tracing::warn!("embedding model download task panicked: {e}"),
                }
            }
            let md = model_dir.clone();
            match spawn_blocking(move || embeddings::load_shared(&md)).await {
                Ok(Ok(shared)) => Some(shared),
                Ok(Err(e)) => {
                    tracing::warn!("embedding model load failed: {e}");
                    // Self-heal a corrupt download: the next run re-downloads
                    // instead of failing the same way forever.
                    let _ = std::fs::remove_dir_all(&model_dir);
                    None
                }
                Err(e) => {
                    tracing::warn!("embedding model load panicked: {e}");
                    None
                }
            }
        } else {
            None
        };

        let Some(shared) = embedder else {
            self.set_phase(Phase::LexicalOnly);
            drop(permit);
            return;
        };
        if let Ok(mut g) = self.embedder.lock() {
            *g = Some(shared.clone());
        }
        self.set_phase(Phase::Embedding);

        let embed = {
            let ws = self.clone();
            let root = root_str.clone();
            spawn_blocking(move || {
                let _prio = thread_priority::BackgroundPriority::begin();
                let sink = |p: IndexProgress| {
                    if let Ok(mut g) = ws.embed_progress.lock() {
                        *g = Some(p);
                    }
                };
                indexer::generate_all_embeddings(ws.db.as_ref(), &shared, Some(&sink), &root)
            })
        };
        match embed.await {
            Ok(Ok((files, vectors))) => {
                tracing::info!(files, vectors, "embeddings complete");
            }
            Ok(Err(e)) => tracing::warn!("embedding generation failed: {e}"),
            Err(e) => tracing::warn!("embedding task panicked: {e}"),
        }
        if let Ok(mut g) = self.embed_progress.lock() {
            *g = None;
        }
        self.set_phase(Phase::Ready);
        drop(permit);
    }

    fn fail(&self, msg: String) {
        tracing::error!("{msg}");
        if let Ok(mut e) = self.error.lock() {
            *e = Some(msg);
        }
        self.set_phase(Phase::Failed);
    }

    fn start_watcher(self: &Arc<Self>) {
        let embedder = self.embedder.clone();
        let source: claudinio_code_intel::watcher::EmbedderSource =
            Arc::new(move || embedder.lock().ok().and_then(|g| g.clone()));
        let sink: claudinio_code_intel::watcher::WatchEventSink = Arc::new(|ev: WatchEvent| {
            tracing::debug!(?ev, "watch");
        });
        match FileWatcher::start(&self.root.to_string_lossy(), &self.db_path, source, sink) {
            Ok(w) => {
                if let Ok(mut g) = self.watcher.lock() {
                    *g = Some(w);
                }
            }
            Err(e) => {
                tracing::warn!("file watcher unavailable (index will not follow edits): {e}");
                if let Ok(mut g) = self.watcher_warning.lock() {
                    *g = Some(format!("Live file watching unavailable: {e}"));
                }
            }
        }
    }

    /// Resolve a tool's `file_path` argument: absolute paths pass through,
    /// relative ones are rooted at the workspace. The index stores whatever
    /// the scan walked, which is `<root>/…`.
    pub fn resolve_path(&self, p: &str) -> String {
        let path = Path::new(p);
        if path.is_absolute() {
            p.to_string()
        } else {
            self.root.join(path).to_string_lossy().to_string()
        }
    }
}
