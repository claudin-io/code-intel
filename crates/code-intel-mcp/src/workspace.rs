//! One indexed workspace: its SQLite index, the shared embedder, the file
//! watcher that keeps the index fresh, and the phase the pipeline is in.
//!
//! The pipeline is the one `open_workspace` runs in Claudinio Code
//! (`src-tauri/src/commands/code_intel.rs`), minus the UI: scan first so
//! lexical search works within seconds, load the model in parallel, embed in
//! the background, then watch. Every step is best-effort — a workspace with
//! no embedding model still answers every tool, just lexically.

use claudinio_code_intel::db::IndexDb;
use claudinio_code_intel::embeddings::{self, ModelChoice, SharedEmbedder};
use claudinio_code_intel::indexer::{self, IndexProgress};
use claudinio_code_intel::media::{self, MediaNeeds};
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
    /// Where this workspace's images and audio stand (see `MediaStage`).
    pub media_stage: Mutex<MediaStage>,
    pub watcher_warning: Mutex<Option<String>>,
    watcher: Mutex<Option<FileWatcher>>,
}

/// Content vectors for media come after everything else, from a model that
/// may still be on its way down. Text search does not wait for any of it;
/// this is how `index_status` tells "not yet" from "not at all".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaStage {
    /// No images or audio here, or media content search is not in play
    /// (embeddings off, `CODE_INTEL_MEDIA=0`).
    None,
    /// The media model or an encoder is being downloaded and loaded.
    Loading,
    /// Media files are being embedded.
    Embedding,
    /// Done: files with a content vector match by what they show or sound
    /// like, the rest by name.
    Ready,
    /// Nothing loaded that can encode this workspace's media; it matches by
    /// file name only.
    NameOnly,
}

impl MediaStage {
    pub fn as_str(self) -> &'static str {
        match self {
            MediaStage::None => "none",
            MediaStage::Loading => "loading",
            MediaStage::Embedding => "embedding",
            MediaStage::Ready => "ready",
            MediaStage::NameOnly => "name-only",
        }
    }
}

/// Machine-local cache root. Never inside the workspace: SQLite in WAL mode
/// is unsupported over network filesystems, and one model per project would
/// be the same download again for every repo.
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

/// Where model files live, one directory per model underneath.
pub fn models_root(cache_dir: &Path) -> PathBuf {
    cache_dir.join("models")
}

/// The embedder is loaded once per process and shared by every workspace:
/// EmbeddingGemma 2 — there for the first workspace that has images or audio
/// — is a few hundred megabytes resident, and two open roots must not mean
/// two copies. Keyed by the models directory so a second cache
/// (tests) gets its own.
static EMBEDDER: tokio::sync::Mutex<Option<(PathBuf, SharedEmbedder)>> = tokio::sync::Mutex::const_new(None);

/// The process's embedder, loading it on first use — the text model only.
/// What a workspace's media needs on top of it is `extend_media_once`, which
/// can take minutes on a first run and must not hold text search back.
async fn shared_embedder(models_root: &Path) -> Result<SharedEmbedder, String> {
    let mut slot = EMBEDDER.lock().await;
    if let Some((root, shared)) = slot.as_ref()
        && root == models_root
    {
        return Ok(shared.clone());
    }
    let shared = embeddings::ensure_and_load(models_root, ModelChoice::from_env(), MediaNeeds::NONE).await?;
    *slot = Some((models_root.to_path_buf(), shared.clone()));
    Ok(shared)
}

/// One media set-up at a time per process: two workspaces opening together
/// must not download the same few hundred megabytes twice.
static MEDIA_SETUP: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn extend_media_once(shared: &SharedEmbedder, needs: MediaNeeds) {
    let _one_at_a_time = MEDIA_SETUP.lock().await;
    embeddings::extend_media(shared, needs).await;
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
            media_stage: Mutex::new(MediaStage::None),
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

    pub fn media_stage(&self) -> MediaStage {
        *self.media_stage.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn set_media_stage(&self, stage: MediaStage) {
        *self.media_stage.lock().unwrap_or_else(|e| e.into_inner()) = stage;
    }

    pub fn current_embedder(&self) -> Option<SharedEmbedder> {
        self.embedder.lock().ok().and_then(|g| g.clone())
    }

    /// Scan → model → embed → watch, then media. Runs to completion in the
    /// background; tools read `phase`/`progress` meanwhile.
    pub async fn run_pipeline(self: Arc<Self>, cache_dir: PathBuf, want_embeddings: bool) {
        let root_str = self.root.to_string_lossy().to_string();

        // Model download and load in parallel with the scan: both are
        // best-effort and independent, and the scan is what unblocks the
        // first search.
        //
        // What the workspace's media needs is decided here too, from what it
        // actually contains — a quick walk of file names, long before the
        // scan gets to them — and fetched on its own task as soon as the
        // text model is up: the media model and its encoders are hundreds of
        // megabytes on a first run, and text search is ready without them. A
        // text-only project never starts that task.
        let model = if want_embeddings {
            let root = root_str.clone();
            let models = models_root(&cache_dir);
            let ws = self.clone();
            Some(tokio::spawn(async move {
                let needs = spawn_blocking(move || media::detect_media_needs(&root))
                    .await
                    .unwrap_or(MediaNeeds::NONE);
                let shared = shared_embedder(&models).await?;
                let media = needs.any().then(|| {
                    ws.set_media_stage(MediaStage::Loading);
                    let shared = shared.clone();
                    tokio::spawn(async move { extend_media_once(&shared, needs).await })
                });
                Ok::<_, String>((shared, media))
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

        let embedder = match model {
            Some(task) => match task.await {
                Ok(Ok(loaded)) => Some(loaded),
                Ok(Err(e)) => {
                    tracing::warn!("embedding model unavailable: {e}");
                    None
                }
                Err(e) => {
                    tracing::warn!("embedding model task panicked: {e}");
                    None
                }
            },
            None => None,
        };

        let Some((shared, media_setup)) = embedder else {
            self.set_phase(Phase::LexicalOnly);
            drop(permit);
            return;
        };
        if let Ok(emb) = shared.lock() {
            tracing::info!(model = emb.model_id(), "embedding model ready");
        }
        if let Ok(mut g) = self.embedder.lock() {
            *g = Some(shared.clone());
        }
        self.set_phase(Phase::Embedding);

        self.embedding_pass(&shared, &root_str).await;
        self.set_phase(Phase::Ready);
        drop(permit);

        // Media, last. Whatever the pass above could not encode — because
        // the media model or an encoder had not arrived yet — it marked as
        // done without a content vector; once the set-up task is through,
        // one more pass picks exactly those files up (see
        // `IndexDb::reconcile_embedding_model`) and nothing else.
        let Some(media_setup) = media_setup else { return };
        if let Err(e) = media_setup.await {
            tracing::warn!("media model task panicked: {e}");
        }
        let support = shared.lock().map(|emb| emb.media_support()).unwrap_or(MediaNeeds::NONE);
        if !support.any() {
            self.set_media_stage(MediaStage::NameOnly);
            return;
        }
        if let Ok(emb) = shared.lock() {
            tracing::info!(
                model = emb.media_model().map(|(id, _)| id).unwrap_or("none"),
                images = support.images,
                audio = support.audio,
                "media model ready"
            );
        }
        self.set_media_stage(MediaStage::Embedding);
        if let Ok(permit) = INDEX_SEMAPHORE.acquire().await {
            self.embedding_pass(&shared, &root_str).await;
            drop(permit);
        }
        self.set_media_stage(MediaStage::Ready);
    }

    /// Embed whatever in the index has no vector for its current content.
    async fn embedding_pass(self: &Arc<Self>, shared: &SharedEmbedder, root: &str) {
        let ws = self.clone();
        let shared = shared.clone();
        let root = root.to_string();
        let pass = spawn_blocking(move || {
            let _prio = thread_priority::BackgroundPriority::begin();
            let sink = |p: IndexProgress| {
                if let Ok(mut g) = ws.embed_progress.lock() {
                    *g = Some(p);
                }
            };
            indexer::generate_all_embeddings(ws.db.as_ref(), &shared, Some(&sink), &root)
        });
        match pass.await {
            Ok(Ok((files, vectors))) => tracing::info!(files, vectors, "embeddings complete"),
            Ok(Err(e)) => tracing::warn!("embedding generation failed: {e}"),
            Err(e) => tracing::warn!("embedding task panicked: {e}"),
        }
        if let Ok(mut g) = self.embed_progress.lock() {
            *g = None;
        }
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

/// The pipeline against stand-in models (the fixtures of the library crate):
/// what a server does from "workspace opened" to "everything embedded".
#[cfg(all(test, feature = "embeddings"))]
mod tests {
    use super::*;
    use claudinio_code_intel::gemma2;

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("code-intel")
            .join("tests")
            .join("fixtures")
    }

    /// A cache directory with the stand-in MiniLM, and `gemma2` deciding
    /// what sits where EmbeddingGemma 2 would be downloaded to: the stand-in
    /// (`Some(true)`), files that are not models (`Some(false)`), or nothing.
    /// Nothing is only for workspaces without media — with media it would be
    /// a real download.
    fn cache(gemma2: Option<bool>) -> tempfile::TempDir {
        let cache = tempfile::tempdir().unwrap();
        let models = models_root(cache.path());
        let minilm = embeddings::minilm_dir(&models);
        std::fs::create_dir_all(&minilm).unwrap();
        for name in ["model_quantized.onnx", "tokenizer.json"] {
            std::fs::copy(fixtures().join("minilm-synthetic").join(name), minilm.join(name)).unwrap();
        }
        if let Some(usable) = gemma2 {
            let dir = models.join(gemma2::CACHE_DIRNAME);
            std::fs::create_dir_all(&dir).unwrap();
            for kind in [None, Some(media::MediaKind::Image), Some(media::MediaKind::Audio)] {
                for (_, local, _, _) in gemma2::component_files(kind) {
                    if usable {
                        std::fs::copy(fixtures().join("gemma2-synthetic").join(local), dir.join(local)).unwrap();
                    } else {
                        std::fs::write(dir.join(local), b"not a model").unwrap();
                    }
                }
            }
        }
        cache
    }

    fn project(with_audio: bool) -> tempfile::TempDir {
        let ws = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(ws.path().join("src")).unwrap();
        std::fs::write(
            ws.path().join("src").join("upload.rs"),
            "/// Retries a failed upload with exponential backoff.\npub fn retry_failed_upload(attempts: u32) -> bool { attempts > 0 }\n",
        )
        .unwrap();
        if with_audio {
            // Half a second of a 440 Hz tone, 16-bit mono.
            let rate = 16_000u32;
            let pcm: Vec<u8> = (0..rate / 2)
                .flat_map(|i| {
                    let s = (2.0 * std::f64::consts::PI * 440.0 * i as f64 / rate as f64).sin() * 9_000.0;
                    (s as i16).to_le_bytes()
                })
                .collect();
            let mut wav: Vec<u8> = Vec::new();
            wav.extend_from_slice(b"RIFF");
            wav.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
            wav.extend_from_slice(b"WAVEfmt ");
            wav.extend_from_slice(&16u32.to_le_bytes());
            wav.extend_from_slice(&1u16.to_le_bytes());
            wav.extend_from_slice(&1u16.to_le_bytes());
            wav.extend_from_slice(&rate.to_le_bytes());
            wav.extend_from_slice(&(rate * 2).to_le_bytes());
            wav.extend_from_slice(&2u16.to_le_bytes());
            wav.extend_from_slice(&16u16.to_le_bytes());
            wav.extend_from_slice(b"data");
            wav.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
            wav.extend_from_slice(&pcm);
            std::fs::create_dir_all(ws.path().join("assets")).unwrap();
            std::fs::write(ws.path().join("assets").join("error-beep.wav"), wav).unwrap();
        }
        ws
    }

    async fn run(project: &tempfile::TempDir, cache: &tempfile::TempDir) -> Arc<Workspace> {
        let ws = Workspace::open(project.path().to_path_buf(), cache.path()).unwrap();
        ws.clone().run_pipeline(cache.path().to_path_buf(), true).await;
        ws
    }

    const MINILM: &str = "all-MiniLM-L6-v2";

    /// Code on MiniLM, the clip on EmbeddingGemma 2 with the one encoder a
    /// workspace of sounds needs, in one index.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_workspace_with_audio_ends_with_text_and_media_vectors() {
        let (project, cache) = (project(true), cache(Some(true)));
        let ws = run(&project, &cache).await;
        assert_eq!(ws.phase(), Phase::Ready);
        assert_eq!(ws.media_stage(), MediaStage::Ready);
        assert_eq!(ws.db.embedding_model().as_deref(), Some(MINILM));
        assert_eq!(ws.db.media_embedding_model().as_deref(), Some(gemma2::MODEL_ID));
        assert_eq!(ws.db.media_embedding_count().unwrap(), 1);
        assert!(ws.db.index_stats().unwrap().2 >= 2, "code and the clip");
        assert_eq!(ws.db.embedding_pending_files().unwrap(), 0);
        let embedder = ws.current_embedder().unwrap();
        let support = embedder.lock().unwrap().media_support();
        assert_eq!(support, MediaNeeds { images: false, audio: true }, "no picture, no vision encoder");
    }

    /// Media that cannot get content vectors takes nothing else down with it.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unusable_media_model_leaves_text_search_whole() {
        let (project, cache) = (project(true), cache(Some(false)));
        let ws = run(&project, &cache).await;
        assert_eq!(ws.phase(), Phase::Ready);
        assert_eq!(ws.media_stage(), MediaStage::NameOnly);
        assert_eq!(ws.db.embedding_model().as_deref(), Some(MINILM));
        assert!(ws.db.index_stats().unwrap().2 >= 1);
        assert_eq!(ws.db.media_embedding_count().unwrap(), 0);
        assert_eq!(ws.db.embedding_pending_files().unwrap(), 0);
        assert_eq!(ws.db.search_media("error beep", None, 3).unwrap().len(), 1);
    }

    /// The case most workspaces are: no media, so nothing about the media
    /// model happens at all — not even a directory for it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_text_only_workspace_never_touches_the_media_model() {
        let (project, cache) = (project(false), cache(None));
        let ws = run(&project, &cache).await;
        assert_eq!(ws.phase(), Phase::Ready);
        assert_eq!(ws.media_stage(), MediaStage::None);
        assert!(ws.db.index_stats().unwrap().2 >= 1);
        assert!(!models_root(cache.path()).join(gemma2::CACHE_DIRNAME).exists());
    }
}
