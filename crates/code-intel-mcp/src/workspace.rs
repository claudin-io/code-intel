//! One indexed workspace: its SQLite index, the shared embedder, the file
//! watcher that keeps the index fresh, and the phase the pipeline is in.
//!
//! The pipeline is the one `open_workspace` runs in Claudinio Code
//! (`src-tauri/src/commands/code_intel.rs`), minus the UI: scan first so
//! lexical search works within seconds, load the model in parallel, embed in
//! the background, then watch. Every step is best-effort — a workspace with
//! no embedding model still answers every tool, just lexically.
//!
//! After it comes the upgrade (`run_upgrade_worker`): the same text embedded
//! again by the slower, better model, for as long as that takes, with search
//! moving to those vectors once they cover the workspace (`text_set`).

use claudinio_code_intel::db::{IndexDb, VectorSet};
use claudinio_code_intel::embeddings::{self, ModelChoice, SharedEmbedder};
use claudinio_code_intel::indexer::{self, IndexProgress};
use claudinio_code_intel::media::{self, MediaNeeds};
use claudinio_code_intel::watcher::{FileWatcher, WatchEvent};
use claudinio_code_intel::{INDEX_SEMAPHORE, UPGRADE_SEMAPHORE, thread_priority};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::task::spawn_blocking;

/// What a workspace's pipeline is asked to do. Read from the environment
/// once, by the server; passed in, so that nothing below depends on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineOptions {
    /// Load an embedding model at all (`CODE_INTEL_EMBEDDINGS`).
    pub embeddings: bool,
    /// Which models (`CODE_INTEL_MODEL`).
    pub model: ModelChoice,
    /// Embed the text a second time in the background and move search to
    /// those vectors (`CODE_INTEL_UPGRADE`). Only the default model choice,
    /// in a build that can run EmbeddingGemma 2, has an upgrade to make.
    pub upgrade: bool,
}

impl PipelineOptions {
    pub fn from_env() -> Self {
        PipelineOptions {
            embeddings: std::env::var("CODE_INTEL_EMBEDDINGS")
                .map(|v| v != "0")
                .unwrap_or(true),
            model: ModelChoice::from_env(),
            upgrade: embeddings::text_upgrade_enabled(),
        }
    }
}

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
    pub options: PipelineOptions,
    /// What the upgrade is doing (see `UpgradeStage`), and how far a pass in
    /// progress has got.
    pub upgrade_stage: Mutex<UpgradeStage>,
    pub upgrade_progress: Mutex<Option<IndexProgress>>,
    /// Rung by the watcher when a file has been taken in: the upgrade set is
    /// now one file behind.
    upgrade_wake: Arc<tokio::sync::Notify>,
    /// Rounds in which the upgrade model failed on a file, by file id.
    upgrade_failures: Mutex<std::collections::HashMap<i64, u32>>,
    pub watcher_warning: Mutex<Option<String>>,
    watcher: Mutex<Option<FileWatcher>>,
}

/// What the background upgrade of a workspace's text vectors is doing. Which
/// vectors a search uses is a separate question, answered from the index
/// (`Workspace::text_set`): a workspace opened for the second time searches
/// its finished upgrade set while this still says `Waiting`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpgradeStage {
    /// Not in play: turned off, no embeddings, or a model choice or a build
    /// that has no second model to upgrade to.
    Off,
    /// Its turn has not come: the first index does, or another workspace's
    /// upgrade, or another process is at this very index.
    Waiting,
    /// EmbeddingGemma 2 is being downloaded and loaded.
    Loading,
    /// The upgrade set is being filled in, or has files still to take in.
    Embedding,
    /// The upgrade set holds every file as it is now; edits are folded in
    /// as they come.
    Ready,
    /// The model could not be had. Text search stays on the first one.
    Unavailable,
}

impl UpgradeStage {
    pub fn as_str(self) -> &'static str {
        match self {
            UpgradeStage::Off => "off",
            UpgradeStage::Waiting => "waiting",
            UpgradeStage::Loading => "loading",
            UpgradeStage::Embedding => "embedding",
            UpgradeStage::Ready => "ready",
            UpgradeStage::Unavailable => "unavailable",
        }
    }
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

/// The file whose advisory lock says "a process is filling in this index's
/// upgrade set". Next to the index, which it is about.
fn upgrade_lock_path(db_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.upgrade.lock", db_path.display()))
}

enum UpgradeLock {
    /// Ours until the file is dropped — or this process dies.
    Held(std::fs::File),
    /// Another process has it.
    Busy,
    /// No lock to be had here (a filesystem without them): carry on as if
    /// alone, which costs duplicated work at worst.
    Unsupported,
}

fn lock_upgrade(db_path: &Path) -> UpgradeLock {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(upgrade_lock_path(db_path));
    let Ok(file) = file else {
        return UpgradeLock::Unsupported;
    };
    match file.try_lock() {
        Ok(()) => UpgradeLock::Held(file),
        Err(std::fs::TryLockError::WouldBlock) => UpgradeLock::Busy,
        Err(std::fs::TryLockError::Error(_)) => UpgradeLock::Unsupported,
    }
}

/// Whether a workspace run with these options has a text upgrade at all.
fn upgrade_in_play(options: &PipelineOptions) -> bool {
    options.embeddings
        && options.upgrade
        && embeddings::load_plan(options.model, embeddings::gemma2_supported()).companion
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
/// EmbeddingGemma 2 is a few hundred megabytes resident, and two open roots
/// must not mean two copies. Keyed by the models directory so a second cache
/// (tests) gets its own.
static EMBEDDER: tokio::sync::Mutex<Option<(PathBuf, SharedEmbedder)>> = tokio::sync::Mutex::const_new(None);

/// The process's embedder, loading it on first use — the text model only.
/// What a workspace's media needs on top of it is `extend_media_once`, and
/// what the upgrade needs is `ensure_upgrade_once`; either can take minutes
/// on a first run and must not hold text search back.
async fn shared_embedder(models_root: &Path, choice: ModelChoice) -> Result<SharedEmbedder, String> {
    let mut slot = EMBEDDER.lock().await;
    if let Some((root, shared)) = slot.as_ref()
        && root == models_root
    {
        return Ok(shared.clone());
    }
    let shared = embeddings::ensure_and_load(models_root, choice, MediaNeeds::NONE).await?;
    *slot = Some((models_root.to_path_buf(), shared.clone()));
    Ok(shared)
}

/// One EmbeddingGemma 2 set-up at a time per process: two workspaces opening
/// together — or one workspace's media and its upgrade — must not download
/// the same few hundred megabytes twice.
static GEMMA2_SETUP: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn extend_media_once(shared: &SharedEmbedder, needs: MediaNeeds) {
    let _one_at_a_time = GEMMA2_SETUP.lock().await;
    embeddings::extend_media(shared, needs).await;
}

async fn ensure_upgrade_once(shared: &SharedEmbedder) -> bool {
    let _one_at_a_time = GEMMA2_SETUP.lock().await;
    embeddings::ensure_text_upgrade(shared).await
}

/// How long one workspace keeps the upgrade model before the next in line
/// gets it (a turn ends with the file in progress, not mid-file).
const UPGRADE_TURN: std::time::Duration = std::time::Duration::from_secs(60);

/// How long the upgrade waits before looking at the index again when nothing
/// told it to: the watcher normally does, the moment a file changes.
const UPGRADE_POLL: std::time::Duration = std::time::Duration::from_secs(60);

/// Rounds the upgrade model may fail on one file before the file is left to
/// keyword search.
const UPGRADE_ATTEMPTS: u32 = 3;

/// How long before another attempt at a model that could not be had — a
/// laptop that was offline when the editor started is usually not for long.
const UPGRADE_RETRY: std::time::Duration = std::time::Duration::from_secs(30 * 60);

impl Workspace {
    pub fn open(root: PathBuf, cache_dir: &Path, options: PipelineOptions) -> Result<Arc<Self>, String> {
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
            options,
            upgrade_stage: Mutex::new(if upgrade_in_play(&options) {
                UpgradeStage::Waiting
            } else {
                UpgradeStage::Off
            }),
            upgrade_progress: Mutex::new(None),
            upgrade_wake: Arc::new(tokio::sync::Notify::new()),
            upgrade_failures: Mutex::new(std::collections::HashMap::new()),
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

    pub fn upgrade_stage(&self) -> UpgradeStage {
        *self.upgrade_stage.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn set_upgrade_stage(&self, stage: UpgradeStage) {
        *self.upgrade_stage.lock().unwrap_or_else(|e| e.into_inner()) = stage;
    }

    /// The set of text vectors a search should use right now; the query must
    /// be embedded by the model that wrote it.
    ///
    /// The upgrade set, when three things hold: the upgrade is in play, its
    /// model is loaded in this process — a query has to come from it — and
    /// the set is complete and all but current (`UpgradeState::usable`).
    /// Everything else is the first model's set, which is always there: that
    /// is the fallback, and it needs no decision to be taken.
    pub fn text_set(&self) -> VectorSet {
        if !upgrade_in_play(&self.options) {
            return VectorSet::Primary;
        }
        let model = self
            .current_embedder()
            .and_then(|shared| shared.lock().ok().and_then(|guard| guard.upgrade_model()));
        match (model, self.db.upgrade_state()) {
            (Some((model, _)), Ok(state)) if state.usable(model) => VectorSet::Upgrade,
            _ => VectorSet::Primary,
        }
    }

    /// One round of the upgrade: bring EmbeddingGemma 2 in if it is not here
    /// yet, then embed whatever the upgrade set lacks. Returns where that
    /// left things. Safe to call again at any time; with nothing to do it
    /// does nothing.
    pub async fn upgrade_once(self: &Arc<Self>) -> UpgradeStage {
        if !upgrade_in_play(&self.options) {
            return UpgradeStage::Off;
        }
        let stage = self.upgrade_round().await;
        self.set_upgrade_stage(stage);
        stage
    }

    async fn upgrade_round(self: &Arc<Self>) -> UpgradeStage {
        // No text model at all: nothing was embedded, nothing to upgrade.
        let Some(shared) = self.current_embedder() else {
            return UpgradeStage::Unavailable;
        };
        let loaded = shared.lock().map(|g| g.upgrade_model().is_some()).unwrap_or(false);
        if !loaded {
            self.set_upgrade_stage(UpgradeStage::Loading);
            if !ensure_upgrade_once(&shared).await {
                return UpgradeStage::Unavailable;
            }
            tracing::info!("upgrade model ready");
        }

        // One process at a time: two agents open on the same repository are
        // two servers over one index, and each would otherwise spend the
        // hour redoing what the other had just done, on the same cores. What
        // the other one embeds lands in the same set; this one finds it
        // there, and only needed the model to query it with.
        let _this_process = match lock_upgrade(&self.db_path) {
            UpgradeLock::Held(file) => Some(file),
            UpgradeLock::Unsupported => None,
            UpgradeLock::Busy => {
                return match self.db.upgrade_state() {
                    Ok(state) if state.complete && state.pending_files == 0 => UpgradeStage::Ready,
                    _ => UpgradeStage::Waiting,
                };
            }
        };

        // In turns of a minute or so: every open workspace shares the one
        // model, and one that only has an edited file to fold in should not
        // wait behind another's first hour.
        let root = self.root.to_string_lossy().to_string();
        let (mut files, mut vectors, mut left) = (0i64, 0i64, 0i64);
        let mut after: Option<i64> = None;
        loop {
            self.set_upgrade_stage(UpgradeStage::Waiting);
            let Ok(turn) = UPGRADE_SEMAPHORE.acquire().await else {
                break;
            };
            self.set_upgrade_stage(UpgradeStage::Embedding);
            let (ws, shared, root) = (self.clone(), shared.clone(), root.clone());
            let pass = spawn_blocking(move || {
                let _prio = thread_priority::BackgroundPriority::begin();
                let sink = |p: IndexProgress| {
                    if let Ok(mut g) = ws.upgrade_progress.lock() {
                        *g = Some(p);
                    }
                };
                let turn = indexer::UpgradeTurn {
                    budget: Some(UPGRADE_TURN),
                    after,
                };
                indexer::generate_upgrade_embeddings(ws.db.as_ref(), &shared, Some(&sink), &root, turn)
            })
            .await;
            drop(turn);
            // Whatever went wrong in a turn — the index busy under another
            // process, a panic — is no reason to call the model unavailable:
            // the round ends here and the next one is a wake or a minute off.
            let pass = match pass {
                Ok(Ok(pass)) => pass,
                Ok(Err(e)) => {
                    tracing::warn!("upgrade turn failed, will retry: {e}");
                    break;
                }
                Err(e) => {
                    tracing::warn!("upgrade task panicked, will retry: {e}");
                    break;
                }
            };
            files += pass.files;
            vectors += pass.vectors;
            left += pass.left;
            self.give_up_on_failing_files(&pass.failed);
            // The next turn takes up where this one stopped — not at the
            // head of the queue, where a file that cannot be finished would
            // be all that is ever attempted.
            if pass.remaining > 0 && pass.last_file.is_some() && pass.last_file != after {
                after = pass.last_file;
                continue;
            }
            break;
        }
        if let Ok(mut g) = self.upgrade_progress.lock() {
            *g = None;
        }
        // Where that leaves things is what the index says, not what the last
        // turn hoped: "ready" means every file, as it is now.
        let stage = match self.db.upgrade_state() {
            Ok(state) if state.complete && state.pending_files == 0 => UpgradeStage::Ready,
            _ => UpgradeStage::Embedding,
        };
        if files > 0 || left > 0 {
            tracing::info!(files, vectors, left, state = stage.as_str(), "upgrade round done");
        }
        stage
    }

    /// A file the model has failed on in several rounds is recorded as done
    /// without its vectors: found by its text, tried again when it changes,
    /// and no longer in the way of the set ever being complete.
    fn give_up_on_failing_files(&self, failed: &[i64]) {
        let Ok(mut counts) = self.upgrade_failures.lock() else {
            return;
        };
        for id in failed {
            let count = counts.entry(*id).or_insert(0);
            *count += 1;
            if *count >= UPGRADE_ATTEMPTS && self.db.skip_upgrade_file(*id).is_ok() {
                tracing::warn!(file_id = *id, "upgrade model keeps failing on a file; leaving it to keyword search");
                counts.remove(id);
            }
        }
    }

    /// The upgrade, for as long as the workspace is open: a first pass that
    /// may run for an hour, then a short one whenever the index has moved on
    /// — the watcher says when. Call it once `run_pipeline` has returned.
    pub async fn run_upgrade_worker(self: Arc<Self>) {
        if !upgrade_in_play(&self.options) {
            return;
        }
        loop {
            if self.upgrade_once().await == UpgradeStage::Unavailable {
                tokio::time::sleep(UPGRADE_RETRY).await;
                continue;
            }
            loop {
                tokio::select! {
                    _ = self.upgrade_wake.notified() => {}
                    _ = tokio::time::sleep(UPGRADE_POLL) => {}
                }
                match self.db.upgrade_state() {
                    // Nothing to do — also when it was another process that
                    // did it while this one waited.
                    Ok(state) if state.complete && state.pending_files == 0 => {
                        self.set_upgrade_stage(UpgradeStage::Ready);
                    }
                    Ok(_) => break,
                    Err(_) => {}
                }
            }
        }
    }

    /// Scan → model → embed → watch, then media. Runs to completion in the
    /// background; tools read `phase`/`progress` meanwhile.
    pub async fn run_pipeline(self: Arc<Self>, cache_dir: PathBuf) {
        let root_str = self.root.to_string_lossy().to_string();
        let want_embeddings = self.options.embeddings;
        let choice = self.options.model;

        // Model download and load in parallel with the scan: both are
        // best-effort and independent, and the scan is what unblocks the
        // first search.
        //
        // What the workspace's media needs is decided here too, from what it
        // actually contains — a quick walk of file names, long before the
        // scan gets to them — and fetched on its own task as soon as the
        // text model is up: the media model and its encoders are hundreds of
        // megabytes on a first run, and text search is ready without them. A
        // text-only project never starts that task; the same model reaches
        // it later, for the upgrade (`run_upgrade_worker`).
        let model = if want_embeddings {
            let root = root_str.clone();
            let models = models_root(&cache_dir);
            let ws = self.clone();
            Some(tokio::spawn(async move {
                let needs = spawn_blocking(move || media::detect_media_needs(&root))
                    .await
                    .unwrap_or(MediaNeeds::NONE);
                let shared = shared_embedder(&models, choice).await?;
                // A workspace opened again, its upgrade set long finished:
                // the model that set is queried with is brought in now, so
                // that the first search of this session is ranked the way
                // the last one of the previous session was.
                if upgrade_in_play(&ws.options) && ws.db.upgrade_state().is_ok_and(|s| s.complete) {
                    let shared = shared.clone();
                    tokio::spawn(async move { ensure_upgrade_once(&shared).await });
                }
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
        let wake = self.upgrade_wake.clone();
        let sink: claudinio_code_intel::watcher::WatchEventSink = Arc::new(move |ev: WatchEvent| {
            tracing::debug!(?ev, "watch");
            if matches!(ev, WatchEvent::Reindexed { .. }) {
                wake.notify_one();
            }
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

    const DEFAULTS: PipelineOptions = PipelineOptions {
        embeddings: true,
        model: ModelChoice::Auto,
        upgrade: true,
    };

    async fn run_with(
        project: &tempfile::TempDir,
        cache: &tempfile::TempDir,
        options: PipelineOptions,
    ) -> Arc<Workspace> {
        let ws = Workspace::open(project.path().to_path_buf(), cache.path(), options).unwrap();
        ws.clone().run_pipeline(cache.path().to_path_buf()).await;
        ws
    }

    /// The pipeline up to "ready" — without the upgrade, which the server
    /// starts afterwards and these tests call when they mean to.
    async fn run(project: &tempfile::TempDir, cache: &tempfile::TempDir) -> Arc<Workspace> {
        run_with(project, cache, DEFAULTS).await
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
    /// The upgrade, start to finish: search is on MiniLM while the second
    /// set is empty, and on EmbeddingGemma 2 once a pass has covered the
    /// workspace — with MiniLM's vectors all still in the index.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_upgrade_moves_text_search_to_the_second_model() {
        let (project, cache) = (project(true), cache(Some(true)));
        let ws = run(&project, &cache).await;
        assert_eq!(ws.phase(), Phase::Ready);
        assert_eq!(ws.upgrade_stage(), UpgradeStage::Waiting);
        assert_eq!(ws.text_set(), VectorSet::Primary);
        let media = ws.db.media_embedding_count().unwrap();
        let text = ws.db.index_stats().unwrap().2 - media;
        assert!(text >= 1);

        assert_eq!(ws.upgrade_once().await, UpgradeStage::Ready);
        assert_eq!(ws.upgrade_stage(), UpgradeStage::Ready);
        assert_eq!(ws.text_set(), VectorSet::Upgrade);
        let state = ws.db.upgrade_state().unwrap();
        assert_eq!(state.model.as_deref(), Some(gemma2::MODEL_ID));
        assert_eq!((state.vectors, state.pending_files), (text, 0), "the code, not the clip");
        assert!(ws.upgrade_progress.lock().unwrap().is_none());

        // The first model's set is what the fallback falls back to.
        assert_eq!(ws.db.embedding_model().as_deref(), Some(MINILM));
        assert_eq!(ws.db.index_stats().unwrap().2, text + media);
        assert_eq!(ws.db.embedding_pending_files().unwrap(), 0);

        // And there is nothing left for a second round to do.
        assert_eq!(ws.upgrade_once().await, UpgradeStage::Ready);
        assert_eq!(ws.db.upgrade_state().unwrap().vectors, text);
    }

    /// The fallback that needs no decision: a second model that cannot be
    /// loaded leaves search exactly where it was.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unusable_second_model_leaves_search_on_the_first() {
        let (project, cache) = (project(false), cache(Some(false)));
        let ws = run(&project, &cache).await;
        assert_eq!(ws.upgrade_once().await, UpgradeStage::Unavailable);
        assert_eq!(ws.text_set(), VectorSet::Primary);
        assert_eq!(ws.phase(), Phase::Ready);
        assert_eq!(ws.db.upgrade_state().unwrap().vectors, 0);
        assert!(ws.db.index_stats().unwrap().2 >= 1);
    }

    /// A finished upgrade set is not used by a process that was told not to:
    /// neither when the upgrade is switched off nor when MiniLM is asked for
    /// by name. And EmbeddingGemma 2 as the text model has no upgrade.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_upgrade_is_off_when_switched_off_or_when_one_model_was_asked_for() {
        let (project, first_cache) = (project(false), cache(Some(true)));
        let ws = run(&project, &first_cache).await;
        assert_eq!(ws.upgrade_once().await, UpgradeStage::Ready);
        assert_eq!(ws.text_set(), VectorSet::Upgrade);
        drop(ws);

        for options in [
            PipelineOptions { upgrade: false, ..DEFAULTS },
            PipelineOptions { model: ModelChoice::MiniLm, ..DEFAULTS },
            PipelineOptions { model: ModelChoice::Gemma2, ..DEFAULTS },
            PipelineOptions { embeddings: false, ..DEFAULTS },
        ] {
            // The same index under another cache directory: a process loads
            // its embedder once, with one choice of models, and these are
            // four processes.
            let cache = cache(Some(true));
            let indexes = cache.path().join("indexes");
            std::fs::create_dir_all(&indexes).unwrap();
            for entry in std::fs::read_dir(first_cache.path().join("indexes")).unwrap() {
                let entry = entry.unwrap();
                std::fs::copy(entry.path(), indexes.join(entry.file_name())).unwrap();
            }
            let ws = run_with(&project, &cache, options).await;
            assert_eq!(ws.upgrade_stage(), UpgradeStage::Off, "{options:?}");
            assert_eq!(ws.upgrade_once().await, UpgradeStage::Off, "{options:?}");
            assert_eq!(ws.text_set(), VectorSet::Primary, "{options:?}");
            assert!(
                ws.db.upgrade_state().unwrap().complete,
                "left in place, for the day it is wanted again: {options:?}"
            );
        }
    }

    /// The worker the server leaves running: after its first pass it sleeps
    /// until the watcher says a file was taken in, then embeds that file.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_worker_folds_an_edited_file_into_the_upgrade_set() {
        let (project, cache) = (project(false), cache(Some(true)));
        let ws = run(&project, &cache).await;
        let worker = tokio::spawn(ws.clone().run_upgrade_worker());

        async fn until(what: &str, done: impl Fn() -> bool) {
            for _ in 0..200 {
                if done() {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
            panic!("timed out waiting for {what}");
        }
        until("the first pass", || ws.text_set() == VectorSet::Upgrade).await;
        let before = ws.db.upgrade_state().unwrap().vectors;

        // What the watcher does for an edit, without waiting on the
        // filesystem to report one.
        let source = project.path().join("src").join("upload.rs");
        std::fs::write(
            &source,
            "/// Retries a failed upload with exponential backoff.\npub fn retry_failed_upload(attempts: u32) -> bool { attempts > 0 }\n\n\
             /// Gives up on an upload for good.\npub fn abandon_upload(id: u64) -> u64 { id }\n",
        )
        .unwrap();
        let shared = ws.current_embedder().unwrap();
        indexer::reindex_file(
            ws.db.as_ref(),
            &source.to_string_lossy(),
            Some(&mut *shared.lock().unwrap()),
            Some(&ws.root.to_string_lossy()),
        )
        .unwrap();
        assert_eq!(ws.db.upgrade_state().unwrap().pending_files, 1);
        assert_eq!(ws.text_set(), VectorSet::Upgrade, "one file behind is still the set to search");
        ws.upgrade_wake.notify_one();

        until("the edited file", || ws.db.upgrade_state().unwrap().pending_files == 0).await;
        let state = ws.db.upgrade_state().unwrap();
        assert!(state.vectors > before, "the new function has a vector in the second set too");
        assert_eq!(state.vectors, ws.db.index_stats().unwrap().2);
        worker.abort();
    }

    /// Two servers on one repository: the second leaves the upgrade to the
    /// first and takes it up when the first is gone.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_upgrade_is_left_to_the_process_that_is_already_at_it() {
        let (project, cache) = (project(false), cache(Some(true)));
        let ws = run(&project, &cache).await;
        let other_process = match lock_upgrade(&ws.db_path) {
            UpgradeLock::Held(file) => file,
            _ => panic!("nothing else holds the lock yet"),
        };
        assert_eq!(ws.upgrade_once().await, UpgradeStage::Waiting);
        assert_eq!(ws.db.upgrade_state().unwrap().vectors, 0);
        assert_eq!(ws.text_set(), VectorSet::Primary);

        drop(other_process);
        assert_eq!(ws.upgrade_once().await, UpgradeStage::Ready);
        assert_eq!(ws.text_set(), VectorSet::Upgrade);
        // And the lock is free again for whoever comes next.
        assert!(matches!(lock_upgrade(&ws.db_path), UpgradeLock::Held(_)));
    }

}
