use crate::db::IndexDb;
use crate::embeddings::SharedEmbedder;
use crate::indexer;
use ignore::gitignore::Gitignore;
use notify::Config;
use notify::RecommendedWatcher;
use notify::RecursiveMode;
use notify::Watcher;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

/// The embedder is loaded once per process and may arrive after the watcher
/// started (it loads in parallel with the first scan), so the watcher asks
/// for it at every batch instead of capturing it at start.
pub type EmbedderSource = Arc<dyn Fn() -> Option<SharedEmbedder> + Send + Sync>;

/// What happened to one file. Claudinio Code forwards these to the UI; the
/// MCP server only logs them.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum WatchEvent {
    Reindexing { file: String },
    Reindexed { file: String, symbols: usize },
    ReindexError { file: String, error: String },
}

pub type WatchEventSink = Arc<dyn Fn(WatchEvent) + Send + Sync>;

/// True if any path component is hidden (starts with `.`) or a well-known
/// junk directory. Mirrors the `.hidden(true)` rule the initial workspace
/// scan already applies (see `indexer::scan_workspace`), so the watcher
/// doesn't reindex `.git`, `.agents`, `.claudinio`, `node_modules`, etc.
fn is_ignored_path(root: &Path, path: &Path, gitignore: &Gitignore) -> bool {
    for component in path.components() {
        if let std::path::Component::Normal(name) = component {
            let name = name.to_string_lossy();
            if name.starts_with('.') || name == "node_modules" || name == "dist" || name == "target"
            {
                return true;
            }
        }
    }
    if let Ok(rel) = path.strip_prefix(root)
        && gitignore.matched(rel, false).is_ignore()
    {
        return true;
    }
    false
}

/// Watch events only for files the indexer can actually parse — derived from
/// the parsers themselves, not a hand-maintained extension list (the old list
/// covered 9 of ~74 indexed languages, so most files went stale until a full
/// workspace reopen).
fn is_indexable_file(path: &Path) -> bool {
    let p = path.to_string_lossy();
    crate::parser::detect_language(&p).is_some()
        || crate::parser::detect_doc_language(&p).is_some()
}

pub struct FileWatcher {
    _watcher: RecommendedWatcher,
}

impl FileWatcher {
    pub fn start(
        root: &str,
        db_path: &Path,
        embedder_source: EmbedderSource,
        on_event: WatchEventSink,
    ) -> Result<Self, String> {
        let db_path = db_path.to_path_buf();
        let root_owned = root.to_string();

        let (gitignore, _) = Gitignore::new(Path::new(root).join(".gitignore"));

        // Single debounced worker: batches of paths are collected here and
        // reindexed serially with one DB connection, instead of spawning a
        // thread (and opening a new SQLite connection) per filesystem event.
        let (tx, rx) = mpsc::channel::<String>();

        {
            let db_p = db_path.clone();
            let ws = root_owned.clone();
            std::thread::spawn(move || {
                // Dedicated thread — hold the demotion for its whole life.
                let _prio = crate::thread_priority::BackgroundPriority::begin();
                loop {
                    // Block for the first path in a batch.
                    let first = match rx.recv() {
                        Ok(p) => p,
                        Err(_) => return,
                    };
                    let mut pending: HashSet<String> = HashSet::new();
                    pending.insert(first);

                    // Coalesce further events for a short window so bursts
                    // (e.g. a save-all, or a build writing many files) collapse
                    // into a single reindex pass.
                    std::thread::sleep(Duration::from_millis(1500));
                    while let Ok(p) = rx.try_recv() {
                        pending.insert(p);
                    }

                    // Defer while a full scan/embedding pass holds the index
                    // semaphore — reindexing on top of it doubled the CPU and
                    // contended the embedder lock. Keep coalescing while we
                    // wait; nothing is dropped.
                    let _permit = loop {
                        match crate::INDEX_SEMAPHORE.try_acquire() {
                            Ok(p) => break p,
                            Err(_) => {
                                std::thread::sleep(Duration::from_secs(2));
                                while let Ok(p) = rx.try_recv() {
                                    pending.insert(p);
                                }
                            }
                        }
                    };

                    let db = match IndexDb::open(&db_p) {
                        Ok(d) => d,
                        Err(_) => continue,
                    };

                    let embedder = embedder_source();

                    for path_str in &pending {
                        let p = Path::new(path_str);
                        if !p.exists() {
                            // Explicit-delete path so the FTS delete triggers
                            // fire (cascades alone leave ghost FTS rows).
                            let _ = db.delete_file(path_str);
                            continue;
                        }

                        on_event(WatchEvent::Reindexing {
                            file: path_str.clone(),
                        });

                        let mut emb = embedder.as_ref().and_then(|e| e.lock().ok());
                        match indexer::reindex_file(&db, path_str, emb.as_deref_mut(), Some(&ws)) {
                            Ok(Some(result)) => {
                                on_event(WatchEvent::Reindexed {
                                    file: path_str.clone(),
                                    symbols: result.symbols.len(),
                                });
                            }
                            Ok(None) => {}
                            Err(e) => {
                                on_event(WatchEvent::ReindexError {
                                    file: path_str.clone(),
                                    error: e,
                                });
                            }
                        }
                    }
                }
            });
        }

        let watch_root = std::path::PathBuf::from(&root_owned);
        let mut watcher =
            notify::recommended_watcher(move |event: Result<notify::Event, notify::Error>| {
                let event = match event {
                    Ok(e) => e,
                    Err(_) => return,
                };

                for p in &event.paths {
                    if !p.is_file() {
                        continue;
                    }
                    if !is_indexable_file(p) {
                        continue;
                    }
                    if is_ignored_path(&watch_root, p, &gitignore) {
                        continue;
                    }
                    let _ = tx.send(p.to_string_lossy().to_string());
                }
            })
            .map_err(|e| format!("watcher create: {e}"))?;

        // The native backends (FSEvents/inotify/ReadDirectoryChangesW) ignore
        // poll_interval, but if notify ever falls back to PollWatcher this is
        // how often the whole workspace tree gets rescanned. 2s here caused
        // sustained background CPU on Windows machines where the native watch
        // failed silently — keep the fallback lazy.
        watcher
            .configure(Config::default().with_poll_interval(Duration::from_secs(60)))
            .map_err(|e| format!("watcher config: {e}"))?;
        eprintln!(
            "[watcher] started for {root_owned} (backend: {})",
            std::any::type_name::<RecommendedWatcher>()
        );

        watcher
            .watch(Path::new(root), RecursiveMode::Recursive)
            .map_err(|e| format!("watcher watch: {e}"))?;

        Ok(FileWatcher { _watcher: watcher })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watcher_accepts_all_indexable_extensions() {
        for accepted in [
            "src/main.go",
            "app/Main.kt",
            "styles/theme.scss",
            "Cargo.toml",
            "src/lib.rs",
            "docs/guide.md",
            "notes.txt",
            "web/app.tsx",
            "api/server.rb",
            "native/window.cpp",
        ] {
            assert!(
                is_indexable_file(Path::new(accepted)),
                "should accept {accepted}"
            );
        }
        for rejected in ["logo.png", "pnpm-lock.lock", "video.mp4", "binary.bin"] {
            assert!(
                !is_indexable_file(Path::new(rejected)),
                "should reject {rejected}"
            );
        }
    }

    #[test]
    fn ignores_hidden_and_junk_dirs() {
        let root = Path::new("/tmp/ws");
        let (gi, _) = Gitignore::new(root.join(".gitignore"));
        assert!(is_ignored_path(root, &root.join(".claudinio/foo.ts"), &gi));
        assert!(is_ignored_path(root, &root.join(".agents/bar.ts"), &gi));
        assert!(is_ignored_path(root, &root.join(".git/hooks/foo.rs"), &gi));
        assert!(is_ignored_path(
            root,
            &root.join("node_modules/pkg/index.js"),
            &gi
        ));
        assert!(is_ignored_path(root, &root.join("dist/bundle.js"), &gi));
        assert!(is_ignored_path(
            root,
            &root.join("target/debug/foo.rs"),
            &gi
        ));
        assert!(!is_ignored_path(root, &root.join("src/main.rs"), &gi));
    }
}
