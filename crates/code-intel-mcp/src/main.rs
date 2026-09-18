//! `claudinio-code-intel` — Claudinio Code's code intelligence as an MCP
//! server over stdio, so Claude Code, Cursor and GitHub Copilot get the same
//! `semantic_search` / `code_search` / `symbol_lookup` / `file_outline` /
//! `find_callers` tools the Claudinio Code agent has.
//!
//! The workspace is `--workspace` / `$CODE_INTEL_WORKSPACE` when given, else
//! the process cwd (the project directory under all three hosts); the client's
//! MCP roots are added on `initialized` when it advertises them.

mod workspace;

use claudinio_code_intel::db::SemanticSearchResult;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::service::{NotificationContext, RoleServer};
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use workspace::{Phase, Workspace};

const SNIPPET_TOP_HITS: usize = 5;
const SNIPPET_MAX_LINES: usize = 40;
const SNIPPET_MAX_CHARS: usize = 2400;

/// How long `semantic_search` waits for the embedding model before running
/// the BM25 leg alone. Same value as the Claudinio Code agent tool.
const MODEL_WAIT_MS: u64 = 5000;

#[derive(Clone)]
struct Options {
    cache_dir: PathBuf,
    embeddings: bool,
}

#[derive(Clone)]
struct CodeIntel {
    tool_router: ToolRouter<Self>,
    opts: Options,
    /// Open workspaces; the first is the default. Roots reported by the
    /// client are appended on `initialized`.
    workspaces: Arc<RwLock<Vec<Arc<Workspace>>>>,
}

// ── tool parameter types ────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct QueryParams {
    /// Search term (identifier, partial name or signature fragment).
    query: String,
    /// Maximum results (default 20).
    #[serde(default)]
    limit: Option<i64>,
    /// Workspace root to search when more than one is open (default: first).
    #[serde(default)]
    workspace: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct SemanticParams {
    /// Natural-language description of the code's functionality, in English
    /// (the index is English-only — translate first).
    query: String,
    /// Maximum results (default 15).
    #[serde(default)]
    limit: Option<i64>,
    /// Workspace root to search when more than one is open (default: first).
    #[serde(default)]
    workspace: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct NameParams {
    /// Exact symbol name (case-insensitive).
    name: String,
    /// Workspace root to search when more than one is open (default: first).
    #[serde(default)]
    workspace: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct FileParams {
    /// Path to the file, absolute or relative to the workspace root.
    file_path: String,
    /// Workspace root when more than one is open (default: the one containing the file).
    #[serde(default)]
    workspace: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct CallersParams {
    /// Name of the called symbol.
    name: String,
    /// File that defines it; callers inside this file are excluded (default: none excluded).
    #[serde(default)]
    file_path: Option<String>,
    /// Workspace root to search when more than one is open (default: first).
    #[serde(default)]
    workspace: Option<String>,
}

#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
struct WorkspaceParams {
    /// Workspace root to report on (default: all open workspaces).
    #[serde(default)]
    workspace: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct OpenParams {
    /// Absolute path of the directory to index.
    path: String,
}

// ── helpers ─────────────────────────────────────────────────────────────

fn pretty<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string_pretty(v).unwrap_or_else(|e| format!("{{\"error\":\"serialize: {e}\"}}"))
}

/// Read the source of the top hits into `snippet`, capped so a tool result
/// never swallows the context window.
fn attach_snippets(results: &mut [SemanticSearchResult]) {
    for r in results.iter_mut().take(SNIPPET_TOP_HITS) {
        let Ok(content) = std::fs::read_to_string(&r.file_path) else {
            continue;
        };
        // Lines are 1-based inclusive in the index.
        let start = r.start_line.max(1) as usize;
        let end = r.end_line.max(r.start_line).max(1) as usize;
        let mut snippet: String = content
            .lines()
            .skip(start - 1)
            .take((end - start + 1).min(SNIPPET_MAX_LINES))
            .collect::<Vec<_>>()
            .join("\n");
        if snippet.len() > SNIPPET_MAX_CHARS {
            let cut = snippet
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|i| *i <= SNIPPET_MAX_CHARS)
                .last()
                .unwrap_or(0);
            snippet.truncate(cut);
            snippet.push_str("\n… [truncated — read the file for the rest]");
        }
        if !snippet.is_empty() {
            r.snippet = Some(snippet);
        }
    }
}

fn canonical(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && let (Some(h), Some(l)) = (
                bytes.get(i + 1).and_then(|c| (*c as char).to_digit(16)),
                bytes.get(i + 2).and_then(|c| (*c as char).to_digit(16)),
            )
        {
            out.push((h * 16 + l) as u8);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `file://` URI (what MCP roots carry) → directory path. Anything else is
/// ignored: a root we cannot walk is not a workspace.
fn root_uri_to_path(uri: &str) -> Option<PathBuf> {
    let rest = uri.strip_prefix("file://")?;
    let rest = rest.strip_prefix("localhost").unwrap_or(rest);
    let decoded = percent_decode(rest);
    #[cfg(windows)]
    let decoded = decoded.trim_start_matches('/').replace('/', "\\");
    let p = PathBuf::from(decoded);
    p.is_dir().then_some(p)
}

impl CodeIntel {
    fn new(opts: Options) -> Self {
        Self {
            tool_router: Self::tool_router(),
            opts,
            workspaces: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Open (and start indexing) a root unless it is already open.
    async fn open_root(&self, root: PathBuf) -> Result<Arc<Workspace>, String> {
        let root = canonical(&root);
        if !root.is_dir() {
            return Err(format!("not a directory: {}", root.display()));
        }
        {
            let list = self.workspaces.read().await;
            if let Some(ws) = list.iter().find(|w| w.root == root) {
                return Ok(ws.clone());
            }
        }
        let ws = Workspace::open(root.clone(), &self.opts.cache_dir)?;
        self.workspaces.write().await.push(ws.clone());
        tracing::info!(root = %root.display(), db = %ws.db_path.display(), "workspace opened");
        tokio::spawn(
            ws.clone()
                .run_pipeline(self.opts.cache_dir.clone(), self.opts.embeddings),
        );
        Ok(ws)
    }

    /// The workspace a tool call refers to: explicit `workspace`, else the
    /// one containing `file_path`, else the first open one.
    async fn pick(
        &self,
        workspace: Option<&str>,
        file_path: Option<&str>,
    ) -> Result<Arc<Workspace>, String> {
        let list = self.workspaces.read().await;
        if list.is_empty() {
            return Err("no workspace open — call open_workspace with a directory".into());
        }
        if let Some(w) = workspace {
            let want = canonical(Path::new(w));
            return list
                .iter()
                .find(|ws| ws.root == want)
                .cloned()
                .ok_or_else(|| {
                    let open: Vec<String> =
                        list.iter().map(|w| w.root.display().to_string()).collect();
                    format!(
                        "workspace {} is not open; open: {}",
                        want.display(),
                        open.join(", ")
                    )
                });
        }
        if let Some(f) = file_path {
            let fp = Path::new(f);
            if fp.is_absolute()
                && let Some(ws) = list.iter().find(|ws| fp.starts_with(&ws.root))
            {
                return Ok(ws.clone());
            }
        }
        Ok(list[0].clone())
    }

    fn require_symbols(ws: &Workspace) -> Result<(), String> {
        if ws.symbols_ready() {
            return Ok(());
        }
        let progress = ws.progress.lock().ok().and_then(|p| p.clone());
        Err(match (ws.phase(), progress) {
            (Phase::Failed, _) => format!(
                "index failed for {}: {}",
                ws.root.display(),
                ws.error.lock().ok().and_then(|e| e.clone()).unwrap_or_default()
            ),
            (_, Some(p)) => format!(
                "index not ready: {} of {} files scanned in {} — retry in a moment (or grep meanwhile)",
                p.files_indexed,
                p.total_files,
                ws.root.display()
            ),
            _ => format!("index not ready for {}", ws.root.display()),
        })
    }
}

// ── tools ───────────────────────────────────────────────────────────────

#[tool_router]
impl CodeIntel {
    #[tool(
        name = "index_status",
        description = "State of the local code index: phase (indexing | embedding | lexical_only | ready | failed), scan/embedding progress and counts of indexed files, symbols and embeddings. Call this when a search tool says the index is not ready."
    )]
    async fn index_status(
        &self,
        Parameters(p): Parameters<WorkspaceParams>,
    ) -> Result<String, String> {
        let selected = match p.workspace {
            Some(w) => vec![self.pick(Some(&w), None).await?],
            None => self.workspaces.read().await.clone(),
        };
        let mut out = Vec::new();
        for ws in selected {
            let (files, symbols, embeddings) = ws.db.index_stats().unwrap_or((0, 0, 0));
            let scan = ws.progress.lock().ok().and_then(|p| p.clone());
            let embed = ws.embed_progress.lock().ok().and_then(|p| p.clone());
            out.push(serde_json::json!({
                "workspace": ws.root,
                "phase": ws.phase(),
                "scan": scan,
                "embedding": embed,
                "files": files,
                "symbols": symbols,
                "embeddings": embeddings,
                "embeddingsPending": ws.db.embedding_pending_files().unwrap_or(0),
                "watcherWarning": ws.watcher_warning.lock().ok().and_then(|w| w.clone()),
                "error": ws.error.lock().ok().and_then(|e| e.clone()),
                "indexDb": ws.db_path,
            }));
        }
        Ok(pretty(&out))
    }

    #[tool(
        name = "open_workspace",
        description = "Index an additional directory (absolute path). The client's workspace is opened automatically at startup; use this only for a directory outside it."
    )]
    async fn open_workspace(&self, Parameters(p): Parameters<OpenParams>) -> Result<String, String> {
        let ws = self.open_root(PathBuf::from(&p.path)).await?;
        Ok(pretty(&serde_json::json!({
            "workspace": ws.root,
            "phase": ws.phase(),
        })))
    }

    #[tool(
        name = "code_search",
        description = "Full-text search across indexed symbol names and signatures (FTS5). Faster and more targeted than grep for finding definitions — prefer this over grep. Searches names/signatures only; for words inside code bodies or docs use semantic_search."
    )]
    async fn code_search(&self, Parameters(p): Parameters<QueryParams>) -> Result<String, String> {
        let ws = self.pick(p.workspace.as_deref(), None).await?;
        Self::require_symbols(&ws)?;
        let results = ws.db.search_symbols(&p.query, p.limit.unwrap_or(20).max(1))?;
        Ok(pretty(&results))
    }

    #[tool(
        name = "symbol_lookup",
        description = "Look up a symbol by exact name across the workspace (case-insensitive). Use when you know the exact symbol name; use code_search for partial names."
    )]
    async fn symbol_lookup(&self, Parameters(p): Parameters<NameParams>) -> Result<String, String> {
        let ws = self.pick(p.workspace.as_deref(), None).await?;
        Self::require_symbols(&ws)?;
        let results = ws.db.lookup_symbols_exact(&p.name, 20)?;
        Ok(pretty(&results))
    }

    #[tool(
        name = "file_outline",
        description = "List all symbols defined in a file (functions, classes, methods, types…) with their line ranges. Use it before reading a file to see its structure at a glance."
    )]
    async fn file_outline(&self, Parameters(p): Parameters<FileParams>) -> Result<String, String> {
        let ws = self.pick(p.workspace.as_deref(), Some(&p.file_path)).await?;
        Self::require_symbols(&ws)?;
        let path = ws.resolve_path(&p.file_path);
        let results = ws.db.symbols_in_file(&path)?;
        Ok(pretty(&results))
    }

    #[tool(
        name = "find_callers",
        description = "Symbols that call or reference a symbol by name, from the call relations tree-sitter recorded at index time. Cheaper than grep for 'who uses this?'; results are the definitions of the callers, not every textual occurrence."
    )]
    async fn find_callers(&self, Parameters(p): Parameters<CallersParams>) -> Result<String, String> {
        let ws = self.pick(p.workspace.as_deref(), p.file_path.as_deref()).await?;
        Self::require_symbols(&ws)?;
        let exclude = p.file_path.map(|f| ws.resolve_path(&f)).unwrap_or_default();
        let results = ws.db.callers_of(&p.name, &exclude)?;
        Ok(pretty(&results))
    }

    #[tool(
        name = "semantic_search",
        description = "Hybrid code & documentation search: BM25 keyword matching over code bodies, docs and file paths, fused with MiniLM semantic embeddings. Finds code by exact identifiers, rare terms and file names AND by meaning/behavior — e.g. 'message queue system' finds a drain/push/queue implementation without an identifier match. Prefer this whenever you don't have a precise symbol name. The index is ENGLISH-ONLY: translate the query to English first. Response is {mode, note?, results}: mode is 'hybrid' or 'lexical-only' (while the embedding model loads), each result has score (relative confidence in (0,1]) and matchType ('hybrid'|'semantic'|'lexical'); the top results include a source snippet. Ranking: semantic_search → code_search (symbol names) → grep (fallback)."
    )]
    async fn semantic_search(
        &self,
        Parameters(p): Parameters<SemanticParams>,
    ) -> Result<String, String> {
        let ws = self.pick(p.workspace.as_deref(), None).await?;
        Self::require_symbols(&ws)?;
        let limit = p.limit.unwrap_or(15).max(1) as usize;

        // Give a model that is still loading a moment, then run whichever leg
        // is available — one code path for every situation.
        let mut model = ws.current_embedder();
        if model.is_none() && matches!(ws.phase(), Phase::Indexing | Phase::Embedding) {
            let mut waited = 0u64;
            while model.is_none() && waited < MODEL_WAIT_MS {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                waited += 500;
                model = ws.current_embedder();
            }
        }
        let query_vec: Option<Vec<f32>> = match model {
            Some(model) => {
                let q = p.query.clone();
                let vec = tokio::task::spawn_blocking(move || {
                    let mut m = model.lock().map_err(|e| format!("embedder lock: {e}"))?;
                    m.encode_query(&q)
                })
                .await
                .map_err(|e| format!("encode task panicked: {e}"))?;
                match vec {
                    Ok(v) => Some(v),
                    Err(e) => {
                        tracing::warn!("query embedding failed, lexical only: {e}");
                        None
                    }
                }
            }
            None => None,
        };

        let mut results = ws.db.search_hybrid(&p.query, query_vec.as_deref(), limit)?;
        attach_snippets(&mut results);

        let pending = ws.db.embedding_pending_files().unwrap_or(0);
        let mode = if query_vec.is_some() { "hybrid" } else { "lexical-only" };
        let note = if query_vec.is_none() {
            Some(if pending > 0 {
                format!(
                    "embedding model unavailable and {pending} files not yet embedded; keyword (BM25) matches only — semantic ranking joins once the model loads"
                )
            } else {
                "embedding model unavailable; keyword (BM25) matches only".to_string()
            })
        } else if pending > 0 {
            Some(format!(
                "{pending} files still embedding in the background; semantic ranking improves when that finishes"
            ))
        } else {
            None
        };
        let mut envelope = serde_json::json!({ "mode": mode, "results": results });
        if let Some(n) = note {
            envelope["note"] = serde_json::json!(n);
        }
        Ok(pretty(&envelope))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for CodeIntel {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new("claudinio-code-intel", env!("CARGO_PKG_VERSION"))
                    .with_title("Claudinio Code Intel")
                    .with_description(
                        "Local hybrid code search: tree-sitter symbols + BM25 + MiniLM embeddings. Nothing leaves the machine.",
                    )
                    .with_website_url("https://github.com/claudin-io/claudinio-code-intel"),
            )
            .with_instructions(
                "Code navigation for the open workspace, indexed locally. Prefer semantic_search \
                 (meaning or a rare term anywhere in code/docs) and code_search (symbol names) over \
                 grep; file_outline before reading a whole file; find_callers for usages. Queries \
                 must be in English. If a tool reports the index is not ready, call index_status \
                 and retry — the first scan takes seconds, embeddings run for a few minutes in the \
                 background and search works lexically meanwhile.",
            )
    }

    #[allow(deprecated)] // roots: SEP-2577 deprecates them, but every host still sends them
    async fn on_initialized(&self, context: NotificationContext<RoleServer>) {
        // The client's roots are the real workspace; cwd was only a guess.
        // Ask only when the client said it can answer — Claude Code advertises
        // roots, older clients reply with an error we would rather not log.
        let advertises_roots = context
            .peer
            .peer_info()
            .and_then(|i| i.capabilities.roots.clone())
            .is_some();
        if !advertises_roots {
            return;
        }
        let roots = match context.peer.list_roots().await {
            Ok(r) => r.roots,
            Err(e) => {
                tracing::debug!("roots/list failed: {e}");
                Vec::new()
            }
        };
        for root in roots {
            if let Some(path) = root_uri_to_path(&root.uri)
                && let Err(e) = self.open_root(path).await
            {
                tracing::warn!("could not open root {}: {e}", root.uri);
            }
        }
    }
}

// ── entry point ─────────────────────────────────────────────────────────

fn usage() -> ! {
    eprintln!(
        "claudinio-code-intel {}\n\n\
         MCP server (stdio) for local hybrid code search.\n\n\
         USAGE: claudinio-code-intel [--workspace <dir>]... [--cache-dir <dir>] [--no-embeddings]\n\
         \x20      claudinio-code-intel index [--workspace <dir>] [--cache-dir <dir>] [--no-embeddings]\n\n\
         The workspace defaults to $CODE_INTEL_WORKSPACE, then the cwd; the client's MCP roots\n\
         are indexed too. `index` builds the index and exits (warm a cache in CI, or debug).\n\
         Env: CODE_INTEL_CACHE_DIR, CODE_INTEL_EMBEDDINGS=0, CODE_INTEL_LOG=<filter>.\n\
         Cache: {}",
        env!("CARGO_PKG_VERSION"),
        workspace::default_cache_dir().display()
    );
    std::process::exit(2)
}

struct Cli {
    workspaces: Vec<PathBuf>,
    opts: Options,
    index_only: bool,
}

fn parse_args() -> Cli {
    let mut args = std::env::args().skip(1);
    let mut cli = Cli {
        workspaces: Vec::new(),
        opts: Options {
            cache_dir: std::env::var_os("CODE_INTEL_CACHE_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(workspace::default_cache_dir),
            embeddings: std::env::var("CODE_INTEL_EMBEDDINGS")
                .map(|v| v != "0")
                .unwrap_or(true),
        },
        index_only: false,
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "index" => cli.index_only = true,
            "--workspace" | "-w" => {
                let v = args.next().unwrap_or_else(|| usage());
                // A host that did not expand its `${...}` placeholder passes
                // it through literally; ignore it rather than fail startup.
                if !v.is_empty() && !v.contains("${") {
                    cli.workspaces.push(PathBuf::from(v));
                }
            }
            "--cache-dir" => {
                cli.opts.cache_dir = PathBuf::from(args.next().unwrap_or_else(|| usage()))
            }
            "--no-embeddings" => cli.opts.embeddings = false,
            "--version" | "-V" => {
                println!("claudinio-code-intel {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0)
            }
            "--help" | "-h" => usage(),
            other => {
                eprintln!("unknown argument: {other}");
                usage()
            }
        }
    }
    if cli.workspaces.is_empty()
        && let Some(v) = std::env::var_os("CODE_INTEL_WORKSPACE")
        && !v.is_empty()
    {
        cli.workspaces.push(PathBuf::from(v));
    }
    cli
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // stdout is the MCP transport; every log line goes to stderr.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("CODE_INTEL_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,ort=warn")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let cli = parse_args();

    if cli.index_only {
        let root = cli
            .workspaces
            .first()
            .cloned()
            .unwrap_or(std::env::current_dir()?);
        let ws = Workspace::open(canonical(&root), &cli.opts.cache_dir)?;
        ws.clone()
            .run_pipeline(cli.opts.cache_dir.clone(), cli.opts.embeddings)
            .await;
        let (files, symbols, embeddings) = ws.db.index_stats().unwrap_or((0, 0, 0));
        println!(
            "{}",
            pretty(&serde_json::json!({
                "workspace": ws.root, "phase": ws.phase(),
                "files": files, "symbols": symbols, "embeddings": embeddings,
                "indexDb": ws.db_path,
                "error": ws.error.lock().ok().and_then(|e| e.clone()),
            }))
        );
        return if ws.phase() == Phase::Failed { std::process::exit(1) } else { Ok(()) };
    }

    let server = CodeIntel::new(cli.opts.clone());

    // Explicit workspaces (or cwd) start indexing before the handshake so the
    // first search lands on a warm index; client roots are added on
    // `initialized` and no-op when they are the same directory.
    let initial = if cli.workspaces.is_empty() {
        vec![std::env::current_dir()?]
    } else {
        cli.workspaces.clone()
    };
    for root in initial {
        if let Err(e) = server.open_root(root).await {
            tracing::warn!("{e}");
        }
    }

    let service = server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}
