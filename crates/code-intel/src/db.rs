use crate::embeddings::ModelProfile;
use crate::media::{MEDIA_KINDS_SQL, MediaKind};
use rusqlite::{Connection, params};
use serde::Serialize;
use std::path::Path;
use std::sync::Mutex;

pub struct IndexDb {
    pub conn: Mutex<Connection>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileRecord {
    pub id: i64,
    pub path: String,
    pub language: Option<String>,
    pub hash: Option<String>,
    pub last_modified: i64,
    pub size: i64,
    /// Content hash the symbol_embeddings for this file were last generated
    /// from. Compared against `hash` to skip re-embedding unchanged files.
    pub embed_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SymbolRecord {
    pub id: i64,
    pub file_id: i64,
    pub name: String,
    pub kind: String,
    pub signature: Option<String>,
    pub start_line: i64,
    pub start_col: i64,
    pub end_line: i64,
    pub end_col: i64,
    pub file_path: Option<String>,
}

/// One embedded chunk of a symbol as stored: the symbol, the chunk's start and
/// end line, and the vector. Both line numbers are 0 for whole-symbol
/// embeddings (headers, small bodies).
pub type EmbeddingRow = (SymbolRecord, i64, i64, Vec<f32>);

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResult {
    pub symbol_id: i64,
    pub name: String,
    pub kind: String,
    pub file_path: String,
    pub start_line: i64,
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticSearchResult {
    pub symbol_id: i64,
    pub name: String,
    pub kind: String,
    pub file_path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub signature: Option<String>,
    pub score: f32,
    /// Which retrieval evidence produced this hit: "hybrid" (both legs),
    /// "semantic" (vector only), or "lexical" (BM25 only).
    pub match_type: String,
    /// Source excerpt of the symbol, filled in by the tool layer for top hits.
    pub snippet: Option<String>,
}

/// An image or audio file matched by `IndexDb::search_media`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MediaSearchResult {
    pub symbol_id: i64,
    /// File name.
    pub name: String,
    /// "image" or "audio".
    pub kind: String,
    pub file_path: String,
    pub score: f32,
    /// "hybrid", "semantic" (content only) or "lexical" (file name only).
    pub match_type: String,
}

/// What `IndexDb::reconcile_embedding_model` had to discard.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmbeddingReset {
    /// The index held text vectors of another model; they were dropped.
    pub model_changed: bool,
    /// The index held media vectors of another model; they were dropped.
    pub media_model_changed: bool,
    /// Media files queued again because an encoder for them is now loaded.
    pub media_requeued: i64,
}

/// One persisted retrieval chunk, as fed to the embedding pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredChunk {
    pub symbol_id: i64,
    pub chunk_index: i64,
    pub start_line: i64,
    pub end_line: i64,
    pub embed_text: String,
}

/// Bump when the index format changes (schema, embedding layout, ignore
/// rules). A mismatched on-disk index is deleted and rebuilt from scratch.
///
/// Not bumped for `index_meta`, media rows and the upgrade set: all are
/// additive, and a model change is handled by `reconcile_embedding_model`,
/// which drops vectors only — a v7 index that stays on MiniLM keeps every one
/// of its embeddings.
const SCHEMA_VERSION: i64 = 7;

const META_EMBED_MODEL: &str = "embed_model";
const META_MEDIA_MODEL: &str = "media_model";
const META_MEDIA_STATE: &str = "media_state";
const META_UPGRADE_MODEL: &str = "upgrade_model";
const META_UPGRADE_COMPLETE: &str = "upgrade_complete";

/// An index can hold the text of a workspace embedded twice.
///
/// The primary set is written first, by a model fast enough that search is
/// useful minutes after a workspace is opened. The upgrade set holds the same
/// chunks embedded by a slower model that ranks better, filled in the
/// background over the following minutes or hours; search moves to it once it
/// covers the workspace (`UpgradeState::usable`) and back whenever the model
/// that wrote it is not there to embed the query. Vectors of the two are
/// never compared with each other, which is why they do not share a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorSet {
    /// `symbol_embeddings`: every index has it. Media vectors live here too.
    Primary,
    /// `symbol_embeddings_upgrade`: text only.
    Upgrade,
}

impl VectorSet {
    fn table(self) -> &'static str {
        match self {
            VectorSet::Primary => "symbol_embeddings",
            VectorSet::Upgrade => "symbol_embeddings_upgrade",
        }
    }
}

/// How far behind the upgrade set may be and still be the one searched: this
/// many files, or one file in twenty, whichever is more.
///
/// Zero would hand search back to the primary set on every save, and to a
/// different ranking with it, for the seconds the slow model needs to catch
/// up; a file it has not reached yet is still found by its text meanwhile.
/// What this allowance does not cover is a change of branch that rewrites
/// half the workspace — there the primary set, current again within a
/// minute, is the better one to search until the upgrade set has caught up.
const UPGRADE_LAG_FILES: i64 = 3;
const UPGRADE_LAG_DIVISOR: i64 = 20;

/// Where an index's upgrade set stands.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UpgradeState {
    /// The model that wrote it, once anything has.
    pub model: Option<String>,
    /// A pass has covered every text file of the workspace at least once.
    pub complete: bool,
    /// Text files whose current content it does not hold.
    pub pending_files: i64,
    pub text_files: i64,
    pub vectors: i64,
}

impl UpgradeState {
    /// Whether search should use this set, given that `model` is the one
    /// loaded to embed queries with.
    pub fn usable(&self, model: &str) -> bool {
        self.complete
            && self.model.as_deref() == Some(model)
            && self.pending_files <= UPGRADE_LAG_FILES.max(self.text_files / UPGRADE_LAG_DIVISOR)
    }
}

/// Tuning knobs for `search_hybrid_with`. `Default` holds the production
/// values, calibrated with `examples/semantic_eval.rs --sweep` — re-run the
/// sweep before changing any of them by hand.
#[derive(Debug, Clone)]
pub struct HybridParams {
    /// RRF smoothing constant: higher flattens the rank curve.
    pub rrf_k: f32,
    /// Leg weights in the fused score.
    pub w_vector: f32,
    pub w_bm25: f32,
    /// Candidates kept per leg (best chunk per symbol) before fusion.
    pub k_candidates: usize,
    /// Entry gate for the vector leg: chunks below this raw cosine never
    /// become candidates. Off-topic queries score ~0.36-0.44 against random
    /// code on MiniLM-class models — do not lower this; BM25 rescues the
    /// relevant hits that sit in that band (sweep 2026-07-20: 0.40 kept
    /// top-3/top-15 intact while halving negative-query leaks vs 0.35).
    pub min_cosine_candidate: f32,
    /// Final gate on the fused+boosted score, in the same [0, 1] range the
    /// old cosine threshold used.
    pub min_hybrid_score: f32,
    /// Floor of distinct query tokens a BM25-only hit must contain (see
    /// `required_token_matches`).
    pub min_bm25_term_matches: usize,
}

impl HybridParams {
    /// The knobs for the model an index was embedded with. Cosine scores are
    /// not comparable across models, so the vector gate is per model; the
    /// rank-based fusion after it is not.
    pub fn for_model(model_id: Option<&str>) -> Self {
        let mut params = HybridParams::default();
        if model_id.is_some_and(|m| m.starts_with("embeddinggemma-2")) {
            params.min_cosine_candidate = GEMMA2_MIN_COSINE_CANDIDATE;
        }
        params
    }
}

/// Vector gate for EmbeddingGemma 2, whose cosines sit in a narrow, high band:
/// nothing in a repository scores under 0.5 against anything, so a MiniLM-sized
/// gate lets every chunk through and every off-topic query returns a full page.
///
/// Measured on Claudinio Code (15.7k chunks, 59 queries, 2026-10-07): a
/// query's best chunk in the file it is looking for scores 0.730-0.869
/// (p25 0.793); chunks of other files have p99 0.748 and p99.9 0.785; the
/// best chunk for an off-topic query scores 0.735, 0.741, 0.749, 0.749 and
/// 0.791. 0.75 is the 99th percentile of unrelated chunks and turns four of
/// those five off-topic queries away at the vector leg, while sitting closer
/// to the weakest relevant score than MiniLM's 0.40 does on its own scale.
///
/// Chosen from those distributions. The sweep of that run stopped at 0.60, so
/// ranking with this gate has not been measured yet: `semantic_eval --sweep`
/// now sweeps the band the model's scores actually occupy — re-run it with
/// EmbeddingGemma 2 as the text model before moving this.
const GEMMA2_MIN_COSINE_CANDIDATE: f32 = 0.75;

/// Gate for an image or audio vector against a text query. Scores across
/// modalities sit on their own scale, and an unrelated picture is never far
/// from a query: on Claudinio Code (18 images, 2026-10-07) the best image for
/// each of 64 code queries scored 0.568-0.687 (median 0.621), so the 0.60
/// this started at would have put an unrelated file under `media` for 54 of
/// them. The two queries that describe an image scored 0.707 and 0.760 on it
/// and 0.586 and 0.578 on the best other one. 0.70 is over every code query
/// and under both descriptions — by a narrow margin, on two queries: a
/// description that matches loosely falls back to the file-name match.
///
/// Audio has no such measurement: that workspace has no sound files. The only
/// numbers are `gemma2_e2e`'s two synthetic clips, whose descriptions scored
/// 0.733 and 0.686 on the right clip and 0.704 and 0.630 on the wrong one —
/// no gate separates those, and real recordings may sit elsewhere.
pub const MEDIA_MIN_COSINE: f32 = 0.70;

impl Default for HybridParams {
    // Calibrated with `semantic_eval --sweep` on 2026-07-20 (59 positives /
    // 5 negatives from real sessions): top-1 67%, top-3 91%, top-15 98%,
    // exact-identifier/basename/body-term 100% top-1, negatives 3/5 empty
    // (the two leaks are genuine repo-vocabulary collisions, scored low).
    fn default() -> Self {
        HybridParams {
            rrf_k: 60.0,
            w_vector: 1.0,
            w_bm25: 1.0,
            k_candidates: 50,
            min_cosine_candidate: 0.40,
            min_hybrid_score: 0.35,
            min_bm25_term_matches: 2,
        }
    }
}

/// Doc sections (markdown headings) embed dense natural language, so they
/// consistently out-score code symbols on NL queries; this penalty keeps them
/// in the results without letting them crowd out the code the agent needs.
const DOC_SECTION_PENALTY: f32 = 0.12;

/// Applied to results living in test files (see `is_test_file`).
const TEST_FILE_PENALTY: f32 = 0.10;

/// Max results kept per source file before `limit` is applied, so a single
/// large file (many symbols) can't dominate the whole ranking.
const MAX_RESULTS_PER_FILE: usize = 3;

const STOPWORDS: &[&str] = &["the", "and", "for", "with"];

/// Lowercase, alphanumeric tokens of at least 3 chars, minus trivial stopwords.
fn tokenize_query(query_text: &str) -> Vec<String> {
    query_text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() >= 3 && !STOPWORDS.contains(&t.as_str()))
        .collect()
}

/// Max quoted terms in a MATCH expression — queries are short NL phrases;
/// beyond this, OR-recall only adds noise and cost.
const MAX_FTS_TERMS: usize = 12;

/// Standalone words dropped from FTS queries. Identifier runs (containing
/// `_`/`-`/`.`) are never filtered — "for" inside delete_symbols_for_file
/// stays part of the phrase.
const FTS_STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "this", "that", "does", "how", "what", "where", "when", "which",
    "code", "file",
];

/// BM25 column weights for chunk_fts (fts_name, fts_path, fts_body): a term
/// hit on the symbol name outweighs one on the path, which outweighs one in
/// the body.
const BM25_W_NAME: f64 = 3.0;
const BM25_W_PATH: f64 = 2.0;
const BM25_W_BODY: f64 = 1.0;

/// Build a safe FTS5 MATCH expression from free text. Runs of
/// `[A-Za-z0-9_.-]` are extracted, stopwords and 1-char runs dropped, each
/// run double-quoted (internal quotes doubled) so FTS5 operators (AND, OR,
/// NOT, NEAR, `*`, `^`, `-`, `:`) can never be injected, then joined with OR
/// for recall — BM25 ranking rewards multi-term hits, so OR does not flood
/// the top ranks. Quoted runs containing `_`/`-`/`.` tokenize into phrases,
/// which is what makes "delete_symbols_for_file" an exact adjacency match.
/// Returns None when nothing usable survives.
fn build_fts_match_query(query_text: &str) -> Option<String> {
    let mut terms: Vec<String> = Vec::new();
    for raw in query_text.split(|c: char| !(c.is_alphanumeric() || "_-.".contains(c))) {
        let run = raw.trim_matches(|c: char| "_-.".contains(c));
        if run.chars().count() < 2 {
            continue;
        }
        let is_identifier = run.contains('_') || run.contains('-') || run.contains('.');
        if !is_identifier && FTS_STOPWORDS.contains(&run.to_lowercase().as_str()) {
            continue;
        }
        let quoted = format!("\"{}\"", run.replace('"', "\"\""));
        if !terms.contains(&quoted) {
            terms.push(quoted);
        }
        if terms.len() >= MAX_FTS_TERMS {
            break;
        }
    }
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" OR "))
    }
}

/// Distinct query tokens a BM25-only hit must contain to survive. Floored at
/// `min_matches` (capped by the token count, so single-token exact-term
/// queries work) and raised to a majority for long NL queries — one
/// incidental common word can't drag junk in via OR-recall.
fn required_token_matches(n_tokens: usize, min_matches: usize) -> usize {
    min_matches.min(n_tokens).max(n_tokens.div_ceil(2)).max(1)
}

/// One candidate from a retrieval leg, reduced to the best chunk per symbol.
struct LegHit {
    symbol_id: i64,
    name: String,
    kind: String,
    signature: Option<String>,
    file_path: String,
    sym_start_line: i64,
    sym_end_line: i64,
    chunk_start: i64,
    chunk_end: i64,
    /// Raw cosine (vector leg only; 0.0 for BM25 hits) — ordering only.
    cosine: f32,
    /// Concatenated fts_* text (BM25 leg only) for the evidence gate.
    fts_text: String,
}

/// True when a hit found only by BM25 has enough lexical evidence to stand
/// without vector support: an exact name/basename/stem token, or at least
/// `required` distinct query tokens present as whole words in its FTS text.
fn bm25_only_hit_has_evidence(tokens: &[String], hit: &LegHit, required: usize) -> bool {
    let (base, stem) = basename_variants(&hit.file_path);
    let name_lower = hit.name.to_lowercase();
    if tokens
        .iter()
        .any(|t| *t == name_lower || *t == base || *t == stem)
    {
        return true;
    }
    let words: std::collections::HashSet<String> = hit
        .fts_text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect();
    tokens.iter().filter(|t| words.contains(t.as_str())).count() >= required
}

/// Returns the file's basename, with and without its extension.
fn basename_variants(file_path: &str) -> (String, String) {
    let base = Path::new(file_path)
        .file_name()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let stem = Path::new(file_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| base.clone());
    (base, stem)
}

/// Highest lexical boost across all query tokens for a given symbol name /
/// file path. Layers don't stack — only the best-matching layer counts.
/// A basename match outranks a symbol-name match: a query naming a file is a
/// strong navigation signal, but short symbol names collide with ordinary
/// query words ("task", "list") and shouldn't dominate the semantic score.
fn lexical_boost(tokens: &[String], name: &str, file_path: &str) -> f32 {
    let name_lower = name.to_lowercase();
    let (base, stem) = basename_variants(file_path);
    let mut boost = 0.0f32;
    for token in tokens {
        if token == &base || token == &stem {
            return 0.25;
        }
        if token == &name_lower {
            boost = boost.max(0.15);
            continue;
        }
        if name_lower.contains(token.as_str())
            || token.contains(name_lower.as_str())
            || base.contains(token.as_str())
            || token.contains(base.as_str())
            || stem.contains(token.as_str())
            || token.contains(stem.as_str())
        {
            boost = boost.max(0.10);
        }
    }
    boost
}

/// Test files answer "how is this used in tests", not "where does this live" —
/// they mirror the vocabulary of the code under test and crowd it out.
fn is_test_file(file_path: &str) -> bool {
    let lower = file_path.to_lowercase();
    let base = lower.rsplit('/').next().unwrap_or(&lower);
    base.contains(".test.")
        || base.contains(".spec.")
        || base.ends_with("_test.rs")
        || base.ends_with("_tests.rs")
        || lower.contains("/tests/")
        || lower.contains("/__tests__/")
}

/// Inline test symbols (Rust `mod tests`, `fn test_*`) live inside production
/// files, so `is_test_file` misses them — catch them by naming convention.
fn is_test_symbol(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.starts_with("test_")
        || lower.ends_with("_test")
        || lower.ends_with("_tests")
        || lower == "tests"
}

impl IndexDb {
    pub fn open(db_path: &Path) -> Result<Self, String> {
        // Ensure the parent directory exists — SQLite cannot create the file
        // when the directory it belongs to doesn't exist yet.
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create db dir {}: {e}", parent.display()))?;
        }
        let mut conn = Connection::open(db_path).map_err(|e| format!("db open: {e}"))?;

        let version: i64 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap_or(0);
        let is_empty: bool = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|c| c == 0)
            .unwrap_or(true);
        if !is_empty && version != SCHEMA_VERSION {
            eprintln!(
                "[index] stale index (version {version}, expected {SCHEMA_VERSION}) — rebuilding {}",
                db_path.display()
            );
            drop(conn);
            let _ = std::fs::remove_file(db_path);
            let base = db_path.display();
            let _ = std::fs::remove_file(format!("{base}-wal"));
            let _ = std::fs::remove_file(format!("{base}-shm"));
            conn = Connection::open(db_path).map_err(|e| format!("db reopen: {e}"))?;
        }

        // recursive_triggers: without it, the internal DELETE half of an
        // INSERT OR REPLACE does not fire delete triggers, which would leave
        // ghost rows in the external-content FTS tables kept in sync below.
        conn.execute_batch(&format!(
            "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA recursive_triggers=ON; PRAGMA user_version={SCHEMA_VERSION};"
        ))
        .map_err(|e| format!("pragma: {e}"))?;
        let db = IndexDb {
            conn: Mutex::new(conn),
        };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                language TEXT,
                hash TEXT,
                last_modified INTEGER,
                size INTEGER,
                embed_hash TEXT
            );

            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'unknown',
                signature TEXT,
                start_line INTEGER,
                start_col INTEGER,
                end_line INTEGER,
                end_col INTEGER,
                doc_comment TEXT,
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY,
                from_symbol_id INTEGER NOT NULL,
                to_symbol_id INTEGER NOT NULL,
                kind TEXT NOT NULL DEFAULT 'calls',
                FOREIGN KEY(from_symbol_id) REFERENCES symbols(id) ON DELETE CASCADE,
                FOREIGN KEY(to_symbol_id) REFERENCES symbols(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_symbols_file_id ON symbols(file_id);
            CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_symbol_id);
            CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_symbol_id);

            CREATE TABLE IF NOT EXISTS symbol_embeddings (
                symbol_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                start_line INTEGER NOT NULL DEFAULT 0,
                end_line INTEGER NOT NULL DEFAULT 0,
                embedding BLOB NOT NULL,
                PRIMARY KEY(symbol_id, chunk_index),
                FOREIGN KEY(symbol_id) REFERENCES symbols(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS symbol_chunks (
                id INTEGER PRIMARY KEY,
                symbol_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_line INTEGER NOT NULL DEFAULT 0,
                end_line INTEGER NOT NULL DEFAULT 0,
                embed_text TEXT NOT NULL,
                fts_name TEXT NOT NULL DEFAULT '',
                fts_path TEXT NOT NULL DEFAULT '',
                fts_body TEXT NOT NULL DEFAULT '',
                UNIQUE(symbol_id, chunk_index),
                FOREIGN KEY(symbol_id) REFERENCES symbols(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_symbol_chunks_symbol ON symbol_chunks(symbol_id);

            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS symbol_embeddings_upgrade (
                symbol_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                start_line INTEGER NOT NULL DEFAULT 0,
                end_line INTEGER NOT NULL DEFAULT 0,
                embedding BLOB NOT NULL,
                PRIMARY KEY(symbol_id, chunk_index),
                FOREIGN KEY(symbol_id) REFERENCES symbols(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS file_upgrade (
                file_id INTEGER PRIMARY KEY,
                hash TEXT NOT NULL,
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
            );
            ",
        )
        .map_err(|e| format!("schema: {e}"))?;

        // symbols_fts is the complete name/signature directory (covers
        // non-embeddable symbols like imports for code_search); chunk_fts
        // covers exactly the retrieval units embeddings cover, enriched with
        // path and split-identifier words, and backs the BM25 leg of hybrid
        // search. unicode61 note: `_`/`-`/`.` are separators, so quoted
        // snake/kebab/dotted terms become adjacency-preserving phrases at
        // query time; camelCase stays one token, which is why split words are
        // stored explicitly in fts_name/fts_body.
        conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
                name, signature,
                content='symbols',
                content_rowid='id'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                fts_name, fts_path, fts_body,
                content='symbol_chunks',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );",
        )
        .map_err(|e| format!("fts5: {e}"))?;

        // Triggers keep the external-content FTS tables in lockstep with
        // their content tables, so no Rust code path can forget the FTS side
        // (the v6 index leaked stale symbols_fts rows on incremental
        // reindex). Deletes must be explicit statements — cascades are not
        // relied on (see delete_symbols_for_file / delete_file).
        conn.execute_batch(
            "CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
                INSERT INTO symbols_fts(rowid, name, signature)
                VALUES (new.id, new.name, new.signature);
            END;
            CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
                INSERT INTO symbols_fts(symbols_fts, rowid, name, signature)
                VALUES ('delete', old.id, old.name, old.signature);
            END;
            CREATE TRIGGER IF NOT EXISTS symbol_chunks_ai AFTER INSERT ON symbol_chunks BEGIN
                INSERT INTO chunk_fts(rowid, fts_name, fts_path, fts_body)
                VALUES (new.id, new.fts_name, new.fts_path, new.fts_body);
            END;
            CREATE TRIGGER IF NOT EXISTS symbol_chunks_ad AFTER DELETE ON symbol_chunks BEGIN
                INSERT INTO chunk_fts(chunk_fts, rowid, fts_name, fts_path, fts_body)
                VALUES ('delete', old.id, old.fts_name, old.fts_path, old.fts_body);
            END;",
        )
        .map_err(|e| format!("fts triggers: {e}"))?;

        Ok(())
    }

    pub fn upsert_file(
        &self,
        path: &str,
        language: &str,
        hash: &str,
        modified: i64,
        size: i64,
    ) -> Result<i64, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "INSERT INTO files (path, language, hash, last_modified, size)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(path) DO UPDATE SET
               language=excluded.language,
               hash=excluded.hash,
               last_modified=excluded.last_modified,
               size=excluded.size",
            params![path, language, hash, modified, size],
        )
        .map_err(|e| format!("upsert file: {e}"))?;

        let id: i64 = conn
            .query_row(
                "SELECT id FROM files WHERE path = ?1",
                params![path],
                |row| row.get(0),
            )
            .map_err(|e| format!("get file id: {e}"))?;
        Ok(id)
    }

    pub fn all_files(&self) -> Result<Vec<FileRecord>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare("SELECT id, path, language, hash, last_modified, size, embed_hash FROM files ORDER BY id")
            .map_err(|e| format!("prepare: {e}"))?;
        let results = stmt
            .query_map([], |row| {
                Ok(FileRecord {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    language: row.get(2)?,
                    hash: row.get(3)?,
                    last_modified: row.get(4)?,
                    size: row.get(5)?,
                    embed_hash: row.get(6)?,
                })
            })
            .map_err(|e| format!("query: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    /// Records that `symbol_embeddings` for this file now reflect `hash`, so a
    /// future scan can skip re-embedding it while the content stays the same.
    pub fn set_embed_hash(&self, file_id: i64, hash: &str) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "UPDATE files SET embed_hash = ?1 WHERE id = ?2",
            params![hash, file_id],
        )
        .map_err(|e| format!("set embed_hash: {e}"))?;
        Ok(())
    }

    fn meta_get(conn: &Connection, key: &str) -> Option<String> {
        conn.query_row("SELECT value FROM index_meta WHERE key = ?1", params![key], |row| row.get(0))
            .ok()
    }

    fn meta_set(conn: &Connection, key: &str, value: &str) -> Result<(), String> {
        conn.execute(
            "INSERT INTO index_meta (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )
        .map(|_| ())
        .map_err(|e| format!("set {key}: {e}"))
    }

    /// The model the stored text vectors came from, if any are stored. An
    /// index written before models were recorded can only hold MiniLM
    /// vectors.
    pub fn embedding_model(&self) -> Option<String> {
        let conn = self.conn.lock().ok()?;
        Self::embedding_model_locked(&conn)
    }

    fn embedding_model_locked(conn: &Connection) -> Option<String> {
        if let Some(model) = Self::meta_get(conn, META_EMBED_MODEL) {
            return Some(model);
        }
        let any: i64 = conn
            .query_row("SELECT EXISTS(SELECT 1 FROM symbol_embeddings)", [], |row| row.get(0))
            .unwrap_or(0);
        (any != 0).then(|| crate::embeddings::MINILM_MODEL_ID.to_string())
    }

    /// The model the stored media vectors came from, if recorded.
    pub fn media_embedding_model(&self) -> Option<String> {
        let conn = self.conn.lock().ok()?;
        Self::meta_get(&conn, META_MEDIA_MODEL)
    }

    /// Make the index consistent with the embedder about to write to it.
    ///
    /// Vectors of two models cannot be compared. Text and media vectors are
    /// tracked separately, since they may come from different models
    /// (`ModelProfile`): if the index holds text vectors of another model
    /// they are dropped and every text file queued for embedding again —
    /// this is what happens the first time a workspace is opened after the
    /// text model changes, in either direction — and the same for media
    /// vectors and the media model. Neither touches the other. Stored
    /// vectors of the wrong length count as another model's too, whatever
    /// the recorded id says: an older binary sharing this cache does not
    /// record what it writes.
    ///
    /// An embedder with no media model at all leaves media vectors where
    /// they are. Nothing can be compared against them meanwhile, and they
    /// are still good when the model that wrote them is back.
    ///
    /// Media files are also queued for a second reason. One that got no
    /// vector — no encoder for its kind was loaded, it was past the cap, it
    /// would not decode — is marked done; whenever the loaded encoders, the
    /// cap or the number of media vectors changes, those are queued again,
    /// without touching any code vector. (The count is what gives a slot
    /// freed by a deleted file to the next one, and it settles after one
    /// retry.)
    ///
    /// One transaction: a process killed halfway must not leave an index
    /// whose vectors are gone but whose files still say "embedded". And one
    /// that takes the write lock before it reads (`BEGIN IMMEDIATE`): it
    /// reads, then writes, and begun the default way it fails outright if
    /// another connection — a second server on the same repository, an hour
    /// into its upgrade — commits in between.
    pub fn reconcile_embedding_model(&self, profile: &ModelProfile) -> Result<EmbeddingReset, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let tx = rusqlite::Transaction::new_unchecked(&conn, rusqlite::TransactionBehavior::Immediate)
            .map_err(|e| format!("begin reconcile: {e}"))?;
        let mut reset = EmbeddingReset::default();

        // Vectors of one role whose length is not `dim` floats.
        let foreign = |media: bool, dim: usize| -> bool {
            let not = if media { "" } else { "NOT" };
            tx.query_row(
                &format!(
                    "SELECT EXISTS(SELECT 1 FROM symbol_embeddings e
                                   JOIN symbols s ON s.id = e.symbol_id
                                   WHERE s.kind {not} IN ({MEDIA_KINDS_SQL})
                                     AND length(e.embedding) != ?1)"
                ),
                params![(dim * 4) as i64],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
                != 0
        };
        // Drop one role's vectors and queue its files; returns how many
        // vectors went.
        let drop_vectors = |media: bool| -> Result<usize, String> {
            let not = if media { "" } else { "NOT" };
            let dropped = tx
                .execute(
                    &format!(
                        "DELETE FROM symbol_embeddings WHERE symbol_id IN
                            (SELECT id FROM symbols WHERE kind {not} IN ({MEDIA_KINDS_SQL}))"
                    ),
                    [],
                )
                .map_err(|e| format!("drop embeddings: {e}"))?;
            let which = if media {
                format!("language IN ({MEDIA_KINDS_SQL})")
            } else {
                format!("language IS NULL OR language NOT IN ({MEDIA_KINDS_SQL})")
            };
            tx.execute(&format!("UPDATE files SET embed_hash = NULL WHERE {which}"), [])
                .map_err(|e| format!("requeue files: {e}"))?;
            Ok(dropped)
        };

        let stored_text = Self::embedding_model_locked(&tx);
        if stored_text.as_deref().is_some_and(|s| s != profile.text_model) || foreign(false, profile.text_dim) {
            eprintln!(
                "[index] embedding model changed ({} -> {}); re-embedding",
                stored_text.as_deref().unwrap_or("unrecorded"),
                profile.text_model
            );
            drop_vectors(false)?;
            reset.model_changed = true;
        }
        Self::meta_set(&tx, META_EMBED_MODEL, profile.text_model)?;

        if let Some((media_model, media_dim)) = profile.media_model {
            // Before the media model was recorded on its own, one model
            // wrote both kinds of vector.
            let stored_media = Self::meta_get(&tx, META_MEDIA_MODEL).or(stored_text);
            if stored_media.as_deref().is_some_and(|s| s != media_model) || foreign(true, media_dim) {
                // Usually there is nothing to drop — an index that never had
                // a media model — so only a real loss is worth a line.
                if drop_vectors(true)? > 0 {
                    eprintln!(
                        "[index] media embedding model changed ({} -> {media_model}); re-embedding media",
                        stored_media.as_deref().unwrap_or("unrecorded")
                    );
                    reset.media_model_changed = true;
                }
            }
            Self::meta_set(&tx, META_MEDIA_MODEL, media_model)?;
        }

        let media = profile.media;
        let mut kinds: Vec<&str> = MediaKind::ALL
            .into_iter()
            .filter(|k| media.has(*k))
            .map(MediaKind::as_str)
            .collect();
        kinds.sort_unstable();
        let media_vectors: i64 = tx
            .query_row(
                &format!(
                    "SELECT count(*) FROM symbol_embeddings e
                     JOIN symbols s ON s.id = e.symbol_id
                     WHERE s.kind IN ({MEDIA_KINDS_SQL})"
                ),
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        let state = format!(
            "{};max={};vectors={media_vectors}",
            kinds.join(","),
            crate::media::max_embedded_media()
        );
        if Self::meta_get(&tx, META_MEDIA_STATE).as_deref() != Some(state.as_str()) {
            for kind in &kinds {
                let n = tx
                    .execute(
                        "UPDATE files SET embed_hash = NULL
                         WHERE language = ?1
                           AND id NOT IN (SELECT s.file_id FROM symbols s
                                          JOIN symbol_embeddings e ON e.symbol_id = s.id)",
                        params![kind],
                    )
                    .map_err(|e| format!("requeue {kind} files: {e}"))?;
                reset.media_requeued += n as i64;
            }
            Self::meta_set(&tx, META_MEDIA_STATE, &state)?;
        }
        tx.commit().map_err(|e| format!("commit reconcile: {e}"))?;
        Ok(reset)
    }

    /// Media files that have a content vector.
    pub fn media_embedding_count(&self) -> Result<i64, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.query_row(
            &format!(
                "SELECT count(*) FROM symbol_embeddings e
                 JOIN symbols s ON s.id = e.symbol_id
                 WHERE s.kind IN ({MEDIA_KINDS_SQL})"
            ),
            [],
            |row| row.get(0),
        )
        .map_err(|e| format!("media_embedding_count: {e}"))
    }

    /// Indexed media files, by kind: `(images, audio)`.
    pub fn media_file_counts(&self) -> Result<(i64, i64), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let count = |kind: MediaKind| -> i64 {
            conn.query_row(
                "SELECT count(*) FROM files WHERE language = ?1",
                params![kind.as_str()],
                |row| row.get(0),
            )
            .unwrap_or(0)
        };
        Ok((count(MediaKind::Image), count(MediaKind::Audio)))
    }

    /// The symbol row standing for a media file.
    pub fn media_symbol_id(&self, file_id: i64) -> Result<Option<i64>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        Ok(conn
            .query_row(
                "SELECT id FROM symbols WHERE file_id = ?1 ORDER BY id LIMIT 1",
                params![file_id],
                |row| row.get(0),
            )
            .ok())
    }

    /// Remove file rows (and, via cascade, their symbols/relations/embeddings)
    /// whose path is not in the current scan set — e.g. node_modules leftovers
    /// indexed before ignore rules existed.
    pub fn prune_files_not_in(
        &self,
        keep: &std::collections::HashSet<String>,
    ) -> Result<i64, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let stale_ids: Vec<i64> = {
            let mut stmt = conn
                .prepare("SELECT id, path FROM files")
                .map_err(|e| format!("prepare: {e}"))?;
            let ids: Vec<i64> = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|e| format!("query: {e}"))?
                .filter_map(|r| r.ok())
                .filter(|(_, path)| !keep.contains(path))
                .map(|(id, _)| id)
                .collect();
            ids
        };
        for id in &stale_ids {
            conn.execute("DELETE FROM files WHERE id = ?1", params![id])
                .map_err(|e| format!("prune file: {e}"))?;
        }
        if !stale_ids.is_empty() {
            // These prune deletes cascade files -> symbols -> chunks, and
            // cascades don't reliably fire the FTS triggers — rebuild both
            // external-content indexes from their content tables (self-heal).
            let _ = conn.execute("INSERT INTO symbols_fts(symbols_fts) VALUES('rebuild')", []);
            let _ = conn.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')", []);
        }
        Ok(stale_ids.len() as i64)
    }

    pub fn file_by_path(&self, path: &str) -> Result<Option<FileRecord>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare("SELECT id, path, language, hash, last_modified, size, embed_hash FROM files WHERE path = ?1")
            .map_err(|e| format!("prepare: {e}"))?;
        let result = stmt
            .query_row(params![path], |row| {
                Ok(FileRecord {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    language: row.get(2)?,
                    hash: row.get(3)?,
                    last_modified: row.get(4)?,
                    size: row.get(5)?,
                    embed_hash: row.get(6)?,
                })
            })
            .ok();
        Ok(result)
    }

    pub fn delete_symbols_for_file(&self, file_id: i64) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute("DELETE FROM relations WHERE from_symbol_id IN (SELECT id FROM symbols WHERE file_id = ?1) OR to_symbol_id IN (SELECT id FROM symbols WHERE file_id = ?1)", params![file_id])
            .map_err(|e| format!("delete relations: {e}"))?;
        conn.execute("DELETE FROM symbol_embeddings WHERE symbol_id IN (SELECT id FROM symbols WHERE file_id = ?1)", params![file_id])
            .map_err(|e| format!("delete embeddings: {e}"))?;
        conn.execute("DELETE FROM symbol_embeddings_upgrade WHERE symbol_id IN (SELECT id FROM symbols WHERE file_id = ?1)", params![file_id])
            .map_err(|e| format!("delete upgrade embeddings: {e}"))?;
        // And the record that the upgrade set had this file. The hash alone
        // cannot carry that: a file edited and then put back as it was has
        // its old hash again and none of its old vectors.
        conn.execute("DELETE FROM file_upgrade WHERE file_id = ?1", params![file_id])
            .map_err(|e| format!("requeue upgrade file: {e}"))?;
        // Explicit deletes (not cascades) so the AFTER DELETE triggers fire
        // and purge the external-content FTS rows deterministically.
        conn.execute("DELETE FROM symbol_chunks WHERE symbol_id IN (SELECT id FROM symbols WHERE file_id = ?1)", params![file_id])
            .map_err(|e| format!("delete chunks: {e}"))?;
        conn.execute("DELETE FROM symbols WHERE file_id = ?1", params![file_id])
            .map_err(|e| format!("delete symbols: {e}"))?;
        Ok(())
    }

    /// Remove a file and everything derived from it. Symbols/chunks are
    /// deleted explicitly first so FTS triggers fire — the FK cascade from
    /// `files` is only a safety net, not the mechanism.
    pub fn delete_file(&self, path: &str) -> Result<(), String> {
        if let Some(file) = self.file_by_path(path)? {
            self.delete_symbols_for_file(file.id)?;
        }
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute("DELETE FROM files WHERE path = ?1", params![path])
            .map_err(|e| format!("delete file: {e}"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert_symbol(
        &self,
        file_id: i64,
        name: &str,
        kind: &str,
        signature: Option<&str>,
        start_line: i64,
        start_col: i64,
        end_line: i64,
        end_col: i64,
        doc_comment: Option<&str>,
    ) -> Result<i64, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "INSERT INTO symbols (file_id, name, kind, signature, start_line, start_col, end_line, end_col, doc_comment)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![file_id, name, kind, signature, start_line, start_col, end_line, end_col, doc_comment],
        )
        .map_err(|e| format!("insert symbol: {e}"))?;
        let id: i64 = conn.last_insert_rowid();
        Ok(id)
    }

    pub fn insert_relation(&self, from_id: i64, to_id: i64, kind: &str) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "INSERT OR IGNORE INTO relations (from_symbol_id, to_symbol_id, kind) VALUES (?1, ?2, ?3)",
            params![from_id, to_id, kind],
        )
        .map_err(|e| format!("insert relation: {e}"))?;
        Ok(())
    }

    /// Symbols by name or signature. Code only: a project with thirty
    /// `logo-*.png` must not answer "logo" with thirty pictures and no
    /// function — media has its own search (`search_media`).
    pub fn search_symbols(&self, query: &str, limit: i64) -> Result<Vec<SearchResult>, String> {
        // Raw agent/user text goes through the sanitizing builder — FTS5
        // operator characters in a query used to be a syntax error here.
        let Some(match_query) = build_fts_match_query(query) else {
            return Ok(vec![]);
        };
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT s.id, s.name, s.kind, f.path, s.start_line, s.signature
                 FROM symbols_fts
                 JOIN symbols s ON s.id = symbols_fts.rowid
                 JOIN files f ON f.id = s.file_id
                 WHERE symbols_fts MATCH ?1 AND s.kind NOT IN ('image','audio')
                 ORDER BY rank
                 LIMIT ?2",
            )
            .map_err(|e| format!("prepare search: {e}"))?;
        let results = stmt
            .query_map(params![match_query, limit], |row| {
                Ok(SearchResult {
                    symbol_id: row.get(0)?,
                    name: row.get(1)?,
                    kind: row.get(2)?,
                    file_path: row.get(3)?,
                    start_line: row.get(4)?,
                    signature: row.get(5)?,
                })
            })
            .map_err(|e| format!("query search: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    /// Exact (case-insensitive) symbol-name lookup — the contract
    /// symbol_lookup advertises, distinct from tokenized FTS matching.
    pub fn lookup_symbols_exact(
        &self,
        name: &str,
        limit: i64,
    ) -> Result<Vec<SearchResult>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT s.id, s.name, s.kind, f.path, s.start_line, s.signature
                 FROM symbols s
                 JOIN files f ON f.id = s.file_id
                 WHERE s.name = ?1 COLLATE NOCASE AND s.kind NOT IN ('image','audio')
                 ORDER BY f.path, s.start_line
                 LIMIT ?2",
            )
            .map_err(|e| format!("prepare lookup: {e}"))?;
        let results = stmt
            .query_map(params![name, limit], |row| {
                Ok(SearchResult {
                    symbol_id: row.get(0)?,
                    name: row.get(1)?,
                    kind: row.get(2)?,
                    file_path: row.get(3)?,
                    start_line: row.get(4)?,
                    signature: row.get(5)?,
                })
            })
            .map_err(|e| format!("query lookup: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    pub fn symbols_in_file(&self, file_path: &str) -> Result<Vec<SymbolRecord>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT s.id, s.file_id, s.name, s.kind, s.signature,
                        s.start_line, s.start_col, s.end_line, s.end_col, f.path
                 FROM symbols s
                 JOIN files f ON f.id = s.file_id
                 WHERE f.path = ?1
                 ORDER BY s.start_line",
            )
            .map_err(|e| format!("prepare: {e}"))?;
        let results = stmt
            .query_map(params![file_path], |row| {
                Ok(SymbolRecord {
                    id: row.get(0)?,
                    file_id: row.get(1)?,
                    name: row.get(2)?,
                    kind: row.get(3)?,
                    signature: row.get(4)?,
                    start_line: row.get(5)?,
                    start_col: row.get(6)?,
                    end_line: row.get(7)?,
                    end_col: row.get(8)?,
                    file_path: row.get(9)?,
                })
            })
            .map_err(|e| format!("query: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    #[allow(dead_code)]
    pub fn callers_of(
        &self,
        symbol_name: &str,
        file_path: &str,
    ) -> Result<Vec<SymbolRecord>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT s.id, s.file_id, s.name, s.kind, s.signature,
                        s.start_line, s.start_col, s.end_line, s.end_col, f.path
                 FROM relations r
                 JOIN symbols s ON s.id = r.from_symbol_id
                 JOIN files f ON f.id = s.file_id
                 JOIN symbols ts ON ts.id = r.to_symbol_id
                 WHERE ts.name = ?1 AND f.path != ?2
                 ORDER BY f.path, s.start_line
                 LIMIT 50",
            )
            .map_err(|e| format!("prepare: {e}"))?;
        let results = stmt
            .query_map(params![symbol_name, file_path], |row| {
                Ok(SymbolRecord {
                    id: row.get(0)?,
                    file_id: row.get(1)?,
                    name: row.get(2)?,
                    kind: row.get(3)?,
                    signature: row.get(4)?,
                    start_line: row.get(5)?,
                    start_col: row.get(6)?,
                    end_line: row.get(7)?,
                    end_col: row.get(8)?,
                    file_path: row.get(9)?,
                })
            })
            .map_err(|e| format!("query: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    /// Insert one retrieval chunk. Plain INSERT by design: callers always run
    /// `delete_symbols_for_file` first, and REPLACE would route the implicit
    /// delete around the FTS trigger on setups without recursive_triggers.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_chunk(
        &self,
        symbol_id: i64,
        chunk_index: i64,
        start_line: i64,
        end_line: i64,
        embed_text: &str,
        fts_name: &str,
        fts_path: &str,
        fts_body: &str,
    ) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "INSERT INTO symbol_chunks (symbol_id, chunk_index, start_line, end_line, embed_text, fts_name, fts_path, fts_body)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![symbol_id, chunk_index, start_line, end_line, embed_text, fts_name, fts_path, fts_body],
        )
        .map_err(|e| format!("insert chunk: {e}"))?;
        Ok(())
    }

    /// Stored retrieval chunks for one file, feeding the background embedding
    /// pass — no re-read or re-parse of the source file is needed.
    pub fn chunks_for_file(&self, file_id: i64) -> Result<Vec<StoredChunk>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT c.symbol_id, c.chunk_index, c.start_line, c.end_line, c.embed_text
                 FROM symbol_chunks c
                 JOIN symbols s ON s.id = c.symbol_id
                 WHERE s.file_id = ?1
                 ORDER BY c.symbol_id, c.chunk_index",
            )
            .map_err(|e| format!("prepare chunks: {e}"))?;
        let results = stmt
            .query_map(params![file_id], |row| {
                Ok(StoredChunk {
                    symbol_id: row.get(0)?,
                    chunk_index: row.get(1)?,
                    start_line: row.get(2)?,
                    end_line: row.get(3)?,
                    embed_text: row.get(4)?,
                })
            })
            .map_err(|e| format!("query chunks: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    pub fn upsert_embedding(
        &self,
        symbol_id: i64,
        chunk_index: i64,
        start_line: i64,
        end_line: i64,
        embedding: &[f32],
    ) -> Result<(), String> {
        self.upsert_embedding_in(VectorSet::Primary, symbol_id, chunk_index, start_line, end_line, embedding)
    }

    pub fn upsert_embedding_in(
        &self,
        set: VectorSet,
        symbol_id: i64,
        chunk_index: i64,
        start_line: i64,
        end_line: i64,
        embedding: &[f32],
    ) -> Result<(), String> {
        let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            &format!(
                "INSERT OR REPLACE INTO {} (symbol_id, chunk_index, start_line, end_line, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                set.table()
            ),
            params![symbol_id, chunk_index, start_line, end_line, bytes],
        )
        .map_err(|e| format!("upsert embedding: {e}"))?;
        Ok(())
    }

    pub fn delete_embeddings_for_file(&self, file_id: i64) -> Result<(), String> {
        self.delete_embeddings_for_file_in(VectorSet::Primary, file_id)
    }

    pub fn delete_embeddings_for_file_in(&self, set: VectorSet, file_id: i64) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            &format!(
                "DELETE FROM {} WHERE symbol_id IN (SELECT id FROM symbols WHERE file_id = ?1)",
                set.table()
            ),
            params![file_id],
        )
        .map_err(|e| format!("delete embeddings: {e}"))?;
        Ok(())
    }

    /// How many vectors of `set` the file's symbols hold.
    pub fn embedding_count_for_file_in(&self, set: VectorSet, file_id: i64) -> Result<i64, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.query_row(
            &format!(
                "SELECT count(*) FROM {} WHERE symbol_id IN (SELECT id FROM symbols WHERE file_id = ?1)",
                set.table()
            ),
            params![file_id],
            |row| row.get(0),
        )
        .map_err(|e| format!("count embeddings: {e}"))
    }

    // ── the upgrade set ─────────────────────────────────────────────────

    /// The model the upgrade set was written by, if it holds anything.
    pub fn upgrade_model(&self) -> Option<String> {
        let conn = self.conn.lock().ok()?;
        Self::meta_get(&conn, META_UPGRADE_MODEL)
    }

    /// Make the upgrade set consistent with the model about to write to it:
    /// vectors of another model (by recorded id, or by length — the id of a
    /// set an interrupted process left behind may never have been written)
    /// are dropped, and with them the record of which files were done.
    /// Returns whether anything was dropped.
    ///
    /// Called at the start of every turn of the upgrade, with the watcher
    /// free to write through its own connection meanwhile. So the usual case
    /// — nothing to change — writes nothing, and the rare one takes the
    /// write lock before it reads: a transaction that read first and wrote
    /// afterwards would fail outright whenever the other connection had
    /// committed in between.
    pub fn reconcile_upgrade_model(&self, model: &str, dim: usize) -> Result<bool, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mismatch = |conn: &Connection| -> Result<(Option<String>, bool), String> {
            let stored = Self::meta_get(conn, META_UPGRADE_MODEL);
            let foreign: i64 = conn
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM symbol_embeddings_upgrade WHERE length(embedding) != ?1)",
                    params![(dim * 4) as i64],
                    |row| row.get(0),
                )
                .map_err(|e| format!("upgrade reconcile: {e}"))?;
            let wrong = stored.as_deref().is_some_and(|s| s != model) || foreign != 0;
            Ok((stored, wrong))
        };
        let (stored, wrong) = mismatch(&conn)?;
        if !wrong && stored.as_deref() == Some(model) {
            return Ok(false);
        }

        let tx = rusqlite::Transaction::new_unchecked(&conn, rusqlite::TransactionBehavior::Immediate)
            .map_err(|e| format!("begin upgrade reconcile: {e}"))?;
        // Again, now that nothing else can write.
        let (stored, wrong) = mismatch(&tx)?;
        if wrong {
            let vectors = tx
                .execute("DELETE FROM symbol_embeddings_upgrade", [])
                .map_err(|e| format!("drop upgrade embeddings: {e}"))?;
            tx.execute("DELETE FROM file_upgrade", [])
                .map_err(|e| format!("requeue upgrade files: {e}"))?;
            Self::meta_set(&tx, META_UPGRADE_COMPLETE, "0")?;
            if vectors > 0 {
                eprintln!(
                    "[index] upgrade embedding model changed ({} -> {model}); re-embedding in the background",
                    stored.as_deref().unwrap_or("unrecorded")
                );
            }
        }
        Self::meta_set(&tx, META_UPGRADE_MODEL, model)?;
        tx.commit().map_err(|e| format!("commit upgrade reconcile: {e}"))?;
        Ok(wrong)
    }

    /// For each file the upgrade set has embedded: the content hash it was
    /// embedded from.
    pub fn upgrade_hashes(&self) -> Result<std::collections::HashMap<i64, String>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare("SELECT file_id, hash FROM file_upgrade")
            .map_err(|e| format!("prepare upgrade hashes: {e}"))?;
        let rows = stmt
            .query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)))
            .map_err(|e| format!("query upgrade hashes: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Records that the upgrade set's vectors for this file reflect `hash`.
    pub fn set_upgrade_hash(&self, file_id: i64, hash: &str) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "INSERT INTO file_upgrade (file_id, hash) VALUES (?1, ?2)
             ON CONFLICT(file_id) DO UPDATE SET hash = excluded.hash",
            params![file_id, hash],
        )
        .map_err(|e| format!("set upgrade hash: {e}"))?;
        Ok(())
    }

    /// Give up on a file for the upgrade set: record it as done in its
    /// current content, with whatever vectors it has. For a file the model
    /// keeps failing on — it stays findable by its text, and one file must
    /// not keep the whole set from ever being complete. It is tried again
    /// when it changes.
    pub fn skip_upgrade_file(&self, file_id: i64) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.execute(
            "INSERT INTO file_upgrade (file_id, hash)
             SELECT id, hash FROM files WHERE id = ?1 AND hash IS NOT NULL
             ON CONFLICT(file_id) DO UPDATE SET hash = excluded.hash",
            params![file_id],
        )
        .map_err(|e| format!("skip upgrade file: {e}"))?;
        Ok(())
    }

    /// Text files whose current content the upgrade set does not hold. An
    /// error is an error: read as zero it would say "nothing left to do".
    fn upgrade_pending_locked(conn: &Connection) -> Result<i64, String> {
        conn.query_row(
            &format!(
                "SELECT count(*) FROM files f
                 LEFT JOIN file_upgrade u ON u.file_id = f.id
                 WHERE (f.language IS NULL OR f.language NOT IN ({MEDIA_KINDS_SQL}))
                   AND (u.hash IS NULL OR u.hash != f.hash)"
            ),
            [],
            |row| row.get(0),
        )
        .map_err(|e| format!("upgrade pending: {e}"))
    }

    pub fn upgrade_state(&self) -> Result<UpgradeState, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let text_files: i64 = conn
            .query_row(
                &format!(
                    "SELECT count(*) FROM files
                     WHERE language IS NULL OR language NOT IN ({MEDIA_KINDS_SQL})"
                ),
                [],
                |row| row.get(0),
            )
            .map_err(|e| format!("upgrade state: {e}"))?;
        let vectors: i64 = conn
            .query_row("SELECT count(*) FROM symbol_embeddings_upgrade", [], |row| row.get(0))
            .map_err(|e| format!("upgrade state: {e}"))?;
        Ok(UpgradeState {
            model: Self::meta_get(&conn, META_UPGRADE_MODEL),
            complete: Self::meta_get(&conn, META_UPGRADE_COMPLETE).as_deref() == Some("1"),
            pending_files: Self::upgrade_pending_locked(&conn)?,
            text_files,
            vectors,
        })
    }

    /// Record that the upgrade set has covered the whole workspace — if it
    /// has: a file edited while the pass ran leaves it one short, and the
    /// next pass is the one that gets to say so. Returns whether it is
    /// complete now.
    pub fn mark_upgrade_complete(&self) -> Result<bool, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        if Self::upgrade_pending_locked(&conn)? != 0 {
            return Ok(Self::meta_get(&conn, META_UPGRADE_COMPLETE).as_deref() == Some("1"));
        }
        Self::meta_set(&conn, META_UPGRADE_COMPLETE, "1")?;
        Ok(true)
    }

    /// Every embedded chunk of the primary set, as stored.
    pub fn load_all_embeddings(&self) -> Result<Vec<EmbeddingRow>, String> {
        self.load_all_embeddings_in(VectorSet::Primary)
    }

    /// Every embedded chunk of one set, as stored. For analysis and tests:
    /// search never loads a set whole (`vector_leg_candidates`).
    pub fn load_all_embeddings_in(&self, set: VectorSet) -> Result<Vec<EmbeddingRow>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT s.id, s.file_id, s.name, s.kind, s.signature,
                        s.start_line, s.start_col, s.end_line, s.end_col, f.path,
                        e.start_line, e.end_line, e.embedding
                 FROM symbols s
                 JOIN files f ON f.id = s.file_id
                 JOIN {} e ON e.symbol_id = s.id",
                set.table()
            ))
            .map_err(|e| format!("prepare: {e}"))?;
        let results = stmt
            .query_map([], |row| {
                let blob: Vec<u8> = row.get(12)?;
                // Blob length defines the dimension — self-describing across model swaps.
                let embedding: Vec<f32> = blob
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&chunk| f32::from_le_bytes(chunk))
                    .collect();
                Ok((
                    SymbolRecord {
                        id: row.get(0)?,
                        file_id: row.get(1)?,
                        name: row.get(2)?,
                        kind: row.get(3)?,
                        signature: row.get(4)?,
                        start_line: row.get(5)?,
                        start_col: row.get(6)?,
                        end_line: row.get(7)?,
                        end_col: row.get(8)?,
                        file_path: row.get(9)?,
                    },
                    row.get::<_, i64>(10)?,
                    row.get::<_, i64>(11)?,
                    embedding,
                ))
            })
            .map_err(|e| format!("query: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    /// Load a single page of code/doc embedding rows, ordered
    /// deterministically. Media rows are left to `search_media`: their scores
    /// against a text query sit on another scale and would distort the ranks.
    pub fn load_embeddings_page(
        &self,
        page_size: i64,
        offset: i64,
    ) -> Result<Vec<EmbeddingRow>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT s.id, s.file_id, s.name, s.kind, s.signature,
                        s.start_line, s.start_col, s.end_line, s.end_col, f.path,
                        e.start_line, e.end_line, e.embedding
                 FROM symbols s
                 JOIN files f ON f.id = s.file_id
                 JOIN symbol_embeddings e ON e.symbol_id = s.id
                 WHERE s.kind NOT IN ('image','audio')
                 ORDER BY s.id, e.start_line
                 LIMIT ? OFFSET ?",
            )
            .map_err(|e| format!("prepare: {e}"))?;
        let results = stmt
            .query_map(params![page_size, offset], |row| {
                let blob: Vec<u8> = row.get(12)?;
                let embedding: Vec<f32> = blob
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&chunk| f32::from_le_bytes(chunk))
                    .collect();
                Ok((
                    SymbolRecord {
                        id: row.get(0)?,
                        file_id: row.get(1)?,
                        name: row.get(2)?,
                        kind: row.get(3)?,
                        signature: row.get(4)?,
                        start_line: row.get(5)?,
                        start_col: row.get(6)?,
                        end_line: row.get(7)?,
                        end_col: row.get(8)?,
                        file_path: row.get(9)?,
                    },
                    row.get::<_, i64>(10)?,
                    row.get::<_, i64>(11)?,
                    embedding,
                ))
            })
            .map_err(|e| format!("query: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    /// Vector leg: cosine of the query against every embedded chunk of one
    /// set, gated at `min_cosine`, reduced to the best chunk per symbol,
    /// ranked by cosine.
    ///
    /// Two steps, because this runs on every search. The scan reads nothing
    /// but the vectors — straight out of SQLite's pages, with no join and no
    /// allocation per row — and only the `k` symbols that survive it are then
    /// looked up by name and path. Loading every row with its symbol and file
    /// first took 0.85 s over 15.7k chunks of 384 dimensions, and 1.0 s over
    /// the same chunks at 768.
    ///
    /// Vectors of another length are another model's and are skipped; so are
    /// media vectors, which `search_media` ranks apart: their scores against
    /// a text query sit on another scale. Ties are broken by symbol id — a
    /// repository full of identical test helpers has many — so that the same
    /// search returns the same ranks twice.
    fn vector_leg_candidates(
        &self,
        set: VectorSet,
        query_vec: &[f32],
        min_cosine: f32,
        k: usize,
    ) -> Result<Vec<LegHit>, String> {
        struct Best {
            cosine: f32,
            chunk_start: i64,
            chunk_end: i64,
        }
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let mut best_per_symbol: std::collections::HashMap<i64, Best> = std::collections::HashMap::new();
        {
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT e.symbol_id, e.start_line, e.end_line, e.embedding
                     FROM {} e
                     WHERE length(e.embedding) = ?1
                       AND e.symbol_id NOT IN (SELECT id FROM symbols WHERE kind IN ({MEDIA_KINDS_SQL}))",
                    set.table()
                ))
                .map_err(|e| format!("prepare vector scan: {e}"))?;
            let mut rows = stmt
                .query(params![(query_vec.len() * 4) as i64])
                .map_err(|e| format!("query vector scan: {e}"))?;
            while let Some(row) = rows.next().map_err(|e| format!("vector scan: {e}"))? {
                let blob = row
                    .get_ref(3)
                    .and_then(|v| v.as_blob().map_err(Into::into))
                    .map_err(|e| format!("vector scan: {e}"))?;
                let dot: f32 = blob
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(query_vec.iter())
                    .map(|(bytes, q)| f32::from_le_bytes(*bytes) * q)
                    .sum();
                let cosine = dot.clamp(0.0, 1.0);
                if cosine < min_cosine {
                    continue;
                }
                let symbol_id: i64 = row.get(0).map_err(|e| format!("vector scan: {e}"))?;
                let chunk_start: i64 = row.get(1).map_err(|e| format!("vector scan: {e}"))?;
                let chunk_end: i64 = row.get(2).map_err(|e| format!("vector scan: {e}"))?;
                let candidate = Best {
                    cosine,
                    chunk_start,
                    chunk_end,
                };
                match best_per_symbol.entry(symbol_id) {
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        // Of two equally good chunks of a symbol, the one
                        // higher up in the file — whatever order the rows
                        // came in.
                        let held = e.get();
                        if cosine > held.cosine || (cosine == held.cosine && chunk_start < held.chunk_start) {
                            e.insert(candidate);
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(candidate);
                    }
                }
            }
        }

        let mut ranked: Vec<(i64, Best)> = best_per_symbol.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.cosine
                .partial_cmp(&a.1.cosine)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        ranked.truncate(k);

        let mut lookup = conn
            .prepare(
                "SELECT s.name, s.kind, s.signature, f.path, s.start_line, s.end_line
                 FROM symbols s
                 JOIN files f ON f.id = s.file_id
                 WHERE s.id = ?1",
            )
            .map_err(|e| format!("prepare vector hits: {e}"))?;
        let mut hits: Vec<LegHit> = Vec::with_capacity(ranked.len());
        for (symbol_id, best) in ranked {
            let hit = lookup.query_row(params![symbol_id], |row| {
                Ok(LegHit {
                    symbol_id,
                    name: row.get(0)?,
                    kind: row.get(1)?,
                    signature: row.get(2)?,
                    file_path: row.get(3)?,
                    sym_start_line: row.get(4)?,
                    sym_end_line: row.get(5)?,
                    chunk_start: best.chunk_start,
                    chunk_end: best.chunk_end,
                    cosine: best.cosine,
                    fts_text: String::new(),
                })
            });
            // A symbol can only be missing if another connection deleted it
            // since the scan a moment ago; it is no longer a result.
            if let Ok(hit) = hit {
                hits.push(hit);
            }
        }
        Ok(hits)
    }

    /// BM25 leg: weighted match over chunk_fts, reduced to the best-ranked
    /// chunk per symbol (rows arrive rank-ordered, so first wins).
    fn bm25_leg_candidates(&self, match_query: &str, k: usize, media: bool) -> Result<Vec<LegHit>, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        // Code and media are ranked apart (see `search_media`).
        let media_filter = if media { "" } else { "NOT" };
        let sql = format!(
            "SELECT c.symbol_id, s.name, s.kind, s.signature, f.path,
                    s.start_line, s.end_line, c.start_line, c.end_line,
                    c.fts_name || ' ' || c.fts_path || ' ' || c.fts_body
             FROM chunk_fts
             JOIN symbol_chunks c ON c.id = chunk_fts.rowid
             JOIN symbols s ON s.id = c.symbol_id
             JOIN files f ON f.id = s.file_id
             WHERE chunk_fts MATCH ?1 AND s.kind {media_filter} IN ({MEDIA_KINDS_SQL})
             ORDER BY bm25(chunk_fts, {BM25_W_NAME}, {BM25_W_PATH}, {BM25_W_BODY})
             LIMIT ?2"
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| format!("prepare bm25: {e}"))?;
        // 2x headroom: several chunks of one symbol can occupy top ranks.
        let raw: Vec<LegHit> = stmt
            .query_map(params![match_query, (k * 2) as i64], |row| {
                Ok(LegHit {
                    symbol_id: row.get(0)?,
                    name: row.get(1)?,
                    kind: row.get(2)?,
                    signature: row.get(3)?,
                    file_path: row.get(4)?,
                    sym_start_line: row.get(5)?,
                    sym_end_line: row.get(6)?,
                    chunk_start: row.get(7)?,
                    chunk_end: row.get(8)?,
                    cosine: 0.0,
                    fts_text: row.get(9)?,
                })
            })
            .map_err(|e| format!("query bm25: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
        let mut hits: Vec<LegHit> = Vec::new();
        for hit in raw {
            if seen.insert(hit.symbol_id) {
                hits.push(hit);
            }
            if hits.len() >= k {
                break;
            }
        }
        Ok(hits)
    }

    pub fn search_hybrid(
        &self,
        query_text: &str,
        query_vec: Option<&[f32]>,
        limit: usize,
    ) -> Result<Vec<SemanticSearchResult>, String> {
        self.search_hybrid_in(VectorSet::Primary, query_text, query_vec, limit)
    }

    /// `search_hybrid` over one of the index's sets of text vectors, with the
    /// knobs of the model that wrote it. `query_vec` must come from that
    /// model.
    pub fn search_hybrid_in(
        &self,
        set: VectorSet,
        query_text: &str,
        query_vec: Option<&[f32]>,
        limit: usize,
    ) -> Result<Vec<SemanticSearchResult>, String> {
        let model = match set {
            VectorSet::Primary => self.embedding_model(),
            VectorSet::Upgrade => self.upgrade_model(),
        };
        let params = HybridParams::for_model(model.as_deref());
        self.search_hybrid_with_in(set, query_text, query_vec, limit, &params)
    }

    /// Images and audio matching a query: by content, when `query_vec` (from
    /// `CodeEmbedder::encode_media_query`) is given and the files have
    /// vectors, and by file name and path. Kept out of `search_hybrid` so
    /// that code ranking is untouched by it; the two legs are fused the same
    /// way, so `score` reads the same in both result lists.
    pub fn search_media(
        &self,
        query_text: &str,
        query_vec: Option<&[f32]>,
        limit: usize,
    ) -> Result<Vec<MediaSearchResult>, String> {
        const K: usize = 20;
        const RRF_K: f32 = 60.0;
        let tokens = tokenize_query(query_text);

        let mut vector_hits: Vec<LegHit> = Vec::new();
        if let Some(qv) = query_vec {
            let conn = self.conn.lock().map_err(|e| e.to_string())?;
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT s.id, s.name, s.kind, f.path, e.embedding
                     FROM symbol_embeddings e
                     JOIN symbols s ON s.id = e.symbol_id
                     JOIN files f ON f.id = s.file_id
                     WHERE s.kind IN ({MEDIA_KINDS_SQL})"
                ))
                .map_err(|e| format!("prepare media scan: {e}"))?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, Vec<u8>>(4)?,
                    ))
                })
                .map_err(|e| format!("query media scan: {e}"))?;
            for (symbol_id, name, kind, file_path, blob) in rows.filter_map(|r| r.ok()) {
                if blob.len() != qv.len() * 4 {
                    continue;
                }
                let dot: f32 = blob
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(qv.iter())
                    .map(|(bytes, q)| f32::from_le_bytes(*bytes) * q)
                    .sum();
                if dot < MEDIA_MIN_COSINE {
                    continue;
                }
                vector_hits.push(LegHit {
                    symbol_id,
                    name,
                    kind,
                    signature: None,
                    file_path,
                    sym_start_line: 0,
                    sym_end_line: 0,
                    chunk_start: 0,
                    chunk_end: 0,
                    cosine: dot,
                    fts_text: String::new(),
                });
            }
            vector_hits.sort_by(|a, b| {
                b.cosine
                    .partial_cmp(&a.cosine)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.symbol_id.cmp(&b.symbol_id))
            });
            vector_hits.truncate(K);
        }

        // A file name is thin evidence, so a name-only hit needs the same
        // share of the query's words a code hit does; one the content leg
        // also found needs none.
        let vector_ids: std::collections::HashSet<i64> = vector_hits.iter().map(|h| h.symbol_id).collect();
        let required = required_token_matches(tokens.len(), HybridParams::default().min_bm25_term_matches);
        let lexical_hits: Vec<LegHit> = match build_fts_match_query(query_text) {
            Some(mq) => self
                .bm25_leg_candidates(&mq, K, true)?
                .into_iter()
                .filter(|h| vector_ids.contains(&h.symbol_id) || bm25_only_hit_has_evidence(&tokens, h, required))
                .collect(),
            None => vec![],
        };

        struct Fused {
            hit: LegHit,
            rrf: f32,
            in_vector: bool,
            in_lexical: bool,
        }
        let mut fused: Vec<Fused> = Vec::new();
        for (i, hit) in vector_hits.into_iter().enumerate() {
            fused.push(Fused {
                hit,
                rrf: 1.0 / (RRF_K + (i + 1) as f32),
                in_vector: true,
                in_lexical: false,
            });
        }
        for (i, hit) in lexical_hits.into_iter().enumerate() {
            let contrib = 1.0 / (RRF_K + (i + 1) as f32);
            match fused.iter_mut().find(|f| f.hit.symbol_id == hit.symbol_id) {
                Some(f) => {
                    f.rrf += contrib;
                    f.in_lexical = true;
                }
                None => fused.push(Fused {
                    hit,
                    rrf: contrib,
                    in_vector: false,
                    in_lexical: true,
                }),
            }
        }
        let norm = 2.0 / (RRF_K + 1.0);
        let mut results: Vec<MediaSearchResult> = fused
            .into_iter()
            .map(|f| MediaSearchResult {
                symbol_id: f.hit.symbol_id,
                name: f.hit.name,
                kind: f.hit.kind,
                file_path: f.hit.file_path,
                score: (f.rrf / norm).clamp(0.0, 1.0),
                match_type: match (f.in_vector, f.in_lexical) {
                    (true, true) => "hybrid",
                    (true, false) => "semantic",
                    _ => "lexical",
                }
                .to_string(),
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.symbol_id.cmp(&b.symbol_id))
        });
        results.truncate(limit);
        Ok(results)
    }

    /// Hybrid retrieval: BM25 over chunk_fts fused with cosine over
    /// symbol_embeddings via normalized Reciprocal Rank Fusion, then the
    /// pre-existing lexical boosts and doc/test penalties. Either leg may be
    /// absent (no vector while the model loads or embeddings are pending; no
    /// BM25 when no query term survives sanitization) — the other leg still
    /// returns results, which is the graceful-degradation story during the
    /// background embedding window.
    pub fn search_hybrid_with(
        &self,
        query_text: &str,
        query_vec: Option<&[f32]>,
        limit: usize,
        params: &HybridParams,
    ) -> Result<Vec<SemanticSearchResult>, String> {
        self.search_hybrid_with_in(VectorSet::Primary, query_text, query_vec, limit, params)
    }

    /// `search_hybrid_with` over one of the index's sets of text vectors.
    pub fn search_hybrid_with_in(
        &self,
        set: VectorSet,
        query_text: &str,
        query_vec: Option<&[f32]>,
        limit: usize,
        params: &HybridParams,
    ) -> Result<Vec<SemanticSearchResult>, String> {
        let tokens = tokenize_query(query_text);
        let match_query = build_fts_match_query(query_text);

        let vector_hits = match query_vec {
            Some(qv) => {
                self.vector_leg_candidates(set, qv, params.min_cosine_candidate, params.k_candidates)?
            }
            None => vec![],
        };
        let bm25_hits = match &match_query {
            Some(mq) => self.bm25_leg_candidates(mq, params.k_candidates, false)?,
            None => vec![],
        };

        // Evidence gate for hits the vector leg doesn't corroborate, applied
        // before ranks are assigned so surviving hits keep dense ranks.
        let vector_ids: std::collections::HashSet<i64> =
            vector_hits.iter().map(|h| h.symbol_id).collect();
        let required = required_token_matches(tokens.len(), params.min_bm25_term_matches);
        let bm25_hits: Vec<LegHit> = bm25_hits
            .into_iter()
            .filter(|h| {
                vector_ids.contains(&h.symbol_id)
                    || bm25_only_hit_has_evidence(&tokens, h, required)
            })
            .collect();

        // Normalized RRF: 1.0 = rank 1 in both legs, 0.5 = rank 1 in exactly
        // one. Vector hits are folded first so their chunk anchors the result
        // line range when both legs agree on a symbol.
        struct Fused {
            hit: LegHit,
            rrf: f32,
            in_vector: bool,
            in_bm25: bool,
        }
        let mut fused: std::collections::HashMap<i64, Fused> = std::collections::HashMap::new();
        for (i, hit) in vector_hits.into_iter().enumerate() {
            let contrib = params.w_vector / (params.rrf_k + (i + 1) as f32);
            fused.insert(
                hit.symbol_id,
                Fused {
                    hit,
                    rrf: contrib,
                    in_vector: true,
                    in_bm25: false,
                },
            );
        }
        for (i, hit) in bm25_hits.into_iter().enumerate() {
            let contrib = params.w_bm25 / (params.rrf_k + (i + 1) as f32);
            match fused.entry(hit.symbol_id) {
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    let f = e.get_mut();
                    f.rrf += contrib;
                    f.in_bm25 = true;
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(Fused {
                        hit,
                        rrf: contrib,
                        in_vector: false,
                        in_bm25: true,
                    });
                }
            }
        }
        let norm_denom = (params.w_vector + params.w_bm25) / (params.rrf_k + 1.0);

        let mut scored: Vec<SemanticSearchResult> = Vec::new();
        for f in fused.into_values() {
            let norm = if norm_denom > 0.0 {
                f.rrf / norm_denom
            } else {
                0.0
            };
            let boost = lexical_boost(&tokens, &f.hit.name, &f.hit.file_path);
            let mut penalty = if f.hit.kind == "doc_section" {
                DOC_SECTION_PENALTY
            } else {
                0.0
            };
            if is_test_file(&f.hit.file_path) || is_test_symbol(&f.hit.name) {
                penalty += TEST_FILE_PENALTY;
            }
            let score = (norm + boost - penalty).clamp(0.0, 1.0);
            if score < params.min_hybrid_score {
                continue;
            }
            let (start_line, end_line) = if f.hit.chunk_start > 0 {
                (f.hit.chunk_start, f.hit.chunk_end)
            } else {
                (f.hit.sym_start_line, f.hit.sym_end_line)
            };
            let match_type = match (f.in_vector, f.in_bm25) {
                (true, true) => "hybrid",
                (true, false) => "semantic",
                _ => "lexical",
            };
            scored.push(SemanticSearchResult {
                symbol_id: f.hit.symbol_id,
                name: f.hit.name,
                kind: f.hit.kind,
                file_path: f.hit.file_path,
                start_line,
                end_line,
                signature: f.hit.signature,
                score,
                match_type: match_type.to_string(),
                snippet: None,
            });
        }
        // Deterministic tiebreak on symbol_id: several hits clamp to 1.0, and
        // without it their order follows HashMap drain order — ranks would
        // flap between identical runs.
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.symbol_id.cmp(&b.symbol_id))
        });

        // Dedupe by file (keep highest-scoring first) before applying limit,
        // so one large file can't occupy the whole ranking.
        let mut per_file_count: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        scored.retain(|r| {
            let count = per_file_count.entry(r.file_path.clone()).or_insert(0);
            *count += 1;
            *count <= MAX_RESULTS_PER_FILE
        });

        scored.truncate(limit);
        Ok(scored)
    }

    /// Number of indexed files whose embeddings are missing or stale
    /// (`embed_hash` absent or behind the content hash). Non-zero means the
    /// background embedding pass hasn't caught up yet, so semantic search may
    /// silently miss content.
    pub fn embedding_pending_files(&self) -> Result<i64, String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        conn.query_row(
            "SELECT count(*) FROM files WHERE embed_hash IS NULL OR embed_hash != hash",
            [],
            |row| row.get(0),
        )
        .map_err(|e| format!("embedding_pending_files: {e}"))
    }

    #[allow(dead_code)]
    pub fn index_stats(&self) -> Result<(i64, i64, i64), String> {
        let conn = self.conn.lock().map_err(|e| e.to_string())?;
        let files: i64 = conn
            .query_row("SELECT count(*) FROM files", [], |row| row.get(0))
            .unwrap_or(0);
        let symbols: i64 = conn
            .query_row("SELECT count(*) FROM symbols", [], |row| row.get(0))
            .unwrap_or(0);
        let embeddings: i64 = conn
            .query_row("SELECT count(*) FROM symbol_embeddings", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);
        Ok((files, symbols, embeddings))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dims: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dims];
        v[hot] = 1.0;
        v
    }

    #[test]
    fn lexical_boost_layers_dont_stack() {
        let tokens = tokenize_query("Icon.tsx component");
        // Exact basename match wins the +0.25 layer even though a substring
        // match would also apply.
        let boost = lexical_boost(&tokens, "Icon", "src/ui/Icon.tsx");
        assert_eq!(boost, 0.25);

        // Exact symbol-name match (no basename match) -> the middle +0.15
        // layer: short names collide with ordinary query words too easily
        // to outrank everything.
        let boost = lexical_boost(&tokens, "icon", "src/ui/other.tsx");
        assert_eq!(boost, 0.15);

        // Only a substring relationship -> the lower +0.10 layer.
        let boost = lexical_boost(&tokens, "IconButton", "src/ui/misc.tsx");
        assert_eq!(boost, 0.10);

        // No relationship at all.
        let boost = lexical_boost(&tokens, "unrelatedThing", "src/ui/misc.rs");
        assert_eq!(boost, 0.0);
    }

    #[test]
    fn test_files_are_detected_across_conventions() {
        assert!(is_test_file("src/components/TasksPanel.test.tsx"));
        assert!(is_test_file("src/lib/ipc.spec.ts"));
        assert!(is_test_file("src-tauri/src/agent/persist_test.rs"));
        assert!(is_test_file("src-tauri/tests/integration.rs"));
        assert!(is_test_file("src/__tests__/App.tsx"));
        assert!(!is_test_file("src/components/TasksPanel.tsx"));
        assert!(!is_test_file("src-tauri/src/agent/tests_helper_naming.rs"));
    }

    #[test]
    fn inline_test_symbols_are_detected() {
        assert!(is_test_symbol("test_golden_pending_ids"));
        assert!(is_test_symbol("golden_tests"));
        assert!(is_test_symbol("tests"));
        assert!(is_test_symbol("roundtrip_test"));
        assert!(!is_test_symbol("TasksPanel"));
        assert!(!is_test_symbol("attestation"));
    }

    #[test]
    fn tokenize_query_drops_short_tokens_and_stopwords() {
        let tokens = tokenize_query("the Icon.tsx and for a with X");
        assert_eq!(tokens, vec!["icon".to_string(), "tsx".to_string()]);
    }

    #[test]
    fn hybrid_dedupes_per_file_and_applies_threshold() {
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let file_id = db
            .upsert_file("src/big_file.ts", "typescript", "hash1", 0, 0)
            .expect("upsert file");

        // Five symbols in the same file, all with a near-perfect embedding
        // match, so dedupe (max 3 per file) — not the score — is what
        // trims them.
        for i in 0..5 {
            let sym_id = db
                .insert_symbol(
                    file_id,
                    &format!("sym{i}"),
                    "function",
                    None,
                    i,
                    0,
                    i,
                    0,
                    None,
                )
                .expect("insert symbol");
            db.upsert_embedding(sym_id, 0, 0, 0, &unit_vec(4, 0))
                .expect("upsert embedding");
        }

        // One symbol in a different file with a low-similarity embedding
        // that falls below the vector-leg cosine gate.
        let other_file_id = db
            .upsert_file("src/other.ts", "typescript", "hash2", 0, 0)
            .expect("upsert file");
        let low_sym_id = db
            .insert_symbol(
                other_file_id,
                "lowMatch",
                "function",
                None,
                0,
                0,
                0,
                0,
                None,
            )
            .expect("insert symbol");
        db.upsert_embedding(low_sym_id, 0, 0, 0, &unit_vec(4, 3))
            .expect("upsert embedding");

        let query_vec = unit_vec(4, 0);
        let results = db
            .search_hybrid("sym", Some(&query_vec), 10)
            .expect("search_hybrid");

        assert_eq!(results.len(), 3, "dedupe should cap results per file at 3");
        assert!(results.iter().all(|r| r.file_path == "src/big_file.ts"));
        let min_score = HybridParams::default().min_hybrid_score;
        assert!(results.iter().all(|r| r.score >= min_score));
        assert!(results.iter().all(|r| r.match_type == "semantic"));
    }

    #[test]
    fn hybrid_can_return_empty() {
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let file_id = db
            .upsert_file("src/only.ts", "typescript", "hash", 0, 0)
            .expect("upsert file");
        let sym_id = db
            .insert_symbol(file_id, "unrelated", "function", None, 1, 0, 3, 0, None)
            .expect("insert symbol");
        // Orthogonal embedding -> cosine ~0; chunk text shares no word with
        // the query -> the BM25 leg matches nothing either.
        db.upsert_embedding(sym_id, 0, 0, 0, &unit_vec(4, 1))
            .expect("upsert embedding");
        db.insert_chunk(
            sym_id,
            0,
            1,
            3,
            "function: unrelated",
            "unrelated",
            "only.ts only",
            "banana orchard code",
        )
        .unwrap();

        let query_vec = unit_vec(4, 0);
        let results = db
            .search_hybrid("totally different query", Some(&query_vec), 10)
            .expect("search_hybrid");
        assert!(results.is_empty());
    }

    #[test]
    fn fts_match_builder_neutralizes_operators() {
        // Operator-laden inputs must produce valid MATCH strings (or None) —
        // never an SQL error surfaced to the caller.
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let file_id = db.upsert_file("src/q.ts", "typescript", "h", 0, 0).unwrap();
        db.insert_symbol(file_id, "fooBar", "function", None, 1, 0, 1, 0, None)
            .unwrap();
        for query in [
            "foo(bar)",
            "a AND b OR c*",
            "\"unbalanced",
            "name:x",
            "-neg",
            "NEAR(a b)",
            "col : val",
            "*)(^",
        ] {
            let result = db.search_symbols(query, 10);
            assert!(result.is_ok(), "query {query:?} errored: {result:?}");
        }
        assert_eq!(build_fts_match_query("*)(^"), None);
        assert_eq!(
            build_fts_match_query("delete_symbols_for_file").as_deref(),
            Some("\"delete_symbols_for_file\"")
        );
        // Sanitized query still finds the symbol: "fooBar" survives as a
        // quoted term and unicode61 folds it to the same token as the name.
        // (Camel-splitting is chunk_fts territory, not symbols_fts.)
        let hits = db.search_symbols("fooBar(arg)", 10).unwrap();
        assert_eq!(hits.len(), 1, "fooBar term should match the fooBar symbol");
    }

    #[test]
    fn hybrid_exact_body_term_ranks_without_vectors() {
        // The headline fix: a term that exists only inside a body must be
        // findable with no vector leg at all (model missing / pending).
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let file_id = db
            .upsert_file("src/thing.ts", "typescript", "h", 0, 0)
            .unwrap();
        let sym_id = db
            .insert_symbol(file_id, "handleThing", "function", None, 1, 0, 9, 0, None)
            .unwrap();
        db.insert_chunk(
            sym_id,
            0,
            1,
            9,
            "embed",
            "handleThing handle thing",
            "thing.ts thing",
            "calls xyzzy_special_case() here",
        )
        .unwrap();

        let results = db
            .search_hybrid("xyzzy_special_case", None, 10)
            .expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "handleThing");
        assert_eq!(results[0].match_type, "lexical");
        assert!(results[0].score >= HybridParams::default().min_hybrid_score);
    }

    #[test]
    fn hybrid_fuses_both_legs_above_single_leg_hits() {
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        // A: vector-only rank 1. B: bm25-only rank 1. C: rank 2 in both —
        // fusion must put C first (2/(k+2) > 1/(k+1) for k=60).
        let fa = db.upsert_file("src/a.ts", "typescript", "h", 0, 0).unwrap();
        let fb = db.upsert_file("src/b.ts", "typescript", "h", 0, 0).unwrap();
        let fc = db.upsert_file("src/c.ts", "typescript", "h", 0, 0).unwrap();
        let sa = db
            .insert_symbol(fa, "alphaFn", "function", None, 1, 0, 2, 0, None)
            .unwrap();
        let sb = db
            .insert_symbol(fb, "betaFn", "function", None, 1, 0, 2, 0, None)
            .unwrap();
        let sc = db
            .insert_symbol(fc, "gammaFn", "function", None, 1, 0, 2, 0, None)
            .unwrap();

        db.upsert_embedding(sa, 0, 1, 2, &unit_vec(4, 0)).unwrap();
        let mixed = {
            // cosine 0.8 against unit(4,0) -> vector rank 2.
            let v = vec![0.8f32, 0.6, 0.0, 0.0];
            v
        };
        db.upsert_embedding(sc, 0, 1, 2, &mixed).unwrap();

        // B mentions the term twice -> bm25 rank 1; C once -> rank 2.
        db.insert_chunk(
            sb,
            0,
            1,
            2,
            "e",
            "betaFn beta fn",
            "b.ts b",
            "zebrafinch pattern zebrafinch pattern",
        )
        .unwrap();
        db.insert_chunk(
            sc,
            0,
            1,
            2,
            "e",
            "gammaFn gamma fn",
            "c.ts c",
            "zebrafinch pattern once",
        )
        .unwrap();

        let results = db
            .search_hybrid("zebrafinch pattern", Some(&unit_vec(4, 0)), 10)
            .expect("search");
        assert_eq!(
            results[0].name, "gammaFn",
            "both-legs hit must outrank single-leg hits"
        );
        assert_eq!(results[0].match_type, "hybrid");
        let names: Vec<&str> = results.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"betaFn"));
        // alphaFn: vector rank 1 but zero lexical evidence — still present as
        // a semantic hit (0.5 norm) since its cosine cleared the gate.
        assert!(names.contains(&"alphaFn"));
    }

    #[test]
    fn hybrid_bm25_only_incidental_word_is_gated() {
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let file_id = db
            .upsert_file("src/pay.ts", "typescript", "h", 0, 0)
            .unwrap();
        let sym_id = db
            .insert_symbol(file_id, "renderList", "function", None, 1, 0, 9, 0, None)
            .unwrap();
        // Contains exactly one of the three query tokens ("payment") — not
        // enough evidence for a BM25-only hit on a multi-word query.
        db.insert_chunk(
            sym_id,
            0,
            1,
            9,
            "e",
            "renderList render list",
            "pay.ts pay",
            "handles payment display rows",
        )
        .unwrap();

        let results = db
            .search_hybrid("stripe payment webhook", None, 10)
            .expect("search");
        assert!(
            results.is_empty(),
            "single incidental word must not pass the gate: {results:?}"
        );
    }

    #[test]
    fn required_token_matches_scales_with_query_length() {
        assert_eq!(
            required_token_matches(1, 2),
            1,
            "single-token exact-term queries pass"
        );
        assert_eq!(required_token_matches(2, 2), 2);
        assert_eq!(required_token_matches(3, 2), 2);
        assert_eq!(required_token_matches(4, 2), 2);
        assert_eq!(
            required_token_matches(5, 2),
            3,
            "long NL queries need a majority"
        );
        assert_eq!(required_token_matches(6, 2), 3);
    }

    /// Count raw FTS index hits WITHOUT joining the content table — ghost
    /// rows (index entries whose content row is gone) are only visible this
    /// way, since joins silently mask them.
    fn fts_raw_count(db: &IndexDb, table: &str, needle: &str) -> i64 {
        let conn = db.conn.lock().unwrap();
        conn.query_row(
            &format!("SELECT count(*) FROM {table} WHERE {table} MATCH ?1"),
            params![needle],
            |row| row.get(0),
        )
        .unwrap()
    }

    #[test]
    fn symbols_fts_has_no_ghosts_after_incremental_reindex() {
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let file_id = db
            .upsert_file("src/a.ts", "typescript", "h1", 0, 0)
            .unwrap();
        db.insert_symbol(file_id, "uniqueAlphaFn", "function", None, 1, 0, 2, 0, None)
            .unwrap();
        assert_eq!(fts_raw_count(&db, "symbols_fts", "uniqueAlphaFn"), 1);

        // Incremental reindex: old symbols deleted, new content inserted.
        db.delete_symbols_for_file(file_id).unwrap();
        db.insert_symbol(file_id, "uniqueBetaFn", "function", None, 1, 0, 2, 0, None)
            .unwrap();

        assert_eq!(
            fts_raw_count(&db, "symbols_fts", "uniqueAlphaFn"),
            0,
            "stale FTS row survived an incremental reindex"
        );
        assert_eq!(fts_raw_count(&db, "symbols_fts", "uniqueBetaFn"), 1);
    }

    #[test]
    fn chunk_fts_purged_by_delete_symbols_for_file_and_delete_file() {
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let file_id = db
            .upsert_file("src/b.ts", "typescript", "h1", 0, 0)
            .unwrap();
        let sym_id = db
            .insert_symbol(file_id, "widgetFactory", "function", None, 1, 0, 9, 0, None)
            .unwrap();
        db.insert_chunk(
            sym_id,
            0,
            1,
            9,
            "embed text",
            "widgetFactory widget factory",
            "b.ts b",
            "xyzzybody content here",
        )
        .unwrap();

        let chunks = db.chunks_for_file(file_id).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].embed_text, "embed text");
        assert_eq!((chunks[0].symbol_id, chunks[0].chunk_index), (sym_id, 0));
        assert_eq!(fts_raw_count(&db, "chunk_fts", "xyzzybody"), 1);

        db.delete_symbols_for_file(file_id).unwrap();
        assert_eq!(
            fts_raw_count(&db, "chunk_fts", "xyzzybody"),
            0,
            "chunk FTS ghost row"
        );
        assert!(db.chunks_for_file(file_id).unwrap().is_empty());

        // Re-insert, then remove the whole file: everything must go.
        let sym_id = db
            .insert_symbol(file_id, "widgetFactory", "function", None, 1, 0, 9, 0, None)
            .unwrap();
        db.insert_chunk(
            sym_id,
            0,
            1,
            9,
            "embed text",
            "widgetFactory",
            "b.ts",
            "xyzzybody again",
        )
        .unwrap();
        db.delete_file("src/b.ts").unwrap();
        assert!(db.file_by_path("src/b.ts").unwrap().is_none());
        assert_eq!(fts_raw_count(&db, "chunk_fts", "xyzzybody"), 0);
        assert_eq!(fts_raw_count(&db, "symbols_fts", "widgetFactory"), 0);
    }

    #[test]
    fn chunks_written_without_embedder_are_readable_for_later_embedding() {
        // Scan-time behavior with no model loaded: chunks + FTS exist so BM25
        // works, symbol_embeddings stays empty until the background pass.
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let file_id = db
            .upsert_file("src/c.ts", "typescript", "h1", 0, 0)
            .unwrap();
        let sym_id = db
            .insert_symbol(file_id, "pendingFn", "function", None, 1, 0, 4, 0, None)
            .unwrap();
        db.insert_chunk(
            sym_id,
            0,
            1,
            4,
            "function: pendingFn | body",
            "pendingFn pending fn",
            "c.ts c",
            "body",
        )
        .unwrap();

        let pending = db.embedding_pending_files().unwrap();
        assert_eq!(pending, 1, "file without embed_hash counts as pending");

        // The background pass reads the stored chunk and writes the vector
        // keyed by (symbol_id, chunk_index).
        let chunks = db.chunks_for_file(file_id).unwrap();
        db.upsert_embedding(
            chunks[0].symbol_id,
            chunks[0].chunk_index,
            chunks[0].start_line,
            chunks[0].end_line,
            &unit_vec(4, 0),
        )
        .unwrap();
        db.set_embed_hash(file_id, "h1").unwrap();
        assert_eq!(db.embedding_pending_files().unwrap(), 0);
    }

    /// What the vector leg promises beyond "the nearest chunks": media and
    /// another model's vectors are not candidates, and equal scores — a
    /// repository has many identical helpers — rank the same way every time.
    #[test]
    fn vector_leg_skips_media_and_foreign_vectors_and_breaks_ties_by_symbol() {
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let code = db.upsert_file("src/a.rs", "rust", "h", 0, 0).unwrap();
        let image = db.upsert_file("assets/logo.png", "image", "media:1:1", 0, 0).unwrap();
        let symbol = |file, name: &str, kind: &str| db.insert_symbol(file, name, kind, None, 1, 0, 40, 0, None).unwrap();
        let twin_a = symbol(code, "twin_a", "function");
        let twin_b = symbol(code, "twin_b", "function");
        let long = symbol(code, "long_fn", "function");
        let other_model = symbol(code, "other_model", "function");
        let logo = symbol(image, "logo.png", "image");

        let query = unit_vec(4, 0);
        let near = vec![0.8f32, 0.6, 0.0, 0.0];
        // Stored in the order that would win if order decided anything.
        db.upsert_embedding(twin_b, 0, 1, 5, &near).unwrap();
        db.upsert_embedding(twin_a, 0, 1, 5, &near).unwrap();
        // Two equally good chunks of one symbol, the lower one first.
        db.upsert_embedding(long, 1, 21, 40, &query).unwrap();
        db.upsert_embedding(long, 0, 1, 20, &query).unwrap();
        // A perfect match that is a picture, and one of another length.
        db.upsert_embedding(logo, 0, 0, 0, &query).unwrap();
        db.upsert_embedding(other_model, 0, 1, 5, &unit_vec(6, 0)).unwrap();

        let hits = db.vector_leg_candidates(VectorSet::Primary, &query, 0.1, 10).unwrap();
        let names: Vec<&str> = hits.iter().map(|h| h.name.as_str()).collect();
        assert_eq!(names, ["long_fn", "twin_a", "twin_b"]);
        assert_eq!((hits[0].chunk_start, hits[0].chunk_end), (1, 20), "the chunk higher up");
        assert_eq!(hits[0].file_path, "src/a.rs");

        // The cut falls between the twins: always on the same side.
        let two = db.vector_leg_candidates(VectorSet::Primary, &query, 0.1, 2).unwrap();
        assert_eq!(two.iter().map(|h| h.symbol_id).collect::<Vec<_>>(), [long, twin_a]);
        // And the gate applies to what is left.
        let gated = db.vector_leg_candidates(VectorSet::Primary, &query, 0.9, 10).unwrap();
        assert_eq!(gated.len(), 1);
    }

    /// The two sets hold vectors of different models for the same chunks: a
    /// search reads the one it is pointed at and never the other.
    #[test]
    fn the_upgrade_set_is_searched_apart_from_the_primary_one() {
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let fa = db.upsert_file("src/a.ts", "typescript", "h", 0, 0).unwrap();
        let fb = db.upsert_file("src/b.ts", "typescript", "h", 0, 0).unwrap();
        let sa = db.insert_symbol(fa, "alphaFn", "function", None, 1, 0, 2, 0, None).unwrap();
        let sb = db.insert_symbol(fb, "betaFn", "function", None, 1, 0, 2, 0, None).unwrap();

        // The first model puts alpha next to the query, the second one beta —
        // and its vectors have another length, as a second model's do.
        db.upsert_embedding(sa, 0, 1, 2, &unit_vec(4, 0)).unwrap();
        db.upsert_embedding(sb, 0, 1, 2, &unit_vec(4, 1)).unwrap();
        assert!(!db.reconcile_upgrade_model("second-model", 6).unwrap());
        db.upsert_embedding_in(VectorSet::Upgrade, sa, 0, 1, 2, &unit_vec(6, 1)).unwrap();
        db.upsert_embedding_in(VectorSet::Upgrade, sb, 0, 1, 2, &unit_vec(6, 0)).unwrap();

        let top = |set, query: &[f32]| -> Vec<String> {
            db.search_hybrid_with_in(set, "", Some(query), 10, &HybridParams::default())
                .unwrap()
                .into_iter()
                .map(|r| r.name)
                .collect()
        };
        assert_eq!(top(VectorSet::Primary, &unit_vec(4, 0)), ["alphaFn"]);
        assert_eq!(top(VectorSet::Upgrade, &unit_vec(6, 0)), ["betaFn"]);
        // A query from the wrong model matches nothing, in either direction.
        assert!(top(VectorSet::Upgrade, &unit_vec(4, 0)).is_empty());
        assert!(top(VectorSet::Primary, &unit_vec(6, 0)).is_empty());

        // The counts every caller already reads are the primary set's.
        assert_eq!(db.index_stats().unwrap().2, 2);
        assert_eq!(db.upgrade_state().unwrap().vectors, 2);
        assert_eq!(db.upgrade_model().as_deref(), Some("second-model"));
        assert_eq!(db.embedding_model().as_deref(), Some(crate::embeddings::MINILM_MODEL_ID));
    }

    /// When search may move to the upgrade set: not before a pass has covered
    /// every text file, not with another model's query, and not while it is
    /// far behind — but a few edited files do not send it back.
    #[test]
    fn the_upgrade_set_is_usable_once_complete_and_while_nearly_current() {
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let files: Vec<i64> = (0..30)
            .map(|i| db.upsert_file(&format!("src/f{i}.rs"), "rust", "v1", 0, 0).unwrap())
            .collect();
        db.upsert_file("assets/logo.png", "image", "media:1:1", 0, 0).unwrap();

        let state = db.upgrade_state().unwrap();
        assert_eq!((state.text_files, state.pending_files, state.complete), (30, 30, false));
        assert!(!state.usable("m"));

        db.reconcile_upgrade_model("m", 4).unwrap();
        for id in &files[..29] {
            db.set_upgrade_hash(*id, "v1").unwrap();
        }
        assert!(!db.mark_upgrade_complete().unwrap(), "one file short");
        assert!(!db.upgrade_state().unwrap().usable("m"));
        db.set_upgrade_hash(files[29], "v1").unwrap();
        assert!(db.mark_upgrade_complete().unwrap());
        let state = db.upgrade_state().unwrap();
        assert_eq!((state.pending_files, state.complete), (0, true), "the image is not text");
        assert!(state.usable("m"));
        assert!(!state.usable("another-model"));

        // Three edited files: still the set to search.
        for i in 0..3 {
            db.upsert_file(&format!("src/f{i}.rs"), "rust", "v2", 1, 0).unwrap();
        }
        let state = db.upgrade_state().unwrap();
        assert_eq!(state.pending_files, 3);
        assert!(state.usable("m"));
        // One more and it is too far behind — until it has caught up.
        db.upsert_file("src/f3.rs", "rust", "v2", 1, 0).unwrap();
        let state = db.upgrade_state().unwrap();
        assert!(state.complete && !state.usable("m"));
        assert!(db.mark_upgrade_complete().unwrap(), "falling behind does not undo completeness");
        for id in &files[..4] {
            db.set_upgrade_hash(*id, "v2").unwrap();
        }
        assert!(db.upgrade_state().unwrap().usable("m"));

        // In a large workspace the allowance is one file in twenty.
        let many = UpgradeState {
            model: Some("m".into()),
            complete: true,
            pending_files: 50,
            text_files: 1000,
            vectors: 0,
        };
        assert!(many.usable("m"));
        assert!(!UpgradeState { pending_files: 51, ..many }.usable("m"));
    }

    #[test]
    fn a_different_upgrade_model_starts_the_set_over_and_only_that_set() {
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let file = db.upsert_file("src/a.rs", "rust", "v1", 0, 0).unwrap();
        let sym = db.insert_symbol(file, "alpha", "function", None, 1, 0, 2, 0, None).unwrap();
        db.upsert_embedding(sym, 0, 1, 2, &unit_vec(4, 0)).unwrap();

        assert!(!db.reconcile_upgrade_model("m1", 6).unwrap());
        db.upsert_embedding_in(VectorSet::Upgrade, sym, 0, 1, 2, &unit_vec(6, 0)).unwrap();
        db.set_upgrade_hash(file, "v1").unwrap();
        assert!(db.mark_upgrade_complete().unwrap());

        // The same model again: nothing happens.
        assert!(!db.reconcile_upgrade_model("m1", 6).unwrap());
        assert!(db.upgrade_state().unwrap().usable("m1"));

        // Another model: the set is emptied and every file queued.
        assert!(db.reconcile_upgrade_model("m2", 6).unwrap());
        let state = db.upgrade_state().unwrap();
        assert_eq!(state.model.as_deref(), Some("m2"));
        assert_eq!((state.vectors, state.pending_files, state.complete), (0, 1, false));
        assert!(db.upgrade_hashes().unwrap().is_empty());

        // Vectors of the wrong length under the right name are another
        // model's too.
        db.upsert_embedding_in(VectorSet::Upgrade, sym, 0, 1, 2, &unit_vec(8, 0)).unwrap();
        assert!(db.reconcile_upgrade_model("m2", 6).unwrap());
        assert_eq!(db.upgrade_state().unwrap().vectors, 0);

        assert_eq!(db.index_stats().unwrap().2, 1, "the primary set was never touched");
    }

    /// Two servers on one repository share an index. One of them checking
    /// the index against its models — it reads, then writes — must not fail
    /// because the other committed in between; and the other may be writing
    /// for an hour.
    #[test]
    fn reconciling_models_survives_another_connection_committing_meanwhile() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.db");
        let (ours, theirs) = (IndexDb::open(&path).unwrap(), IndexDb::open(&path).unwrap());
        let file = ours.upsert_file("src/a.rs", "rust", "v0", 0, 0).unwrap();
        for i in 0..600 {
            let sym = ours.insert_symbol(file, &format!("f{i}"), "function", None, 1, 0, 2, 0, None).unwrap();
            ours.upsert_embedding(sym, 0, 1, 2, &unit_vec(384, i % 384)).unwrap();
        }
        let profile = ModelProfile::text_only(crate::embeddings::MINILM_MODEL_ID, 384);
        ours.reconcile_embedding_model(&profile).unwrap();

        let stop = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let mut i = 0;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    theirs.upsert_file("src/b.rs", "rust", &format!("v{i}"), i, 0).unwrap();
                    i += 1;
                }
            });
            let failures: Vec<String> = (0..150)
                .filter_map(|_| ours.reconcile_embedding_model(&profile).err())
                .collect();
            // Before any assertion: a panic in here would leave the other
            // thread writing for ever.
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            assert!(failures.is_empty(), "{} of 150 failed, e.g. {}", failures.len(), failures[0]);
        });
        assert_eq!(ours.index_stats().unwrap().2, 600);
    }

    /// The upgrade checks its model at the start of every turn while the
    /// watcher writes through a connection of its own: that check must not
    /// fail because the other connection committed, in either order.
    #[test]
    fn checking_the_upgrade_model_survives_another_connection_writing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.db");
        let (pass, watcher) = (IndexDb::open(&path).unwrap(), IndexDb::open(&path).unwrap());
        let file = watcher.upsert_file("src/a.rs", "rust", "v0", 0, 0).unwrap();
        let sym = watcher.insert_symbol(file, "alpha", "function", None, 1, 0, 2, 0, None).unwrap();
        assert!(!pass.reconcile_upgrade_model("m1", 6).unwrap());
        pass.upsert_embedding_in(VectorSet::Upgrade, sym, 0, 1, 2, &unit_vec(6, 0)).unwrap();
        for i in 0..50 {
            watcher.upsert_file("src/a.rs", "rust", &format!("v{i}"), i, 0).unwrap();
            assert!(!pass.reconcile_upgrade_model("m1", 6).unwrap(), "round {i}");
        }
        // The change itself, with the other connection still writing.
        watcher.upsert_file("src/b.rs", "rust", "v0", 0, 0).unwrap();
        assert!(pass.reconcile_upgrade_model("m2", 6).unwrap());
        assert_eq!(watcher.upgrade_state().unwrap().vectors, 0);
        assert_eq!(watcher.upgrade_model().as_deref(), Some("m2"));
    }

    /// Giving up on a file counts it as done in its current content, and
    /// only in that content.
    #[test]
    fn a_skipped_upgrade_file_is_done_until_it_changes() {
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let file = db.upsert_file("src/a.rs", "rust", "v1", 0, 0).unwrap();
        db.reconcile_upgrade_model("m", 6).unwrap();
        assert!(!db.mark_upgrade_complete().unwrap());
        db.skip_upgrade_file(file).unwrap();
        assert!(db.mark_upgrade_complete().unwrap());
        assert_eq!(db.upgrade_hashes().unwrap().get(&file).map(String::as_str), Some("v1"));

        db.upsert_file("src/a.rs", "rust", "v2", 1, 0).unwrap();
        assert_eq!(db.upgrade_state().unwrap().pending_files, 1);
        db.skip_upgrade_file(file).unwrap();
        assert_eq!(db.upgrade_state().unwrap().pending_files, 0);
        // A file that is not there is nothing to skip.
        db.skip_upgrade_file(file + 100).unwrap();
        assert_eq!(db.upgrade_hashes().unwrap().len(), 1);
    }

    #[test]
    fn re_scanning_or_deleting_a_file_takes_its_upgrade_vectors_along() {
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let file = db.upsert_file("src/a.rs", "rust", "v1", 0, 0).unwrap();
        let sym = db.insert_symbol(file, "alpha", "function", None, 1, 0, 2, 0, None).unwrap();
        db.upsert_embedding_in(VectorSet::Upgrade, sym, 0, 1, 2, &unit_vec(6, 0)).unwrap();
        db.set_upgrade_hash(file, "v1").unwrap();

        // A re-scan replaces the file's symbols: vectors keyed by the old
        // ones must not survive to be read as the new ones'. Nor may the
        // record that the file was done — its hash may be the old one again
        // (an edit, undone), and it has no vectors now.
        db.delete_symbols_for_file(file).unwrap();
        let state = db.upgrade_state().unwrap();
        assert_eq!((state.vectors, state.pending_files), (0, 1));
        assert!(db.upgrade_hashes().unwrap().is_empty());

        let sym = db.insert_symbol(file, "alpha", "function", None, 1, 0, 2, 0, None).unwrap();
        db.upsert_embedding_in(VectorSet::Upgrade, sym, 0, 1, 2, &unit_vec(6, 0)).unwrap();
        db.delete_file("src/a.rs").unwrap();
        assert_eq!(db.upgrade_state().unwrap().vectors, 0);
        assert!(db.upgrade_hashes().unwrap().is_empty());
    }

    #[test]
    fn embed_hash_tracks_content_independently_of_re_scan() {
        let db = IndexDb::open(Path::new(":memory:")).expect("open in-memory db");
        let file_id = db
            .upsert_file("src/foo.ts", "typescript", "hash-v1", 0, 0)
            .expect("upsert file");

        // No embed_hash yet — file has never been embedded.
        let file = db.file_by_path("src/foo.ts").unwrap().unwrap();
        assert_eq!(file.embed_hash, None);
        assert_ne!(file.hash, file.embed_hash);

        // Embedding generation records the hash it embedded from.
        db.set_embed_hash(file_id, "hash-v1")
            .expect("set embed_hash");
        let file = db.file_by_path("src/foo.ts").unwrap().unwrap();
        assert_eq!(file.embed_hash.as_deref(), Some("hash-v1"));
        assert_eq!(
            file.hash, file.embed_hash,
            "hash == embed_hash means the file is up to date"
        );

        // Re-scanning with unchanged content re-upserts the same hash and
        // must NOT disturb embed_hash — this is what lets a second
        // generate_all_embeddings pass skip the file entirely.
        db.upsert_file("src/foo.ts", "typescript", "hash-v1", 1, 0)
            .expect("re-upsert unchanged file");
        let file = db.file_by_path("src/foo.ts").unwrap().unwrap();
        assert_eq!(file.embed_hash.as_deref(), Some("hash-v1"));
        assert_eq!(file.hash, file.embed_hash);

        // Editing the file changes hash but leaves embed_hash pointing at
        // the stale content, so the mismatch correctly flags it as pending.
        db.upsert_file("src/foo.ts", "typescript", "hash-v2", 2, 0)
            .expect("re-upsert changed file");
        let file = db.file_by_path("src/foo.ts").unwrap().unwrap();
        assert_eq!(file.hash.as_deref(), Some("hash-v2"));
        assert_eq!(file.embed_hash.as_deref(), Some("hash-v1"));
        assert_ne!(
            file.hash, file.embed_hash,
            "content changed — embedding is now stale"
        );
    }
}
