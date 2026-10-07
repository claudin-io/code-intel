//! Calibration eval for hybrid search: indexes a workspace with each
//! embedding model, runs real queries with known answers, and reports the
//! rank of the expected files per query class, raw cosine distributions and
//! indexing speed. It is how one model is compared against another and how
//! `HybridParams` and the media gate are tuned — update those constants from
//! a run of this, never by hand.
//!
//! Ported from Claudinio Code's `examples/semantic_eval.rs`; the query set
//! next to this file (`semantic_eval_queries.json`) is the one written
//! against the Claudinio Code repository, which is the workspace to point
//! this at.
//!
//!   cargo run --release --example semantic_eval -- <workspace_root> [queries.json]
//!       [--model auto|minilm|embeddinggemma2|both]
//!                                               default: both — `auto` (what the server runs: MiniLM
//!                                               for text, EmbeddingGemma 2 beside it for media) and
//!                                               then `embeddinggemma2` (that model for text as well);
//!                                               `auto` alone in a candle build
//!       [--models-dir <dir>]                    default: the plugin's own cache, so nothing is fetched twice
//!       [--sweep]                               grid over the fusion gates, one row per combination
//!       [--no-vector]                           BM25 leg only: the pending-embeddings window
//!       [--report <file>]                       also write everything printed to <file>
//!
//! Models are downloaded on first use. The text model of every run is named
//! in its output, and `embeddinggemma2` has no fallback, so a model that
//! fails to load is reported as failed instead of being quietly measured as
//! the other one.

use claudinio_code_intel::db::{HybridParams, IndexDb, MEDIA_MIN_COSINE};
use claudinio_code_intel::embeddings::{self, ModelChoice, SharedEmbedder};
use claudinio_code_intel::indexer::{self, IndexProgress};
use claudinio_code_intel::media;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(serde::Deserialize)]
struct EvalSet {
    positive: Vec<PositiveCase>,
    negative: Vec<NegativeCase>,
    /// Queries whose answer is an image or audio file of the workspace.
    #[serde(default)]
    media: Vec<MediaCase>,
}

#[derive(serde::Deserialize)]
struct PositiveCase {
    query: String,
    expected: Vec<String>,
    #[serde(default = "default_class")]
    class: String,
}

fn default_class() -> String {
    "concept".into()
}

#[derive(serde::Deserialize)]
struct NegativeCase {
    query: String,
}

#[derive(serde::Deserialize)]
struct MediaCase {
    query: String,
    expected: Vec<String>,
}

#[derive(Default, Clone)]
struct ClassStats {
    n: usize,
    top1: usize,
    top3: usize,
    top15: usize,
}

struct Summary {
    per_class: BTreeMap<String, ClassStats>,
    neg_empty: usize,
    neg_total: usize,
    relevant_scores: Vec<f32>,
    irrelevant_scores: Vec<f32>,
}

impl Summary {
    fn overall(&self) -> ClassStats {
        let mut o = ClassStats::default();
        for s in self.per_class.values() {
            o.n += s.n;
            o.top1 += s.top1;
            o.top3 += s.top3;
            o.top15 += s.top15;
        }
        o
    }
}

/// Everything printed, kept so `--report` can write it out.
#[derive(Default)]
struct Report {
    text: String,
}

impl Report {
    fn line(&mut self, s: impl AsRef<str>) {
        println!("{}", s.as_ref());
        self.text.push_str(s.as_ref());
        self.text.push('\n');
    }
}

struct Options {
    root: String,
    queries: PathBuf,
    models_root: PathBuf,
    sweep: bool,
    no_vector: bool,
    report: Option<PathBuf>,
    models: Vec<ModelChoice>,
}

fn usage() -> ! {
    eprintln!(
        "usage: semantic_eval <workspace_root> [queries.json] [--model auto|minilm|embeddinggemma2|both] \
         [--models-dir <dir>] [--sweep] [--no-vector] [--report <file>]"
    );
    std::process::exit(1)
}

fn parse_args() -> Options {
    let mut positional: Vec<String> = Vec::new();
    let mut model = String::from("both");
    let mut models_root: Option<PathBuf> = None;
    let mut report = None;
    let (mut sweep, mut no_vector) = (false, false);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--sweep" => sweep = true,
            "--no-vector" => no_vector = true,
            "--model" => model = args.next().unwrap_or_else(|| usage()),
            "--models-dir" => {
                models_root = Some(PathBuf::from(args.next().unwrap_or_else(|| usage())))
            }
            "--report" => report = Some(PathBuf::from(args.next().unwrap_or_else(|| usage()))),
            other if other.starts_with("--") => usage(),
            _ => positional.push(a),
        }
    }
    if positional.is_empty() || positional.len() > 2 {
        usage();
    }
    let models = match model.as_str() {
        "both" if embeddings::gemma2_supported() => vec![ModelChoice::Auto, ModelChoice::Gemma2],
        "both" => vec![ModelChoice::Auto],
        other => match ModelChoice::parse(other) {
            None => usage(),
            Some(choice) => vec![choice],
        },
    };
    Options {
        root: positional[0].clone(),
        queries: positional.get(1).map(PathBuf::from).unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/semantic_eval_queries.json")
        }),
        // The same directory the MCP server uses (`<cache>/claudinio-code-intel/models`).
        models_root: models_root.unwrap_or_else(|| {
            std::env::var_os("CODE_INTEL_CACHE_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    dirs::cache_dir()
                        .unwrap_or_else(std::env::temp_dir)
                        .join("claudinio-code-intel")
                })
                .join("models")
        }),
        sweep,
        no_vector,
        report,
        models,
    }
}

fn main() {
    let opts = parse_args();
    let eval: EvalSet =
        serde_json::from_str(&std::fs::read_to_string(&opts.queries).expect("read queries.json"))
            .expect("parse queries.json");
    let mut report = Report::default();
    report.line(format!(
        "semantic_eval — workspace {} — {} threads per model run — {} positive / {} negative / {} media queries",
        opts.root,
        embeddings::intra_threads(),
        eval.positive.len(),
        eval.negative.len(),
        eval.media.len()
    ));

    if opts.no_vector {
        run_model(&opts, &eval, None, &mut report);
    } else {
        for choice in &opts.models {
            run_model(&opts, &eval, Some(*choice), &mut report);
        }
    }

    if let Some(path) = &opts.report {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(path, &report.text).expect("write report");
        eprintln!("report written to {}", path.display());
    }
}

fn run_model(opts: &Options, eval: &EvalSet, choice: Option<ModelChoice>, report: &mut Report) {
    let root = opts.root.as_str();
    report.line(String::new());
    report.line("################################################################");
    report.line(match choice {
        Some(c) => format!("# model: {c:?}"),
        None => "# no model (BM25 leg only)".to_string(),
    });
    report.line("################################################################");

    let needs = media::detect_media_needs(root);
    let shared: Option<SharedEmbedder> = match choice {
        Some(choice) => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");
            let t = Instant::now();
            match rt.block_on(embeddings::ensure_and_load(
                &opts.models_root,
                choice,
                needs,
            )) {
                Ok(shared) => {
                    let (id, media_model, support) = {
                        let g = shared.lock().unwrap();
                        (g.model_id(), g.media_model(), g.media_support())
                    };
                    report.line(format!(
                        "loaded in {:.1}s (download included if it was not cached); text: {id}; media: {}; encoders: image={} audio={}",
                        t.elapsed().as_secs_f32(),
                        media_model.map(|(id, _)| id).unwrap_or("none"),
                        support.images,
                        support.audio
                    ));
                    Some(shared)
                }
                Err(e) => {
                    report.line(format!("FAILED to load {choice:?}: {e}"));
                    return;
                }
            }
        }
        None => None,
    };

    let db_path = std::env::temp_dir().join(format!(
        "semantic_eval_{}_{}.db",
        std::process::id(),
        choice
            .map(|c| format!("{c:?}"))
            .unwrap_or_else(|| "bm25".into())
    ));
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", db_path.display()));
    }
    let db = IndexDb::open(&db_path).expect("open db");

    let t = Instant::now();
    let (files, symbols) = indexer::scan_workspace(&db, root, None, None, None).expect("scan");
    report.line(format!(
        "scanned {files} files, {symbols} symbols in {:.1}s",
        t.elapsed().as_secs_f32()
    ));

    if let Some(shared) = &shared {
        let t = Instant::now();
        let progress = |p: IndexProgress| {
            if p.files_indexed % 50 == 0 {
                eprintln!(
                    "  embedding {}/{} files, {} vectors",
                    p.files_indexed, p.total_files, p.symbols_indexed
                );
            }
        };
        indexer::generate_all_embeddings(&db, shared, Some(&progress), root).expect("embed");
        let secs = t.elapsed().as_secs_f32();
        let vectors = db.index_stats().map(|s| s.2).unwrap_or(0);
        let media_vectors = db.media_embedding_count().unwrap_or(0);
        let (images, audio) = db.media_file_counts().unwrap_or((0, 0));
        report.line(format!(
            "embedded {vectors} vectors in {secs:.1}s; media: {images} images, {audio} audio, {media_vectors} with a content vector"
        ));
        // The pass above mixes text chunks and media files. Time a few media
        // files on their own to tell the two apart: what is left of the
        // total is the text.
        let media_secs = media_timing(shared, root, report);
        let text_vectors = vectors - media_vectors;
        let text_secs = (secs - media_secs * media_vectors as f32).max(0.001);
        report.line(format!(
            "text: {text_vectors} vectors in about {text_secs:.1}s ({:.1} vectors/s)",
            text_vectors as f32 / text_secs
        ));
    }

    let encode = |text: &str| -> Option<Vec<f32>> {
        shared
            .as_ref()
            .map(|s| s.lock().unwrap().encode_query(text).expect("encode query"))
    };
    // Encode every query once — the sweep re-runs search, not the model.
    let t = Instant::now();
    let pos_vecs: Vec<Option<Vec<f32>>> = eval.positive.iter().map(|c| encode(&c.query)).collect();
    let neg_vecs: Vec<Option<Vec<f32>>> = eval.negative.iter().map(|c| encode(&c.query)).collect();
    if shared.is_some() {
        let n = (pos_vecs.len() + neg_vecs.len()).max(1);
        report.line(format!(
            "query encoding: {:.0} ms per query",
            t.elapsed().as_secs_f32() * 1000.0 / n as f32
        ));
    }

    let model_id = shared.as_ref().map(|s| s.lock().unwrap().model_id());
    let params = HybridParams::for_model(model_id);
    report.line(format!(
        "\nparams in use: rrf_k={} w_vector={} w_bm25={} min_cosine_candidate={} min_hybrid_score={}",
        params.rrf_k, params.w_vector, params.w_bm25, params.min_cosine_candidate, params.min_hybrid_score
    ));
    let s = run_eval(&db, eval, &pos_vecs, &neg_vecs, &params, Some(report));
    print_summary(&s, shared.is_some(), report);

    if shared.is_some() {
        let band = cosine_analysis(&db, eval, &pos_vecs, &neg_vecs, report);
        if opts.sweep {
            sweep(&db, eval, &pos_vecs, &neg_vecs, &params, band, report);
        }
        if let Some(shared) = &shared {
            media_analysis(&db, eval, shared, report);
        }
    }

    drop(db);
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", db_path.display()));
    }
}

/// Seconds to embed one media file, averaged over a few of the workspace's
/// own (0 when there is nothing to time). Per kind in the report: an image
/// and a sound cost differently.
fn media_timing(shared: &SharedEmbedder, root: &str, report: &mut Report) -> f32 {
    const SAMPLE: usize = 5;
    let support = shared.lock().unwrap().media_support();
    let files = media::media_files(root);
    let (mut total, mut n) = (0f32, 0usize);
    for kind in media::MediaKind::ALL {
        if !support.has(kind) {
            continue;
        }
        let mut times: Vec<f32> = Vec::new();
        for (path, _) in files.iter().filter(|(_, k)| *k == kind) {
            if times.len() == SAMPLE {
                break;
            }
            let t = Instant::now();
            if embeddings::embed_media_file(shared, Path::new(path), kind).is_ok() {
                times.push(t.elapsed().as_secs_f32());
            }
        }
        if times.is_empty() {
            continue;
        }
        let sum: f32 = times.iter().sum();
        report.line(format!(
            "{}: {:.2}s per file (mean of {}; slowest {:.2}s)",
            kind.as_str(),
            sum / times.len() as f32,
            times.len(),
            times.iter().cloned().fold(0.0, f32::max)
        ));
        total += sum;
        n += times.len();
    }
    if n == 0 { 0.0 } else { total / n as f32 }
}

fn print_summary(s: &Summary, hybrid: bool, report: &mut Report) {
    let o = s.overall();
    let pct = |a: usize, n: usize| 100 * a / n.max(1);
    report.line(format!(
        "\n=== SUMMARY ({}) ===",
        if hybrid { "hybrid" } else { "BM25-only" }
    ));
    report.line(format!("positive queries: {}", o.n));
    report.line(format!(
        "  top-1 (unique file): {} ({}%)",
        o.top1,
        pct(o.top1, o.n)
    ));
    report.line(format!(
        "  top-3 (unique file): {} ({}%)",
        o.top3,
        pct(o.top3, o.n)
    ));
    report.line(format!(
        "  anywhere in top-15:  {} ({}%)",
        o.top15,
        pct(o.top15, o.n)
    ));
    report.line(format!(
        "negative queries empty: {}/{}",
        s.neg_empty, s.neg_total
    ));
    report.line("\nper class (top1/top3/top15 of n):");
    for (class, c) in &s.per_class {
        report.line(format!(
            "  {:16} {:2}/{:2}/{:2} of {:2}  ({}% / {}% / {}%)",
            class,
            c.top1,
            c.top3,
            c.top15,
            c.n,
            pct(c.top1, c.n),
            pct(c.top3, c.n),
            pct(c.top15, c.n),
        ));
    }
    report.line(dist("relevant score dist  ", &s.relevant_scores));
    report.line(dist("irrelevant score dist", &s.irrelevant_scores));
}

/// The cosine values a sweep tries: nine across the band where the gate can
/// do anything — from the 90th percentile of unrelated chunks up to the upper
/// quartile of the chunks queries are looking for — plus the gate in use.
///
/// Measured, not fixed, because models put their scores in different places:
/// MiniLM's band is 0.20-0.59, EmbeddingGemma 2's is 0.71-0.83, and a fixed
/// 0.20-0.60 grid run against the latter printed 54 identical rows.
fn cosine_grid(band: Option<(f32, f32)>, in_use: f32) -> Vec<f32> {
    const STEPS: usize = 9;
    let (low, high) = match band {
        Some((low, high)) if high > low => (low, high),
        _ => (0.20, 0.60),
    };
    let mut grid: Vec<f32> = (0..STEPS)
        .map(|i| low + (high - low) * i as f32 / (STEPS - 1) as f32)
        .chain([in_use])
        .map(|c| (c * 1000.0).round() / 1000.0)
        .collect();
    grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    grid.dedup();
    grid
}

/// The decisive levers are the two gates (vector-leg cosine entry and the
/// final hybrid score) plus the BM25 weight; rrf_k trades against
/// min_hybrid_score on the same axis, so it stays at the default.
fn sweep(
    db: &IndexDb,
    eval: &EvalSet,
    pos_vecs: &[Option<Vec<f32>>],
    neg_vecs: &[Option<Vec<f32>>],
    base: &HybridParams,
    band: Option<(f32, f32)>,
    report: &mut Report,
) {
    let grid = cosine_grid(band, base.min_cosine_candidate);
    report.line("\n=== SWEEP ===");
    report.line("min_hyb  w_bm25  min_cos |  top1  top3  top15  neg  | exact-id top1");
    for min_hybrid_score in [0.35f32, 0.40, 0.45] {
        for w_bm25 in [0.7f32, 1.0] {
            for &min_cosine_candidate in &grid {
                let params = HybridParams {
                    w_bm25,
                    min_hybrid_score,
                    min_cosine_candidate,
                    ..base.clone()
                };
                let s = run_eval(db, eval, pos_vecs, neg_vecs, &params, None);
                let o = s.overall();
                let exact = s
                    .per_class
                    .get("exact-identifier")
                    .cloned()
                    .unwrap_or_default();
                report.line(format!(
                    "{:7} {:7} {:8} | {:4}% {:4}% {:5}% {:2}/{} | {:3}%",
                    min_hybrid_score,
                    w_bm25,
                    min_cosine_candidate,
                    100 * o.top1 / o.n.max(1),
                    100 * o.top3 / o.n.max(1),
                    100 * o.top15 / o.n.max(1),
                    s.neg_empty,
                    s.neg_total,
                    100 * exact.top1 / exact.n.max(1),
                ));
            }
        }
    }
}

fn basename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Where the model's raw cosines sit, before any fusion: how a query scores
/// against the best chunk of the file it is looking for, against everything
/// else, and what an off-topic query's best match looks like. The vector gate
/// (`min_cosine_candidate`) belongs between the last two and under the first.
///
/// Returns the band a sweep of that gate should cover: the 90th percentile of
/// unrelated chunks and the upper quartile of the sought ones.
fn cosine_analysis(
    db: &IndexDb,
    eval: &EvalSet,
    pos_vecs: &[Option<Vec<f32>>],
    neg_vecs: &[Option<Vec<f32>>],
    report: &mut Report,
) -> Option<(f32, f32)> {
    let rows: Vec<(String, Vec<f32>)> = db
        .load_all_embeddings()
        .expect("load embeddings")
        .into_iter()
        .filter(|(sym, _, _, _)| sym.kind != "image" && sym.kind != "audio")
        .map(|(sym, _, _, v)| (basename(&sym.file_path.unwrap_or_default()), v))
        .collect();
    if rows.is_empty() {
        return None;
    }
    let mut relevant_best = Vec::new();
    let mut background = Vec::new();
    for (case, qv) in eval.positive.iter().zip(pos_vecs) {
        let Some(qv) = qv else { continue };
        let mut best: Option<f32> = None;
        for (i, (base, v)) in rows.iter().enumerate() {
            if v.len() != qv.len() {
                continue;
            }
            let c = dot(qv, v);
            if case.expected.iter().any(|e| e == base) {
                best = Some(best.map_or(c, |b: f32| b.max(c)));
            } else if i % 7 == 0 {
                // A sample is plenty for percentiles.
                background.push(c);
            }
        }
        if let Some(b) = best {
            relevant_best.push(b);
        }
    }
    let mut negative_top = Vec::new();
    for qv in neg_vecs.iter().flatten() {
        let top = rows
            .iter()
            .filter(|(_, v)| v.len() == qv.len())
            .map(|(_, v)| dot(qv, v))
            .fold(f32::MIN, f32::max);
        negative_top.push(top);
    }
    report.line("\n=== RAW COSINE (vector leg, before fusion) ===");
    report.line(dist("best chunk of an expected file ", &relevant_best));
    report.line(dist("chunks of other files (sample) ", &background));
    report.line(dist("off-topic query, its best chunk", &negative_top));
    background.sort_by(|a, b| a.partial_cmp(b).unwrap());
    relevant_best.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if background.is_empty() || relevant_best.is_empty() {
        return None;
    }
    let p = |q: f64| background[((background.len() - 1) as f64 * q) as usize];
    report.line(format!(
        "other-file chunks, upper tail: p90={:.3} p99={:.3} p99.9={:.3}",
        p(0.90),
        p(0.99),
        p(0.999)
    ));
    Some((p(0.90), relevant_best[(relevant_best.len() - 1) * 3 / 4]))
}

/// Text queries against the workspace's image and audio vectors. Two
/// questions: do code queries stay under the media gate (they should match no
/// picture), and do the media queries clear it for the file they describe?
fn media_analysis(db: &IndexDb, eval: &EvalSet, shared: &SharedEmbedder, report: &mut Report) {
    let rows: Vec<(String, Vec<f32>)> = db
        .load_all_embeddings()
        .expect("load embeddings")
        .into_iter()
        .filter(|(sym, _, _, _)| sym.kind == "image" || sym.kind == "audio")
        .map(|(sym, _, _, v)| (basename(&sym.file_path.unwrap_or_default()), v))
        .collect();
    if rows.is_empty() {
        return;
    }
    let media_query = |text: &str| -> Option<Vec<f32>> {
        shared
            .lock()
            .unwrap()
            .encode_media_query(text)
            .expect("encode media query")
    };
    report.line(format!(
        "\n=== MEDIA ({} files with a content vector; gate {MEDIA_MIN_COSINE}) ===",
        rows.len()
    ));

    let mut code_query_max = Vec::new();
    let code_queries = eval
        .positive
        .iter()
        .map(|c| c.query.as_str())
        .chain(eval.negative.iter().map(|c| c.query.as_str()));
    for q in code_queries {
        let Some(qv) = media_query(q) else { return };
        code_query_max.push(
            rows.iter()
                .map(|(_, v)| dot(&qv, v))
                .fold(f32::MIN, f32::max),
        );
    }
    report.line(dist("code query, its best media file", &code_query_max));
    let leaks = code_query_max
        .iter()
        .filter(|c| **c >= MEDIA_MIN_COSINE)
        .count();
    report.line(format!(
        "code queries over the gate: {leaks}/{} (each would show an unrelated file under `media`)",
        code_query_max.len()
    ));

    for case in &eval.media {
        let Some(qv) = media_query(&case.query) else {
            return;
        };
        let mut scored: Vec<(f32, &str)> = rows
            .iter()
            .map(|(b, v)| (dot(&qv, v), b.as_str()))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let is_expected = |name: &str| case.expected.iter().any(|e| e == name);
        let rank = scored
            .iter()
            .position(|(_, b)| is_expected(b))
            .map(|i| i + 1);
        let best_expected = scored.iter().find(|(_, b)| is_expected(b)).map(|(c, _)| *c);
        let best_other = scored
            .iter()
            .find(|(_, b)| !is_expected(b))
            .map(|(c, _)| *c);
        report.line(format!(
            "[rank={}] {:60} expected={} other={}  top: {}",
            rank.map(|r| r.to_string()).unwrap_or_else(|| "MISS".into()),
            truncate(&case.query, 58),
            best_expected
                .map(|c| format!("{c:.3}"))
                .unwrap_or_else(|| "-".into()),
            best_other
                .map(|c| format!("{c:.3}"))
                .unwrap_or_else(|| "-".into()),
            scored
                .iter()
                .take(3)
                .map(|(c, b)| format!("{b} ({c:.3})"))
                .collect::<Vec<_>>()
                .join(" | ")
        ));
        // What the tool itself would return for this query.
        let hits = db
            .search_media(&case.query, Some(&qv), 3)
            .expect("search media");
        report.line(format!(
            "          search_media -> {}",
            if hits.is_empty() {
                "(nothing)".to_string()
            } else {
                hits.iter()
                    .map(|h| format!("{} ({:.2}/{})", h.name, h.score, &h.match_type[..1]))
                    .collect::<Vec<_>>()
                    .join(" | ")
            }
        ));
    }
}

fn run_eval(
    db: &IndexDb,
    eval: &EvalSet,
    pos_vecs: &[Option<Vec<f32>>],
    neg_vecs: &[Option<Vec<f32>>],
    params: &HybridParams,
    mut verbose: Option<&mut Report>,
) -> Summary {
    let mut summary = Summary {
        per_class: BTreeMap::new(),
        neg_empty: 0,
        neg_total: eval.negative.len(),
        relevant_scores: Vec::new(),
        irrelevant_scores: Vec::new(),
    };

    if let Some(r) = verbose.as_deref_mut() {
        r.line("\n=== POSITIVE QUERIES ===");
    }
    for (case, qvec) in eval.positive.iter().zip(pos_vecs.iter()) {
        let results = db
            .search_hybrid_with(&case.query, qvec.as_deref(), 15, params)
            .expect("search");

        // unique-file rank of the first expected basename
        let mut seen_files: Vec<String> = Vec::new();
        let mut rank: Option<usize> = None;
        for r in &results {
            let base = basename(&r.file_path);
            if !seen_files.contains(&base) {
                seen_files.push(base.clone());
            }
            let file_rank = seen_files.iter().position(|f| f == &base).unwrap() + 1;
            // A hit is either the expected file itself, or a doc section whose
            // title names it (e.g. "6. Frontend: `src/App.tsx` — Settings
            // modal") — the agent follows that pointer just the same.
            let is_hit = case.expected.iter().any(|e| e == &base)
                || (r.kind == "doc_section"
                    && case.expected.iter().any(|e| r.name.contains(e.as_str())));
            if is_hit {
                if rank.is_none() {
                    rank = Some(file_rank);
                }
                summary.relevant_scores.push(r.score);
            } else {
                summary.irrelevant_scores.push(r.score);
            }
        }
        let stats = summary.per_class.entry(case.class.clone()).or_default();
        stats.n += 1;
        if rank == Some(1) {
            stats.top1 += 1;
        }
        if matches!(rank, Some(r) if r <= 3) {
            stats.top3 += 1;
        }
        if rank.is_some() {
            stats.top15 += 1;
        }
        if let Some(r) = verbose.as_deref_mut() {
            let top: Vec<String> = results
                .iter()
                .take(3)
                .map(|r| {
                    format!(
                        "{}:{} ({:.3}/{})",
                        basename(&r.file_path),
                        r.name,
                        r.score,
                        &r.match_type[..1]
                    )
                })
                .collect();
            r.line(format!(
                "[rank={}] [{}] {:52} -> {}",
                rank.map(|r| r.to_string()).unwrap_or_else(|| "MISS".into()),
                &case.class[..case.class.len().min(8)],
                truncate(&case.query, 50),
                top.join(" | ")
            ));
        }
    }

    if let Some(r) = verbose.as_deref_mut() {
        r.line("\n=== NEGATIVE QUERIES (expect empty) ===");
    }
    for (case, qvec) in eval.negative.iter().zip(neg_vecs.iter()) {
        let results = db
            .search_hybrid_with(&case.query, qvec.as_deref(), 15, params)
            .expect("search");
        if results.is_empty() {
            summary.neg_empty += 1;
        }
        if let Some(r) = verbose.as_deref_mut() {
            r.line(format!(
                "[{}] {:60} -> {} results (top score {})",
                if results.is_empty() { "OK  " } else { "LEAK" },
                truncate(&case.query, 58),
                results.len(),
                results
                    .first()
                    .map(|r| format!("{:.3}", r.score))
                    .unwrap_or_else(|| "-".into())
            ));
        }
    }

    summary
}

fn dist(label: &str, scores: &[f32]) -> String {
    if scores.is_empty() {
        return format!("{label}: (none)");
    }
    let mut scores = scores.to_vec();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f32| scores[((scores.len() - 1) as f32 * q) as usize];
    format!(
        "{label}: n={:4} min={:.3} p25={:.3} p50={:.3} p75={:.3} max={:.3}",
        scores.len(),
        scores[0],
        p(0.25),
        p(0.5),
        p(0.75),
        scores[scores.len() - 1]
    )
}

fn truncate(s: &str, n: usize) -> String {
    match s.char_indices().nth(n) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_string(),
    }
}
