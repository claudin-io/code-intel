#[cfg(feature = "embeddings")]
use ort::{session::Session, value::Tensor};
use crate::media::{MediaKind, MediaNeeds};
use std::path::Path;
#[cfg(any(feature = "embeddings", feature = "embeddings-candle"))]
use tokenizers::Tokenizer;

/// Config for a single-vector bi-encoder embedding model. Swapping candidates
/// is changing `ACTIVE_MODEL` below to point at a different const — everything
/// else (download, load, encode) reads from this struct.
struct ModelConfig {
    /// HF repo id, e.g. "Xenova/bge-small-en-v1.5".
    repo: &'static str,
    /// (remote path relative to `resolve/main/`, local filename in cache_dir,
    /// sha256 hex, size in bytes). Quantized ONNX exports from Xenova live
    /// under an `onnx/` subfolder in the repo, but we flatten them into
    /// cache_dir for simplicity. Hash/size pin the exact artifacts the app
    /// was calibrated with — the fallback download verifies against them
    /// (keep in sync with scripts/fetch_embedding_model.py).
    files: &'static [(&'static str, &'static str, &'static str, u64)],
    /// Local filename (from `files`) of the ONNX model, used to check
    /// presence and to load the session.
    model_filename: &'static str,
    /// Local filename (from `files`) of the tokenizer.
    tokenizer_filename: &'static str,
    /// Prefix prepended to queries only (not documents) before encoding.
    /// bge-family models are trained with an instruction prefix for queries;
    /// MiniLM has none.
    query_prefix: &'static str,
    /// Short name used to namespace the on-disk cache dir per model, so
    /// switching ACTIVE_MODEL doesn't silently reuse a stale cache.
    cache_dirname: &'static str,
}

#[allow(dead_code)]
const BGE_SMALL: ModelConfig = ModelConfig {
    repo: "Xenova/bge-small-en-v1.5",
    files: &[
        (
            "onnx/model_quantized.onnx",
            "model_quantized.onnx",
            "6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4",
            34_014_426,
        ),
        (
            "tokenizer.json",
            "tokenizer.json",
            "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            711_396,
        ),
        (
            "config.json",
            "config.json",
            "fa73f90bf92c8cace1fbcb709626306f2bdbc9ea3e5b5f94b440df9b6aa56350",
            683,
        ),
    ],
    model_filename: "model_quantized.onnx",
    tokenizer_filename: "tokenizer.json",
    query_prefix: "Represent this sentence for searching relevant passages: ",
    cache_dirname: "bge-small-en-v1.5",
};

const MINILM_L6: ModelConfig = ModelConfig {
    repo: "Xenova/all-MiniLM-L6-v2",
    files: &[
        (
            "onnx/model_quantized.onnx",
            "model_quantized.onnx",
            "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1",
            22_972_370,
        ),
        (
            "tokenizer.json",
            "tokenizer.json",
            "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
            711_661,
        ),
        (
            "config.json",
            "config.json",
            "7135149f7cffa1a573466c6e4d8423ed73b62fd2332c575bf738a0d033f70df7",
            650,
        ),
    ],
    model_filename: "model_quantized.onnx",
    tokenizer_filename: "tokenizer.json",
    query_prefix: "",
    cache_dirname: "all-MiniLM-L6-v2",
};

/// Single-vector bi-encoder in active use. Change this to `BGE_SMALL` to try
/// the other candidate — nothing else in this file needs to change. MiniLM won
/// the calibration eval (see examples/semantic_eval.rs): better top-3 rank,
/// wider score spread, less noise on off-topic queries, faster indexing.
const ACTIVE_MODEL: ModelConfig = MINILM_L6;

/// Cache-dir name derived from the active model config, so callers can
/// namespace the on-disk cache per model (e.g. `models/{this}`).
pub fn model_cache_dirname() -> &'static str {
    ACTIVE_MODEL.cache_dirname
}

/// Local filename of the active model's ONNX file, so callers can check for
/// its presence without re-hardcoding the name.
pub fn model_filename() -> &'static str {
    ACTIVE_MODEL.model_filename
}

// Bounded low: attention memory grows with seq^2, and embedding texts are
// already capped to ~800 chars of body (see MAX_BODY_CHARS below), so a much
// larger window only inflates padding and peak memory without adding signal.
const MAX_LENGTH: usize = 512;

#[cfg(feature = "embeddings")]
pub struct MiniLmEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    output_name: String,
    /// Whether the loaded model's inputs include `token_type_ids`, detected
    /// at load time from `session.inputs()` rather than assumed.
    wants_token_type_ids: bool,
}

/// Pure-Rust embedding backend (candle). Selected with
/// `--no-default-features --features embeddings-candle`.
///
/// Exists because `ort`'s prebuilt ONNX Runtime contains x86-64-v3
/// instructions (BMI2 `SHLX`) that fault on pre-Haswell CPUs. candle is
/// compiled from source with baseline codegen, so it runs anywhere.
/// Produces the same 384-dim mean-pooled + L2-normalized vectors as the ORT
/// path, from the same all-MiniLM-L6-v2 weights (safetensors instead of ONNX).
#[cfg(all(feature = "embeddings-candle", not(feature = "embeddings")))]
pub struct MiniLmEmbedder {
    model: candle_transformers::models::bert::BertModel,
    tokenizer: Tokenizer,
    device: candle_core::Device,
}

#[cfg(all(feature = "embeddings-candle", not(feature = "embeddings")))]
impl MiniLmEmbedder {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        use candle_core::{DType, Device};
        use candle_nn::VarBuilder;
        use candle_transformers::models::bert::{BertModel, Config};

        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join(ACTIVE_MODEL.tokenizer_filename);
        let weights_path = model_dir.join("model.safetensors");

        for (p, what) in [
            (&config_path, "config.json"),
            (&tokenizer_path, "tokenizer"),
            (&weights_path, "model.safetensors"),
        ] {
            if !p.exists() {
                return Err(format!("{what} not found at {}", p.display()));
            }
        }

        let config_json =
            std::fs::read_to_string(&config_path).map_err(|e| format!("read config.json: {e}"))?;
        let config: Config =
            serde_json::from_str(&config_json).map_err(|e| format!("parse bert config: {e}"))?;

        let device = Device::Cpu;
        // SAFETY: mmap of a file we just verified exists; candle requires unsafe
        // here because the mapping is invalidated if the file changes underneath.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| format!("load safetensors: {e}"))?
        };
        let model = BertModel::load(vb, &config).map_err(|e| format!("build bert: {e}"))?;

        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| format!("tokenizer load: {e}"))?;

        Ok(MiniLmEmbedder {
            model,
            tokenizer,
            device,
        })
    }

    pub fn encode(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        use candle_core::Tensor as CTensor;

        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenization/padding is identical to the ORT path so both backends
        // see the same inputs.
        let encoding = self
            .tokenizer
            .encode_batch(
                texts
                    .iter()
                    .map(|t| tokenizers::EncodeInput::Single(t.to_string().into()))
                    .collect(),
                true,
            )
            .map_err(|e| format!("tokenize: {e}"))?;

        let batch_size = encoding.len();
        let mut padded_len = 0;
        for enc in &encoding {
            padded_len = padded_len.max(enc.len().min(MAX_LENGTH));
        }
        if padded_len == 0 {
            padded_len = 1;
        }

        let mut input_ids = vec![0u32; batch_size * padded_len];
        let mut attention_mask = vec![0i64; batch_size * padded_len];
        for (b, enc) in encoding.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let len = ids.len().min(MAX_LENGTH);
            for i in 0..len {
                input_ids[b * padded_len + i] = ids[i];
                attention_mask[b * padded_len + i] = mask[i] as i64;
            }
        }

        let ids_t = CTensor::from_vec(input_ids, (batch_size, padded_len), &self.device)
            .map_err(|e| format!("input_ids tensor: {e}"))?;
        let type_ids_t = ids_t
            .zeros_like()
            .map_err(|e| format!("token_type_ids tensor: {e}"))?;
        let mask_u8: Vec<u32> = attention_mask.iter().map(|v| *v as u32).collect();
        let mask_t = CTensor::from_vec(mask_u8, (batch_size, padded_len), &self.device)
            .map_err(|e| format!("attention_mask tensor: {e}"))?;

        // [batch, seq, hidden]
        let out = self
            .model
            .forward(&ids_t, &type_ids_t, Some(&mask_t))
            .map_err(|e| format!("bert forward: {e}"))?;
        let hidden = out.dim(2).map_err(|e| format!("output dim: {e}"))?;
        let flat: Vec<f32> = out
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| format!("extract output: {e}"))?;

        if hidden == 0 || flat.len() < batch_size * padded_len * hidden {
            return Err(format!(
                "unexpected output tensor: hidden {hidden}, len {}",
                flat.len()
            ));
        }

        // Same masked mean-pool + L2 normalize as the ORT path.
        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let mut sum = vec![0f32; hidden];
            let mut count: f32 = 0.0;
            for s in 0..padded_len {
                if attention_mask[b * padded_len + s] > 0 {
                    let offset = (b * padded_len + s) * hidden;
                    for d in 0..hidden {
                        sum[d] += flat[offset + d];
                    }
                    count += 1.0;
                }
            }
            if count > 0.0 {
                for v in sum.iter_mut() {
                    *v /= count;
                }
            }
            let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for v in sum.iter_mut() {
                *v /= norm;
            }
            results.push(sum);
        }

        Ok(results)
    }

    pub fn encode_query(&mut self, text: &str) -> Result<Vec<f32>, String> {
        let prefixed;
        let query = if ACTIVE_MODEL.query_prefix.is_empty() {
            text
        } else {
            prefixed = format!("{}{}", ACTIVE_MODEL.query_prefix, text);
            &prefixed
        };
        let mut vecs = self.encode(&[query])?;
        vecs.pop().ok_or("empty encode result".into())
    }
}

/// Stub used when the crate is built with no embedding backend at all.
/// Construction always fails, so callers fall back to their existing
/// "embedding model unavailable" path and the app runs without semantic search.
#[cfg(not(any(feature = "embeddings", feature = "embeddings-candle")))]
pub struct MiniLmEmbedder {
    _never: std::convert::Infallible,
}

#[cfg(not(any(feature = "embeddings", feature = "embeddings-candle")))]
impl MiniLmEmbedder {
    const DISABLED: &'static str =
        "semantic search disabled: built without the `embeddings` feature (no ONNX Runtime)";

    pub fn load(_model_dir: &Path) -> Result<Self, String> {
        Err(Self::DISABLED.to_string())
    }

    pub fn encode(&mut self, _texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        // Unreachable: `load` never yields a value, so no instance can exist.
        match self._never {}
    }

    pub fn encode_query(&mut self, _text: &str) -> Result<Vec<f32>, String> {
        match self._never {}
    }
}

#[cfg(feature = "embeddings")]
impl MiniLmEmbedder {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let model_path = model_dir.join(ACTIVE_MODEL.model_filename);
        let tokenizer_path = model_dir.join(ACTIVE_MODEL.tokenizer_filename);

        if !model_path.exists() {
            return Err(format!(
                "model not found at {}. Call ensure_model_downloaded first.",
                model_path.display()
            ));
        }
        if !tokenizer_path.exists() {
            return Err(format!(
                "tokenizer not found at {}. Call ensure_model_downloaded first.",
                tokenizer_path.display()
            ));
        }

        let session = Session::builder()
            .map_err(|e| format!("ort builder: {e}"))?
            // Memory pattern pre-allocates and retains buffers sized for the
            // largest batch ever seen, so peak memory never shrinks back down.
            // We batch inputs ourselves (see EMBED_BATCH_SIZE in indexer.rs),
            // so disable it and let the allocator release memory between runs.
            .with_memory_pattern(false)
            .map_err(|e| format!("ort memory pattern: {e}"))?
            // Cap ONNX threading: the default (one intra-op thread per
            // physical core) saturates the whole machine during indexing and
            // starves the WebView UI thread — Windows then flags the window
            // as "Not responding". Embedding is background work; keep it slow
            // and polite. `CODE_INTEL_THREADS` raises the cap where there is
            // no UI to starve.
            .with_intra_threads(intra_threads())
            .map_err(|e| format!("ort intra threads: {e}"))?
            .with_inter_threads(1)
            .map_err(|e| format!("ort inter threads: {e}"))?
            .commit_from_file(&model_path)
            .map_err(|e| format!("ort load model: {e}"))?;

        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| format!("tokenizer load: {e}"))?;

        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or("model has no outputs")?;

        // Some BERT-family ONNX exports require a token_type_ids input in
        // addition to input_ids/attention_mask; others don't. Detect it from
        // the model itself rather than assuming either way.
        let wants_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");

        Ok(MiniLmEmbedder {
            session,
            tokenizer,
            output_name,
            wants_token_type_ids,
        })
    }

    pub fn encode(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encoding = self
            .tokenizer
            .encode_batch(
                texts
                    .iter()
                    .map(|t| tokenizers::EncodeInput::Single(t.to_string().into()))
                    .collect(),
                true,
            )
            .map_err(|e| format!("tokenize: {e}"))?;

        let batch_size = encoding.len();
        let mut padded_len = 0;
        for enc in &encoding {
            padded_len = padded_len.max(enc.len().min(MAX_LENGTH));
        }
        if padded_len == 0 {
            padded_len = 1;
        }

        let mut input_ids = vec![0i64; batch_size * padded_len];
        let mut attention_mask = vec![0i64; batch_size * padded_len];

        for (b, enc) in encoding.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let len = ids.len().min(MAX_LENGTH);
            for i in 0..len {
                input_ids[b * padded_len + i] = ids[i] as i64;
                attention_mask[b * padded_len + i] = mask[i] as i64;
            }
        }

        let ids_tensor = Tensor::from_array((
            vec![batch_size as i64, padded_len as i64],
            input_ids.clone(),
        ))
        .map_err(|e| format!("input_ids tensor: {e}"))?;
        let mask_tensor = Tensor::from_array((
            vec![batch_size as i64, padded_len as i64],
            attention_mask.clone(),
        ))
        .map_err(|e| format!("attention_mask tensor: {e}"))?;

        let mut inputs_map: std::collections::HashMap<String, ort::value::DynValue> =
            std::collections::HashMap::new();
        inputs_map.insert("input_ids".to_string(), ids_tensor.into());
        inputs_map.insert("attention_mask".to_string(), mask_tensor.into());

        if self.wants_token_type_ids {
            let token_type_ids = vec![0i64; batch_size * padded_len];
            let type_tensor =
                Tensor::from_array((vec![batch_size as i64, padded_len as i64], token_type_ids))
                    .map_err(|e| format!("token_type_ids tensor: {e}"))?;
            inputs_map.insert("token_type_ids".to_string(), type_tensor.into());
        }

        let ort_outs = self
            .session
            .run(inputs_map)
            .map_err(|e| format!("ort run: {e}"))?;

        let (shape, flat) = ort_outs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract output: {e}"))?;

        // Hidden size comes from the model output, not a constant: [batch, seq, hidden].
        let hidden = shape
            .last()
            .map(|d| *d as usize)
            .filter(|d| *d > 0)
            .unwrap_or_else(|| flat.len() / (batch_size * padded_len));
        if hidden == 0 || flat.len() < batch_size * padded_len * hidden {
            return Err(format!(
                "unexpected output tensor: shape {shape:?}, len {}",
                flat.len()
            ));
        }

        // Bi-encoder: mean-pool token embeddings into a single vector per
        // text, then L2-normalize so cosine similarity is a plain dot product.
        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let mut sum = vec![0f32; hidden];
            let mut count: f32 = 0.0;
            for s in 0..padded_len {
                if attention_mask[b * padded_len + s] > 0 {
                    let offset = (b * padded_len + s) * hidden;
                    for d in 0..hidden {
                        sum[d] += flat[offset + d];
                    }
                    count += 1.0;
                }
            }
            if count > 0.0 {
                for v in sum.iter_mut() {
                    *v /= count;
                }
            }
            let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for v in sum.iter_mut() {
                *v /= norm;
            }
            results.push(sum);
        }

        Ok(results)
    }

    pub fn encode_query(&mut self, text: &str) -> Result<Vec<f32>, String> {
        let prefixed;
        let query = if ACTIVE_MODEL.query_prefix.is_empty() {
            text
        } else {
            prefixed = format!("{}{}", ACTIVE_MODEL.query_prefix, text);
            &prefixed
        };
        let mut vecs = self.encode(&[query])?;
        vecs.pop().ok_or("empty encode result".into())
    }
}

/// Max chars of symbol body included in the embedding text. Bounds tokenizer
/// and inference cost — beyond this, extra body adds latency, not signal.
const MAX_BODY_CHARS: usize = 800;

pub fn build_embedding_text(
    kind: &str,
    name: &str,
    parent_context: Option<&str>,
    doc: Option<&str>,
    body: Option<&str>,
) -> String {
    let mut parts = vec![format!("{kind}: {name}")];
    if let Some(ctx) = parent_context {
        let trimmed = ctx.trim();
        if !trimmed.is_empty() {
            parts.push(format!("context: {trimmed}"));
        }
    }
    if let Some(d) = doc {
        let trimmed = d.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }
    if let Some(b) = body {
        let trimmed = b.trim();
        if !trimmed.is_empty() {
            let cut = trimmed
                .char_indices()
                .nth(MAX_BODY_CHARS)
                .map(|(i, _)| i)
                .unwrap_or(trimmed.len());
            parts.push(trimmed[..cut].to_string());
        }
    }
    parts.join(" | ")
}

/// Lines of overlap between consecutive chunks so a match spanning a chunk
/// boundary still lands in at least one chunk.
const CHUNK_OVERLAP_LINES: usize = 2;

/// One embeddable slice of a symbol. Large symbol bodies (big components,
/// long functions) are split into several chunks so content deep inside them
/// is searchable — a single mean-pooled vector of the first MAX_BODY_CHARS
/// can't represent a 300-line component. The `fts_*` fields mirror the same
/// slice into the chunk_fts BM25 index so both retrieval legs cover
/// byte-identical units.
#[derive(Debug, Clone)]
pub struct EmbedChunk {
    pub chunk_index: i64,
    /// Absolute 1-based file lines covered by this chunk's body slice.
    pub start_line: i64,
    pub end_line: i64,
    pub text: String,
    /// Symbol name plus its split words ("buildChunks build chunks").
    pub fts_name: String,
    /// Basename/stem/dir words of the owning file.
    pub fts_path: String,
    /// Doc comment + body slice + resolved i18n copy + split camel words.
    pub fts_body: String,
}

/// Max chars of resolved i18n copy appended to one chunk's embedding text.
const MAX_I18N_CHARS: usize = 400;

/// Resolve i18n keys referenced in a code slice into their user-visible copy.
/// Framework-agnostic: instead of parsing call syntax (`t(...)`,
/// `NSLocalizedString(...)`, `I18n.t(...)`, `$t(...)`, ...), every quoted
/// string literal in the slice is checked for exact membership in the dict —
/// translation keys are distinctive enough that membership is the filter.
fn resolve_i18n_keys(slice: &str, dict: &std::collections::HashMap<String, String>) -> String {
    let bytes = slice.as_bytes();
    let mut out: Vec<&str> = Vec::new();
    let mut total = 0usize;
    let mut i = 0usize;
    while i < bytes.len() && total < MAX_I18N_CHARS {
        let b = bytes[i];
        if (b == b'"' || b == b'\'')
            && let Some(end) = slice[i + 1..].find(b as char)
        {
            let literal = &slice[i + 1..i + 1 + end];
            if !literal.is_empty()
                && literal.len() <= 128
                && literal
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || ".-_".contains(c))
                && let Some(value) = dict.get(literal)
                && !out.contains(&value.as_str())
            {
                total += value.len() + 1;
                out.push(value);
            }
            i += 1 + end + 1;
            continue;
        }
        i += 1;
    }
    out.join(" ")
}

/// Path-derived terms for the chunk FTS index: basename, stem, split words of
/// the stem and of the last three parent directories — so "ToolBody.tsx list
/// rendering" or "tool renderers" match on path evidence alone.
fn build_fts_path_terms(file_path: &str, basename: &str, stem: &str) -> String {
    let mut terms: Vec<String> = Vec::new();
    for t in [basename, stem] {
        if !t.is_empty() && !terms.iter().any(|x| x == t) {
            terms.push(t.to_string());
        }
    }
    for w in crate::text::split_identifier_words(stem) {
        if !terms.contains(&w) {
            terms.push(w);
        }
    }
    let dirs: Vec<String> = Path::new(file_path)
        .parent()
        .map(|p| {
            p.components()
                .filter_map(|c| match c {
                    std::path::Component::Normal(os) => Some(os.to_string_lossy().to_string()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    for dir in dirs.iter().rev().take(3) {
        for w in crate::text::split_identifier_words(dir) {
            if !terms.contains(&w) {
                terms.push(w);
            }
        }
    }
    terms.join(" ")
}

/// Split a symbol into embedding chunks. Line-based and language-agnostic:
/// works identically for any tree-sitter grammar (and doc sections), since it
/// only sees the extracted body text. Every chunk repeats the symbol header
/// (kind/name/context/doc) so it stays anchored to its symbol. When an i18n
/// dict is given, copy referenced via `t("key")` inside a chunk is resolved
/// and appended to that chunk's text, so user-visible wording is searchable.
/// Each chunk also carries `fts_*` mirrors of the same content for the BM25
/// leg — see EmbedChunk.
#[allow(clippy::too_many_arguments)]
pub fn build_embedding_chunks(
    kind: &str,
    name: &str,
    file_path: &str,
    parent_context: Option<&str>,
    doc: Option<&str>,
    body: Option<&str>,
    symbol_start_line: i64,
    symbol_end_line: i64,
    i18n: Option<&std::collections::HashMap<String, String>>,
) -> Vec<EmbedChunk> {
    let path = Path::new(file_path);
    let basename = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let name_words = crate::text::identifier_word_string(name);

    // The split words go into the embed header too: MiniLM's WordPiece
    // subwords camelCase poorly, and NL queries say "build embedding chunks",
    // not "buildEmbeddingChunks".
    let mut header_parts = vec![match &name_words {
        Some(words) => format!("{kind}: {name} ({words})"),
        None => format!("{kind}: {name}"),
    }];
    if !stem.is_empty() {
        header_parts.push(format!("file: {stem}"));
    }
    if let Some(ctx) = parent_context {
        let trimmed = ctx.trim();
        if !trimmed.is_empty() {
            header_parts.push(format!("context: {trimmed}"));
        }
    }
    let doc_trimmed = doc.map(str::trim).filter(|d| !d.is_empty());
    if let Some(d) = doc_trimmed {
        header_parts.push(d.to_string());
    }
    let header = header_parts.join(" | ");

    let fts_name = match &name_words {
        Some(words) => format!("{name} {words}"),
        None => name.to_string(),
    };
    let fts_path = build_fts_path_terms(file_path, &basename, &stem);
    let fts_body_for = |slice: &str, i18n_copy: &str| -> String {
        let mut out = String::new();
        if let Some(d) = doc_trimmed {
            out.push_str(d);
        }
        if !slice.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(slice);
        }
        if !i18n_copy.is_empty() {
            out.push('\n');
            out.push_str(i18n_copy);
        }
        let split = crate::text::body_split_words(
            slice,
            crate::text::FTS_BODY_SPLIT_CAP_CHARS,
        );
        if !split.is_empty() {
            out.push('\n');
            out.push_str(&split);
        }
        out
    };

    let body = body.map(str::trim_end).unwrap_or("");
    if body.trim().is_empty() {
        return vec![EmbedChunk {
            chunk_index: 0,
            start_line: symbol_start_line,
            end_line: symbol_end_line,
            text: header,
            fts_name,
            fts_path,
            fts_body: fts_body_for("", ""),
        }];
    }

    let lines: Vec<&str> = body.lines().collect();
    // The body is the tail of the symbol's line range (for code it's the whole
    // range; for doc sections it starts after the heading), so anchor absolute
    // line numbers from the end. This holds for every language uniformly.
    let body_first_line = (symbol_end_line - lines.len() as i64 + 1).max(symbol_start_line);

    let mut chunks: Vec<EmbedChunk> = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let mut chars = 0usize;
        let mut j = i;
        while j < lines.len() {
            let line_len = lines[j].chars().count() + 1;
            if chars + line_len > MAX_BODY_CHARS && j > i {
                break;
            }
            chars += line_len;
            j += 1;
        }
        let slice = lines[i..j].join("\n");
        let slice = slice.trim();
        if !slice.is_empty() {
            let i18n_copy = match i18n {
                Some(dict) => resolve_i18n_keys(slice, dict),
                None => String::new(),
            };
            let mut text = format!("{header} | {slice}");
            if !i18n_copy.is_empty() {
                text.push_str(" | i18n: ");
                text.push_str(&i18n_copy);
            }
            chunks.push(EmbedChunk {
                chunk_index: chunks.len() as i64,
                start_line: body_first_line + i as i64,
                end_line: body_first_line + j as i64 - 1,
                text,
                fts_name: fts_name.clone(),
                fts_path: fts_path.clone(),
                fts_body: fts_body_for(slice, &i18n_copy),
            });
        }
        if j >= lines.len() {
            break;
        }
        i = j.saturating_sub(CHUNK_OVERLAP_LINES).max(i + 1);
    }

    if chunks.is_empty() {
        chunks.push(EmbedChunk {
            chunk_index: 0,
            start_line: symbol_start_line,
            end_line: symbol_end_line,
            text: header,
            fts_name,
            fts_path,
            fts_body: fts_body_for("", ""),
        });
    }
    chunks
}

const DOWNLOAD_RETRIES: usize = 3;

/// Runtime fallback download — the normal path is the model bundled as a
/// Tauri resource (fetched at build time by scripts/fetch_embedding_model.py).
/// Streams to a `.part` file with an incremental sha256, verifies size+hash
/// against the pinned values, then atomically renames into place, with
/// retries — a corrupt or truncated download can never become the cache.
/// Weights for the candle backend. Xenova's ONNX-conversion repo does not
/// host safetensors, so these come from the upstream sentence-transformers
/// repo, pinned to a commit like everything else in this table (same pins as
/// Claudinio Code's `scripts/fetch_embedding_model.py --with-candle`).
#[cfg(all(feature = "embeddings-candle", not(feature = "embeddings")))]
const CANDLE_WEIGHTS: (&str, &str, &str, u64) = (
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/model.safetensors",
    "model.safetensors",
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    90_868_376,
);

/// The file whose presence means "model is in the cache" for the backend
/// this binary was built with.
pub fn model_marker_filename() -> &'static str {
    #[cfg(all(feature = "embeddings-candle", not(feature = "embeddings")))]
    {
        CANDLE_WEIGHTS.1
    }
    #[cfg(not(all(feature = "embeddings-candle", not(feature = "embeddings"))))]
    {
        ACTIVE_MODEL.model_filename
    }
}

/// `(url, local filename, sha256, size)` of every file the active backend
/// loads. The ORT build never fetches the 87 MB safetensors; the candle build
/// never fetches the ONNX graph.
pub fn required_model_files() -> Vec<(String, &'static str, &'static str, u64)> {
    let base_url = format!("https://huggingface.co/{}/resolve/main", ACTIVE_MODEL.repo);
    let out: Vec<(String, &'static str, &'static str, u64)> = ACTIVE_MODEL
        .files
        .iter()
        .filter(|(_, local, _, _)| {
            let is_graph = *local == ACTIVE_MODEL.model_filename;
            !is_graph || cfg!(not(all(feature = "embeddings-candle", not(feature = "embeddings"))))
        })
        .map(|(remote, local, sha, len)| (format!("{base_url}/{remote}"), *local, *sha, *len))
        .collect();
    #[cfg(all(feature = "embeddings-candle", not(feature = "embeddings")))]
    {
        let mut out = out;
        out.push((
            CANDLE_WEIGHTS.0.to_string(),
            CANDLE_WEIGHTS.1,
            CANDLE_WEIGHTS.2,
            CANDLE_WEIGHTS.3,
        ));
        return out;
    }
    #[cfg(not(all(feature = "embeddings-candle", not(feature = "embeddings"))))]
    out
}

pub async fn ensure_model_downloaded(cache_dir: &Path) -> Result<(), String> {
    if cache_dir.join(model_marker_filename()).exists() {
        return Ok(());
    }

    // A repeated hit here means the cache path is not persisting between runs
    // (suspected on some Windows setups) — every index would re-download the
    // model from HuggingFace.
    eprintln!(
        "[embeddings] model not in cache, downloading {} to {}",
        ACTIVE_MODEL.repo,
        cache_dir.display()
    );

    std::fs::create_dir_all(cache_dir).map_err(|e| format!("create model dir: {e}"))?;

    for (url, local_filename, sha256_hex, expected_len) in required_model_files() {
        let dest = cache_dir.join(local_filename);
        if dest.exists() {
            continue;
        }

        crate::download::download_verified_with_retries(
            &url,
            &dest,
            local_filename,
            sha256_hex,
            expected_len,
            DOWNLOAD_RETRIES,
        )
        .await
        .map_err(|e| {
            format!(
                "{e}. If the error is a hash mismatch, the model files changed upstream — \
                 update the pinned hashes in embeddings.rs and fetch_embedding_model.py."
            )
        })?;
    }

    Ok(())
}

// ── the embedder the rest of the crate talks to ─────────────────────────

/// Threads one model run may use. Two by default — indexing is a background
/// job, and in Claudinio Code it shares the machine with a UI. A headless
/// server on a many-core machine can afford more: `CODE_INTEL_THREADS`.
pub const DEFAULT_INTRA_THREADS: usize = 2;

pub fn intra_threads() -> usize {
    parse_threads(std::env::var("CODE_INTEL_THREADS").ok().as_deref())
}

fn parse_threads(value: Option<&str>) -> usize {
    value
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| (1..=64).contains(n))
        .unwrap_or(DEFAULT_INTRA_THREADS)
}

/// How MiniLM vectors are labelled in an index (see
/// `IndexDb::reconcile_embedding_model`).
pub const MINILM_MODEL_ID: &str = "all-MiniLM-L6-v2";

/// Size of a MiniLM vector (its hidden size; the model has no projection).
pub const MINILM_DIM: usize = 384;

enum Engine {
    MiniLm(Box<MiniLmEmbedder>),
    #[cfg(feature = "embeddings")]
    Gemma2(Box<crate::gemma2::Gemma2Embedder>),
}

/// Whichever embedding model this process ended up with.
///
/// Every build has MiniLM: text only, 384 dimensions, 23 MB. The ONNX Runtime
/// build prefers EmbeddingGemma 2, which also places images and audio in the
/// same space as text, and keeps MiniLM as the fallback for when that model
/// cannot be downloaded or loaded. Callers do not branch on which one they
/// got: the index records `model_id`, and media calls simply report that the
/// model at hand cannot do them.
pub struct CodeEmbedder {
    engine: Engine,
}

const NO_MEDIA: &str = "the loaded embedding model has no encoder for this kind of file";

impl CodeEmbedder {
    /// MiniLM from `model_dir` — what this function has always loaded.
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        Ok(CodeEmbedder {
            engine: Engine::MiniLm(Box::new(MiniLmEmbedder::load(model_dir)?)),
        })
    }

    /// EmbeddingGemma 2's text model from `model_dir`.
    #[cfg(feature = "embeddings")]
    pub fn load_gemma2(model_dir: &Path) -> Result<Self, String> {
        Ok(CodeEmbedder {
            engine: Engine::Gemma2(Box::new(crate::gemma2::Gemma2Embedder::load(model_dir)?)),
        })
    }

    /// Identifies the vectors this embedder produces. Vectors of different
    /// ids are not comparable and never share an index.
    pub fn model_id(&self) -> &'static str {
        match &self.engine {
            Engine::MiniLm(_) => MINILM_MODEL_ID,
            #[cfg(feature = "embeddings")]
            Engine::Gemma2(_) => crate::gemma2::MODEL_ID,
        }
    }

    /// Length of every vector this embedder produces.
    pub fn embedding_dim(&self) -> usize {
        match &self.engine {
            Engine::MiniLm(_) => MINILM_DIM,
            #[cfg(feature = "embeddings")]
            Engine::Gemma2(_) => crate::gemma2::EMBED_DIM,
        }
    }

    /// Embed index texts (code chunks, doc sections).
    pub fn encode(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        match &mut self.engine {
            Engine::MiniLm(m) => m.encode(texts),
            #[cfg(feature = "embeddings")]
            Engine::Gemma2(g) => g.encode_documents(texts),
        }
    }

    /// Embed a search query, to compare against `encode` vectors.
    pub fn encode_query(&mut self, text: &str) -> Result<Vec<f32>, String> {
        match &mut self.engine {
            Engine::MiniLm(m) => m.encode_query(text),
            #[cfg(feature = "embeddings")]
            Engine::Gemma2(g) => g.encode_query(text, crate::gemma2::QueryTask::Code),
        }
    }

    /// Embed a search query to compare against image and audio vectors.
    /// `None` when the model has no shared text/media space at all.
    pub fn encode_media_query(&mut self, text: &str) -> Result<Option<Vec<f32>>, String> {
        match &mut self.engine {
            Engine::MiniLm(_) => {
                let _ = text;
                Ok(None)
            }
            #[cfg(feature = "embeddings")]
            Engine::Gemma2(g) => g.encode_query(text, crate::gemma2::QueryTask::Media).map(Some),
        }
    }

    /// The kinds of media this embedder can turn into vectors right now.
    pub fn media_support(&self) -> MediaNeeds {
        match &self.engine {
            Engine::MiniLm(_) => MediaNeeds::NONE,
            #[cfg(feature = "embeddings")]
            Engine::Gemma2(g) => g.media_support(),
        }
    }

    /// Embed one image or audio file. Holds the embedder for the whole run —
    /// right for the watcher's one file at a time; bulk indexing uses the
    /// free function `embed_media_file`, which lets searches in between.
    pub fn embed_media_file(&mut self, path: &Path, kind: MediaKind) -> Result<Vec<f32>, String> {
        match &mut self.engine {
            Engine::MiniLm(_) => {
                let _ = (path, kind);
                Err(NO_MEDIA.into())
            }
            #[cfg(feature = "embeddings")]
            Engine::Gemma2(g) => {
                if !g.media_support().has(kind) {
                    return Err(NO_MEDIA.into());
                }
                let soft = crate::gemma2::encode_media_file(&g.encoders(), path, kind)?;
                g.embed_soft_tokens(kind, &soft)
            }
        }
    }
}

pub type SharedEmbedder = std::sync::Arc<std::sync::Mutex<CodeEmbedder>>;

/// MiniLM from `cache_dir`, shared.
pub fn load_shared(cache_dir: &Path) -> Result<SharedEmbedder, String> {
    let embedder = CodeEmbedder::load(cache_dir)?;
    Ok(std::sync::Arc::new(std::sync::Mutex::new(embedder)))
}

/// Embed one media file with a shared embedder. Decoding and the encoder run
/// — seconds for an image — happen outside the embedder's lock, which is
/// taken only for the short text-model pass at the end, so a search issued
/// meanwhile is not held up.
pub fn embed_media_file(shared: &SharedEmbedder, path: &Path, kind: MediaKind) -> Result<Vec<f32>, String> {
    #[cfg(feature = "embeddings")]
    {
        let encoders = {
            let guard = shared.lock().map_err(|e| format!("embedder lock poisoned: {e}"))?;
            match &guard.engine {
                Engine::Gemma2(g) if g.media_support().has(kind) => g.encoders(),
                _ => return Err(NO_MEDIA.into()),
            }
        };
        let soft = crate::gemma2::encode_media_file(&encoders, path, kind)?;
        let mut guard = shared.lock().map_err(|e| format!("embedder lock poisoned: {e}"))?;
        match &mut guard.engine {
            Engine::Gemma2(g) => g.embed_soft_tokens(kind, &soft),
            Engine::MiniLm(_) => Err(NO_MEDIA.into()),
        }
    }
    #[cfg(not(feature = "embeddings"))]
    {
        let _ = (shared, path, kind);
        Err(NO_MEDIA.into())
    }
}

/// Which model to use. `CODE_INTEL_MODEL` sets it; the default is `Auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelChoice {
    /// EmbeddingGemma 2 where this build can run it, MiniLM otherwise — and
    /// MiniLM whenever the larger model fails to download or load.
    Auto,
    /// MiniLM only: the behaviour before EmbeddingGemma 2 existed.
    MiniLm,
    /// EmbeddingGemma 2 or nothing. No fallback, so a comparison between the
    /// two models can never quietly measure MiniLM twice.
    Gemma2,
}

impl ModelChoice {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Some(ModelChoice::Auto),
            "minilm" | "all-minilm-l6-v2" => Some(ModelChoice::MiniLm),
            "embeddinggemma2" | "embeddinggemma-2" | "gemma2" => Some(ModelChoice::Gemma2),
            _ => None,
        }
    }

    pub fn from_env() -> Self {
        match std::env::var("CODE_INTEL_MODEL") {
            Ok(v) => Self::parse(&v).unwrap_or_else(|| {
                eprintln!("[embeddings] unknown CODE_INTEL_MODEL={v:?}; using auto (auto | minilm | embeddinggemma2)");
                ModelChoice::Auto
            }),
            Err(_) => ModelChoice::Auto,
        }
    }
}

/// Whether this binary can run EmbeddingGemma 2 at all.
pub const fn gemma2_supported() -> bool {
    cfg!(feature = "embeddings")
}

/// Directory of the MiniLM files under a models root (`<cache>/models`).
pub fn minilm_dir(models_root: &Path) -> std::path::PathBuf {
    models_root.join(model_cache_dirname())
}

async fn load_minilm(models_root: &Path) -> Result<SharedEmbedder, String> {
    let dir = minilm_dir(models_root);
    ensure_model_downloaded(&dir).await?;
    let load_dir = dir.clone();
    match tokio::task::spawn_blocking(move || load_shared(&load_dir)).await {
        Ok(Ok(shared)) => Ok(shared),
        Ok(Err(e)) => {
            // Self-heal a corrupt download: the next run fetches it again
            // instead of failing the same way forever.
            let _ = std::fs::remove_dir_all(&dir);
            Err(e)
        }
        Err(e) => Err(format!("embedding model load panicked: {e}")),
    }
}

#[cfg(feature = "embeddings")]
async fn load_gemma2(models_root: &Path, needs: MediaNeeds) -> Result<SharedEmbedder, String> {
    let dir = models_root.join(crate::gemma2::CACHE_DIRNAME);
    crate::gemma2::ensure_component_downloaded(&dir, None).await?;
    let load_dir = dir.clone();
    // Not removed on failure, unlike MiniLM: these files were hash-checked on
    // the way in, so a load error means this runtime cannot run the graph —
    // and re-downloading 175 MB on every start would not change that.
    let embedder = tokio::task::spawn_blocking(move || CodeEmbedder::load_gemma2(&load_dir))
        .await
        .map_err(|e| format!("EmbeddingGemma 2 load panicked: {e}"))??;
    let shared = std::sync::Arc::new(std::sync::Mutex::new(embedder));
    extend_media(&shared, needs).await;
    Ok(shared)
}

/// A model family, in the order `ensure_and_load` tries them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Gemma2,
    MiniLm,
}

/// Which models to try, in order, for a choice and a build. This is the whole
/// fallback policy: the first that downloads and loads wins.
pub fn load_plan(choice: ModelChoice, gemma2_supported: bool) -> &'static [ModelFamily] {
    match (choice, gemma2_supported) {
        (ModelChoice::Auto, true) => &[ModelFamily::Gemma2, ModelFamily::MiniLm],
        (ModelChoice::Auto, false) | (ModelChoice::MiniLm, _) => &[ModelFamily::MiniLm],
        (ModelChoice::Gemma2, true) => &[ModelFamily::Gemma2],
        (ModelChoice::Gemma2, false) => &[],
    }
}

/// Download (as needed) and load the embedder for a workspace with the given
/// media. The text model is fetched in every case; a media encoder only when
/// `needs` says the workspace holds that kind of file.
pub async fn ensure_and_load(
    models_root: &Path,
    choice: ModelChoice,
    needs: MediaNeeds,
) -> Result<SharedEmbedder, String> {
    let plan = load_plan(choice, gemma2_supported());
    let mut last_error =
        String::from("EmbeddingGemma 2 needs the ONNX Runtime build; this binary has MiniLM only");
    for (i, family) in plan.iter().enumerate() {
        let attempt = match family {
            #[cfg(feature = "embeddings")]
            ModelFamily::Gemma2 => load_gemma2(models_root, needs).await,
            #[cfg(not(feature = "embeddings"))]
            ModelFamily::Gemma2 => Err(last_error.clone()),
            ModelFamily::MiniLm => load_minilm(models_root).await,
        };
        match attempt {
            Ok(shared) => return Ok(shared),
            Err(e) => {
                if let Some(next) = plan.get(i + 1) {
                    eprintln!("[embeddings] {family:?} unavailable, falling back to {next:?}: {e}");
                }
                last_error = e;
            }
        }
    }
    let _ = needs;
    Err(last_error)
}

/// Give an already-loaded embedder the media encoders `needs` asks for and it
/// lacks — a second workspace with images joining a process that started on a
/// text-only one. Best-effort: a failed encoder leaves that kind of media
/// findable by file name only, and everything else as it was.
pub async fn extend_media(shared: &SharedEmbedder, needs: MediaNeeds) {
    #[cfg(feature = "embeddings")]
    for kind in MediaKind::ALL {
        if !needs.has(kind) {
            continue;
        }
        // Cheap checks under the embedder's lock; the download and the load
        // run without it.
        let slot = {
            let Ok(guard) = shared.lock() else { return };
            match &guard.engine {
                Engine::Gemma2(g) if !g.media_support().has(kind) => {
                    g.encoder_slot(kind).map(|slot| (slot, g.dir().to_path_buf()))
                }
                _ => continue,
            }
        };
        let (encoders, dir) = match slot {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[embeddings] {} search unavailable: {e}", kind.as_str());
                continue;
            }
        };
        if let Err(e) = crate::gemma2::ensure_component_downloaded(&dir, Some(kind)).await {
            eprintln!("[embeddings] {} encoder download failed: {e}", kind.as_str());
            continue;
        }
        let loaded = tokio::task::spawn_blocking(move || {
            encoders
                .lock()
                .map_err(|e| format!("encoders lock poisoned: {e}"))?
                .load(&dir, kind)
        })
        .await;
        match loaded {
            Ok(Ok(())) => eprintln!("[embeddings] {} encoder ready", kind.as_str()),
            Ok(Err(e)) => eprintln!("[embeddings] {} encoder failed to load: {e}", kind.as_str()),
            Err(e) => eprintln!("[embeddings] {} encoder load panicked: {e}", kind.as_str()),
        }
    }
    #[cfg(not(feature = "embeddings"))]
    {
        let _ = (shared, needs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fallback policy, for every choice in both kinds of build. The
    /// default must reach MiniLM whenever the larger model is not an option,
    /// and an explicit choice must never be swapped for the other model.
    #[test]
    fn load_plan_falls_back_to_minilm_only_under_auto() {
        use ModelFamily::*;
        assert_eq!(load_plan(ModelChoice::Auto, true), &[Gemma2, MiniLm]);
        assert_eq!(load_plan(ModelChoice::Auto, false), &[MiniLm]);
        assert_eq!(load_plan(ModelChoice::MiniLm, true), &[MiniLm]);
        assert_eq!(load_plan(ModelChoice::MiniLm, false), &[MiniLm]);
        assert_eq!(load_plan(ModelChoice::Gemma2, true), &[Gemma2]);
        assert!(load_plan(ModelChoice::Gemma2, false).is_empty());
        // This build's own answer: the candle and no-backend builds must not
        // even try the ONNX-only model.
        let here = load_plan(ModelChoice::Auto, gemma2_supported());
        assert_eq!(here.contains(&Gemma2), cfg!(feature = "embeddings"));
        assert_eq!(here.last(), Some(&MiniLm));
    }

    #[test]
    fn thread_cap_defaults_to_two_and_ignores_nonsense() {
        assert_eq!(parse_threads(None), 2);
        assert_eq!(parse_threads(Some("8")), 8);
        assert_eq!(parse_threads(Some(" 4 ")), 4);
        for bad in ["0", "-1", "many", "", "1000"] {
            assert_eq!(parse_threads(Some(bad)), 2, "{bad:?}");
        }
    }

    #[test]
    fn model_choice_parses_the_documented_values() {
        assert_eq!(ModelChoice::parse("auto"), Some(ModelChoice::Auto));
        assert_eq!(ModelChoice::parse(""), Some(ModelChoice::Auto));
        assert_eq!(ModelChoice::parse(" MiniLM "), Some(ModelChoice::MiniLm));
        assert_eq!(ModelChoice::parse("embeddinggemma2"), Some(ModelChoice::Gemma2));
        assert_eq!(ModelChoice::parse("EmbeddingGemma-2"), Some(ModelChoice::Gemma2));
        assert_eq!(ModelChoice::parse("bge"), None);
    }

    /// Each backend must fetch exactly the files it loads: the ORT build the
    /// ONNX graph, the candle build the safetensors weights — and both the
    /// tokenizer and config. The marker file is the one `load` needs.
    #[test]
    fn required_files_match_the_compiled_backend() {
        let files = required_model_files();
        let names: Vec<&str> = files.iter().map(|f| f.1).collect();
        assert!(names.contains(&"tokenizer.json"));
        assert!(names.contains(&"config.json"));
        assert!(names.contains(&model_marker_filename()));
        let candle = cfg!(all(feature = "embeddings-candle", not(feature = "embeddings")));
        assert_eq!(names.contains(&"model.safetensors"), candle);
        assert_eq!(names.contains(&"model_quantized.onnx"), !candle);
        for f in &files {
            assert!(f.0.starts_with("https://huggingface.co/"), "{}", f.0);
            assert_eq!(f.2.len(), 64, "sha256 hex for {}", f.1);
        }
    }

    /// Proves the candle backend produces *semantically useful* vectors, not
    /// just correctly-shaped ones: related code must score higher against a
    /// query than unrelated code, and vectors must be unit-length 384-dim.
    #[test]
    #[cfg(all(feature = "embeddings-candle", not(feature = "embeddings")))]
    fn candle_embeddings_are_normalized_and_semantically_ordered() {
        let dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("models/{}", model_cache_dirname()));
        if !dir.join("model.safetensors").exists() {
            eprintln!("safetensors weights not present, skipping");
            return;
        }
        let mut e = MiniLmEmbedder::load(&dir).expect("load candle model");

        let docs = [
            "fn refresh_token_if_stale(session: &mut Session) { /* renew expired auth */ }",
            "fn quicksort<T: Ord>(v: &mut [T]) { /* sort slice in place */ }",
        ];
        let vecs = e.encode(&docs).expect("encode docs");
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0].len(), 384, "MiniLM hidden size");

        for v in &vecs {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "not L2-normalized: {norm}");
        }

        // Query about auth/session expiry must rank the auth snippet above the
        // sorting one. This is the property that makes semantic search work at
        // all -- it fails loudly if the weights or pooling are wrong.
        let q = e
            .encode_query("where do we handle expired login sessions")
            .expect("encode query");
        let cos = |a: &Vec<f32>| -> f32 { a.iter().zip(q.iter()).map(|(x, y)| x * y).sum() };
        let auth_score = cos(&vecs[0]);
        let sort_score = cos(&vecs[1]);
        eprintln!("auth={auth_score:.4} sort={sort_score:.4}");
        assert!(
            auth_score > sort_score,
            "semantic ordering wrong: auth {auth_score} should beat sort {sort_score}"
        );
    }

    #[test]
    #[cfg(feature = "embeddings")]
    fn encode_produces_normalized_model_dim_vectors() {
        let dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("models/{}", model_cache_dirname()));
        if !dir.join(ACTIVE_MODEL.model_filename).exists() {
            eprintln!("model not present, skipping");
            return;
        }
        let mut e = MiniLmEmbedder::load(&dir).expect("load model");
        for o in e.session.outputs() {
            eprintln!("model output: {}", o.name());
        }
        for i in e.session.inputs() {
            eprintln!("model input: {}", i.name());
        }
        let vecs = e
            .encode(&[
                "fn hello_world() {}",
                "struct FileWatcher that reindexes files",
            ])
            .expect("encode");
        assert_eq!(vecs.len(), 2);
        eprintln!("embedding dim = {}", vecs[0].len());
        assert!(vecs[0].len() >= 32, "dim too small: {}", vecs[0].len());
        assert_eq!(vecs[0].len(), vecs[1].len());
        for v in &vecs {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "not normalized: {norm}");
        }
        // Distinct texts must not collapse to the same vector.
        assert!(
            vecs[0]
                .iter()
                .zip(&vecs[1])
                .any(|(a, b)| (a - b).abs() > 1e-3)
        );
    }

    #[test]
    fn small_body_yields_single_chunk_with_symbol_lines() {
        let chunks = build_embedding_chunks(
            "function",
            "foo",
            "",
            None,
            None,
            Some("let x = 1;"),
            10,
            10,
            None,
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (10, 10));
        assert!(chunks[0].text.contains("function: foo"));
        assert!(chunks[0].text.contains("let x = 1;"));
    }

    #[test]
    fn no_body_yields_header_only_chunk() {
        let chunks = build_embedding_chunks(
            "struct",
            "Config",
            "",
            Some("mod db"),
            None,
            None,
            5,
            8,
            None,
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (5, 8));
        assert_eq!(chunks[0].text, "struct: Config | context: mod db");
    }

    #[test]
    fn large_body_splits_into_overlapping_chunks_with_absolute_lines() {
        // 100 lines of ~40 chars -> several chunks under MAX_BODY_CHARS (800).
        let body: Vec<String> = (0..100)
            .map(|i| format!("line {i} {}", "x".repeat(32)))
            .collect();
        let body = body.join("\n");
        // Symbol spans lines 50..149 and body covers the whole range.
        let chunks = build_embedding_chunks(
            "function",
            "big",
            "",
            None,
            None,
            Some(&body),
            50,
            149,
            None,
        );
        assert!(
            chunks.len() > 2,
            "expected multiple chunks, got {}",
            chunks.len()
        );
        assert_eq!(chunks[0].start_line, 50);
        assert_eq!(chunks.last().unwrap().end_line, 149);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.chunk_index, i as i64);
            assert!(c.text.starts_with("function: big | "));
            assert!(c.start_line >= 50 && c.end_line <= 149 && c.start_line <= c.end_line);
        }
        // Consecutive chunks overlap by CHUNK_OVERLAP_LINES.
        for w in chunks.windows(2) {
            assert_eq!(
                w[1].start_line,
                w[0].end_line + 1 - CHUNK_OVERLAP_LINES as i64
            );
        }
        // Deep content (line 90) is present in some chunk even though it is
        // far beyond the first MAX_BODY_CHARS of the body.
        assert!(chunks.iter().any(|c| c.text.contains("line 90")));
    }

    #[test]
    fn i18n_keys_resolve_into_chunk_text_framework_agnostic() {
        let mut dict = std::collections::HashMap::new();
        dict.insert(
            "onboarding.features.agent.title".to_string(),
            "Agent-first coding".to_string(),
        );
        dict.insert("app.title".to_string(), "Claudinio Code".to_string());
        // t("...") (web), NSLocalizedString (iOS), I18n.t (Rails) all reduce
        // to a quoted literal that is a dict key.
        let body = concat!(
            "const a = t(\"onboarding.features.agent.title\");\n",
            "let b = NSLocalizedString('app.title', comment: '');\n",
            "let c = other(\"not.a.key\");"
        );
        let chunks = build_embedding_chunks(
            "function",
            "Wizard",
            "",
            None,
            None,
            Some(body),
            1,
            3,
            Some(&dict),
        );
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("i18n: "));
        assert!(chunks[0].text.contains("Agent-first coding"));
        assert!(chunks[0].text.contains("Claudinio Code"));
        assert!(!chunks[0].text.contains("not.a.key\" resolved"));

        // Without a dict, text is unchanged (no i18n marker).
        let plain =
            build_embedding_chunks("function", "Wizard", "", None, None, Some(body), 1, 3, None);
        assert!(!plain[0].text.contains("i18n:"));
    }

    #[test]
    fn i18n_resolution_is_capped_and_deduped() {
        let mut dict = std::collections::HashMap::new();
        dict.insert("k".to_string(), "x".repeat(300));
        dict.insert("k2".to_string(), "y".repeat(300));
        let body = "t(\"k\") t(\"k\") t(\"k2\") t(\"k2\")";
        let out = resolve_i18n_keys(body, &dict);
        // Deduped: each value once; capped near MAX_I18N_CHARS.
        assert!(out.len() <= MAX_I18N_CHARS + 310);
        assert_eq!(out.matches(&"x".repeat(300)).count(), 1);
    }

    #[test]
    fn doc_section_body_anchors_lines_from_symbol_end() {
        // Doc sections: body starts after the heading line, so absolute lines
        // are anchored from end_line (body = last N lines of the range).
        let body = "para one\npara two\npara three";
        let chunks = build_embedding_chunks(
            "doc_section",
            "Intro",
            "",
            None,
            None,
            Some(body),
            4,
            7,
            None,
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (5, 7));
    }

    #[test]
    fn chunk_headers_and_fts_fields_carry_name_words_and_path() {
        let chunks = build_embedding_chunks(
            "function",
            "buildEmbeddingChunks",
            "src/code_intel/embeddings.rs",
            None,
            Some("Splits symbols"),
            Some("let toolBody = attachSnippets();"),
            1,
            2,
            None,
        );
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert!(
            c.text
                .contains("function: buildEmbeddingChunks (build embedding chunks)")
        );
        assert!(c.text.contains("file: embeddings"));
        assert_eq!(c.fts_name, "buildEmbeddingChunks build embedding chunks");
        assert!(c.fts_path.contains("embeddings.rs"));
        assert!(
            c.fts_path.contains("intel"),
            "dir words missing: {}",
            c.fts_path
        );
        assert!(c.fts_body.contains("Splits symbols"), "doc missing");
        assert!(c.fts_body.contains("attachSnippets"), "raw body missing");
        assert!(
            c.fts_body.contains("tool body"),
            "split words missing: {}",
            c.fts_body
        );
        assert!(c.fts_body.contains("attach snippets"));
    }
}
