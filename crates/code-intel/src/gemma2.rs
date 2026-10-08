//! EmbeddingGemma 2 through ONNX Runtime: one vector space for code, prose,
//! images and audio.
//!
//! The model is three ONNX graphs. The text model is always needed; the
//! vision and audio encoders are separate downloads (109 MB and 189 MB) that
//! turn an image or a clip into "soft tokens" the text model reads in place
//! of placeholder tokens. A workspace with no images never downloads the
//! vision encoder, and one with no audio never downloads the audio encoder.
//!
//! The contract with the graphs is the one transformers.js 4.3.1 implements
//! (`EmbeddingGemma2Model`): the text model takes `input_ids`,
//! `attention_mask` and one `*_features` input per modality — an empty
//! `[0, hidden]` tensor for the modalities a sequence does not use — and
//! returns the pooled, normalized `sentence_embedding`. Nothing here is
//! assumed silently: `load` checks the inputs and outputs it relies on and
//! runs one real encode, so a graph this code does not understand fails at
//! load time, where the caller carries on without it, rather than at search
//! time.
//!
//! By default this model is loaded beside MiniLM (`embeddings::CodeEmbedder`),
//! which writes the index first. The text half below then does two jobs: it
//! embeds the soft tokens of a picture or a clip and the queries compared
//! against them, and it embeds the workspace's text a second time, in the
//! background, into the index's upgrade set (`db::VectorSet`).

use crate::media::{MediaKind, MediaNeeds};
use crate::media_prep::{self, AudioFeatures, ImagePatches};
use ort::session::Session;
use ort::value::{DynValue, Tensor, ValueType};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

/// Stored with the index. A different quantization or output size produces
/// different vectors, so either one changes this id — and an index built
/// under another id is re-embedded instead of being searched with vectors
/// that no longer line up.
pub const MODEL_ID: &str = "embeddinggemma-2-q4-768";

/// Size of the stored vectors. The model is trained so that a prefix of its
/// 768-d output is itself a usable embedding (512, 256 or 128); lowering this
/// trades recall for index size. Keep `MODEL_ID` in step with it.
pub const EMBED_DIM: usize = 768;

pub const CACHE_DIRNAME: &str = "embeddinggemma-2-onnx-q4";

const REPO: &str = "onnx-community/embeddinggemma-2-ONNX";
/// Every file is fetched from this commit and checked against its sha256, so
/// an upstream re-export can never change what a released binary loads.
const REVISION: &str = "daa72c51243991dfcaf9f9137d2c573d8f7790c0";

/// `(path in the repo, local file name, sha256, size in bytes)`.
pub type PinnedFile = (&'static str, &'static str, &'static str, u64);

const TEXT_GRAPH: &str = "model_q4.onnx";
const VISION_GRAPH: &str = "vision_encoder_q4.onnx";
const AUDIO_GRAPH: &str = "audio_encoder_q4.onnx";
const TOKENIZER: &str = "tokenizer.json";

// Each graph keeps its weights in an `.onnx_data` file it refers to by name,
// so the local names must stay the upstream ones. The weights are listed
// first: the graph file is what marks a component as complete.
const TEXT_FILES: &[PinnedFile] = &[
    (
        "tokenizer.json",
        TOKENIZER,
        "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
        32_170_510,
    ),
    (
        "onnx/model_q4.onnx_data",
        "model_q4.onnx_data",
        "c3975f2d1ab7a1878ae31a7d7a9b7804a827aff3800b60dfceafce21cac3df49",
        174_028_800,
    ),
    (
        "onnx/model_q4.onnx",
        TEXT_GRAPH,
        "f9eeba97acddf139b8ee2ddf04bc30dceafa88de93fadf74d7644e0d61a477a9",
        490_742,
    ),
];

const VISION_FILES: &[PinnedFile] = &[
    (
        "onnx/vision_encoder_q4.onnx_data",
        "vision_encoder_q4.onnx_data",
        "0a9d6c927334f152a33dd90874f65d6ea5228999abe6a450d3f7813677fa704c",
        108_957_696,
    ),
    (
        "onnx/vision_encoder_q4.onnx",
        VISION_GRAPH,
        "7ea284226d4938f0ad921ab091f1d80a9ca699aa802984ef5cd5eec4f4761d96",
        159_400,
    ),
];

const AUDIO_FILES: &[PinnedFile] = &[
    (
        "onnx/audio_encoder_q4.onnx_data",
        "audio_encoder_q4.onnx_data",
        "ba9328e6341360974083085b44b2bba265003bda564740f7c4c23ed9928f17e2",
        189_075_968,
    ),
    (
        "onnx/audio_encoder_q4.onnx",
        AUDIO_GRAPH,
        "c4cce3370e72262280d00293cac038896050a03a8e9a27e10b061ca97510296e",
        245_543,
    ),
];

/// The files of one component: the text model (`None`) or a media encoder.
pub fn component_files(component: Option<MediaKind>) -> &'static [PinnedFile] {
    match component {
        None => TEXT_FILES,
        Some(MediaKind::Image) => VISION_FILES,
        Some(MediaKind::Audio) => AUDIO_FILES,
    }
}

pub fn file_url(remote: &str) -> String {
    format!("https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}")
}

pub fn component_present(dir: &Path, component: Option<MediaKind>) -> bool {
    component_files(component)
        .iter()
        .all(|(_, local, _, _)| dir.join(local).exists())
}

/// Download whatever is missing of one component. Files land through a
/// verified `.part` rename, so a file that exists is a file that is whole.
pub async fn ensure_component_downloaded(
    dir: &Path,
    component: Option<MediaKind>,
) -> Result<(), String> {
    if component_present(dir, component) {
        return Ok(());
    }
    let what = match component {
        None => "text model",
        Some(MediaKind::Image) => "vision encoder",
        Some(MediaKind::Audio) => "audio encoder",
    };
    eprintln!(
        "[embeddings] downloading EmbeddingGemma 2 {what} to {}",
        dir.display()
    );
    std::fs::create_dir_all(dir).map_err(|e| format!("create model dir: {e}"))?;
    for (remote, local, sha256, len) in component_files(component) {
        let dest = dir.join(local);
        if dest.exists() {
            continue;
        }
        crate::download::download_verified_with_retries(
            &file_url(remote),
            &dest,
            local,
            sha256,
            *len,
            crate::download::DEFAULT_RETRIES,
        )
        .await?;
    }
    Ok(())
}

// ── prompts ─────────────────────────────────────────────────────────────

// The model is trained with a task prefix on every text; leaving it out
// measurably hurts retrieval. Media is passed without any prefix.
const DOC_PREFIX: &str = "title: none | text: ";
const CODE_QUERY_PREFIX: &str = "task: code retrieval | query: ";
const MEDIA_QUERY_PREFIX: &str = "task: search result | query: ";

/// What a query is looking for; it selects the task prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryTask {
    /// Code and documentation.
    Code,
    /// Images and audio.
    Media,
}

/// Index texts are capped at ~800 characters of body (see
/// `embeddings::build_embedding_chunks`), so a longer window would only pad.
const TEXT_MAX_TOKENS: usize = 512;

/// Soft-token width when the graph does not declare it.
const DEFAULT_HIDDEN: usize = 512;

const OUTPUT: &str = "sentence_embedding";

fn build_session(path: &Path) -> Result<Session, String> {
    Session::builder()
        .map_err(|e| format!("ort builder: {e}"))?
        // Same settings and reasons as the MiniLM session: no retained
        // buffers between runs, and a thread cap (`embeddings::intra_threads`)
        // so indexing stays a background job instead of taking the whole
        // machine.
        .with_memory_pattern(false)
        .map_err(|e| format!("ort memory pattern: {e}"))?
        .with_intra_threads(crate::embeddings::intra_threads())
        .map_err(|e| format!("ort intra threads: {e}"))?
        .with_inter_threads(1)
        .map_err(|e| format!("ort inter threads: {e}"))?
        .commit_from_file(path)
        .map_err(|e| format!("ort load {}: {e}", path.display()))
}

fn input_names(session: &Session) -> Vec<String> {
    session
        .inputs()
        .iter()
        .map(|i| i.name().to_string())
        .collect()
}

fn output_name(session: &Session, wanted: &str) -> Result<String, String> {
    let names: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();
    if names.iter().any(|n| n == wanted) {
        Ok(wanted.to_string())
    } else {
        Err(format!(
            "graph has no `{wanted}` output (outputs: {})",
            names.join(", ")
        ))
    }
}

fn require_inputs(session: &Session, what: &str, wanted: &[&str]) -> Result<(), String> {
    let names = input_names(session);
    for w in wanted {
        if !names.iter().any(|n| n == w) {
            return Err(format!(
                "{what} has no `{w}` input (inputs: {})",
                names.join(", ")
            ));
        }
    }
    Ok(())
}

fn f32_tensor(shape: Vec<i64>, data: Vec<f32>) -> Result<DynValue, String> {
    Tensor::from_array((shape, data))
        .map(Into::into)
        .map_err(|e| format!("f32 tensor: {e}"))
}

fn i64_tensor(shape: Vec<i64>, data: Vec<i64>) -> Result<DynValue, String> {
    Tensor::from_array((shape, data))
        .map(Into::into)
        .map_err(|e| format!("i64 tensor: {e}"))
}

/// Keep the leading `EMBED_DIM` values and re-normalize, so cosine similarity
/// stays a plain dot product. A non-finite value means the run went wrong —
/// that vector must never reach the index.
fn finalize(row: &[f32]) -> Result<Vec<f32>, String> {
    if row.len() < EMBED_DIM {
        return Err(format!(
            "model returned {} dimensions, expected at least {EMBED_DIM}",
            row.len()
        ));
    }
    let mut v = row[..EMBED_DIM].to_vec();
    if v.iter().any(|x| !x.is_finite()) {
        return Err("model returned a non-finite embedding".into());
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-6 {
        return Err("model returned a zero embedding".into());
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    Ok(v)
}

// ── media encoders ──────────────────────────────────────────────────────

/// The vision and audio encoders, behind their own lock: encoding an image
/// takes seconds, and a search must not wait for it. The text model — the
/// only thing a query needs — has a lock of its own (`Gemma2Embedder`).
#[derive(Default)]
pub struct MediaEncoders {
    vision: Option<Session>,
    audio: Option<Session>,
    loaded: Arc<LoadedEncoders>,
}

pub type SharedEncoders = Arc<Mutex<MediaEncoders>>;

/// Which encoders are loaded, readable without `MediaEncoders`' lock. That
/// lock is held for a whole encoder run, and "is there a vision encoder?" is
/// asked by threads holding the embedder's lock (a status call, the watcher)
/// — which must never end up waiting on an image being encoded.
#[derive(Default)]
struct LoadedEncoders {
    vision: std::sync::atomic::AtomicBool,
    audio: std::sync::atomic::AtomicBool,
}

impl LoadedEncoders {
    fn support(&self) -> MediaNeeds {
        use std::sync::atomic::Ordering::Relaxed;
        MediaNeeds {
            images: self.vision.load(Relaxed),
            audio: self.audio.load(Relaxed),
        }
    }
}

impl MediaEncoders {
    pub fn has(&self, kind: MediaKind) -> bool {
        match kind {
            MediaKind::Image => self.vision.is_some(),
            MediaKind::Audio => self.audio.is_some(),
        }
    }

    pub fn support(&self) -> MediaNeeds {
        MediaNeeds {
            images: self.vision.is_some(),
            audio: self.audio.is_some(),
        }
    }

    /// Load one encoder from `dir` unless it is already loaded.
    pub fn load(&mut self, dir: &Path, kind: MediaKind) -> Result<(), String> {
        if self.has(kind) {
            return Ok(());
        }
        if !component_present(dir, Some(kind)) {
            return Err(format!(
                "{} encoder not downloaded to {}",
                kind.as_str(),
                dir.display()
            ));
        }
        match kind {
            MediaKind::Image => {
                let session = build_session(&dir.join(VISION_GRAPH))?;
                require_inputs(
                    &session,
                    "vision encoder",
                    &["pixel_values", "pixel_position_ids"],
                )?;
                output_name(&session, "image_features")?;
                self.vision = Some(session);
                self.loaded
                    .vision
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
            MediaKind::Audio => {
                let session = build_session(&dir.join(AUDIO_GRAPH))?;
                require_inputs(
                    &session,
                    "audio encoder",
                    &["input_features", "input_features_mask"],
                )?;
                output_name(&session, "audio_features")?;
                self.audio = Some(session);
                self.loaded
                    .audio
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// Soft tokens of one image: `soft_tokens * hidden` values.
    pub fn encode_image(&mut self, patches: &ImagePatches) -> Result<Vec<f32>, String> {
        let session = self.vision.as_mut().ok_or("vision encoder not loaded")?;
        let n = patches.max_patches as i64;
        let mut inputs: HashMap<String, DynValue> = HashMap::new();
        inputs.insert(
            "pixel_values".into(),
            f32_tensor(
                vec![1, n, media_prep::PATCH_DIM as i64],
                patches.pixel_values.clone(),
            )?,
        );
        inputs.insert(
            "pixel_position_ids".into(),
            i64_tensor(vec![1, n, 2], patches.position_ids.clone())?,
        );
        let outs = session
            .run(inputs)
            .map_err(|e| format!("vision encoder run: {e}"))?;
        let (_, flat) = outs["image_features"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("vision encoder output: {e}"))?;
        Ok(flat.to_vec())
    }

    /// Soft tokens of one clip: `soft_tokens * hidden` values.
    pub fn encode_audio(&mut self, audio: &AudioFeatures) -> Result<Vec<f32>, String> {
        let session = self.audio.as_mut().ok_or("audio encoder not loaded")?;
        let frames = audio.frames as i64;
        let mut inputs: HashMap<String, DynValue> = HashMap::new();
        inputs.insert(
            "input_features".into(),
            f32_tensor(
                vec![1, frames, media_prep::MEL_BINS as i64],
                audio.features.clone(),
            )?,
        );
        let mask: DynValue = Tensor::from_array((vec![1, frames], audio.mask.clone()))
            .map(Into::into)
            .map_err(|e| format!("mask tensor: {e}"))?;
        inputs.insert("input_features_mask".into(), mask);
        let outs = session
            .run(inputs)
            .map_err(|e| format!("audio encoder run: {e}"))?;
        let (_, flat) = outs["audio_features"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("audio encoder output: {e}"))?;
        Ok(flat.to_vec())
    }
}

/// Decode a media file and run it through its encoder. Only the encoders'
/// lock is held, and only for the encoder run itself.
pub fn encode_media_file(
    encoders: &SharedEncoders,
    path: &Path,
    kind: MediaKind,
) -> Result<Vec<f32>, String> {
    match kind {
        MediaKind::Image => {
            let patches = media_prep::load_image_patches(path, media_prep::image_token_budget())?;
            let mut enc = encoders
                .lock()
                .map_err(|e| format!("encoders lock poisoned: {e}"))?;
            enc.encode_image(&patches)
        }
        MediaKind::Audio => {
            let features = media_prep::load_audio_features(path)?;
            let mut enc = encoders
                .lock()
                .map_err(|e| format!("encoders lock poisoned: {e}"))?;
            enc.encode_audio(&features)
        }
    }
}

// ── the embedder ────────────────────────────────────────────────────────

/// The text model, shareable: a run locks its session and nothing else, so
/// one thread can be embedding a workspace in the background, a batch at a
/// time, while another embeds a query in between two batches — and neither
/// holds whatever lock the caller keeps its embedders under.
pub struct Gemma2Embedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    /// Width of one soft token (the text model's hidden size).
    hidden: usize,
    /// The text model's soft-token inputs (`image_features`, `audio_features`,
    /// `video_features`). All of them must be fed on every run.
    feature_inputs: Vec<String>,
    encoders: SharedEncoders,
    loaded: Arc<LoadedEncoders>,
    dir: PathBuf,
}

impl Gemma2Embedder {
    /// Load the text model from `dir`. Media encoders are added afterwards
    /// with `load_encoder`, so a failure there costs media search only.
    pub fn load(dir: &Path) -> Result<Self, String> {
        if !component_present(dir, None) {
            return Err(format!(
                "EmbeddingGemma 2 not downloaded to {}",
                dir.display()
            ));
        }
        let session = build_session(&dir.join(TEXT_GRAPH))?;
        require_inputs(&session, "text model", &["input_ids", "attention_mask"])?;
        output_name(&session, OUTPUT)?;

        let mut feature_inputs = Vec::new();
        let mut hidden = DEFAULT_HIDDEN;
        for input in session.inputs() {
            let name = input.name();
            if name == "input_ids" || name == "attention_mask" {
                continue;
            }
            if !name.ends_with("_features") {
                return Err(format!(
                    "text model has an input this build does not know how to feed: `{name}`"
                ));
            }
            if let ValueType::Tensor { shape, .. } = input.dtype()
                && let Some(last) = shape.last()
                && *last > 0
            {
                hidden = *last as usize;
            }
            feature_inputs.push(name.to_string());
        }

        let mut tokenizer = Tokenizer::from_file(dir.join(TOKENIZER))
            .map_err(|e| format!("tokenizer load: {e}"))?;
        // Lengths are handled here, per kind of input: text is cut to
        // `TEXT_MAX_TOKENS` (see `clip_ids`), a media sequence must never be
        // cut at all. Whatever the tokenizer file configures would apply to
        // both.
        tokenizer
            .with_truncation(None)
            .map_err(|e| format!("tokenizer truncation: {e}"))?;
        tokenizer.with_padding(None);

        let encoders = MediaEncoders::default();
        let loaded = encoders.loaded.clone();
        let embedder = Gemma2Embedder {
            session: Mutex::new(session),
            tokenizer,
            hidden,
            feature_inputs,
            encoders: Arc::new(Mutex::new(encoders)),
            loaded,
            dir: dir.to_path_buf(),
        };
        // One real run: if this build feeds the graph wrongly it must surface
        // here, not on the first search.
        embedder
            .encode_documents(&["fn main() {}"])
            .map_err(|e| format!("EmbeddingGemma 2 self-test failed: {e}"))?;
        Ok(embedder)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn encoders(&self) -> SharedEncoders {
        self.encoders.clone()
    }

    /// The encoders loaded so far. Never blocks, even mid-encode.
    pub fn media_support(&self) -> MediaNeeds {
        self.loaded.support()
    }

    /// Where a media encoder of this kind gets loaded — provided the text
    /// model has the matching soft-token input and the tokenizer its
    /// placeholder; without both the encoder would be dead weight.
    pub fn encoder_slot(&self, kind: MediaKind) -> Result<SharedEncoders, String> {
        let (_, token, _) = placeholder_tokens(kind);
        if self.tokenizer.token_to_id(token).is_none() {
            return Err(format!("tokenizer has no `{token}` token"));
        }
        let input = feature_input(kind);
        if !self.feature_inputs.iter().any(|n| n == input) {
            return Err(format!("text model has no `{input}` input"));
        }
        Ok(self.encoders.clone())
    }

    /// Load a media encoder that has already been downloaded to this model's
    /// directory.
    pub fn load_encoder(&self, kind: MediaKind) -> Result<(), String> {
        self.encoder_slot(kind)?
            .lock()
            .map_err(|e| format!("encoders lock poisoned: {e}"))?
            .load(&self.dir, kind)
    }

    fn run_text(
        &self,
        ids: Vec<i64>,
        mask: Vec<i64>,
        batch: usize,
        seq: usize,
        soft: Option<(&str, &[f32])>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut inputs: HashMap<String, DynValue> = HashMap::new();
        inputs.insert(
            "input_ids".into(),
            i64_tensor(vec![batch as i64, seq as i64], ids)?,
        );
        inputs.insert(
            "attention_mask".into(),
            i64_tensor(vec![batch as i64, seq as i64], mask)?,
        );
        for name in &self.feature_inputs {
            // A modality the sequence does not use still gets its input: zero
            // rows of soft tokens.
            let data: Vec<f32> = match soft {
                Some((used, data)) if used == name => data.to_vec(),
                _ => Vec::new(),
            };
            let rows = (data.len() / self.hidden) as i64;
            inputs.insert(
                name.clone(),
                f32_tensor(vec![rows, self.hidden as i64], data)?,
            );
        }
        let mut session = self
            .session
            .lock()
            .map_err(|e| format!("text model lock poisoned: {e}"))?;
        let outs = session.run(inputs).map_err(|e| format!("ort run: {e}"))?;
        let (shape, flat) = outs[OUTPUT]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract {OUTPUT}: {e}"))?;
        let dim = shape.last().map(|d| *d as usize).unwrap_or(0);
        if dim == 0 || flat.len() != batch * dim {
            return Err(format!(
                "unexpected {OUTPUT} shape {shape:?} for a batch of {batch}"
            ));
        }
        flat.chunks_exact(dim).map(finalize).collect()
    }

    fn encode_prefixed(&self, prefix: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(
                texts
                    .iter()
                    .map(|t| tokenizers::EncodeInput::Single(format!("{prefix}{t}").into()))
                    .collect(),
                true,
            )
            .map_err(|e| format!("tokenize: {e}"))?;

        let rows: Vec<Vec<i64>> = encodings
            .iter()
            .map(|e| clip_ids(e.get_ids(), e.get_special_tokens_mask(), TEXT_MAX_TOKENS))
            .collect();
        let batch = rows.len();
        let seq = rows.iter().map(Vec::len).max().unwrap_or(0).max(1);
        // Padding is token 0 with a zero mask, which is `<pad>` for Gemma.
        let mut ids = vec![0i64; batch * seq];
        let mut mask = vec![0i64; batch * seq];
        for (b, row) in rows.iter().enumerate() {
            ids[b * seq..b * seq + row.len()].copy_from_slice(row);
            mask[b * seq..b * seq + row.len()].fill(1);
        }
        self.run_text(ids, mask, batch, seq, None)
    }

    /// Embed index texts (code chunks, doc sections).
    pub fn encode_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        self.encode_prefixed(DOC_PREFIX, texts)
    }

    pub fn encode_query(&self, text: &str, task: QueryTask) -> Result<Vec<f32>, String> {
        let prefix = match task {
            QueryTask::Code => CODE_QUERY_PREFIX,
            QueryTask::Media => MEDIA_QUERY_PREFIX,
        };
        self.encode_prefixed(prefix, &[text])?
            .pop()
            .ok_or_else(|| "empty encode result".to_string())
    }

    /// The token sequence standing for one media item: its opening marker,
    /// one placeholder per soft token, its closing marker.
    fn media_sequence(&self, kind: MediaKind, soft_tokens: usize) -> Result<Vec<i64>, String> {
        let (open, token, close) = placeholder_tokens(kind);
        let token_id =
            self.tokenizer
                .token_to_id(token)
                .ok_or_else(|| format!("tokenizer has no `{token}` token"))? as i64;
        let text = format!("{open}{}{close}", token.repeat(soft_tokens));
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenize media sequence: {e}"))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|i| *i as i64).collect();
        // The text model pairs placeholders with soft tokens one to one; any
        // other count would silently misalign them.
        let placeholders = ids.iter().filter(|i| **i == token_id).count();
        if placeholders != soft_tokens {
            return Err(format!(
                "tokenizer produced {placeholders} `{token}` placeholders for {soft_tokens} soft tokens"
            ));
        }
        Ok(ids)
    }

    /// Embed one media item from the soft tokens its encoder produced.
    pub fn embed_soft_tokens(&self, kind: MediaKind, soft: &[f32]) -> Result<Vec<f32>, String> {
        if soft.is_empty() || !soft.len().is_multiple_of(self.hidden) {
            return Err(format!(
                "{} encoder returned {} values, not a multiple of the soft-token width {}",
                kind.as_str(),
                soft.len(),
                self.hidden
            ));
        }
        let ids = self.media_sequence(kind, soft.len() / self.hidden)?;
        let seq = ids.len();
        self.run_text(
            ids,
            vec![1i64; seq],
            1,
            seq,
            Some((feature_input(kind), soft)),
        )?
        .pop()
        .ok_or_else(|| "empty encode result".to_string())
    }
}

/// Cut a token sequence to `max` tokens. A special token closing the
/// sequence (an end-of-sequence marker) is kept: the model was trained with
/// it in place, and plain truncation would be the one case that drops it.
fn clip_ids(ids: &[u32], special: &[u32], max: usize) -> Vec<i64> {
    if ids.len() <= max {
        return ids.iter().map(|i| *i as i64).collect();
    }
    let mut out: Vec<i64> = ids[..max].iter().map(|i| *i as i64).collect();
    let last = ids.len() - 1;
    if special.get(last) == Some(&1)
        && let Some(slot) = out.last_mut()
    {
        *slot = ids[last] as i64;
    }
    out
}

fn feature_input(kind: MediaKind) -> &'static str {
    match kind {
        MediaKind::Image => "image_features",
        MediaKind::Audio => "audio_features",
    }
}

/// `(opening marker, placeholder, closing marker)` as the tokenizer names them.
fn placeholder_tokens(kind: MediaKind) -> (&'static str, &'static str, &'static str) {
    match kind {
        MediaKind::Image => ("<|image>", "<|image|>", "<image|>"),
        MediaKind::Audio => ("<|audio>", "<|audio|>", "<audio|>"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins are only useful if they are well-formed and complete: every file
    /// has a 64-hex digest and a size, comes from the pinned commit, and each
    /// graph ships with the weights file it refers to.
    #[test]
    fn every_component_is_pinned_to_one_revision_with_its_weights() {
        for component in [None, Some(MediaKind::Image), Some(MediaKind::Audio)] {
            let files = component_files(component);
            for (remote, local, sha, len) in files {
                assert_eq!(sha.len(), 64, "{local}");
                assert!(
                    sha.chars()
                        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                    "{local}"
                );
                assert!(*len > 0, "{local}");
                assert_eq!(
                    Path::new(remote).file_name().unwrap().to_str().unwrap(),
                    *local
                );
                assert!(file_url(remote).contains(&format!("/resolve/{REVISION}/")));
            }
            let graph = files
                .iter()
                .find(|f| f.1.ends_with(".onnx"))
                .expect("a graph");
            let weights = format!("{}_data", graph.1);
            assert!(
                files.iter().any(|f| f.1 == weights),
                "{} needs {weights}",
                graph.1
            );
            // The graph is last, so its presence means the component is whole.
            assert_eq!(files.last().unwrap().1, graph.1);
        }
        assert!(
            MODEL_ID.ends_with(&EMBED_DIM.to_string()),
            "MODEL_ID must change with EMBED_DIM"
        );
    }

    #[test]
    fn a_component_is_present_only_when_all_of_its_files_are() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!component_present(dir.path(), None));
        for (_, local, _, _) in VISION_FILES {
            assert!(!component_present(dir.path(), Some(MediaKind::Image)));
            std::fs::write(dir.path().join(local), b"x").unwrap();
        }
        assert!(component_present(dir.path(), Some(MediaKind::Image)));
        assert!(!component_present(dir.path(), Some(MediaKind::Audio)));
        assert!(!component_present(dir.path(), None));
    }

    #[test]
    fn long_inputs_are_cut_but_keep_their_closing_token() {
        let ids: Vec<u32> = (10..20).collect();
        let none = vec![0u32; 10];
        let mut bos_eos = none.clone();
        bos_eos[0] = 1;
        bos_eos[9] = 1;
        // Short enough: untouched.
        assert_eq!(clip_ids(&ids, &bos_eos, 10).len(), 10);
        // Cut, with a closing special token: it replaces the last kept token.
        assert_eq!(clip_ids(&ids, &bos_eos, 4), vec![10, 11, 12, 19]);
        // Cut, with no closing special token: a plain prefix.
        assert_eq!(clip_ids(&ids, &none, 4), vec![10, 11, 12, 13]);
    }

    #[test]
    fn stored_vectors_are_truncated_and_unit_length() {
        let mut row = vec![0f32; 1024];
        row[0] = 3.0;
        row[1] = 4.0;
        row[1000] = 100.0; // beyond EMBED_DIM: dropped before normalizing
        let v = finalize(&row).unwrap();
        assert_eq!(v.len(), EMBED_DIM);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);

        assert!(finalize(&vec![0f32; EMBED_DIM]).is_err(), "zero vector");
        let mut nan = vec![1f32; EMBED_DIM];
        nan[5] = f32::NAN;
        assert!(finalize(&nan).is_err(), "non-finite");
        assert!(finalize(&[1.0; 10]).is_err(), "too short");
    }
}
