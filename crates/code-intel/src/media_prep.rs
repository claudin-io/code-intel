//! Turning an image or an audio file into what EmbeddingGemma 2's encoders
//! take as input. Pure Rust on purpose: every decoder here compiles on all
//! the platforms the plugin ships for, with no system library to find.
//!
//! Both halves are ports of the reference preprocessing in transformers.js
//! 4.3.1 (`Gemma4ImageProcessor`, `Gemma4AudioFeatureExtractor`) with the
//! values of the model's `processor_config.json`. `tests/fixtures/
//! gemma2_preprocessing.json` holds that implementation's output for a set of
//! deterministic inputs, and the tests below hold this one to it.

use std::path::Path;

// ── images ──────────────────────────────────────────────────────────────

/// Side of one square patch, in pixels.
pub const PATCH_SIZE: usize = 16;
/// The vision encoder pools `POOL x POOL` patches into one soft token.
pub const POOL: usize = 3;
/// Values per patch: `PATCH_SIZE * PATCH_SIZE * 3` (RGB).
pub const PATCH_DIM: usize = PATCH_SIZE * PATCH_SIZE * 3;

/// Soft-token budgets the vision encoder was trained with. More tokens means
/// a larger resize target: finer detail, quadratically more attention work.
pub const IMAGE_TOKEN_BUDGETS: [usize; 5] = [70, 140, 280, 560, 1120];
/// The model's reference budget (`max_soft_tokens` in its processor config),
/// and the one its published image-retrieval scores were measured at.
pub const DEFAULT_IMAGE_TOKENS: usize = 280;

/// `CODE_INTEL_IMAGE_TOKENS` trades detail for indexing speed; anything that
/// is not one of the trained budgets is ignored.
pub fn image_token_budget() -> usize {
    std::env::var("CODE_INTEL_IMAGE_TOKENS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| IMAGE_TOKEN_BUDGETS.contains(v))
        .unwrap_or(DEFAULT_IMAGE_TOKENS)
}

/// One image as the vision encoder wants it: a fixed-length, zero-padded list
/// of flattened patches plus each patch's `(column, row)`, `-1` for padding.
#[derive(Debug, Clone)]
pub struct ImagePatches {
    /// `max_patches * PATCH_DIM` values in `[0, 1]`.
    pub pixel_values: Vec<f32>,
    /// `max_patches * 2` values.
    pub position_ids: Vec<i64>,
    pub max_patches: usize,
    /// Patches that hold image content (the rest is padding).
    pub patches: usize,
    /// Soft tokens the encoder returns for this image.
    pub soft_tokens: usize,
}

/// Size to resize to: the largest `(height, width)` with the image's aspect
/// ratio whose patch count fits the budget, both sides a multiple of
/// `POOL * PATCH_SIZE`. Small images are scaled *up* to it — the encoder
/// always sees roughly the same number of patches.
pub fn target_size(
    height: usize,
    width: usize,
    max_patches: usize,
) -> Result<(usize, usize), String> {
    if height == 0 || width == 0 {
        return Err("image has a zero dimension".into());
    }
    let side_mult = POOL * PATCH_SIZE;
    let target_px = (max_patches * PATCH_SIZE * PATCH_SIZE) as f64;
    let factor = (target_px / (height as f64 * width as f64)).sqrt();
    let mut th = ((factor * height as f64) / side_mult as f64).floor() as usize * side_mult;
    let mut tw = ((factor * width as f64) / side_mult as f64).floor() as usize * side_mult;
    if th == 0 && tw == 0 {
        return Err("image cannot be resized to a non-empty patch grid".into());
    }
    // Extreme aspect ratios round one side down to nothing: give that side a
    // single pooled row/column and let the other take what the budget allows.
    let max_side = (max_patches / (POOL * POOL)) * side_mult;
    if th == 0 {
        th = side_mult;
        tw = ((width / height) * side_mult).min(max_side);
    } else if tw == 0 {
        tw = side_mult;
        th = ((height / width) * side_mult).min(max_side);
    }
    if th == 0 || tw == 0 {
        return Err("image cannot be resized to a non-empty patch grid".into());
    }
    Ok((th, tw))
}

/// Cut an RGB image (row-major, 3 bytes per pixel, sides already multiples of
/// `PATCH_SIZE`) into patches. Within a patch the order is row, column,
/// channel — the image's own byte order.
pub fn patchify(rgb: &[u8], height: usize, width: usize, soft_token_budget: usize) -> ImagePatches {
    let max_patches = soft_token_budget * POOL * POOL;
    let ph = height / PATCH_SIZE;
    let pw = width / PATCH_SIZE;
    let patches = (ph * pw).min(max_patches);

    let mut pixel_values = vec![0f32; max_patches * PATCH_DIM];
    let mut position_ids = vec![-1i64; max_patches * 2];
    const SCALE: f32 = 1.0 / 255.0;
    let mut out = 0usize;
    let mut idx = 0usize;
    'rows: for row in 0..ph {
        for col in 0..pw {
            if idx >= patches {
                break 'rows;
            }
            for dy in 0..PATCH_SIZE {
                let start = ((row * PATCH_SIZE + dy) * width + col * PATCH_SIZE) * 3;
                for &byte in &rgb[start..start + PATCH_SIZE * 3] {
                    pixel_values[out] = byte as f32 * SCALE;
                    out += 1;
                }
            }
            position_ids[idx * 2] = col as i64;
            position_ids[idx * 2 + 1] = row as i64;
            idx += 1;
        }
    }

    ImagePatches {
        pixel_values,
        position_ids,
        max_patches,
        patches,
        soft_tokens: patches / (POOL * POOL),
    }
}

/// Decoded pixels are capped so a pathological file (a 30000x30000 PNG a few
/// kilobytes long) cannot take the process down.
const MAX_DECODE_BYTES: u64 = 512 * 1024 * 1024;

/// Decoders parse whatever a repository contains, and a corrupt file can
/// trip an internal assertion in one of them (fuzzing found one in WAV header
/// handling: a zeroed sample rate). That must cost the file its vector, not
/// the server its life.
fn guarded<T>(what: &str, decode: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(decode))
        .unwrap_or_else(|_| Err(format!("{what} decoder panicked on this file")))
}

/// Decode, flatten transparency, resize and patchify an image file.
pub fn load_image_patches(path: &Path, soft_token_budget: usize) -> Result<ImagePatches, String> {
    guarded("image", || decode_image_patches(path, soft_token_budget))
}

fn decode_image_patches(path: &Path, soft_token_budget: usize) -> Result<ImagePatches, String> {
    let mut reader = image::ImageReader::open(path)
        .map_err(|e| format!("open image: {e}"))?
        .with_guessed_format()
        .map_err(|e| format!("read image header: {e}"))?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_DECODE_BYTES);
    reader.limits(limits);
    let decoded = reader.decode().map_err(|e| format!("decode image: {e}"))?;
    rgb_to_patches(flatten_on_white(&decoded), soft_token_budget)
}

/// Transparent pixels are composited over white. Icons and logos — most of
/// what a repository's images are — are drawn on transparency, and simply
/// dropping the alpha channel leaves whatever colour the encoder stored
/// underneath, usually black: the shape disappears into its background.
fn flatten_on_white(img: &image::DynamicImage) -> image::RgbImage {
    if !img.color().has_alpha() {
        return img.to_rgb8();
    }
    let rgba = img.to_rgba8();
    let mut out = image::RgbImage::new(rgba.width(), rgba.height());
    for (dst, src) in out.pixels_mut().zip(rgba.pixels()) {
        let a = src[3] as u32;
        for c in 0..3 {
            dst[c] = ((src[c] as u32 * a + 255 * (255 - a) + 127) / 255) as u8;
        }
    }
    out
}

pub fn rgb_to_patches(
    img: image::RgbImage,
    soft_token_budget: usize,
) -> Result<ImagePatches, String> {
    let max_patches = soft_token_budget * POOL * POOL;
    let (h, w) = (img.height() as usize, img.width() as usize);
    let (th, tw) = target_size(h, w, max_patches)?;
    let img = if (th, tw) != (h, w) {
        // Catmull-Rom is the bicubic the reference pipeline asks for
        // (`resample: 3`).
        image::imageops::resize(
            &img,
            tw as u32,
            th as u32,
            image::imageops::FilterType::CatmullRom,
        )
    } else {
        img
    };
    Ok(patchify(img.as_raw(), th, tw, soft_token_budget))
}

// ── audio ───────────────────────────────────────────────────────────────

pub const SAMPLE_RATE: usize = 16_000;
/// Analysis window: 20 ms, advanced 10 ms at a time.
const FRAME_LENGTH: usize = 320;
const HOP_LENGTH: usize = 160;
const FFT_LENGTH: usize = 512;
const FREQ_BINS: usize = FFT_LENGTH / 2 + 1;
/// Mel channels per frame.
pub const MEL_BINS: usize = 128;
const MEL_MAX_HZ: f64 = 8000.0;
const MEL_FLOOR: f32 = 0.001;
/// The waveform is zero-padded to a multiple of this before framing.
const PAD_MULTIPLE: usize = 128;
/// Clips are cut at 30 s, as the reference feature extractor does. At 25 soft
/// tokens per second that is 750 tokens, and enough to tell what a sound is.
pub const MAX_AUDIO_SECONDS: usize = 30;
const MAX_SAMPLES: usize = MAX_AUDIO_SECONDS * SAMPLE_RATE;
/// The audio encoder halves the frame rate twice: one soft token per 4 frames.
const FRAMES_PER_TOKEN: usize = 4;

/// One clip as the audio encoder wants it: log-mel frames and which of them
/// are real audio rather than padding.
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    /// `frames * MEL_BINS` values.
    pub features: Vec<f32>,
    pub mask: Vec<bool>,
    pub frames: usize,
    /// Soft tokens the encoder returns for this clip.
    pub soft_tokens: usize,
}

/// In-place radix-2 FFT. `FFT_LENGTH` is a power of two and this runs once
/// per 10 ms of audio, so a dependency would buy nothing.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two() && im.len() == n);
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = -2.0 * std::f64::consts::PI / len as f64;
        let (w_im, w_re) = angle.sin_cos();
        for start in (0..n).step_by(len) {
            let (mut cur_re, mut cur_im) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let t_re = re[b] * cur_re - im[b] * cur_im;
                let t_im = re[b] * cur_im + im[b] * cur_re;
                re[b] = re[a] - t_re;
                im[b] = im[a] - t_im;
                re[a] += t_re;
                im[a] += t_im;
                let next_re = cur_re * w_re - cur_im * w_im;
                cur_im = cur_re * w_im + cur_im * w_re;
                cur_re = next_re;
            }
        }
        len <<= 1;
    }
}

/// Triangular mel filters (HTK scale, no area normalization), `MEL_BINS` rows
/// of `FREQ_BINS` weights.
fn mel_filter_bank() -> Vec<f32> {
    let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10f64.powf(mel / 2595.0) - 1.0);
    let mel_max = hz_to_mel(MEL_MAX_HZ);
    let points = MEL_BINS + 2;
    let filter_hz: Vec<f64> = (0..points)
        .map(|i| mel_to_hz(mel_max * i as f64 / (points - 1) as f64))
        .collect();
    let nyquist = (SAMPLE_RATE / 2) as f64;
    let mut bank = vec![0f32; MEL_BINS * FREQ_BINS];
    for bin in 0..FREQ_BINS {
        let hz = nyquist * bin as f64 / (FREQ_BINS - 1) as f64;
        for m in 0..MEL_BINS {
            let rising = (hz - filter_hz[m]) / (filter_hz[m + 1] - filter_hz[m]);
            let falling = (filter_hz[m + 2] - hz) / (filter_hz[m + 2] - filter_hz[m + 1]);
            bank[m * FREQ_BINS + bin] = rising.min(falling).max(0.0) as f32;
        }
    }
    bank
}

/// Log-mel features of a mono 16 kHz waveform.
pub fn mel_features(wave: &[f32]) -> Result<AudioFeatures, String> {
    let wave = &wave[..wave.len().min(MAX_SAMPLES)];
    let real_len = wave.len();
    let padded_len = real_len.div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE;

    // "Semicausal" framing: half a frame of silence in front, none behind,
    // and a frame only counts if the sample just past it exists.
    let lead = FRAME_LENGTH / 2;
    let total = padded_len + lead;
    if total < FRAME_LENGTH + 1 {
        return Err("audio clip is too short to analyse".into());
    }
    let frames = (total - (FRAME_LENGTH + 1)) / HOP_LENGTH + 1;

    let sample = |i: usize| -> f64 {
        // Index into [lead zeros][real samples][padding zeros].
        if i >= lead && i - lead < real_len {
            wave[i - lead] as f64
        } else {
            0.0
        }
    };
    // Periodic Hann window.
    let window: Vec<f64> = (0..FRAME_LENGTH)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FRAME_LENGTH as f64).cos())
        .collect();
    let bank = mel_filter_bank();

    let mut features = vec![0f32; frames * MEL_BINS];
    let mut mask = vec![false; frames];
    let mut re = vec![0f64; FFT_LENGTH];
    let mut im = vec![0f64; FFT_LENGTH];
    let mut magnitude = vec![0f32; FREQ_BINS];
    for frame in 0..frames {
        let offset = frame * HOP_LENGTH;
        // A frame is real audio when the sample one past its end is.
        let probe = offset + FRAME_LENGTH;
        mask[frame] = probe >= lead && probe - lead < real_len;
        if !mask[frame] {
            // Frames that reach into the padding are zeroed, not analysed.
            continue;
        }
        re.fill(0.0);
        im.fill(0.0);
        for (j, w) in window.iter().enumerate() {
            re[j] = sample(offset + j) * w;
        }
        fft(&mut re, &mut im);
        for (bin, mag) in magnitude.iter_mut().enumerate() {
            *mag = (re[bin] * re[bin] + im[bin] * im[bin]).sqrt() as f32;
        }
        let row = &mut features[frame * MEL_BINS..(frame + 1) * MEL_BINS];
        for (m, out) in row.iter_mut().enumerate() {
            let weights = &bank[m * FREQ_BINS..(m + 1) * FREQ_BINS];
            let energy: f32 = weights
                .iter()
                .zip(magnitude.iter())
                .map(|(w, x)| w * x)
                .sum();
            *out = (energy + MEL_FLOOR).ln();
        }
    }

    let soft_tokens = mask
        .iter()
        .step_by(FRAMES_PER_TOKEN)
        .filter(|m| **m)
        .count();
    if soft_tokens == 0 {
        return Err("audio clip is too short to analyse".into());
    }
    Ok(AudioFeatures {
        features,
        mask,
        frames,
        soft_tokens,
    })
}

/// Resample by windowed-sinc interpolation. Downsampling lowers the cutoff
/// to the new Nyquist frequency so content above it does not fold back.
pub fn resample(input: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || input.is_empty() {
        return input.to_vec();
    }
    const ZERO_CROSSINGS: f64 = 16.0;
    let ratio = to_hz as f64 / from_hz as f64;
    let cutoff = ratio.min(1.0);
    let half_width = ZERO_CROSSINGS / cutoff;
    let out_len = ((input.len() as f64) * ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let center = i as f64 / ratio;
        let lo = (center - half_width).ceil().max(0.0) as usize;
        let hi = ((center + half_width).floor() as usize).min(input.len() - 1);
        let mut acc = 0f64;
        for (k, x) in input.iter().enumerate().take(hi + 1).skip(lo) {
            let d = k as f64 - center;
            let x_sinc = std::f64::consts::PI * d * cutoff;
            let sinc = if x_sinc.abs() < 1e-9 {
                1.0
            } else {
                x_sinc.sin() / x_sinc
            };
            let hann = 0.5 + 0.5 * (std::f64::consts::PI * d / half_width).cos();
            acc += *x as f64 * sinc * hann * cutoff;
        }
        out.push(acc as f32);
    }
    out
}

/// Decode the first `MAX_AUDIO_SECONDS` of an audio file to mono 16 kHz.
pub fn decode_audio_mono_16k(path: &Path) -> Result<Vec<f32>, String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).map_err(|e| format!("open audio: {e}"))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("unrecognized audio format: {e}"))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("no decodable audio track")?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("unsupported audio codec: {e}"))?;

    let mut mono: Vec<f32> = Vec::new();
    let mut source_rate: u32 = track.codec_params.sample_rate.unwrap_or(0);
    let mut wanted = usize::MAX;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // End of stream is reported as an I/O error; a reset means the
            // stream changed shape, which a one-shot read just stops at.
            Err(SymphoniaError::IoError(_)) | Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(format!("read audio: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // A corrupt packet is skipped, as a player would.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(_)) | Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(format!("decode audio: {e}")),
        };
        let spec = *decoded.spec();
        let channels = spec.channels.count().max(1);
        if source_rate == 0 {
            source_rate = spec.rate;
        }
        if wanted == usize::MAX && source_rate > 0 {
            wanted = MAX_AUDIO_SECONDS * source_rate as usize;
        }
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        for frame in buf.samples().chunks_exact(channels) {
            mono.push(frame.iter().sum::<f32>() / channels as f32);
        }
        if mono.len() >= wanted {
            mono.truncate(wanted);
            break;
        }
    }
    if mono.is_empty() || source_rate == 0 {
        return Err("audio file holds no samples".into());
    }
    Ok(resample(&mono, source_rate, SAMPLE_RATE as u32))
}

pub fn load_audio_features(path: &Path) -> Result<AudioFeatures, String> {
    guarded("audio", || mel_features(&decode_audio_mono_16k(path)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generator the fixture's inputs were made with (see
    /// `tests/fixtures/gemma2_preprocessing.mjs`).
    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.0
        }
    }

    fn fixture() -> serde_json::Value {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma2_preprocessing.json");
        serde_json::from_str(&std::fs::read_to_string(path).expect("fixture present"))
            .expect("fixture parses")
    }

    fn floats(v: &serde_json::Value) -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    }

    fn reference_wave(samples: usize) -> Vec<f32> {
        let mut rng = Lcg(samples as u32);
        (0..samples)
            .map(|i| {
                let noise = rng.next() as f64 / 4_294_967_296.0 * 2.0 - 1.0;
                let t = i as f64 / 16_000.0;
                (0.5 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                    + 0.25 * (2.0 * std::f64::consts::PI * 3100.0 * t).sin()
                    + 0.05 * noise) as f32
            })
            .collect()
    }

    /// Frame count, padding mask, token count and the mel values themselves
    /// must match the reference feature extractor. A mismatch here means the
    /// audio encoder is being fed something it was never trained on.
    #[test]
    fn mel_features_match_the_reference_extractor() {
        let fx = fixture();
        for case in fx["audio"].as_array().unwrap() {
            let samples = case["samples"].as_u64().unwrap() as usize;
            let got = mel_features(&reference_wave(samples)).expect("features");
            assert_eq!(
                got.frames as u64,
                case["frames"].as_u64().unwrap(),
                "frames for {samples}"
            );
            assert_eq!(
                got.soft_tokens as u64,
                case["tokens"].as_u64().unwrap(),
                "tokens for {samples}"
            );
            let mask: Vec<bool> = case["mask"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| m.as_u64().unwrap() == 1)
                .collect();
            assert_eq!(got.mask, mask, "mask for {samples}");

            let row = |f: usize| &got.features[f * MEL_BINS..(f + 1) * MEL_BINS];
            let mid = case["mid_index"].as_u64().unwrap() as usize;
            for (name, frame, want) in [
                ("first", 0, floats(&case["frame0"])),
                ("middle", mid, floats(&case["frame_mid"])),
                ("last", got.frames - 1, floats(&case["frame_last"])),
            ] {
                for (bin, (g, w)) in row(frame).iter().zip(want.iter()).enumerate() {
                    assert!(
                        (*g as f64 - w).abs() < 2e-3,
                        "{samples} samples, {name} frame, mel bin {bin}: got {g}, reference {w}"
                    );
                }
            }
            let sum: f64 = got.features.iter().map(|v| *v as f64).sum();
            let want_sum = case["sum"].as_f64().unwrap();
            assert!(
                (sum - want_sum).abs() < 1e-3 * want_sum.abs().max(1.0),
                "{samples} samples: feature sum {sum} vs reference {want_sum}"
            );
        }
    }

    #[test]
    fn clips_too_short_for_one_token_are_an_error_not_a_panic() {
        assert!(mel_features(&[]).is_err());
        assert!(mel_features(&[0.1; 100]).is_err());
        // Longer than 30 s is cut, not rejected: 750 tokens at 25 per second.
        let long = vec![0.01f32; 31 * SAMPLE_RATE];
        assert_eq!(mel_features(&long).unwrap().soft_tokens, 750);
    }

    fn reference_image(w: u32, h: u32) -> image::RgbImage {
        let mut rng = Lcg(w.wrapping_mul(100_003).wrapping_add(h));
        let data: Vec<u8> = (0..w * h * 3).map(|_| (rng.next() >> 24) as u8).collect();
        image::RgbImage::from_raw(w, h, data).unwrap()
    }

    /// Resize target, patch count, soft-token count and position ids must
    /// match the reference image processor for every aspect ratio, including
    /// the degenerate ones; where no resize happens the pixels must be
    /// identical too.
    #[test]
    fn image_patches_match_the_reference_processor() {
        let fx = fixture();
        for case in fx["image"].as_array().unwrap() {
            let (w, h) = (
                case["w"].as_u64().unwrap() as u32,
                case["h"].as_u64().unwrap() as u32,
            );
            let tag = format!("{w}x{h}");
            let (th, tw) =
                target_size(h as usize, w as usize, DEFAULT_IMAGE_TOKENS * POOL * POOL).unwrap();
            assert_eq!(
                tw as u64,
                case["target_w"].as_u64().unwrap(),
                "target width of {tag}"
            );
            assert_eq!(
                th as u64,
                case["target_h"].as_u64().unwrap(),
                "target height of {tag}"
            );

            let got = rgb_to_patches(reference_image(w, h), DEFAULT_IMAGE_TOKENS).unwrap();
            assert_eq!(got.max_patches, 2520);
            assert_eq!(got.pixel_values.len(), 2520 * PATCH_DIM);
            assert_eq!(
                got.patches as u64,
                case["patches"].as_u64().unwrap(),
                "patches of {tag}"
            );
            assert_eq!(
                got.soft_tokens as u64,
                case["soft_tokens"].as_u64().unwrap(),
                "soft tokens of {tag}"
            );

            let ints = |v: &serde_json::Value| -> Vec<i64> {
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect()
            };
            assert_eq!(
                &got.position_ids[..8],
                &ints(&case["pos_head"])[..],
                "first positions of {tag}"
            );
            let last = got.patches - 1;
            assert_eq!(
                &got.position_ids[last * 2..last * 2 + 2],
                &ints(&case["pos_tail_real"])[..],
                "{tag}"
            );
            if got.patches < got.max_patches {
                assert_eq!(
                    &got.position_ids[got.patches * 2..got.patches * 2 + 2],
                    &[-1, -1],
                    "padding of {tag}"
                );
            }

            // An untouched image must agree value for value. A resized one is
            // compared on overall brightness: within 0.2% when shrinking or
            // enlarging mildly. Under heavy enlargement the reference (sharp's
            // affine transform) blends the outer source pixels with a black
            // background, so it comes out up to ~2.5% darker than a resize
            // that replicates the edge, as this one and Pillow do.
            let sum: f64 = got.pixel_values.iter().map(|v| *v as f64).sum();
            let want_sum = case["sum"].as_f64().unwrap();
            let resized = (th, tw) != (h as usize, w as usize);
            let enlargement = (tw as f64 / w as f64).max(th as f64 / h as f64);
            let tolerance = match (resized, enlargement > 2.0) {
                (false, _) => 1e-6,
                (true, false) => 0.002,
                (true, true) => 0.04,
            };
            assert!(
                (sum - want_sum).abs() <= tolerance * want_sum,
                "pixel sum of {tag}: got {sum}, reference {want_sum}"
            );
            if !resized {
                for (i, (g, w)) in got
                    .pixel_values
                    .iter()
                    .zip(floats(&case["first_patch"]))
                    .enumerate()
                {
                    assert!((*g as f64 - w).abs() < 1e-6, "{tag} first patch value {i}");
                }
                let start = last * PATCH_DIM;
                for (i, (g, w)) in got.pixel_values[start..]
                    .iter()
                    .zip(floats(&case["last_real_patch"]))
                    .enumerate()
                {
                    assert!((*g as f64 - w).abs() < 1e-6, "{tag} last patch value {i}");
                }
            }
        }
    }

    #[test]
    fn smaller_token_budgets_shrink_the_resize_target() {
        let img = reference_image(640, 480);
        let small = rgb_to_patches(img.clone(), 70).unwrap();
        let large = rgb_to_patches(img, 280).unwrap();
        assert!(small.soft_tokens <= 70 && small.soft_tokens > 50);
        assert!(large.soft_tokens <= 280 && large.soft_tokens > 240);
        assert_eq!(small.max_patches, 630);
    }

    /// A transparent icon must keep its shape: fully transparent pixels become
    /// white whatever colour is stored underneath them.
    #[test]
    fn transparency_is_flattened_onto_white() {
        let mut rgba = image::RgbaImage::new(2, 1);
        rgba.put_pixel(0, 0, image::Rgba([10, 20, 30, 0]));
        rgba.put_pixel(1, 0, image::Rgba([10, 20, 30, 255]));
        let flat = flatten_on_white(&image::DynamicImage::ImageRgba8(rgba));
        assert_eq!(flat.get_pixel(0, 0).0, [255, 255, 255]);
        assert_eq!(flat.get_pixel(1, 0).0, [10, 20, 30]);
    }

    /// End to end through the real decoders: a PNG and a WAV written to disk
    /// come back as encoder inputs of the right shape.
    #[test]
    fn files_on_disk_decode_into_encoder_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let png = dir.path().join("square.png");
        reference_image(100, 60).save(&png).unwrap();
        let patches = load_image_patches(&png, DEFAULT_IMAGE_TOKENS).unwrap();
        assert!(patches.soft_tokens > 200);

        // 0.5 s of 44.1 kHz stereo, 16-bit PCM.
        let wav = dir.path().join("tone.wav");
        let (rate, channels, seconds) = (44_100u32, 2u16, 0.5f64);
        let frames = (rate as f64 * seconds) as usize;
        let mut pcm: Vec<u8> = Vec::new();
        for i in 0..frames {
            let s = ((2.0 * std::f64::consts::PI * 440.0 * i as f64 / rate as f64).sin() * 12_000.0)
                as i16;
            for _ in 0..channels {
                pcm.extend_from_slice(&s.to_le_bytes());
            }
        }
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
        bytes.extend_from_slice(&(channels * 2).to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&pcm);
        std::fs::write(&wav, bytes).unwrap();

        let wave = decode_audio_mono_16k(&wav).unwrap();
        assert!(
            (wave.len() as i64 - 8000).abs() <= 2,
            "0.5 s at 16 kHz, got {}",
            wave.len()
        );
        // The tone survives resampling at its amplitude (12000 / 32768).
        let peak = wave[1000..7000].iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!((peak - 0.366).abs() < 0.02, "peak {peak}");
        let features = load_audio_features(&wav).unwrap();
        // 49 whole frames in 0.5 s, one token per four of them.
        assert_eq!(features.soft_tokens, 13);

        assert!(load_image_patches(&wav, DEFAULT_IMAGE_TOKENS).is_err());
        assert!(decode_audio_mono_16k(&png).is_err());
    }

    fn wav_bytes(rate: u32, channels: u16, samples: usize) -> Vec<u8> {
        let pcm = vec![0x10u8; samples * channels as usize * 2];
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate.wrapping_mul(channels as u32 * 2)).to_le_bytes());
        bytes.extend_from_slice(&(channels * 2).to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&pcm);
        bytes
    }

    /// Headers a healthy file never has. Whatever the decoder makes of them —
    /// including an assertion failure inside it — the caller gets a `Result`.
    #[test]
    fn hostile_headers_are_errors_never_panics() {
        let dir = tempfile::tempdir().unwrap();
        for (name, bytes) in [
            ("rate0.wav", wav_bytes(0, 1, 4000)),
            ("channels0.wav", wav_bytes(16_000, 0, 4000)),
            ("rate_max.wav", wav_bytes(u32::MAX, 2, 4000)),
            ("empty.wav", Vec::new()),
            ("riff_only.wav", b"RIFF".to_vec()),
        ] {
            let path = dir.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            // The call returning at all is the assertion; an `Ok` is fine too.
            let _ = load_audio_features(&path);
            let _ = load_image_patches(&path, DEFAULT_IMAGE_TOKENS);
        }
        assert!(guarded::<()>("test", || panic!("decoder bug")).is_err());
        assert_eq!(guarded("test", || Ok(7)), Ok(7));
    }
}
