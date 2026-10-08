// Regenerates gemma2_preprocessing.json: the output of the reference
// preprocessing (transformers.js) for deterministic inputs that
// src/media_prep.rs rebuilds bit for bit in its tests.
//
//   npm install --ignore-scripts @huggingface/transformers@4.3.1
//   node gemma2_preprocessing.mjs
//
// The configs are the values of processor_config.json in
// onnx-community/embeddinggemma-2-ONNX at the revision pinned in src/gemma2.rs.
import { Gemma4AudioFeatureExtractor, Gemma4ImageProcessor, RawImage } from '@huggingface/transformers';
import fs from 'node:fs';

// Deterministic generators reproduced bit-for-bit in the Rust tests.
function lcg(seed) { let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s; }; }

const fe = new Gemma4AudioFeatureExtractor({
  dither: 0.0, feature_extractor_type: 'Gemma4AudioFeatureExtractor', feature_size: 128, fft_length: 512,
  fft_overdrive: false, frame_length: 320, hop_length: 160, input_scale_factor: 1.0, max_frequency: 8000.0,
  mel_floor: 0.001, min_frequency: 0.0, padding_side: 'right', padding_value: 0.0, per_bin_mean: null,
  per_bin_stddev: null, preemphasis: 0.0, preemphasis_htk_flavor: true, return_attention_mask: true, sampling_rate: 16000,
});

const audio_cases = [];
for (const n of [9000, 4096, 16000, 333]) {
  const r = lcg(n);
  const wave = new Float32Array(n);
  for (let i = 0; i < n; ++i) {
    const noise = (r() / 4294967296) * 2 - 1;
    wave[i] = Math.fround(0.5 * Math.sin(2 * Math.PI * 440 * i / 16000) + 0.25 * Math.sin(2 * Math.PI * 3100 * i / 16000) + 0.05 * noise);
  }
  const out = await fe(wave);
  const [, frames, feats] = out.input_features.dims;
  const mask = Array.from(out.input_features_mask.data, Number);
  let tokens = 0; for (let i = 0; i < mask.length; i += 4) tokens += mask[i];
  const data = out.input_features.data;
  // Keep the fixture small: every feature of 3 frames + a checksum of everything.
  let sum = 0; for (const v of data) sum += v;
  const pick = (f) => Array.from(data.slice(f * feats, (f + 1) * feats));
  audio_cases.push({ samples: n, frames, feats, mask, tokens, sum,
    frame0: pick(0), frame_mid: pick(Math.floor(frames / 2)), mid_index: Math.floor(frames / 2), frame_last: pick(frames - 1) });
}

const ip = new Gemma4ImageProcessor({ do_convert_rgb: true, do_normalize: false, do_rescale: true, do_resize: true,
  max_soft_tokens: 280, patch_size: 16, pooling_kernel_size: 3, resample: 3, rescale_factor: 0.00392156862745098 });

const image_cases = [];
// Sizes that are already valid targets are not resized, so patches must match exactly.
for (const [w, h] of [[96, 48], [48, 144], [1920, 1080], [640, 480], [300, 200], [32, 32], [4000, 100], [17, 900], [768, 768], [800, 801]]) {
  const r = lcg(w * 100003 + h);
  const data = new Uint8ClampedArray(w * h * 3);
  for (let i = 0; i < data.length; ++i) data[i] = r() >>> 24;
  const out = await ip(new RawImage(data, w, h, 3));
  const pv = out.pixel_values.data, pos = out.image_position_ids.data;
  const n = out.num_soft_tokens_per_image[0];
  let real = 0; while (real < pos.length / 2 && pos[real * 2] !== -1n) ++real;
  let max_col = 0, max_row = 0;
  for (let i = 0; i < real; ++i) { max_col = Math.max(max_col, Number(pos[2 * i])); max_row = Math.max(max_row, Number(pos[2 * i + 1])); }
  let sum = 0; for (const v of pv) sum += v;
  image_cases.push({ w, h, dims: out.pixel_values.dims, soft_tokens: n, patches: real, target_w: (max_col + 1) * 16, target_h: (max_row + 1) * 16, sum,
    first_patch: Array.from(pv.slice(0, 24)), last_real_patch: Array.from(pv.slice((real - 1) * 768, (real - 1) * 768 + 24)),
    pos_head: Array.from(pos.slice(0, 8), Number), pos_tail_real: Array.from(pos.slice((real - 1) * 2, real * 2), Number), pos_first_pad: Array.from(pos.slice(real * 2, real * 2 + 2), Number) });
}
fs.writeFileSync('gemma2_preprocessing.json', JSON.stringify({ source: '@huggingface/transformers 4.3.1', audio: audio_cases, image: image_cases }));
console.log(audio_cases.map(c => [c.samples, c.frames, c.tokens, c.sum]), image_cases.map(c => [c.w, c.h, c.target_w, c.target_h, c.soft_tokens, c.patches]));
