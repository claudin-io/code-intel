"""Builds the stand-in EmbeddingGemma 2 used by tests/gemma2_synthetic.rs.

Three tiny ONNX graphs and a tokenizer with the *same contract* as the real
model (same file names, input and output names, dtypes, dynamic shapes,
external weight files, placeholder tokens) and arithmetic simple enough to
reason about. They let the tests drive the real loading and inference code
through ONNX Runtime without downloading 470 MB of weights:

  model_q4.onnx           input_ids, attention_mask, {image,audio,video}_features
                          -> last_hidden_state, sentence_embedding
                          Token embeddings, with each placeholder token's row
                          replaced by the next soft token of its modality;
                          masked mean pooling; projection to 768; L2 norm.
  vision_encoder_q4.onnx  pixel_values, pixel_position_ids -> image_features
                          One soft token per 9 real patches.
  audio_encoder_q4.onnx   input_features, input_features_mask -> audio_features
                          One soft token per 4 frames, real frames only.

They say nothing about retrieval quality. That is tests/gemma2_e2e.rs, which
runs the real weights.

    pip install onnx numpy tokenizers && python generate.py
"""
import json
import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nph
from tokenizers import Tokenizer, models, pre_tokenizers, processors, AddedToken

HIDDEN, OUT = 8, 768
rng = np.random.default_rng(7)

# ── tokenizer ───────────────────────────────────────────────────────────
specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<|image>", "<image|>", "<|image|>",
            "<|audio>", "<audio|>", "<|audio|>", "<|video|>"]
chars = list("abcdefghijklmnopqrstuvwxyz0123456789_:|.,(){}[]<>=+-*/&!?'\"#%;")
vocab = {t: i for i, t in enumerate(specials)}
for c in chars:
    vocab.setdefault(c, len(vocab))
for c in chars:
    vocab.setdefault("##" + c, len(vocab))
tok = Tokenizer(models.WordPiece(vocab, unk_token="<unk>", max_input_chars_per_word=200))
tok.normalizer = None
from tokenizers import normalizers
tok.normalizer = normalizers.Lowercase()
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.add_special_tokens([AddedToken(t, special=True) for t in specials])
tok.post_processor = processors.TemplateProcessing(
    single="<bos> $A", special_tokens=[("<bos>", vocab["<bos>"])])
tok.save("tokenizer.json")
V = len(vocab)
IMG, AUD, VID = vocab["<|image|>"], vocab["<|audio|>"], vocab["<|video|>"]


def save(graph, name):
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    # Weights go to `<name>_data`, as in the real export (small shape constants
    # must stay inline: shape inference reads them).
    onnx.save_model(model, name, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=name + "_data", size_threshold=1024)


def const(name, arr):
    return nph.from_array(np.asarray(arr), name)


# ── text model ──────────────────────────────────────────────────────────
nodes, inits = [], []
inits += [
    const("E", rng.standard_normal((V, HIDDEN)).astype(np.float32)),
    const("W", rng.standard_normal((HIDDEN, OUT)).astype(np.float32)),
    const("flat_shape", np.array([-1, HIDDEN], np.int64)),
    const("minus1", np.array([-1], np.int64)),
    const("axis2", np.array([2], np.int64)),
    const("axis1", np.array([1], np.int64)),
    const("eps", np.array(1e-9, np.float32)),
]
nodes += [
    h.make_node("Gather", ["E", "input_ids"], ["emb"]),
    h.make_node("Shape", ["emb"], ["emb_shape"]),
    h.make_node("Reshape", ["emb", "flat_shape"], ["flat0"]),
    h.make_node("Reshape", ["input_ids", "minus1"], ["ids_flat"]),
]
prev = "flat0"
for i, (name, token) in enumerate([("image", IMG), ("audio", AUD), ("video", VID)]):
    inits.append(const(f"{name}_id", np.array(token, np.int64)))
    nodes += [
        h.make_node("Equal", ["ids_flat", f"{name}_id"], [f"{name}_is"]),
        h.make_node("NonZero", [f"{name}_is"], [f"{name}_nz"]),          # [1, N]
        h.make_node("Transpose", [f"{name}_nz"], [f"{name}_idx"], perm=[1, 0]),  # [N, 1]
        h.make_node("ScatterND", [prev, f"{name}_idx", f"{name}_features"], [f"flat{i + 1}"]),
    ]
    prev = f"flat{i + 1}"
nodes += [
    h.make_node("Reshape", [prev, "emb_shape"], ["last_hidden_state"]),
    h.make_node("Cast", ["attention_mask"], ["mask_f"], to=T.FLOAT),
    h.make_node("Unsqueeze", ["mask_f", "axis2"], ["mask_3d"]),
    h.make_node("Mul", ["last_hidden_state", "mask_3d"], ["masked"]),
    h.make_node("ReduceSum", ["masked", "axis1"], ["summed"], keepdims=0),
    h.make_node("ReduceSum", ["mask_3d", "axis1"], ["count"], keepdims=0),
    h.make_node("Add", ["count", "eps"], ["count_eps"]),
    h.make_node("Div", ["summed", "count_eps"], ["pooled"]),
    h.make_node("MatMul", ["pooled", "W"], ["projected"]),
    h.make_node("LpNormalization", ["projected"], ["sentence_embedding"], axis=1, p=2),
]
save(h.make_graph(
    nodes, "embedding_gemma2_text",
    [h.make_tensor_value_info("input_ids", T.INT64, ["batch", "seq"]),
     h.make_tensor_value_info("attention_mask", T.INT64, ["batch", "seq"]),
     h.make_tensor_value_info("image_features", T.FLOAT, ["num_image_tokens", HIDDEN]),
     h.make_tensor_value_info("audio_features", T.FLOAT, ["num_audio_tokens", HIDDEN]),
     h.make_tensor_value_info("video_features", T.FLOAT, ["num_video_tokens", HIDDEN])],
    [h.make_tensor_value_info("last_hidden_state", T.FLOAT, ["batch", "seq", HIDDEN]),
     h.make_tensor_value_info("sentence_embedding", T.FLOAT, ["batch", OUT])],
    inits), "model_q4.onnx")

# ── vision encoder ──────────────────────────────────────────────────────
PATCH_DIM, POOL2 = 768, 9
inits = [
    const("Wv", (rng.standard_normal((PATCH_DIM, HIDDEN)) / 8).astype(np.float32)),
    const("zero", np.array([0], np.int64)), const("one", np.array([1], np.int64)),
    const("nine", np.array([POOL2], np.int64)), const("zero_s", np.array(0, np.int64)),
    const("pool_shape", np.array([-1, POOL2, HIDDEN], np.int64)),
    const("ax0", np.array([0], np.int64)), const("ax1", np.array([1], np.int64)),
]
nodes = [
    h.make_node("Squeeze", ["pixel_values", "ax0"], ["px"]),                 # [P, 768]
    h.make_node("MatMul", ["px", "Wv"], ["proj"]),                           # [P, H]
    h.make_node("Squeeze", ["pixel_position_ids", "ax0"], ["pos"]),          # [P, 2]
    h.make_node("Slice", ["pos", "zero", "one", "ax1"], ["pos_x"]),          # [P, 1]
    h.make_node("GreaterOrEqual", ["pos_x", "zero_s"], ["valid"]),
    h.make_node("Cast", ["valid"], ["valid_i"], to=T.INT64),
    h.make_node("ReduceSum", ["valid_i"], ["k"], keepdims=0),               # real patches
    h.make_node("Reshape", ["k", "one"], ["k1"]),
    h.make_node("Div", ["k1", "nine"], ["n"]),                               # soft tokens
    h.make_node("Mul", ["n", "nine"], ["used"]),
    h.make_node("Slice", ["proj", "zero", "used", "ax0"], ["real"]),
    h.make_node("Reshape", ["real", "pool_shape"], ["grouped"]),
    h.make_node("ReduceMean", ["grouped"], ["image_features"], axes=[1], keepdims=0),
]
save(h.make_graph(
    nodes, "embedding_gemma2_vision",
    [h.make_tensor_value_info("pixel_values", T.FLOAT, ["batch", "patches", PATCH_DIM]),
     h.make_tensor_value_info("pixel_position_ids", T.INT64, ["batch", "patches", 2])],
    [h.make_tensor_value_info("image_features", T.FLOAT, ["num_image_tokens", HIDDEN])],
    inits), "vision_encoder_q4.onnx")

# ── audio encoder ───────────────────────────────────────────────────────
MEL, STRIDE = 128, 4
inits = [
    const("Wa", (rng.standard_normal((MEL, HIDDEN)) / 8).astype(np.float32)),
    const("zero", np.array([0], np.int64)), const("big", np.array([2**62], np.int64)),
    const("four", np.array([STRIDE], np.int64)), const("one", np.array([1], np.int64)),
    const("ax0", np.array([0], np.int64)),
]
nodes = [
    h.make_node("Squeeze", ["input_features", "ax0"], ["feats"]),            # [T, 128]
    h.make_node("MatMul", ["feats", "Wa"], ["proj"]),                        # [T, H]
    h.make_node("Slice", ["proj", "zero", "big", "ax0", "four"], ["strided"]),
    h.make_node("Squeeze", ["input_features_mask", "ax0"], ["mask"]),        # [T]
    h.make_node("Cast", ["mask"], ["mask_i"], to=T.INT64),
    h.make_node("Slice", ["mask_i", "zero", "big", "ax0", "four"], ["mask_s"]),
    h.make_node("ReduceSum", ["mask_s"], ["n"], keepdims=0),
    h.make_node("Reshape", ["n", "one"], ["n1"]),
    h.make_node("Slice", ["strided", "zero", "n1", "ax0"], ["audio_features"]),
]
save(h.make_graph(
    nodes, "embedding_gemma2_audio",
    [h.make_tensor_value_info("input_features", T.FLOAT, ["batch", "frames", MEL]),
     h.make_tensor_value_info("input_features_mask", T.BOOL, ["batch", "frames"])],
    [h.make_tensor_value_info("audio_features", T.FLOAT, ["num_audio_tokens", HIDDEN])],
    inits), "audio_encoder_q4.onnx")

print(json.dumps({"vocab": V, "image": IMG, "audio": AUD, "hidden": HIDDEN}))
