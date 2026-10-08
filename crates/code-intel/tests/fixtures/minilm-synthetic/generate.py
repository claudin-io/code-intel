"""Builds the stand-in MiniLM used by tests/gemma2_synthetic.rs.

One tiny ONNX graph and a tokenizer with the contract of the real
all-MiniLM-L6-v2 export (same file names, `input_ids` / `attention_mask` in,
per-token hidden states of size 384 out). With it the tests can load the
default pairing — MiniLM for text, EmbeddingGemma 2 beside it for media —
through ONNX Runtime without downloading either model.

A token's hidden state is its embedding and nothing else, so a text's vector
is a bag of its characters: strings that share letters score high. That is
enough to tell one function from another, and says nothing about the real
model's retrieval quality.

    pip install onnx numpy tokenizers && python generate.py
"""
import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nph
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, AddedToken

INNER, HIDDEN = 16, 384
rng = np.random.default_rng(11)

specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
chars = list("abcdefghijklmnopqrstuvwxyz0123456789_:|.,(){}[]<>=+-*/&!?'\"#%;")
vocab = {t: i for i, t in enumerate(specials)}
for c in chars:
    vocab.setdefault(c, len(vocab))
for c in chars:
    vocab.setdefault("##" + c, len(vocab))
tok = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=200))
tok.normalizer = normalizers.Lowercase()
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.add_special_tokens([AddedToken(t, special=True) for t in specials])
tok.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
tok.save("tokenizer.json")

# `##x` shares most of `x`'s embedding: where a letter sits in a word should
# matter less than which letter it is.
base = rng.standard_normal((len(specials) + len(chars), INNER)).astype(np.float32)
cont = base[len(specials):] + 0.2 * rng.standard_normal((len(chars), INNER)).astype(np.float32)
E = np.concatenate([base, cont])
E[:len(specials)] *= 0.05  # the frame tokens are in every text
W = rng.standard_normal((INNER, HIDDEN)).astype(np.float32)

graph = h.make_graph(
    [h.make_node("Gather", ["E", "input_ids"], ["emb"]),
     h.make_node("MatMul", ["emb", "W"], ["last_hidden_state"])],
    "minilm_stand_in",
    [h.make_tensor_value_info("input_ids", T.INT64, ["batch", "seq"]),
     h.make_tensor_value_info("attention_mask", T.INT64, ["batch", "seq"])],
    [h.make_tensor_value_info("last_hidden_state", T.FLOAT, ["batch", "seq", HIDDEN])],
    [nph.from_array(E, "E"), nph.from_array(W, "W")])
model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save_model(model, "model_quantized.onnx")
print({"vocab": len(vocab), "hidden": HIDDEN})
