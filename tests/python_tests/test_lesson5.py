#!/usr/bin/env python3
"""Lesson 5: AI Compiler Frontend — Model Construction & ONNX Export.

Two parallel paths are validated:

Path B (PyTorch + ONNX, Lesson 5 §3-§7) — exercises the Qwen3-0.6B reference
implementation (``joy/python/joy/onnx_export/qwen3_model.py``) and its ONNX
export pipeline (``joy/python/joy/onnx_export/export.py``):

  Test 1: PyTorch reference matches Qwen3 hyper-parameters
  Test 2: Sub-modules produce correct shapes (RMSNorm / MLP / Attention)
  Test 3: GQA + Q/K RMSNorm + RoPE intermediate shapes
  Test 4: Tied lm_head / embed_tokens
  Test 5: End-to-end forward shape  [B, S, V]
  Test 6: torch.onnx.export to joy/tests/onnx_model/qwen3_tiny.onnx
  Test 7: onnx.checker.check_model passes
  Test 8: Key ONNX operators (MatMul / Softmax / Sigmoid / Pow / Sqrt /
          ReduceMean / Gather / Reshape / Transpose / Expand) present
  Test 9: Dynamic axes (batch / seq) preserved
  Test 10: Tiny vs real-Qwen3 config sanity

Path A (Manual Graph, Lesson 5 §2) — exercises the zero-dependency Joy IR
builder (``joy/python/joy/builder/``):

  Test 11: ``joy.builder`` produces a syntactically well-formed Joy MLIR
           module for a Qwen3-tiny decoder block, with the expected op
           histogram and attribute serialization.

Usage:
    python3 joy/tests/python_tests/test_lesson5.py
    python3 joy/tests/python_tests/test_lesson5.py --keep-onnx --print-nodes
    python3 joy/tests/python_tests/test_lesson5.py --print-builder-ir
"""

import argparse
import os
import re
import sys
from collections import Counter

cur_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(cur_path, "../.."))
sys.path.insert(0, os.path.join(project_root, "python"))

import torch

from joy.onnx_export import (
    Qwen3Config, Qwen3Model, Qwen3DecoderLayer, Qwen3Attention,
    Qwen3MLP, Qwen3RMSNorm,
    apply_rotary_pos_emb, repeat_kv, build_rope_cache,
    TINY_QWEN3_CFG, REAL_QWEN3_06B_CFG, export_qwen3_to_onnx,
)

# Path A: zero-dependency Joy IR builder used by Lesson 5 §2 and reused
# as oracle by Lesson 6 ONNX Parser tests.
from joy.builder import Graph, ops
from joy.builder.ops.norm import fused_add_rms_norm


ONNX_OUT_DIR = os.path.join(project_root, "tests", "onnx_model")
TINY_ONNX_PATH = os.path.join(ONNX_OUT_DIR, "qwen3_tiny.onnx")


# ============================================================================
# Test 1: Qwen3 hyper-parameter constants
# ============================================================================
def test_qwen3_06b_config():
    print("\n" + "=" * 60)
    print("  Test 1: Real Qwen3-0.6B config (matches HF model card)")
    print("=" * 60)

    cfg = REAL_QWEN3_06B_CFG
    expected = {
        "vocab_size":           151936,
        "hidden_size":          1024,
        "num_hidden_layers":    28,
        "num_attention_heads":  16,
        "num_key_value_heads":  8,
        "head_dim":             128,
        "intermediate_size":    3072,
        "rms_norm_eps":         1e-6,
        "tie_word_embeddings":  True,
    }
    all_pass = True
    for k, v in expected.items():
        got = getattr(cfg, k)
        ok = got == v
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {k} = {got} (expected {v})")

    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    ok_gqa = n_rep == 2
    print(f"  {'PASS' if ok_gqa else 'FAIL'}  GQA n_rep = "
          f"{n_rep} (expected 2)")
    assert all_pass and ok_gqa, "[Lesson5] Test 1 failed"
    print("\n[Lesson5]: ================== Test 1 PASSED ==================")


# ============================================================================
# Test 2: Sub-module shape checks (RMSNorm / MLP)
# ============================================================================
def test_submodule_shapes():
    print("\n" + "=" * 60)
    print("  Test 2: RMSNorm / MLP sub-module shapes")
    print("=" * 60)

    cfg = TINY_QWEN3_CFG
    B, S = 2, 8

    rms = Qwen3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    x = torch.randn(B, S, cfg.hidden_size)
    y = rms(x)
    ok_rms = y.shape == x.shape
    print(f"  {'PASS' if ok_rms else 'FAIL'}  RMSNorm preserves shape "
          f"{list(y.shape)}")

    mlp = Qwen3MLP(cfg)
    z = mlp(x)
    ok_mlp = z.shape == x.shape
    print(f"  {'PASS' if ok_mlp else 'FAIL'}  MLP preserves outer shape "
          f"{list(z.shape)}")

    assert ok_rms and ok_mlp, "[Lesson5] Test 2 failed"
    print("\n[Lesson5]: ================== Test 2 PASSED ==================")


# ============================================================================
# Test 3: Attention intermediate shapes (GQA + Q/K Norm + RoPE)
# ============================================================================
def test_attention_intermediate_shapes():
    print("\n" + "=" * 60)
    print("  Test 3: Attention intermediate shapes")
    print("=" * 60)

    cfg = TINY_QWEN3_CFG
    B, S = 2, 8
    attn = Qwen3Attention(cfg)

    x = torch.randn(B, S, cfg.hidden_size)
    cos, sin = build_rope_cache(S, cfg.head_dim, theta=cfg.rope_theta)
    cos = cos.unsqueeze(0).expand(B, S, cfg.head_dim).contiguous()
    sin = sin.unsqueeze(0).expand(B, S, cfg.head_dim).contiguous()

    q = attn.q_proj(x).view(B, S, cfg.num_attention_heads, cfg.head_dim)
    k = attn.k_proj(x).view(B, S, cfg.num_key_value_heads, cfg.head_dim)
    v = attn.v_proj(x).view(B, S, cfg.num_key_value_heads, cfg.head_dim)
    checks = [
        (list(q.shape) == [B, S, cfg.num_attention_heads, cfg.head_dim],
         f"q.view shape == [B, S, NH, D] -> {list(q.shape)}"),
        (list(k.shape) == [B, S, cfg.num_key_value_heads, cfg.head_dim],
         f"k.view shape == [B, S, NKV, D] -> {list(k.shape)}"),
    ]

    q = attn.q_norm(q).transpose(1, 2)
    k = attn.k_norm(k).transpose(1, 2)
    v = v.transpose(1, 2)
    q = apply_rotary_pos_emb(q, cos, sin)
    k = apply_rotary_pos_emb(k, cos, sin)
    checks.append((list(q.shape) ==
                   [B, cfg.num_attention_heads, S, cfg.head_dim],
                   f"after RoPE q.shape == [B, NH, S, D] -> {list(q.shape)}"))

    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    k_expanded = repeat_kv(k, n_rep)
    v_expanded = repeat_kv(v, n_rep)
    checks.append((list(k_expanded.shape) ==
                   [B, cfg.num_attention_heads, S, cfg.head_dim],
                   f"repeat_kv: K expanded -> {list(k_expanded.shape)}"))
    checks.append((list(v_expanded.shape) ==
                   [B, cfg.num_attention_heads, S, cfg.head_dim],
                   f"repeat_kv: V expanded -> {list(v_expanded.shape)}"))

    out = attn(x, cos, sin)
    checks.append((list(out.shape) == [B, S, cfg.hidden_size],
                   f"o_proj output shape -> {list(out.shape)}"))

    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson5] Test 3 failed"
    print("\n[Lesson5]: ================== Test 3 PASSED ==================")


# ============================================================================
# Test 4: Tied embedding / lm_head
# ============================================================================
def test_tied_embedding():
    print("\n" + "=" * 60)
    print("  Test 4: Tied embed_tokens / lm_head weight sharing")
    print("=" * 60)

    cfg = TINY_QWEN3_CFG
    model = Qwen3Model(cfg)
    tied = (model.lm_head.weight.data_ptr() ==
            model.embed_tokens.weight.data_ptr())
    print(f"  {'PASS' if tied else 'FAIL'}  lm_head.weight is "
          f"embed_tokens.weight (shared storage)")

    cfg_untied = Qwen3Config(**{**cfg.__dict__, "tie_word_embeddings": False})
    model2 = Qwen3Model(cfg_untied)
    untied = (model2.lm_head.weight.data_ptr() !=
              model2.embed_tokens.weight.data_ptr())
    print(f"  {'PASS' if untied else 'FAIL'}  untied config keeps the two "
          f"matrices separate")

    assert tied and untied, "[Lesson5] Test 4 failed"
    print("\n[Lesson5]: ================== Test 4 PASSED ==================")


# ============================================================================
# Test 5: End-to-end forward
# ============================================================================
def test_e2e_forward():
    print("\n" + "=" * 60)
    print("  Test 5: End-to-end forward (Qwen3-tiny)")
    print("=" * 60)

    cfg = TINY_QWEN3_CFG
    torch.manual_seed(0)
    model = Qwen3Model(cfg).eval()

    B, S = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (B, S), dtype=torch.long)
    cos, sin = build_rope_cache(S, cfg.head_dim, theta=cfg.rope_theta)
    cos = cos.unsqueeze(0).expand(B, S, cfg.head_dim).contiguous()
    sin = sin.unsqueeze(0).expand(B, S, cfg.head_dim).contiguous()

    with torch.no_grad():
        logits = model(ids, cos, sin)

    ok_shape = list(logits.shape) == [B, S, cfg.vocab_size]
    ok_finite = torch.isfinite(logits).all().item()
    print(f"  {'PASS' if ok_shape else 'FAIL'}  logits shape "
          f"== [B={B}, S={S}, V={cfg.vocab_size}] -> {list(logits.shape)}")
    print(f"  {'PASS' if ok_finite else 'FAIL'}  logits are all finite "
          f"(no NaN/Inf)")
    assert ok_shape and ok_finite, "[Lesson5] Test 5 failed"
    print("\n[Lesson5]: ================== Test 5 PASSED ==================")


# ============================================================================
# Test 6: ONNX export
# ============================================================================
def test_onnx_export():
    print("\n" + "=" * 60)
    print("  Test 6: ONNX Export (Qwen3-tiny -> tests/onnx_model/)")
    print("=" * 60)

    cfg = TINY_QWEN3_CFG
    path, _ = export_qwen3_to_onnx(cfg, TINY_ONNX_PATH,
                                   batch_size=1, seq_len=16,
                                   opset_version=17)

    ok_exists = os.path.exists(path)
    size = os.path.getsize(path) if ok_exists else 0
    ok_size = size > 0
    print(f"  {'PASS' if ok_exists else 'FAIL'}  ONNX file written to {path}")
    print(f"  {'PASS' if ok_size else 'FAIL'}  ONNX file size = "
          f"{size / 1024:.1f} KB")
    assert ok_exists and ok_size, "[Lesson5] Test 6 failed"
    print("\n[Lesson5]: ================== Test 6 PASSED ==================")
    return path


# ============================================================================
# Test 7: onnx.checker.check_model passes
# ============================================================================
def test_onnx_checker(path):
    print("\n" + "=" * 60)
    print("  Test 7: onnx.checker.check_model")
    print("=" * 60)

    import onnx
    model = onnx.load(path)
    try:
        onnx.checker.check_model(model)
        ok_checker = True
        err = None
    except Exception as e:
        ok_checker = False
        err = str(e)
    print(f"  {'PASS' if ok_checker else 'FAIL'}  onnx.checker passes"
          + (f" ({err})" if err else ""))

    opset = model.opset_import[0].version
    ok_opset = opset == 17
    print(f"  {'PASS' if ok_opset else 'FAIL'}  opset version = "
          f"{opset} (expected 17)")

    n_init = len(model.graph.initializer)
    ok_init = n_init > 0
    print(f"  {'PASS' if ok_init else 'FAIL'}  graph has "
          f"{n_init} weight initializers")

    assert ok_checker and ok_opset and ok_init, "[Lesson5] Test 7 failed"
    print("\n[Lesson5]: ================== Test 7 PASSED ==================")
    return model


# ============================================================================
# Test 8: Key ONNX operators present
# ============================================================================
def test_key_onnx_ops(onnx_model, print_nodes=False):
    print("\n" + "=" * 60)
    print("  Test 8: Key ONNX op_types appear in the graph")
    print("=" * 60)

    counts = Counter(n.op_type for n in onnx_model.graph.node)
    if print_nodes:
        print("  Full op_type histogram:")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {k}: {v}")

    cfg = TINY_QWEN3_CFG
    n_layers = cfg.num_hidden_layers
    n_rmsnorm = n_layers * 4 + 1   # in_ln + post_ln + q_norm + k_norm + final
    # MatMul: per layer Q/K/V/O + gate/up/down + 2 attn-MatMuls = 9
    # Plus 1 for lm_head (Qwen3 has tied embed but lm_head is still a MatMul)
    n_matmul_min = n_layers * 9

    # Strict presence checks (count > 0) for every algebra primitive we
    # expect a Qwen3-style network to expose at the ONNX level.
    presence = [
        ("MatMul",     "Linear projections / attention MatMuls"),
        ("Softmax",    "attention softmax"),
        ("Sigmoid",    "SiLU = x * Sigmoid(x)"),
        ("Pow",        "RMSNorm Pow(2)"),
        ("ReduceMean", "RMSNorm mean over last dim"),
        ("Sqrt",       "RMSNorm rsqrt(var + eps)"),
        ("Gather",     "Embedding (= Gather along axis=0)"),
        ("Reshape",    "view / repeat_kv reshape"),
        ("Transpose",  "head reordering"),
        ("Expand",     "repeat_kv expand"),
        ("Add",        "residual + Add"),
        ("Mul",        "RoPE / RMSNorm scale / GQA"),
    ]

    all_pass = True
    for op, src in presence:
        c = counts.get(op, 0)
        ok = c > 0
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {op}: {c}  [{src}]")

    # Cardinality checks
    ok_softmax = counts.get("Softmax", 0) == n_layers
    print(f"  {'PASS' if ok_softmax else 'FAIL'}  Softmax count = "
          f"{counts.get('Softmax', 0)} (expected num_layers={n_layers})")

    ok_sigmoid = counts.get("Sigmoid", 0) == n_layers
    print(f"  {'PASS' if ok_sigmoid else 'FAIL'}  Sigmoid count = "
          f"{counts.get('Sigmoid', 0)} (expected num_layers={n_layers})")

    ok_pow = counts.get("Pow", 0) == n_rmsnorm
    print(f"  {'PASS' if ok_pow else 'FAIL'}  Pow count = "
          f"{counts.get('Pow', 0)} (expected #RMSNorm={n_rmsnorm})")

    ok_matmul = counts.get("MatMul", 0) >= n_matmul_min
    print(f"  {'PASS' if ok_matmul else 'FAIL'}  MatMul count >= "
          f"{n_matmul_min}: got {counts.get('MatMul', 0)}")

    assert all_pass and ok_softmax and ok_sigmoid and ok_pow and ok_matmul, \
        "[Lesson5] Test 8 failed"
    print("\n[Lesson5]: ================== Test 8 PASSED ==================")


# ============================================================================
# Test 9: dynamic axes preserved
# ============================================================================
def test_dynamic_axes(onnx_model):
    print("\n" + "=" * 60)
    print("  Test 9: Dynamic axes (batch / seq) preserved in ONNX")
    print("=" * 60)

    def get_dims(value_info):
        return [(d.dim_value if d.dim_value > 0 else d.dim_param)
                for d in value_info.type.tensor_type.shape.dim]

    name2dims = {x.name: get_dims(x) for x in onnx_model.graph.input}
    out_dims = {x.name: get_dims(x) for x in onnx_model.graph.output}

    cases = [
        ("input_ids", ["batch", "seq"]),
        ("cos",       ["batch", "seq", TINY_QWEN3_CFG.head_dim]),
        ("sin",       ["batch", "seq", TINY_QWEN3_CFG.head_dim]),
    ]
    all_pass = True
    for name, expected in cases:
        got = name2dims[name]
        ok = got == expected
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  input {name} dims = "
              f"{got} (expected {expected})")

    expected_out = ["batch", "seq", TINY_QWEN3_CFG.vocab_size]
    got_out = out_dims["logits"]
    ok_out = got_out == expected_out
    if not ok_out:
        all_pass = False
    print(f"  {'PASS' if ok_out else 'FAIL'}  output logits dims = "
          f"{got_out} (expected {expected_out})")

    assert all_pass, "[Lesson5] Test 9 failed"
    print("\n[Lesson5]: ================== Test 9 PASSED ==================")


# ============================================================================
# Test 10: Tiny vs real Qwen3-0.6B structural equivalence
# ============================================================================
def test_tiny_vs_real_structure():
    print("\n" + "=" * 60)
    print("  Test 10: TINY_QWEN3_CFG mirrors real Qwen3-0.6B (structurally)")
    print("=" * 60)

    tiny = TINY_QWEN3_CFG
    real = REAL_QWEN3_06B_CFG
    same_structure_fields = [
        "rms_norm_eps",
        "tie_word_embeddings",
    ]
    proportional_fields = [
        ("num_attention_heads", "num_key_value_heads"),  # GQA ratio
    ]

    all_pass = True
    for k in same_structure_fields:
        ok = getattr(tiny, k) == getattr(real, k)
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {k}: "
              f"tiny={getattr(tiny, k)} == real={getattr(real, k)}")

    for big, small in proportional_fields:
        tiny_ratio = getattr(tiny, big) // getattr(tiny, small)
        real_ratio = getattr(real, big) // getattr(real, small)
        ok = tiny_ratio == real_ratio
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  GQA ratio "
              f"{big}/{small}: tiny={tiny_ratio}, real={real_ratio}")

    assert all_pass, "[Lesson5] Test 10 failed"
    print("\n[Lesson5]: ================== Test 10 PASSED ==================")


# ============================================================================
# Test 11: Manual Graph (Path A) — joy.builder constructs a Qwen3-tiny block
# ============================================================================
def _build_qwen3_tiny_layer_with_builder(cfg=TINY_QWEN3_CFG, batch=1, seq=16):
    """Replays the §2.4 example end-to-end and returns (graph, ir_text).

    Mirrors the same algebra as ``Qwen3DecoderLayer``:
        x = x + self_attn(input_layernorm(x), cos, sin)
        x = fused_add_rms_norm post-attn → MLP → residual add
    The output is a pure Joy Tensor IR module, no PyTorch / ONNX involved.
    """
    B, S = batch, seq
    H     = cfg.hidden_size
    NH    = cfg.num_attention_heads
    KVH   = cfg.num_key_value_heads
    D     = cfg.head_dim
    INTER = cfg.intermediate_size
    n_rep = NH // KVH

    g = Graph("qwen3_tiny_layer")
    x      = g.input([B, S, H],    "f16")
    cos    = g.input([B, S, D],    "f16")
    sin    = g.input([B, S, D],    "f16")
    w_in   = g.input([H],          "f16")
    w_q    = g.input([NH * D,  H], "f16")
    w_k    = g.input([KVH * D, H], "f16")
    w_v    = g.input([KVH * D, H], "f16")
    w_o    = g.input([H, NH * D],  "f16")
    w_qn   = g.input([D],          "f16")
    w_kn   = g.input([D],          "f16")
    w_post = g.input([H],          "f16")
    w_gate = g.input([INTER, H],   "f16")
    w_up   = g.input([INTER, H],   "f16")
    w_down = g.input([H, INTER],   "f16")

    # Attention sub-block
    h_norm = ops.rms_norm(x, w_in, epsilon=cfg.rms_norm_eps)
    q = ops.reshape(ops.linear(h_norm, w_q), [B, S, NH,  D])
    k = ops.reshape(ops.linear(h_norm, w_k), [B, S, KVH, D])
    v = ops.reshape(ops.linear(h_norm, w_v), [B, S, KVH, D])
    q = ops.rms_norm(q, w_qn, epsilon=cfg.rms_norm_eps)
    k = ops.rms_norm(k, w_kn, epsilon=cfg.rms_norm_eps)
    q = ops.transpose(q, [0, 2, 1, 3])
    k = ops.transpose(k, [0, 2, 1, 3])
    v = ops.transpose(v, [0, 2, 1, 3])
    q = ops.apply_rotary_emb(q, cos, sin)
    k = ops.apply_rotary_emb(k, cos, sin)
    k = ops.repeat_kv(k, n_rep=n_rep)
    v = ops.repeat_kv(v, n_rep=n_rep)
    scores  = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2]))
    weights = ops.softmax(scores, axis=-1)
    attn    = ops.matmul(weights, v)
    attn    = ops.transpose(attn, [0, 2, 1, 3])
    attn    = ops.reshape(attn, [B, S, NH * D])
    attn    = ops.linear(attn, w_o)

    # Residual + Post-Norm + MLP
    post, _ = fused_add_rms_norm(attn, x, w_post, epsilon=cfg.rms_norm_eps)
    gate    = ops.silu(ops.linear(post, w_gate))
    mlp_in  = ops.mul(gate, ops.linear(post, w_up))
    mlp_out = ops.linear(mlp_in, w_down)
    out     = ops.add(post, mlp_out)

    g.set_outputs([out])
    return g, g.get_ir()


def test_manual_graph_builder(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 11: Manual Graph (Path A) via joy.builder")
    print("=" * 60)

    cfg = TINY_QWEN3_CFG
    B, S = 1, 16
    g, ir = _build_qwen3_tiny_layer_with_builder(cfg, batch=B, seq=S)

    if print_ir:
        print("--- Generated Joy MLIR -------------------------------------")
        print(ir)
        print("------------------------------------------------------------")

    # 1) IR文本结构合法性
    structural = [
        ("module {",                                "MLIR module wrapper"),
        ("func.func @qwen3_tiny_layer",             "function definition"),
        ("return ",                                  "explicit return op"),
        (f"tensor<{B}x{S}x{cfg.hidden_size}xf16>",  "hidden tensor type"),
    ]
    all_pass = True
    for needle, desc in structural:
        ok = needle in ir
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  contains '{needle}'  [{desc}]")

    # 2) op histogram via Graph.get_op_stats()
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    stats = g.get_op_stats()
    expected = {
        # 3 RMSNorms: input_layernorm + q_norm + k_norm  (post-norm is fused
        # into joy.fused_add_rms_norm, so it doesn't count here).
        "joy.rms_norm":            3,
        # Linear projections: q,k,v,o (attention) + gate,up,down (MLP) = 7
        "joy.linear":              7,
        # MatMul: Q@K^T and (·)@V
        "joy.matmul":              2,
        "joy.softmax":             1,
        "joy.silu":                1,
        # 5 transposes: 3 for QKV head reorder, 1 inside scores, 1 to merge heads
        "joy.transpose":           5,
        # reshape: 3 for QKV view + 1 to merge heads back to [B, S, NH*D]
        "joy.reshape":             4,
        "joy.repeat_kv":           2,
        "joy.apply_rotary_emb":    2,
        "joy.fused_add_rms_norm":  1,
        # gate * up
        "joy.mul":                 1,
        # final residual add (residual1 is already inside fused_add_rms_norm)
        "joy.add":                 1,
    }
    for op, want in sorted(expected.items()):
        got = stats.get(op, 0)
        ok = got == want
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {op}: got {got} (expected {want})")

    # The op set matches exactly — no surprise extra ops sneak in.
    extra_ops = sorted(set(stats) - set(expected))
    ok_extra = len(extra_ops) == 0
    if not ok_extra:
        all_pass = False
    print(f"  {'PASS' if ok_extra else 'FAIL'}  no extra ops "
          f"(unexpected: {extra_ops})")

    # 3) Attribute serialization sanity (regex check)
    eps_pattern = r'epsilon = 1\.000000e-06 : f32'
    ok_eps = bool(re.search(eps_pattern, ir))
    print(f"  {'PASS' if ok_eps else 'FAIL'}  epsilon attr serialised "
          f"as '{eps_pattern}'")
    if not ok_eps:
        all_pass = False

    # softmax(axis=-1) on a rank-4 tensor [B, NH, S, S] => axis = 3 in IR.
    axis_pattern = r'axis = 3 : i64'
    ok_axis = bool(re.search(axis_pattern, ir))
    print(f"  {'PASS' if ok_axis else 'FAIL'}  softmax axis serialised "
          f"as '{axis_pattern}'")
    if not ok_axis:
        all_pass = False

    perm_pattern = r'permutation = dense<\[0, 2, 1, 3\]> : tensor<4xi64>'
    ok_perm = bool(re.search(perm_pattern, ir))
    print(f"  {'PASS' if ok_perm else 'FAIL'}  transpose permutation "
          f"serialised as '{perm_pattern}'")
    if not ok_perm:
        all_pass = False

    n_rep_pattern = rf'n_rep = {n_rep} : i64'
    ok_nrep = bool(re.search(n_rep_pattern, ir))
    print(f"  {'PASS' if ok_nrep else 'FAIL'}  repeat_kv n_rep serialised "
          f"as '{n_rep_pattern}'")
    if not ok_nrep:
        all_pass = False

    # 4) SSA value count sanity — every line in the body should produce a
    # uniquely numbered %N (excluding %argN which are inputs).
    body_lines = [ln for ln in ir.splitlines() if ln.startswith("    %")]
    ssa_ids = set()
    for ln in body_lines:
        m = re.match(r'    (%\d+(?:, %\d+)*)', ln)
        if m:
            for tok in m.group(1).split(", "):
                ssa_ids.add(tok)
    n_body_results = sum(
        len(re.match(r'    (%\d+(?:, %\d+)*)', ln).group(1).split(", "))
        for ln in body_lines if re.match(r'    (%\d+(?:, %\d+)*)', ln)
    )
    ok_ssa = len(ssa_ids) == n_body_results > 0
    print(f"  {'PASS' if ok_ssa else 'FAIL'}  SSA result names are unique "
          f"({len(ssa_ids)} distinct, {n_body_results} produced)")
    if not ok_ssa:
        all_pass = False

    assert all_pass, "[Lesson5] Test 11 failed"
    print("\n[Lesson5]: ================== Test 11 PASSED ==================")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Lesson 5 test runner")
    parser.add_argument("--keep-onnx", action="store_true",
                        help="Keep qwen3_tiny.onnx after running")
    parser.add_argument("--print-nodes", action="store_true",
                        help="Print full ONNX op_type histogram")
    parser.add_argument("--print-builder-ir", action="store_true",
                        help="Print the Joy MLIR text generated by joy.builder")
    args = parser.parse_args()

    print("=" * 60)
    print("  Lesson 5: AI Compiler Frontend — Model Construction & ONNX Export")
    print("=" * 60)

    # ---- Path B: PyTorch + ONNX ------------------------------------------
    test_qwen3_06b_config()
    test_submodule_shapes()
    test_attention_intermediate_shapes()
    test_tied_embedding()
    test_e2e_forward()
    path = test_onnx_export()
    onnx_model = test_onnx_checker(path)
    test_key_onnx_ops(onnx_model, print_nodes=args.print_nodes)
    test_dynamic_axes(onnx_model)
    test_tiny_vs_real_structure()

    if not args.keep_onnx and os.path.exists(path):
        # ONNX file is kept by default to make Netron exploration easy.
        pass

    # ---- Path A: Manual Graph builder ------------------------------------
    test_manual_graph_builder(print_ir=args.print_builder_ir)

    print("\n" + "=" * 60)
    print("  ALL LESSON 5 TESTS PASSED!")
    print(f"  Exported ONNX: {TINY_ONNX_PATH}")
    print(f"  Visualize via:  netron {TINY_ONNX_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
