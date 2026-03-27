#!/usr/bin/env python3
"""Lesson 15: Qwen3-0.6B end-to-end run on the joy GPU backend.

Lesson 15 walks through two phases:

Phase A (compilation pipeline) - lowers a Qwen3-0.6B graph through
  Joy (tensor)
    -> OpFusion (add + rms_norm -> fuse_add_rmsnorm)
    -> Joyl (memref)
    -> Codegen (rms_norm / fuse_add_rmsnorm -> GPU kernels)
    -> Joyh (GPU custom calls)
  and verifies that every operator type is present in each dialect and
  that the final Joyh IR uses the correct ``joy_gpu_*`` custom-call
  targets.  This is "drive the compiler"; no GPU is touched here.

Phase B (end-to-end inference) - actually runs the full Qwen3-0.6B
  network on the GPU backend by dispatching every operator through the
  ``joy_gpu_*`` extern "C" entry points exposed by
  ``libjoy_gpu_runtime.so`` (built from ``joy/lib/runtime/gpu`` +
  ``joy/lib/backend/gpu``).  Prefills a sentiment-classification prompt
  ("快乐") and greedy-decodes a few tokens; the run passes when the
  model emits "[正向]" within the decoded window.  See
  ``qwen3_gpu_runner.py`` for the executor.

Usage:
    # Both phases (compile pipeline + GPU inference)
    python3 tests/python_tests/test_lesson15.py fp16

    # Compilation pipeline only (the legacy mode)
    python3 tests/python_tests/test_lesson15.py fp16 --no-inference

    # GPU inference only, with a custom HF model path
    python3 tests/python_tests/test_lesson15.py fp16 --inference-only \\
                                                 --model-path /path/to/Qwen3-0.6B

    # Quick IR-only smoke check (NUM_LAYERS=1, ~1s, no GPU required)
    python3 tests/python_tests/test_lesson15.py fp16 --quick-ir-check

    # Custom prompt
    python3 tests/python_tests/test_lesson15.py fp16 \\
                                                 --prompt "悲伤" \\
                                                 --expect "[负向]"

    # Custom HF model path
    python3 tests/python_tests/test_lesson15.py fp16 \\
                                                 --model-path /path/to/Qwen3-0.6B
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

cur_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(cur_path, "../.."))
sys.path.insert(0, os.path.join(project_root, "python"))
sys.path.insert(0, cur_path)

from joy.builder import Graph
from joy.builder import ops

# ============================================================================
# Qwen3-0.6B model hyper-parameters
# ============================================================================
VOCAB_SIZE = 151936
HIDDEN_SIZE = 1024
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_SIZE = 3072
NUM_LAYERS = 28
RMS_NORM_EPS = 1e-6
NUM_KV_GROUPS = NUM_HEADS // NUM_KV_HEADS  # 2


# ============================================================================
# Attention layer
# ============================================================================
def build_self_attention(graph, hidden, layer_idx,
                         batch_size, seq_len, dtype, cos, sin):
    p = f"layer{layer_idx}.self_attn"

    q_w = graph.input([NUM_HEADS * HEAD_DIM, HIDDEN_SIZE],    dtype, name=f"{p}.q_proj.weight")
    k_w = graph.input([NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE], dtype, name=f"{p}.k_proj.weight")
    v_w = graph.input([NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE], dtype, name=f"{p}.v_proj.weight")

    q = ops.linear(hidden, q_w)
    k = ops.linear(hidden, k_w)
    v = ops.linear(hidden, v_w)

    q = ops.reshape(q, [batch_size, seq_len, NUM_HEADS,    HEAD_DIM])
    k = ops.reshape(k, [batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM])
    v = ops.reshape(v, [batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM])

    q_norm_w = graph.input([HEAD_DIM], dtype, name=f"{p}.q_norm.weight")
    k_norm_w = graph.input([HEAD_DIM], dtype, name=f"{p}.k_norm.weight")
    q = ops.rms_norm(q, q_norm_w, epsilon=RMS_NORM_EPS)
    k = ops.rms_norm(k, k_norm_w, epsilon=RMS_NORM_EPS)

    q = ops.transpose(q, [0, 2, 1, 3])
    k = ops.transpose(k, [0, 2, 1, 3])
    v = ops.transpose(v, [0, 2, 1, 3])

    q = ops.apply_rotary_emb(q, cos, sin)
    k = ops.apply_rotary_emb(k, cos, sin)

    k = ops.repeat_kv(k, n_rep=NUM_KV_GROUPS)
    v = ops.repeat_kv(v, n_rep=NUM_KV_GROUPS)

    k_t = ops.transpose(k, [0, 1, 3, 2])
    attn_weights = ops.matmul(q, k_t)
    attn_weights = ops.softmax(attn_weights, axis=-1)
    attn_out = ops.matmul(attn_weights, v)

    attn_out = ops.transpose(attn_out, [0, 2, 1, 3])
    attn_out = ops.reshape(attn_out, [batch_size, seq_len, NUM_HEADS * HEAD_DIM])

    o_w = graph.input([HIDDEN_SIZE, NUM_HEADS * HEAD_DIM], dtype, name=f"{p}.o_proj.weight")
    attn_out = ops.linear(attn_out, o_w)

    return attn_out


# ============================================================================
# MLP block
# ============================================================================
def build_mlp(graph, hidden, layer_idx, batch_size, seq_len, dtype):
    p = f"layer{layer_idx}.mlp"
    gate_w = graph.input([INTERMEDIATE_SIZE, HIDDEN_SIZE], dtype, name=f"{p}.gate_proj.weight")
    up_w   = graph.input([INTERMEDIATE_SIZE, HIDDEN_SIZE], dtype, name=f"{p}.up_proj.weight")
    down_w = graph.input([HIDDEN_SIZE, INTERMEDIATE_SIZE], dtype, name=f"{p}.down_proj.weight")

    gate = ops.linear(hidden, gate_w)
    gate = ops.silu(gate)
    up   = ops.linear(hidden, up_w)
    gate_up = ops.mul(gate, up)
    out  = ops.linear(gate_up, down_w)
    return out


# ============================================================================
# Decoder layer
# ============================================================================
def build_qwen3_06b_layer(graph, hidden, layer_idx,
                           batch_size, seq_len, dtype, cos, sin):
    p = f"layer{layer_idx}"

    residual = hidden
    ln_w = graph.input([HIDDEN_SIZE], dtype, name=f"{p}.input_layernorm.weight")
    hidden = ops.rms_norm(hidden, ln_w, epsilon=RMS_NORM_EPS)

    attn_out = build_self_attention(graph, hidden, layer_idx,
                                    batch_size, seq_len, dtype, cos, sin)

    hidden = ops.add(residual, attn_out)

    residual = hidden
    post_ln_w = graph.input([HIDDEN_SIZE], dtype,
                            name=f"{p}.post_attention_layernorm.weight")
    hidden = ops.rms_norm(hidden, post_ln_w, epsilon=RMS_NORM_EPS)

    mlp_out = build_mlp(graph, hidden, layer_idx, batch_size, seq_len, dtype)

    hidden = ops.add(residual, mlp_out)

    return hidden


# ============================================================================
# Full model
# ============================================================================
def build_qwen3_06b(graph, batch_size, seq_len, dtype, num_layers=None):
    if num_layers is None:
        num_layers = NUM_LAYERS

    input_ids = graph.input([batch_size, seq_len], "i64", name="input_ids")
    cos = graph.input([batch_size, seq_len, HEAD_DIM], dtype,
                      name="rotary_emb.cos")
    sin = graph.input([batch_size, seq_len, HEAD_DIM], dtype,
                      name="rotary_emb.sin")

    embed_w = graph.input([VOCAB_SIZE, HIDDEN_SIZE], dtype,
                          name="model.embed_tokens.weight")
    hidden = ops.embedding(input_ids, embed_w)

    for layer_idx in range(num_layers):
        hidden = build_qwen3_06b_layer(
            graph, hidden, layer_idx,
            batch_size, seq_len, dtype, cos, sin)

    final_ln_w = graph.input([HIDDEN_SIZE], dtype, name="model.norm.weight")
    hidden = ops.rms_norm(hidden, final_ln_w, epsilon=RMS_NORM_EPS)

    lm_head_w = graph.input([VOCAB_SIZE, HIDDEN_SIZE], dtype,
                            name="lm_head.weight")
    logits = ops.linear(hidden, lm_head_w)

    return logits


# ============================================================================
# joy-opt helpers
# ============================================================================
def _run_joy_opt(input_ir, passes, timeout=120):
    """Run joy-opt with the given passes on input_ir text, return stdout."""
    joy_opt_path = os.path.join(project_root, "build", "bin", "joy-opt")
    if not os.path.exists(joy_opt_path):
        print(f"  WARNING: joy-opt not found at {joy_opt_path}")
        return None

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mlir")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(input_ir)
        cmd = [joy_opt_path] + passes + [tmp_path]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout)
        if result.returncode != 0:
            print(f"  ERROR: joy-opt failed (rc={result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-15:]:
                    print(f"    {line}")
            return None
        return result.stdout
    except Exception as e:
        print(f"  ERROR: Failed to run joy-opt: {e}")
        return None
    finally:
        os.unlink(tmp_path)


def apply_op_fusion(joy_ir):
    """Apply OpFusion pass (add+rms_norm -> fuse_add_rmsnorm)."""
    return _run_joy_opt(joy_ir, ["--joy-op-fusion"])


def lower_joy_to_joyl(joy_ir):
    """Lower Joy dialect IR to Joyl dialect."""
    return _run_joy_opt(joy_ir, ["--lower-joy-to-joyl"])


def lower_joyl_to_joyh(joyl_ir):
    """Lower Joyl dialect IR to Joyh via codegen + custom calls."""
    return _run_joy_opt(joyl_ir,
                        ["--codegen-rms-norm", "--lower-joyl-to-joyh"])


# ============================================================================
# Phase A: compile-pipeline checks
# ============================================================================
def test_qwen3_06b_joy(batch_size, dtype, print_ir=False,
                       print_joy_opt_ir=False,
                       print_joyl_ir=False, print_joyh_ir=False,
                       print_ir_all=False,
                       num_layers=None,
                       seq_len=64):
    print("\n" + "=" * 60)
    print("  Testing Qwen3-0.6B Model -> Joy Dialect Conversion")
    n_layers_resolved = num_layers if num_layers is not None else NUM_LAYERS
    print(f"  batch_size={batch_size}, seq_len={seq_len}, dtype={dtype}")
    print(f"  layers={n_layers_resolved}, num_heads={NUM_HEADS}")
    print(f"  num_kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"  (GQA n_rep={NUM_KV_GROUPS})")
    print("=" * 60)

    graph = Graph(name="qwen3_06b")
    logits = build_qwen3_06b(graph, batch_size, seq_len=seq_len, dtype=dtype,
                             num_layers=num_layers)
    graph.set_outputs([logits])

    ir = graph.get_ir()

    # ---- Phase 1: Verify Joy dialect ----
    required_ops = [
        ("joy.embedding",        "embed_tokens"),
        ("joy.rms_norm",         "Qwen3RMSNorm"),
        ("joy.linear",           "q/k/v/o_proj, gate/up/down_proj, lm_head"),
        ("joy.reshape",          "view() in Qwen3Attention"),
        ("joy.transpose",        "transpose() in Qwen3Attention"),
        ("joy.apply_rotary_emb", "apply_rotary_pos_emb()"),
        ("joy.repeat_kv",        "repeat_kv()"),
        ("joy.matmul",           "Q@K^T and attn@V"),
        ("joy.softmax",          "softmax"),
        ("joy.silu",             "SiLU in Qwen3MLP"),
        ("joy.mul",              "act_fn(gate) * up_proj"),
        ("joy.add",              "residual connections"),
    ]

    print("\nVerifying Joy dialect operations:")
    all_pass = True
    for op_name, source in required_ops:
        count = ir.count(f'"{op_name}"(')
        ok = count > 0
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {op_name}: {count}  [{source}]")

    stats = graph.get_op_stats()
    print(f"\nOperation statistics:")
    for op_name, count in sorted(stats.items()):
        print(f"  {op_name}: {count}")

    if print_ir or print_ir_all:
        print("\n--- MLIR IR (joy dialect) ---")
        print(ir)
        print("--- end of MLIR IR (joy dialect) ---")

    assert all_pass, \
        "[JOY PyTest]: XXXXXXXXXXXXXXXXXX Joy dialect check failed XXXXXXXXXXXXXXXXXX\n"
    print(f"\n[JOY PyTest]: ================== Joy dialect compare success ==================")

    # ==================================================================
    # Phase 2: OpFusion (add + rms_norm -> fuse_add_rmsnorm)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Applying OpFusion (add + rms_norm -> fuse_add_rmsnorm)")
    print("=" * 60)

    fused_ir = apply_op_fusion(ir)
    if fused_ir is None:
        print("  SKIP: OpFusion not available")
        return

    fuse_count = fused_ir.count('"joy.fuse_add_rmsnorm"(')
    remaining_add = fused_ir.count('"joy.add"(')
    remaining_rms = fused_ir.count('"joy.rms_norm"(')

    print(f"\n  joy.fuse_add_rmsnorm : {fuse_count}")
    print(f"  remaining joy.add    : {remaining_add}")
    print(f"  remaining joy.rms_norm: {remaining_rms}")

    fusion_ok = fuse_count > 0 and remaining_add == 0
    print(f"\n  {'PASS' if fuse_count > 0 else 'FAIL'}  "
          f"fuse_add_rmsnorm ops created")
    print(f"  {'PASS' if remaining_add == 0 else 'FAIL'}  "
          f"all add ops fused (none remaining)")
    print(f"  {'PASS' if remaining_rms > 0 else 'FAIL'}  "
          f"standalone rms_norm ops remain (q_norm/k_norm/layer0)")

    if print_joy_opt_ir or print_ir_all:
        print("\n--- MLIR IR (joy dialect after OpFusion) ---")
        print(fused_ir)
        print("--- end of MLIR IR (joy dialect after OpFusion) ---")

    assert fusion_ok, \
        "[JOY PyTest]: XXXXXXXXXXXXXXXXXX OpFusion failed XXXXXXXXXXXXXXXXXX\n"
    print(f"\n[JOY PyTest]: ================== OpFusion success ==================")

    # ==================================================================
    # Phase 3: Lower Joy -> Joyl (tensor -> memref)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Lowering Joy -> Joyl (tensor -> memref)")
    print("=" * 60)

    joyl_ir = lower_joy_to_joyl(fused_ir)
    if joyl_ir is None:
        print("  SKIP: Joyl lowering not available")
        return

    joyl_required_ops = [
        ("joyl.embedding",          "embed_tokens"),
        ("joyl.rms_norm",           "standalone Qwen3RMSNorm"),
        ("joyl.fuse_add_rmsnorm",   "fused add+rms_norm"),
        ("joyl.linear",             "linear projections"),
        ("joyl.reshape",            "reshape"),
        ("joyl.transpose",          "transpose"),
        ("joyl.apply_rotary_emb",   "rotary embedding"),
        ("joyl.repeat_kv",          "repeat_kv"),
        ("joyl.matmul",             "matmul"),
        ("joyl.softmax",            "softmax"),
        ("joyl.silu",               "SiLU activation"),
        ("joyl.mul",                "element-wise mul"),
    ]

    print("\nVerifying Joyl dialect operations:")
    joyl_pass = True
    for op_name, source in joyl_required_ops:
        count = joyl_ir.count(f'"{op_name}"(')
        ok = count > 0
        if not ok:
            joyl_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {op_name}: {count}  [{source}]")

    has_memref = "memref<" in joyl_ir or "memref." in joyl_ir
    no_joyl_add = '"joyl.add"' not in joyl_ir
    print(f"\n  memref types present     : {'YES' if has_memref else 'NO'}")
    print(f"  joyl.add absent (fused)  : {'YES' if no_joyl_add else 'NO'}")

    if print_joyl_ir or print_ir_all:
        print("\n--- MLIR IR (joyl dialect) ---")
        print(joyl_ir)
        print("--- end of MLIR IR (joyl dialect) ---")

    assert joyl_pass and has_memref, \
        "[JOY PyTest]: XXXXXXXXXXXXXXXXXX joyl lowering failed XXXXXXXXXXXXXXXXXX\n"
    print(f"\n[JOY PyTest]: ================== Joyl lowering success ==================")

    # ==================================================================
    # Phase 4: Lower Joyl -> Joyh (codegen + custom calls)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Lowering Joyl -> Joyh (codegen + GPU custom calls)")
    print("=" * 60)

    joyh_ir = lower_joyl_to_joyh(joyl_ir)
    if joyh_ir is None:
        print("  SKIP: Joyh lowering not available")
        return

    # Custom call targets (all ops EXCEPT rms_norm and fuse_add_rmsnorm)
    joyh_custom_call_targets = [
        ("joy_gpu_embedding",        "embed_tokens"),
        ("joy_gpu_linear",           "linear projections"),
        ("joy_gpu_reshape",          "reshape"),
        ("joy_gpu_transpose",        "transpose"),
        ("joy_gpu_apply_rotary_emb", "rotary embedding"),
        ("joy_gpu_repeat_kv",        "repeat_kv"),
        ("joy_gpu_matmul",           "matmul"),
        ("joy_gpu_softmax",          "softmax"),
        ("joy_gpu_silu",             "SiLU"),
        ("joy_gpu_mul",              "element-wise mul"),
    ]

    print("\nVerifying Joyh custom call targets:")
    joyh_pass = True
    for target_name, source in joyh_custom_call_targets:
        count = joyh_ir.count(f'call_target_name = "{target_name}"')
        ok = count > 0
        if not ok:
            joyh_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {target_name}: {count}  [{source}]")

    # Codegen'd kernel verification
    print("\nVerifying codegen'd GPU kernels:")

    has_rms_kernel = "@joy_rms_norm_kernel" in joyh_ir
    has_fuse_kernel = "@joy_fuse_add_rmsnorm_kernel" in joyh_ir
    has_gpu_kernel_attr = "gpu_kernel" in joyh_ir
    has_math_rsqrt = "math.rsqrt" in joyh_ir
    has_scf_for = "scf.for" in joyh_ir
    rms_kernel_refs = joyh_ir.count("@joy_rms_norm_kernel")
    fuse_kernel_refs = joyh_ir.count("@joy_fuse_add_rmsnorm_kernel")
    no_rms_custom_call = 'call_target_name = "joy_gpu_rms_norm"' not in joyh_ir
    no_add_custom_call = 'call_target_name = "joy_gpu_add"' not in joyh_ir
    no_fuse_custom_call = ('call_target_name = "joy_gpu_fuse_add_rmsnorm"'
                           not in joyh_ir)

    print(f"  {'PASS' if has_rms_kernel else 'FAIL'}  "
          f"@joy_rms_norm_kernel present")
    print(f"  {'PASS' if has_fuse_kernel else 'FAIL'}  "
          f"@joy_fuse_add_rmsnorm_kernel present")
    print(f"  {'PASS' if has_gpu_kernel_attr else 'FAIL'}  "
          f"gpu_kernel attribute present")
    print(f"  {'PASS' if has_math_rsqrt else 'FAIL'}  "
          f"math.rsqrt in kernel body")
    print(f"  {'PASS' if has_scf_for else 'FAIL'}  "
          f"scf.for loops in kernel body")
    print(f"  {'PASS' if no_rms_custom_call else 'FAIL'}  "
          f"rms_norm NOT in custom_call (codegen'd)")
    print(f"  {'PASS' if no_add_custom_call else 'FAIL'}  "
          f"add NOT in custom_call (fused)")
    print(f"  {'PASS' if no_fuse_custom_call else 'FAIL'}  "
          f"fuse_add_rmsnorm NOT in custom_call (codegen'd)")
    print(f"  rms_norm kernel refs           : {rms_kernel_refs} "
          f"(1 def + {rms_kernel_refs - 1} calls)")
    print(f"  fuse_add_rmsnorm kernel refs   : {fuse_kernel_refs} "
          f"(1 def + {fuse_kernel_refs - 1} calls)")

    codegen_pass = (has_rms_kernel and has_fuse_kernel and
                    has_gpu_kernel_attr and has_math_rsqrt and
                    has_scf_for and no_rms_custom_call and
                    no_add_custom_call and no_fuse_custom_call)

    has_custom_call = '"joyh.custom_call"' in joyh_ir
    no_joyl = '"joyl.' not in joyh_ir
    has_backend_gpu = 'backend = "gpu"' in joyh_ir
    print(f"\n  joyh.custom_call present : {'YES' if has_custom_call else 'NO'}")
    print(f"  joyl ops removed         : {'YES' if no_joyl else 'NO'}")
    print(f"  backend = gpu            : {'YES' if has_backend_gpu else 'NO'}")

    if print_joyh_ir or print_ir_all:
        print("\n--- MLIR IR (joyh dialect) ---")
        print(joyh_ir)
        print("--- end of MLIR IR (joyh dialect) ---")

    assert (joyh_pass and codegen_pass and has_custom_call
            and no_joyl and has_backend_gpu), \
        "[JOY PyTest]: XXXXXXXXXXXXXXXXXX joyh lowering failed XXXXXXXXXXXXXXXXXX\n"
    print(f"\n[JOY PyTest]: ================== Joyh lowering success ==================\n")


def is_fp16(argv):
    return "fp16" in argv


# ============================================================================
# Lesson-15 extra: quick smoke check that does NOT need GPU or HF weights
# ============================================================================
def test_quick_ir_smoke(dtype: str, *, print_ir: bool = False) -> None:
    """Run Phase A with ``NUM_LAYERS=1`` so the whole pipeline fits in
    ~1 second.

    This is the cheapest possible Lesson-15 regression: it exercises
    every Pass (OpFusion / Joy->Joyl / Codegen / Joyl->Joyh) without
    paying the 2-3 s cost of the full 28-layer IR.  No GPU is touched.
    """
    print("\n" + "=" * 60)
    print("  Lesson 15 quick IR-only smoke (NUM_LAYERS=1, seq_len=8)")
    print("=" * 60)
    t0 = time.perf_counter()
    test_qwen3_06b_joy(
        batch_size=1, dtype=dtype,
        print_ir=print_ir, print_joy_opt_ir=False,
        print_joyl_ir=False, print_joyh_ir=False,
        print_ir_all=False,
        num_layers=1, seq_len=8)
    dt = time.perf_counter() - t0
    print(f"\n[JOY PyTest]: ================== "
          f"quick IR smoke success ({dt:.2f}s) ==================\n")


# ============================================================================
# Phase B - End-to-end GPU inference (prefill + decode)
# ============================================================================
DEFAULT_MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca")

# Reserved control tokens used by the chat template (matches the
# reference Qwen3 chat prompt format).  Their values are part of the
# Qwen3 vocabulary.
THINK_START_ID = 151667     # <think>
THINK_END_ID = 151668       # </think>
DOUBLE_NEWLINE_ID = 271     # '\n\n'


def _build_classification_prompt(query: str):
    """Construct the same chat-template prompt the reference test uses.

    Returns ``(prefix_text, prompt_ids_no_template)`` where the second
    element is the list of token ids that will be fed into the model.
    The prompt closes with the four "thinking pacifier" tokens so the
    assistant can directly emit the classification label.
    """
    prefix_text = (
        "<|im_start|>system\n"
        "请对输入词语进行情感分类，格式为[正向]或[负向]<|im_end|>\n"
        "<|im_start|>user\n" + query + "<|im_end|>\n"
        "<|im_start|>assistant\n")
    return prefix_text


def _resolve_model_path(arg_path):
    if arg_path and os.path.isdir(arg_path):
        return arg_path
    if os.environ.get("QWEN3_MODEL_PATH") and \
            os.path.isdir(os.environ["QWEN3_MODEL_PATH"]):
        return os.environ["QWEN3_MODEL_PATH"]
    if os.path.isdir(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH

    # Fallback: try resolving a snapshot folder under HF_HUB_CACHE.
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        cand = os.path.join(HF_HUB_CACHE, "models--Qwen--Qwen3-0.6B",
                            "snapshots")
        if os.path.isdir(cand):
            for snap in os.listdir(cand):
                cand_full = os.path.join(cand, snap)
                if os.path.isfile(os.path.join(cand_full, "config.json")):
                    return cand_full
    except Exception:
        pass
    return None


def test_qwen3_06b_inference(query: str, expect: str,
                             max_new_tokens: int,
                             model_path,
                             verbose: bool = True) -> None:
    """Execute the full Qwen3-0.6B network on the GPU backend.

    Builds prompt -> uploads weights -> runs prefill (one full forward
    pass on the prompt) -> greedy-decodes ``max_new_tokens`` more tokens
    by recomputing the forward each step (no KV cache).  Asserts that
    ``expect`` appears in the decoded text.
    """
    print("\n" + "=" * 60)
    print("  Phase B: End-to-end Qwen3-0.6B inference on GPU backend")
    print("=" * 60)

    model_path = _resolve_model_path(model_path)
    if model_path is None:
        print("  SKIP: Qwen3-0.6B model weights not found.")
        print("        Set --model-path /path/to/Qwen3-0.6B (HF format) or "
              "QWEN3_MODEL_PATH env var, or download via:")
        print('        python3 -c "from huggingface_hub import snapshot_download; '
              'snapshot_download(\'Qwen/Qwen3-0.6B\')"')
        return False
    print(f"  model path        : {model_path}")

    # Defer heavy imports so Phase A is not penalised when --no-inference.
    try:
        import numpy as np
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"  SKIP: missing python dependency: {e}")
        return False

    try:
        from qwen3_gpu_runner import Qwen3GpuRunner
    except FileNotFoundError as e:
        print(f"  SKIP: GPU runtime library not built ({e}).  Run "
              "scripts/build.sh first.")
        return False

    # ---- Build prompt ----
    print(f"  query             : {query!r}")
    print(f"  expect substring  : {expect!r}")

    print("\n  Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_path)
    prefix_text = _build_classification_prompt(query)
    prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
    prompt_ids = (prefix_ids
                  + [THINK_START_ID, DOUBLE_NEWLINE_ID,
                     THINK_END_ID, DOUBLE_NEWLINE_ID])
    print(f"  prompt length     : {len(prompt_ids)} tokens")

    # ---- Build runner (uploads ~3 GB of weights) ----
    print("\n  Building GPU runner (uploading weights)...")
    t0 = time.perf_counter()
    runner = Qwen3GpuRunner.from_pretrained(model_path)
    print(f"  runner ready in   : {time.perf_counter() - t0:.2f} s")

    # ---- Prefill + decode ----
    print(f"\n  Generating up to {max_new_tokens} tokens...")
    t_gen = time.perf_counter()
    new_ids = runner.generate(
        np.array([prompt_ids], dtype=np.int64),
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id,
        verbose=verbose)
    gen_dt = time.perf_counter() - t_gen
    text = tok.decode(new_ids[0])
    print(f"\n  generated_ids     : {new_ids[0].tolist()}")
    print(f"  generated_text    : {text!r}")
    print(f"  total gen wall    : {gen_dt:.2f} s "
          f"({len(new_ids[0]) / max(gen_dt, 1e-3):.1f} tok/s)")

    # ---- Verify ----
    ok = expect in text
    print(f"\n  expect {expect!r} present? : "
          f"{'YES' if ok else 'NO'}")
    assert ok, (
        f"[JOY PyTest]: XXXXXXXXXXXXXXXXXX inference failed: "
        f"output {text!r} does not contain {expect!r} XXXXXXXXXXXXXXXXXX\n")
    print(f"\n[JOY PyTest]: ================== "
          f"GPU inference success ==================")
    return True


def main(argv):
    parser = argparse.ArgumentParser(
        description="Lesson 15: Qwen3-0.6B Joy Dialect + GPU Inference Test")
    # Phase A flags (existing)
    parser.add_argument("--print-joy-ir", action="store_true",
                        help="Print the full generated Joy MLIR IR")
    parser.add_argument("--print-joy-opt-ir", action="store_true",
                        help="Print Joy IR after OpFusion pass")
    parser.add_argument("--print-joyl-ir", action="store_true",
                        help="Print the lowered Joyl MLIR IR")
    parser.add_argument("--print-joyh-ir", action="store_true",
                        help="Print the lowered Joyh MLIR IR")
    parser.add_argument("--print-ir-all", action="store_true",
                        help="Print IR after every pass")
    # Phase A/B selection
    parser.add_argument("--no-inference", action="store_true",
                        help="Skip Phase B (GPU inference); run only the "
                             "compilation pipeline checks")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip Phase A (compilation checks); run only "
                             "the end-to-end GPU inference")
    parser.add_argument("--quick-ir-check", action="store_true",
                        help="Run a single-layer (~1s) Phase A IR smoke "
                             "check and exit; skips Phase A 28-layer + "
                             "Phase B entirely.  Useful for fast CI.")
    # Phase B flags
    parser.add_argument("--model-path", default=None,
                        help="Path to a HF-format Qwen3-0.6B checkpoint "
                             "(defaults to the local HF cache)")
    parser.add_argument("--prompt", default="快乐",
                        help="Word to classify (sentiment classification)")
    parser.add_argument("--expect", default="[正向]",
                        help="Substring that must appear in the generated "
                             "text for the test to pass")
    parser.add_argument("--max-new-tokens", type=int, default=8,
                        help="Maximum number of decode steps")
    args, _ = parser.parse_known_args(argv[1:])

    fp16_flag = is_fp16(argv)
    dtype = "f16" if fp16_flag else "f32"

    print(f"Run fp16 flag      : {fp16_flag}")
    print(f"Using dtype        : {dtype}")
    print(f"Device             : gpu")
    print(f"Print Joy IR       : {args.print_joy_ir}")
    print(f"Print Joy Opt IR   : {args.print_joy_opt_ir}")
    print(f"Print Joyl IR      : {args.print_joyl_ir}")
    print(f"Print Joyh IR      : {args.print_joyh_ir}")
    print(f"Print IR All       : {args.print_ir_all}")
    print(f"Quick IR check     : {args.quick_ir_check}")
    print(f"Run Phase A (IR)   : "
          f"{not args.inference_only and not args.quick_ir_check}")
    print(f"Run Phase B (exec) : "
          f"{not args.no_inference and not args.quick_ir_check}")

    # ---- Lesson-15 fast path: 1-layer IR smoke ----
    if args.quick_ir_check:
        test_quick_ir_smoke(dtype, print_ir=args.print_joy_ir)
        return

    # ---- Phase A: compile pipeline ----
    if not args.inference_only:
        for b in [1]:
            print(f"\nStart batch {b} compile-pipeline check:")
            test_qwen3_06b_joy(b, dtype,
                               print_ir=args.print_joy_ir,
                               print_joy_opt_ir=args.print_joy_opt_ir,
                               print_joyl_ir=args.print_joyl_ir,
                               print_joyh_ir=args.print_joyh_ir,
                               print_ir_all=args.print_ir_all)

    # ---- Phase B: actually execute the network on GPU ----
    if not args.no_inference:
        test_qwen3_06b_inference(
            query=args.prompt,
            expect=args.expect,
            max_new_tokens=args.max_new_tokens,
            model_path=args.model_path)


if __name__ == "__main__":
    main(sys.argv)
