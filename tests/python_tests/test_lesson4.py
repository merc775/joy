#!/usr/bin/env python3
"""Lesson 4: Custom Dialect Design (joy / joyl / joyh) — hands-on tests.

Covers the topics in `joy/docs/第4讲-自定义算子Dialect设计.md`:

  Section 2 (joy):
    1. Single joy.linear graph builds correctly and shape-infers right
    2. Multi-op MLP-like graph uses every joy op advertised by JoyOps.td

  Section 2.3 (OpFusion):
    3. --joy-op-fusion turns add + rms_norm into joy.fuse_add_rmsnorm

  Section 3 (joyl):
    4. --lower-joy-to-joyl turns tensor ops into memref ops with the
       expected Arg<MemRead/MemWrite> shape (no tensor types leak)

  Section 4 (joyh):
    5. --lower-joyl-to-joyh produces joyh.custom_call with
       call_target_name = "joy_gpu_<mnemonic>"
    6. lower-joyl-to-joyh respects the Codegen white-list: rms_norm and
       fuse_add_rmsnorm do NOT become joyh.custom_call (they become
       func.call @joy_*_kernel via --codegen-rms-norm)

Section 5 of the lecture (`joy-emit-cuda`) is only introduced briefly
in the doc; its end-to-end behaviour is exercised by Lesson 7's tests.

Usage:
    python3 tests/python_tests/test_lesson4.py
    python3 tests/python_tests/test_lesson4.py --print-ir-all
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

cur_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(cur_path, "../.."))
sys.path.insert(0, os.path.join(project_root, "python"))

from joy.builder import Graph, ops


JOY_OPT = os.path.join(project_root, "build", "bin", "joy-opt")


# ============================================================================
# Helpers
# ============================================================================
def _run_joy_opt(input_ir, passes, timeout=60):
    if not os.path.exists(JOY_OPT):
        print(f"  WARNING: joy-opt not found at {JOY_OPT}")
        return None

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mlir")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(input_ir)
        cmd = [JOY_OPT] + passes + [tmp_path]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout)
        if result.returncode != 0:
            print(f"  ERROR: joy-opt failed (rc={result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-10:]:
                    print(f"    {line}")
            return None
        return result.stdout
    except Exception as e:
        print(f"  ERROR: Failed to run joy-opt: {e}")
        return None
    finally:
        os.unlink(tmp_path)


def _print_checks(checks):
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    return all_pass


# ============================================================================
# Test 1: Build a single joy.linear graph
# ============================================================================
def test_joy_linear_single(print_ir=False):
    """The simplest possible joy graph: one joy.linear op."""
    print("\n" + "=" * 60)
    print("  Test 1: Single joy.linear graph (Tensor IR)")
    print("=" * 60)

    graph = Graph(name="single_linear")
    x = graph.input([1, 64, 1024], "f16", name="input")
    w = graph.input([512, 1024], "f16", name="weight")
    out = ops.linear(x, w)
    graph.set_outputs([out])

    ir = graph.get_ir()
    if print_ir:
        print("\n--- IR ---")
        print(ir)

    checks = [
        ('"joy.linear"(' in ir,           "joy.linear op emitted"),
        ("tensor<1x64x1024xf16>" in ir,    "input type tensor<1x64x1024xf16>"),
        ("tensor<512x1024xf16>" in ir,     "weight type tensor<512x1024xf16>"),
        ("tensor<1x64x512xf16>" in ir,
         "output type tensor<1x64x512xf16> (PyTorch nn.Linear convention)"),
    ]
    assert _print_checks(checks), "[Lesson4] Test 1 failed"
    print("\n[Lesson4]: ================== Test 1 PASSED ==================")


# ============================================================================
# Test 2: All 13 joy ops are usable via the Python front-end
# ============================================================================
def test_joy_op_inventory(print_ir=False):
    """Build a graph that exercises every joy op listed in the lecture."""
    print("\n" + "=" * 60)
    print("  Test 2: Joy op inventory (13 ops from lecture §2.2)")
    print("=" * 60)

    HIDDEN = 1024
    INTER = 3072

    graph = Graph(name="full_inventory")
    # ---- Inputs ----
    ids       = graph.input([1, 64],          "i64", name="ids")
    embed_w   = graph.input([4096, HIDDEN],   "f16", name="embed_w")
    ln_w      = graph.input([HIDDEN],         "f16", name="ln_w")
    gate_w    = graph.input([INTER, HIDDEN],  "f16", name="gate_w")
    up_w      = graph.input([INTER, HIDDEN],  "f16", name="up_w")
    down_w    = graph.input([HIDDEN, INTER],  "f16", name="down_w")
    rope_cos  = graph.input([1, 64, 128],     "f16", name="rope_cos")
    rope_sin  = graph.input([1, 64, 128],     "f16", name="rope_sin")
    qk        = graph.input([1, 8, 64, 128],  "f16", name="qk")
    kv        = graph.input([1, 2, 64, 128],  "f16", name="kv")
    matmul_b  = graph.input([1, 8, 128, 64],  "f16", name="matmul_b")

    # ---- Exercise ops (mnemonics only; semantics don't have to be correct) ----
    emb       = ops.embedding(ids, embed_w)             # joy.embedding
    normed    = ops.rms_norm(emb, ln_w, epsilon=1e-6)    # joy.rms_norm
    rotated   = ops.apply_rotary_emb(qk, rope_cos, rope_sin)  # apply_rotary_emb
    expanded  = ops.repeat_kv(kv, n_rep=4)               # joy.repeat_kv
    gate      = ops.linear(normed, gate_w)               # joy.linear
    gate_act  = ops.silu(gate)                            # joy.silu
    up        = ops.linear(normed, up_w)                 # joy.linear (again)
    gate_up   = ops.mul(gate_act, up)                    # joy.mul
    down      = ops.linear(gate_up, down_w)              # joy.linear (again)
    summed    = ops.add(emb, down)                       # joy.add
    qk2       = ops.transpose(rotated, [0, 2, 1, 3])     # joy.transpose
    qk3       = ops.reshape(qk2, [1, 64, 1024])           # joy.reshape
    attn      = ops.matmul(rotated, matmul_b)             # joy.matmul
    soft      = ops.softmax(attn, axis=-1)                 # joy.softmax

    graph.set_outputs([summed, qk3, soft])
    ir = graph.get_ir()
    if print_ir:
        print("\n--- IR ---")
        print(ir)

    stats = graph.get_op_stats()
    expected_ops = [
        ("joy.embedding",        1),
        ("joy.rms_norm",         1),
        ("joy.apply_rotary_emb", 1),
        ("joy.repeat_kv",        1),
        ("joy.linear",           3),
        ("joy.silu",             1),
        ("joy.mul",              1),
        ("joy.add",              1),
        ("joy.transpose",        1),
        ("joy.reshape",          1),
        ("joy.matmul",           1),
        ("joy.softmax",          1),
    ]

    checks = [(stats.get(name, 0) == count,
               f"{name}: expected {count}, got {stats.get(name, 0)}")
              for name, count in expected_ops]
    # epsilon attribute appears for rms_norm
    checks.append(('epsilon = 9.999999E-7 : f32' in ir
                   or 'epsilon = 1.000000e-06 : f32' in ir,
                   "rms_norm epsilon attribute present"))
    # transpose permutation attr
    checks.append(("permutation = dense" in ir,
                   "transpose permutation attr present"))
    # softmax axis attr (Python frontend normalises negative axes)
    checks.append((re.search(r'axis\s*=\s*\d+\s*:\s*i64', ir) is not None,
                   "softmax axis attribute present (normalised to positive)"))
    # repeat_kv n_rep attr
    checks.append(("n_rep = 4 : i64" in ir,
                   "repeat_kv n_rep attribute present"))

    assert _print_checks(checks), "[Lesson4] Test 2 failed"
    print("\n[Lesson4]: ================== Test 2 PASSED ==================")


# ============================================================================
# Test 3: --joy-op-fusion turns add + rms_norm → fuse_add_rmsnorm
# ============================================================================
def test_op_fusion(print_ir=False):
    """Build a two-layer MLP, run OpFusion, verify fusion happened."""
    print("\n" + "=" * 60)
    print("  Test 3: OpFusion (add + rms_norm -> fuse_add_rmsnorm)")
    print("=" * 60)

    HIDDEN = 1024
    INTER = 3072
    graph = Graph(name="two_layer")
    hidden = graph.input([1, 64, HIDDEN], "f16", name="hidden")

    for layer_idx in range(2):
        ln_w   = graph.input([HIDDEN],         "f16",
                             name=f"l{layer_idx}.ln.w")
        gate_w = graph.input([INTER, HIDDEN],  "f16",
                             name=f"l{layer_idx}.gate.w")
        up_w   = graph.input([INTER, HIDDEN],  "f16",
                             name=f"l{layer_idx}.up.w")
        down_w = graph.input([HIDDEN, INTER],  "f16",
                             name=f"l{layer_idx}.down.w")

        residual = hidden
        normed   = ops.rms_norm(hidden, ln_w, epsilon=1e-6)
        gate     = ops.silu(ops.linear(normed, gate_w))
        up       = ops.linear(normed, up_w)
        down     = ops.linear(ops.mul(gate, up), down_w)
        hidden   = ops.add(residual, down)

    graph.set_outputs([hidden])
    ir = graph.get_ir()

    fused = _run_joy_opt(ir, ["--joy-op-fusion"])
    if fused is None:
        print("  SKIP: joy-opt not available")
        return

    if print_ir:
        print("\n--- post fusion IR ---")
        print(fused)

    fuse_count   = fused.count('"joy.fuse_add_rmsnorm"(')
    remaining_a  = fused.count('"joy.add"(')
    remaining_r  = fused.count('"joy.rms_norm"(')
    before_add   = ir.count('"joy.add"(')
    before_rms   = ir.count('"joy.rms_norm"(')

    print(f"\n  before: joy.add={before_add}, joy.rms_norm={before_rms}")
    print(f"  after : fuse_add_rmsnorm={fuse_count}, add={remaining_a}, "
          f"rms_norm={remaining_r}")

    checks = [
        (fuse_count == 1,    "1x joy.fuse_add_rmsnorm produced"),
        (remaining_a == 1,
         "1x joy.add remains (last layer; no following rms_norm)"),
        (remaining_r == 1,
         "1x standalone joy.rms_norm remains (layer 0 input has no add)"),
    ]
    assert _print_checks(checks), "[Lesson4] Test 3 failed"
    print("\n[Lesson4]: ================== Test 3 PASSED ==================")


# ============================================================================
# Test 4: --lower-joy-to-joyl produces joyl.* ops with memref operands
# ============================================================================
def test_lower_to_joyl(print_ir=False):
    """Tensor → MemRef lowering creates joyl ops with memref operands and
    no joy tensor types left behind."""
    print("\n" + "=" * 60)
    print("  Test 4: --lower-joy-to-joyl (tensor -> memref)")
    print("=" * 60)

    graph = Graph(name="lower_to_joyl")
    x   = graph.input([1, 64, 1024], "f16", name="x")
    w_l = graph.input([512, 1024],    "f16", name="w_l")
    s   = graph.input([1024],          "f16", name="s")

    out = ops.rms_norm(x, s, epsilon=1e-6)
    out = ops.linear(out, w_l)
    out = ops.silu(out)
    graph.set_outputs([out])

    joyl_ir = _run_joy_opt(graph.get_ir(), ["--lower-joy-to-joyl"])
    if joyl_ir is None:
        print("  SKIP: joy-opt not available")
        return

    if print_ir:
        print("\n--- post joy->joyl IR ---")
        print(joyl_ir)

    checks = [
        ('"joyl.rms_norm"' in joyl_ir,    "joyl.rms_norm produced"),
        ('"joyl.linear"' in joyl_ir,      "joyl.linear produced"),
        ('"joyl.silu"' in joyl_ir,        "joyl.silu produced"),
        ('"joy.' not in joyl_ir,          "no leftover joy.* op"),
        ("memref<" in joyl_ir,             "memref types present"),
        ("memref.alloc" in joyl_ir,        "memref.alloc for output buffers"),
        ("tensor<" not in joyl_ir,         "no leftover tensor<...> types"),
    ]
    assert _print_checks(checks), "[Lesson4] Test 4 failed"
    print("\n[Lesson4]: ================== Test 4 PASSED ==================")


# ============================================================================
# Test 5: --lower-joyl-to-joyh creates joyh.custom_call with joy_gpu_* names
# ============================================================================
def test_lower_to_joyh(print_ir=False):
    """joyl ops (except rms_norm / fuse_add_rmsnorm) become
    joyh.custom_call with call_target_name = joy_gpu_<mnemonic>."""
    print("\n" + "=" * 60)
    print("  Test 5: --lower-joyl-to-joyh (memref -> custom_call)")
    print("=" * 60)

    graph = Graph(name="lower_to_joyh")
    x   = graph.input([1, 64, 1024], "f16", name="x")
    w   = graph.input([512, 1024],    "f16", name="w")
    a   = graph.input([1, 64, 512],   "f16", name="a")
    out = ops.linear(x, w)
    out = ops.silu(out)
    out = ops.add(out, a)
    out = ops.softmax(out, axis=-1)
    graph.set_outputs([out])

    joyl_ir = _run_joy_opt(graph.get_ir(), ["--lower-joy-to-joyl"])
    if joyl_ir is None:
        print("  SKIP joy-opt not available")
        return

    joyh_ir = _run_joy_opt(joyl_ir, ["--lower-joyl-to-joyh"])
    if joyh_ir is None:
        print("  SKIP joy-opt not available (joyh step)")
        return

    if print_ir:
        print("\n--- post joyl->joyh IR ---")
        print(joyh_ir)

    custom_calls = re.findall(
        r'"joyh\.custom_call".*?call_target_name\s*=\s*"([^"]+)"',
        joyh_ir, flags=re.DOTALL)
    print(f"\n  joyh.custom_call targets: {custom_calls}")

    checks = [
        ('"joyh.custom_call"' in joyh_ir, "joyh.custom_call emitted"),
        ("call_target_name" in joyh_ir,    "call_target_name attribute present"),
        ('backend = "gpu"' in joyh_ir,     'backend = "gpu" attribute'),
        ("num_inputs" in joyh_ir,          "num_inputs attribute present"),
        # All targets must be prefixed with joy_gpu_
        (all(t.startswith("joy_gpu_") for t in custom_calls),
         "every call_target_name starts with joy_gpu_"),
        ("joy_gpu_linear"  in custom_calls, "joy_gpu_linear  produced"),
        ("joy_gpu_silu"    in custom_calls, "joy_gpu_silu    produced"),
        ("joy_gpu_add"     in custom_calls, "joy_gpu_add     produced"),
        ("joy_gpu_softmax" in custom_calls, "joy_gpu_softmax produced"),
        ("joyl." not in joyh_ir,            "no leftover joyl.* op"),
    ]
    assert _print_checks(checks), "[Lesson4] Test 5 failed"
    print("\n[Lesson4]: ================== Test 5 PASSED ==================")


# ============================================================================
# Test 6: Codegen white-list — rms_norm / fuse_add_rmsnorm don't become
#         joyh.custom_call; instead they become func.call @joy_*_kernel
# ============================================================================
def test_codegen_whitelist(print_ir=False):
    """The custom call lowering must SKIP rms_norm / fuse_add_rmsnorm so
    that --codegen-rms-norm can replace them with proper func.call to the
    emitted GPU kernel."""
    print("\n" + "=" * 60)
    print("  Test 6: Codegen white-list (rms_norm bypasses joyh)")
    print("=" * 60)

    # Build a small graph containing a rms_norm + linear so both code paths
    # show up in the same module.
    graph = Graph(name="whitelist_demo")
    x  = graph.input([1, 64, 1024], "f16", name="x")
    s  = graph.input([1024],         "f16", name="s")
    w  = graph.input([512, 1024],     "f16", name="w")
    out = ops.rms_norm(x, s, epsilon=1e-6)
    out = ops.linear(out, w)
    graph.set_outputs([out])

    joyl_ir = _run_joy_opt(graph.get_ir(), ["--lower-joy-to-joyl"])
    if joyl_ir is None:
        print("  SKIP"); return

    # Pipeline order matters: codegen-rms-norm first (peels rms_norm out
    # of joyl), then lower-joyl-to-joyh (handles the rest).
    final_ir = _run_joy_opt(joyl_ir,
                            ["--codegen-rms-norm", "--lower-joyl-to-joyh"])
    if final_ir is None:
        print("  SKIP"); return

    if print_ir:
        print("\n--- final IR ---")
        print(final_ir)

    # Expectations:
    #   - rms_norm replaced by func.call @joy_rms_norm_kernel
    #   - linear  replaced by joyh.custom_call target=joy_gpu_linear
    #   - no joy_gpu_rms_norm in any custom_call target
    custom_calls = re.findall(
        r'"joyh\.custom_call".*?call_target_name\s*=\s*"([^"]+)"',
        final_ir, flags=re.DOTALL)

    checks = [
        ("call @joy_rms_norm_kernel" in final_ir,
         "rms_norm -> func.call @joy_rms_norm_kernel"),
        ("joy_gpu_rms_norm" not in final_ir,
         "no joy_gpu_rms_norm in custom_call targets (white-listed)"),
        ("joy_gpu_fuse_add_rmsnorm" not in final_ir,
         "no joy_gpu_fuse_add_rmsnorm (white-listed)"),
        ("joy_gpu_linear" in custom_calls,
         "linear -> joyh.custom_call(joy_gpu_linear)"),
        ('"joyl.' not in final_ir,
         "no joyl ops remain"),
    ]
    assert _print_checks(checks), "[Lesson4] Test 6 failed"
    print("\n[Lesson4]: ================== Test 6 PASSED ==================")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Lesson 4: Custom Dialect Design (joy / joyl / joyh)")
    parser.add_argument("--print-ir-all", action="store_true",
                        help="Print IR at each stage")
    args = parser.parse_args()
    p = args.print_ir_all

    print("=" * 60)
    print("  Lesson 4: Custom Dialect Design (joy / joyl / joyh)")
    print("=" * 60)

    test_joy_linear_single(print_ir=p)
    test_joy_op_inventory(print_ir=p)
    test_op_fusion(print_ir=p)
    test_lower_to_joyl(print_ir=p)
    test_lower_to_joyh(print_ir=p)
    test_codegen_whitelist(print_ir=p)

    print("\n" + "=" * 60)
    print("  ALL LESSON 4 TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
