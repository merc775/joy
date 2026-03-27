#!/usr/bin/env python3
"""Lesson 10: Auto operator codegen vs external library call — hands-on tests.

This test suite mirrors the two final-stage lowering paths in Joy:

  Part A — Auto codegen (joy/lib/optimizer/CodegenRMSNormPass.cpp)
           joyl.rms_norm           -> @joy_rms_norm_kernel + func.call
           joyl.fuse_add_rmsnorm   -> @joy_fuse_add_rmsnorm_kernel + func.call
           All kernel bodies are pure MLIR (arith / scf / math / memref / func).

  Part B — External library call (joy/lib/optimizer/LowerJoylToJoyhPass.cpp)
           Every other joyl.* op   -> joyh.custom_call {
                                          call_target_name = "joy_gpu_<op>",
                                          backend = "gpu",
                                          num_inputs = ...
                                      }

  Part C — Coexistence + workflow
           A full backend pipeline produces a joyh IR that contains BOTH
           codegen'd kernel functions/calls AND custom_calls.  The pipeline
           is idempotent and stops cleanly when there's nothing to codegen.

The lowering stops at the joyh dialect; emitting CUDA C / compiling to GPU
binary is the subject of the next lecture.

Usage:
    python3 tests/python_tests/test_lesson10.py
    python3 tests/python_tests/test_lesson10.py --print-ir-all
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

from joy.builder import Graph
from joy.builder import ops


JOY_OPT = os.path.join(project_root, "build", "bin", "joy-opt")


# ============================================================================
# Helpers
# ============================================================================
def _run_joy_opt(input_ir, passes, timeout=30):
    """Run joy-opt with the given passes on input_ir text, return stdout."""
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


def _count(ir, pattern):
    """Count occurrences of a substring in IR text."""
    return ir.count(pattern)


def _build_joy_ir_and_lower(graph):
    """Build Joy IR, apply fusion + lowering to Joyl, return Joyl IR."""
    joy_ir = graph.get_ir()
    fused = _run_joy_opt(joy_ir, ["--joy-optimization-pipeline"])
    assert fused is not None, "optimization pipeline failed"
    joyl_ir = _run_joy_opt(fused, ["--lower-joy-to-joyl"])
    assert joyl_ir is not None, "joy→joyl lowering failed"
    return joyl_ir


def _report(checks):
    """Print a list of (ok, description) pairs; return True iff all PASS."""
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    return all_pass


# ============================================================================
# Part A — Auto Codegen tests
# ============================================================================

# Test 1: joyl.rms_norm → @joy_rms_norm_kernel (single op, single call)
# ----------------------------------------------------------------------------
def test_rms_norm_codegen(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 1: joyl.rms_norm auto-codegen → @joy_rms_norm_kernel")
    print("=" * 60)

    H = 1024
    graph = Graph(name="rms_codegen_test")
    x = graph.input([1, 64, H], "f16")
    scale = graph.input([H], "f16")
    normed = ops.rms_norm(x, scale, epsilon=1e-6)
    graph.set_outputs([normed])
    joyl_ir = _build_joy_ir_and_lower(graph)

    rms_before = _count(joyl_ir, '"joyl.rms_norm"(')
    print(f"\n  Joyl IR: joyl.rms_norm count = {rms_before}")

    codegen_ir = _run_joy_opt(joyl_ir, ["--codegen-rms-norm"])
    assert codegen_ir is not None, "[Lesson10] Test 1 failed: codegen returned None"

    if print_ir:
        print("\n--- IR after Codegen ---")
        print(codegen_ir)
        print("--- end ---")

    checks = [
        (rms_before == 1,
         f"joyl IR contains exactly 1 joyl.rms_norm (got {rms_before})"),
        ("func.func private @joy_rms_norm_kernel" in codegen_ir,
         "@joy_rms_norm_kernel function defined (kernel-only IR)"),
        ("gpu_kernel" in codegen_ir,
         "gpu_kernel attribute set on the kernel function"),
        ('kernel_name = "rms_norm"' in codegen_ir,
         'kernel_name = "rms_norm" attribute set'),
        ("math.rsqrt" in codegen_ir,
         "math.rsqrt in kernel body (computation expanded into IR)"),
        ("scf.for" in codegen_ir,
         "scf.for loops in kernel body"),
        ("memref.load" in codegen_ir and "memref.store" in codegen_ir,
         "memref.load / memref.store inside kernel body"),
        ("arith.extf" in codegen_ir and "arith.truncf" in codegen_ir,
         "arith.extf / arith.truncf (f16↔f32 promotion/demotion)"),
        ("call @joy_rms_norm_kernel" in codegen_ir,
         "func.call @joy_rms_norm_kernel issued at the call site"),
        (_count(codegen_ir, '"joyl.rms_norm"(') == 0,
         "original joyl.rms_norm fully eliminated"),
    ]

    assert _report(checks), "[Lesson10] Test 1 failed"
    print("\n[Lesson10]: ================== Test 1 PASSED ==================")


# Test 2: joyl.fuse_add_rmsnorm → @joy_fuse_add_rmsnorm_kernel
# ----------------------------------------------------------------------------
def test_fuse_add_rmsnorm_codegen(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 2: joyl.fuse_add_rmsnorm auto-codegen "
          "(cross-op fusion materialised in IR)")
    print("=" * 60)

    H = 1024
    graph = Graph(name="fuse_codegen_test")
    hidden = graph.input([1, 64, H], "f16")
    residual = graph.input([1, 64, H], "f16")
    ln_w = graph.input([H], "f16")
    w = graph.input([512, H], "f16")

    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=1e-6)
    out = ops.linear(normed, w)
    graph.set_outputs([out])

    joyl_ir = _build_joy_ir_and_lower(graph)

    fuse_before = _count(joyl_ir, '"joyl.fuse_add_rmsnorm"(')
    print(f"\n  Joyl IR: joyl.fuse_add_rmsnorm count = {fuse_before}")

    codegen_ir = _run_joy_opt(joyl_ir, ["--codegen-rms-norm"])
    assert codegen_ir is not None, "[Lesson10] Test 2 failed: codegen returned None"

    if print_ir:
        print("\n--- IR after Codegen ---")
        print(codegen_ir)
        print("--- end ---")

    checks = [
        (fuse_before == 1,
         f"OpFusion produced 1 joyl.fuse_add_rmsnorm (got {fuse_before})"),
        ("func.func private @joy_fuse_add_rmsnorm_kernel" in codegen_ir,
         "@joy_fuse_add_rmsnorm_kernel function defined"),
        ("gpu_kernel" in codegen_ir,
         "gpu_kernel attribute present"),
        ('kernel_name = "fuse_add_rmsnorm"' in codegen_ir,
         'kernel_name = "fuse_add_rmsnorm" attribute set'),
        ("call @joy_fuse_add_rmsnorm_kernel" in codegen_ir,
         "func.call @joy_fuse_add_rmsnorm_kernel issued"),
        (_count(codegen_ir, '"joyl.fuse_add_rmsnorm"(') == 0,
         "original joyl.fuse_add_rmsnorm eliminated"),
        ('"joyl.linear"(' in codegen_ir,
         "joyl.linear left untouched (codegen pass white-list works)"),
    ]

    assert _report(checks), "[Lesson10] Test 2 failed"
    print("\n[Lesson10]: ================== Test 2 PASSED ==================")


# Test 3: kernel body structure — every helper op in CodegenRMSNormPass appears
# ----------------------------------------------------------------------------
def test_kernel_body_structure(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 3: Codegen'd kernel body uses pure public dialects")
    print("=" * 60)

    H = 1024
    graph = Graph(name="kernel_body_test")
    x = graph.input([1, 64, H], "f16")
    scale = graph.input([H], "f16")
    normed = ops.rms_norm(x, scale, epsilon=1e-6)
    graph.set_outputs([normed])

    joyl_ir = _build_joy_ir_and_lower(graph)
    codegen_ir = _run_joy_opt(joyl_ir, ["--codegen-rms-norm"])
    assert codegen_ir is not None, "[Lesson10] Test 3 failed"

    if print_ir:
        print("\n--- Codegen IR ---")
        print(codegen_ir)
        print("--- end ---")

    expected_ops = {
        # kernel body
        "memref.dim":            "memref.dim queries dynamic rows/cols",
        "arith.index_cast":      "arith.index_cast (index → i64)",
        "arith.sitofp":          "arith.sitofp (i64 → f32 for mean)",
        "arith.divf":            "arith.divf (sum_sq / cols)",
        "arith.addf":            "arith.addf (sum reduction / + eps)",
        "arith.mulf":            "arith.mulf (square + scale)",
        "scf.for":               "scf.for (outer + inner loops)",
        "scf.yield":             "scf.yield (iter_args fold)",
        "math.rsqrt":            "math.rsqrt (1/sqrt computation)",
        "arith.extf":            "arith.extf (f16 → f32 promote)",
        "arith.truncf":          "arith.truncf (f32 → f16 demote)",
        "memref.load":           "memref.load inside kernel body",
        "memref.store":          "memref.store inside kernel body",
        # call site
        "memref.collapse_shape": "memref.collapse_shape (N-D → 2D at call site)",
        "memref.cast":           "memref.cast (static → dynamic shape)",
        "arith.constant":        "arith.constant for epsilon at call site",
    }

    checks = [(op in codegen_ir, desc) for op, desc in expected_ops.items()]

    # In addition: only public dialects + joy_*_kernel function names
    # (no joy/joyl/joyh op inside the kernel body)
    body_match = re.search(
        r"func\.func\s+private\s+@joy_rms_norm_kernel.*?^\s*\}\s*$",
        codegen_ir, flags=re.DOTALL | re.MULTILINE)
    body = body_match.group(0) if body_match else ""
    checks.append((bool(body),
                   "located @joy_rms_norm_kernel function body"))
    checks.append(("joy."  not in body,
                   "no joy.*  ops inside kernel body"))
    checks.append(("joyl." not in body,
                   "no joyl.* ops inside kernel body"))
    checks.append(("joyh." not in body,
                   "no joyh.* ops inside kernel body"))

    assert _report(checks), "[Lesson10] Test 3 failed"
    print("\n[Lesson10]: ================== Test 3 PASSED ==================")


# Test 4: multiple call sites share a single kernel function definition
# ----------------------------------------------------------------------------
def test_kernel_function_is_shared(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 4: One kernel function definition, multiple call sites")
    print("=" * 60)

    H = 1024
    graph = Graph(name="shared_kernel_test")
    x = graph.input([1, 64, H], "f16")
    s1 = graph.input([H], "f16")
    s2 = graph.input([H], "f16")
    residual = graph.input([1, 64, H], "f16")
    w = graph.input([512, H], "f16")

    normed1 = ops.rms_norm(x, s1, epsilon=1e-6)        # standalone rms_norm
    added = ops.add(normed1, residual)
    fused = ops.rms_norm(added, s2, epsilon=1e-6)      # → fuse_add_rmsnorm
    out = ops.linear(fused, w)
    graph.set_outputs([out])

    joyl_ir = _build_joy_ir_and_lower(graph)
    codegen_ir = _run_joy_opt(joyl_ir, ["--codegen-rms-norm"])
    assert codegen_ir is not None, "[Lesson10] Test 4 failed"

    if print_ir:
        print("\n--- Codegen IR ---")
        print(codegen_ir)
        print("--- end ---")

    rms_kernel_def = _count(codegen_ir, "func.func private @joy_rms_norm_kernel")
    fuse_kernel_def = _count(codegen_ir,
                              "func.func private @joy_fuse_add_rmsnorm_kernel")
    rms_calls = _count(codegen_ir, "call @joy_rms_norm_kernel")
    fuse_calls = _count(codegen_ir, "call @joy_fuse_add_rmsnorm_kernel")

    print(f"\n  Definitions / call sites:")
    print(f"    @joy_rms_norm_kernel        : def={rms_kernel_def} call={rms_calls}")
    print(f"    @joy_fuse_add_rmsnorm_kernel: def={fuse_kernel_def} call={fuse_calls}")

    checks = [
        (rms_kernel_def == 1,
         f"1 rms_norm kernel definition (got {rms_kernel_def}) — shared, not duplicated"),
        (fuse_kernel_def == 1,
         f"1 fuse_add_rmsnorm kernel definition (got {fuse_kernel_def})"),
        (rms_calls == 1,
         f"1 rms_norm call site (the standalone op, got {rms_calls})"),
        (fuse_calls == 1,
         f"1 fuse_add_rmsnorm call site (the fused op, got {fuse_calls})"),
        (_count(codegen_ir, '"joyl.rms_norm"(') == 0,
         "all joyl.rms_norm eliminated"),
        (_count(codegen_ir, '"joyl.fuse_add_rmsnorm"(') == 0,
         "all joyl.fuse_add_rmsnorm eliminated"),
    ]
    assert _report(checks), "[Lesson10] Test 4 failed"
    print("\n[Lesson10]: ================== Test 4 PASSED ==================")


# ============================================================================
# Part B — External-library CustomCall tests
# ============================================================================

# Test 5: joyl.linear → joyh.custom_call{joy_gpu_linear} (cuBLAS dispatch)
# ----------------------------------------------------------------------------
def test_linear_to_custom_call(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 5: joyl.linear → joyh.custom_call (external library path)")
    print("=" * 60)

    graph = Graph(name="linear_cc_test")
    x = graph.input([1, 64, 1024], "f16")
    w = graph.input([512, 1024], "f16")
    y = ops.linear(x, w)
    graph.set_outputs([y])

    joy_ir = graph.get_ir()
    joyl_ir = _run_joy_opt(joy_ir, ["--lower-joy-to-joyl"])
    assert joyl_ir is not None, "[Lesson10] Test 5: joy→joyl failed"

    joyh_ir = _run_joy_opt(joyl_ir, ["--lower-joyl-to-joyh"])
    assert joyh_ir is not None, "[Lesson10] Test 5: joyl→joyh failed"

    if print_ir:
        print("\n--- Joyh IR ---")
        print(joyh_ir)
        print("--- end ---")

    checks = [
        ('call_target_name = "joy_gpu_linear"' in joyh_ir,
         'call_target_name = "joy_gpu_linear" (mnemonic→joy_gpu_*)'),
        ('backend = "gpu"' in joyh_ir,
         'backend = "gpu" attribute present'),
        ("num_inputs = 2" in joyh_ir,
         "num_inputs = 2 (input + weight)"),
        ('"joyh.custom_call"(' in joyh_ir,
         "joyh.custom_call op produced"),
        ('"joyl.linear"(' not in joyh_ir,
         "joyl.linear eliminated"),
        # original memref types are preserved verbatim (no computation expansion)
        ("memref<1x64x1024xf16>" in joyh_ir,
         "input memref<1x64x1024xf16> preserved unchanged"),
        ("memref<512x1024xf16>" in joyh_ir,
         "weight memref<512x1024xf16> preserved unchanged"),
        ("memref<1x64x512xf16>" in joyh_ir,
         "output memref<1x64x512xf16> preserved unchanged"),
        # CustomCall does NOT introduce any arithmetic ops
        ("math.rsqrt" not in joyh_ir,
         "no math.rsqrt produced (computation stays opaque)"),
        ("scf.for" not in joyh_ir,
         "no scf.for produced (computation stays opaque)"),
    ]
    assert _report(checks), "[Lesson10] Test 5 failed"
    print("\n[Lesson10]: ================== Test 5 PASSED ==================")


# Test 6: original attributes are forwarded onto joyh.custom_call
# ----------------------------------------------------------------------------
def test_attribute_forwarding(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 6: Attribute forwarding onto joyh.custom_call")
    print("=" * 60)

    H, D = 1024, 64
    graph = Graph(name="attr_forward_test")

    # softmax has axis attribute
    x = graph.input([1, 64, H], "f16")
    sm = ops.softmax(x, axis=-1)

    # transpose has permutation attribute
    sm4 = graph.input([2, 4, 8, 16], "f16")
    tp = ops.transpose(sm4, [0, 2, 1, 3])

    # repeat_kv has n_rep attribute
    kv = graph.input([1, 2, 32, D], "f16")
    rkv = ops.repeat_kv(kv, n_rep=4)

    graph.set_outputs([sm, tp, rkv])

    joy_ir = graph.get_ir()
    joyl_ir = _run_joy_opt(joy_ir, ["--lower-joy-to-joyl"])
    assert joyl_ir is not None, "[Lesson10] Test 6: joy→joyl failed"

    joyh_ir = _run_joy_opt(joyl_ir, ["--lower-joyl-to-joyh"])
    assert joyh_ir is not None, "[Lesson10] Test 6: joyl→joyh failed"

    if print_ir:
        print("\n--- Joyh IR ---")
        print(joyh_ir)
        print("--- end ---")

    # MLIR prints op attributes in alphabetical order, so we scan
    # each custom_call line individually and check both directions.
    def _find_attr_in_call(ir, target, attr_name):
        """Find `attr_name = <value> : <type>` within the custom_call whose
        call_target_name == target.  Returns the captured value string."""
        for line in ir.splitlines():
            if (f'call_target_name = "{target}"' in line
                    and "joyh.custom_call" in line):
                m = re.search(rf'{attr_name}\s*=\s*(-?\d+)\s*:\s*i64', line)
                if m:
                    return m.group(1)
        return None

    def _find_attr_present(ir, target, attr_name):
        """Return True iff the custom_call line for `target` mentions attr_name."""
        for line in ir.splitlines():
            if (f'call_target_name = "{target}"' in line
                    and "joyh.custom_call" in line):
                return bool(re.search(rf'\b{attr_name}\b\s*=', line))
        return False

    softmax_axis_str = _find_attr_in_call(joyh_ir, "joy_gpu_softmax", "axis")
    n_rep_str        = _find_attr_in_call(joyh_ir, "joy_gpu_repeat_kv", "n_rep")
    transpose_perm   = _find_attr_present(joyh_ir, "joy_gpu_transpose",
                                          "permutation")

    checks = [
        (softmax_axis_str is not None,
         "softmax axis attribute carried onto custom_call"),
        (softmax_axis_str in {"2", "-1"},
         f"softmax axis is correct (got {softmax_axis_str!r}, "
         f"expected 2 (axis=-1 normalised) or -1)"),
        (transpose_perm,
         "transpose permutation attribute carried onto custom_call"),
        (n_rep_str is not None,
         "repeat_kv n_rep attribute carried onto custom_call"),
        (n_rep_str == "4",
         f"n_rep value preserved (got {n_rep_str!r}, expected '4')"),
        ('"joyl.softmax"('   not in joyh_ir, "joyl.softmax eliminated"),
        ('"joyl.transpose"(' not in joyh_ir, "joyl.transpose eliminated"),
        ('"joyl.repeat_kv"(' not in joyh_ir, "joyl.repeat_kv eliminated"),
    ]
    assert _report(checks), "[Lesson10] Test 6 failed"
    print("\n[Lesson10]: ================== Test 6 PASSED ==================")


# Test 7: MLP-style subgraph → one custom_call per op (5 total)
# ----------------------------------------------------------------------------
def test_mlp_subgraph_custom_calls(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 7: MLP subgraph — 5 ops → 5 joyh.custom_call's")
    print("=" * 60)

    H, INTER = 1024, 3072
    graph = Graph(name="mlp_cc_test")
    x = graph.input([1, 64, H], "f16")
    gate_w = graph.input([INTER, H], "f16")
    up_w = graph.input([INTER, H], "f16")
    down_w = graph.input([H, INTER], "f16")

    gate = ops.linear(x, gate_w)
    gate = ops.silu(gate)
    up = ops.linear(x, up_w)
    gate_up = ops.mul(gate, up)
    out = ops.linear(gate_up, down_w)
    graph.set_outputs([out])

    joy_ir = graph.get_ir()
    joyl_ir = _run_joy_opt(joy_ir, ["--lower-joy-to-joyl"])
    assert joyl_ir is not None, "[Lesson10] Test 7: joy→joyl failed"

    joyh_ir = _run_joy_opt(joyl_ir, ["--lower-joyl-to-joyh"])
    assert joyh_ir is not None, "[Lesson10] Test 7: joyl→joyh failed"

    if print_ir:
        print("\n--- Joyh IR ---")
        print(joyh_ir)
        print("--- end ---")

    expected_targets = {
        "joy_gpu_linear": 3,
        "joy_gpu_silu":   1,
        "joy_gpu_mul":    1,
    }
    checks = []
    for target, expected in expected_targets.items():
        actual = _count(joyh_ir, f'call_target_name = "{target}"')
        checks.append((actual == expected,
                       f'call_target_name = "{target}": expected {expected}, got {actual}'))

    joyl_remaining = _count(joyh_ir, '"joyl.')
    checks.append((joyl_remaining == 0,
                   f"all joyl ops eliminated ({joyl_remaining} remaining)"))

    total_cc = _count(joyh_ir, '"joyh.custom_call"(')
    checks.append((total_cc == 5,
                   f"5 custom_calls total (3 linear + 1 silu + 1 mul), got {total_cc}"))

    # CustomCall path keeps computation opaque — no math.rsqrt / scf.for / arith.divf
    # introduced (those would only appear via Codegen).
    checks.append(("math.rsqrt" not in joyh_ir,
                   "no math.rsqrt — external-library path never expands computation"))
    checks.append(("scf.for" not in joyh_ir,
                   "no scf.for — external-library path never expands computation"))

    assert _report(checks), "[Lesson10] Test 7 failed"
    print("\n[Lesson10]: ================== Test 7 PASSED ==================")


# ============================================================================
# Part C — Coexistence & workflow
# ============================================================================

# Test 8: Full backend pipeline — codegen + customcall coexist in joyh IR
# ----------------------------------------------------------------------------
def test_full_backend_pipeline(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 8: Full backend pipeline — codegen + customcall coexist")
    print("=" * 60)

    H = 1024
    graph = Graph(name="full_pipeline_test")
    hidden = graph.input([1, 64, H], "f16")
    residual = graph.input([1, 64, H], "f16")
    ln_w = graph.input([H], "f16")
    w = graph.input([512, H], "f16")

    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=1e-6)
    out = ops.linear(normed, w)
    graph.set_outputs([out])

    joyl_ir = _build_joy_ir_and_lower(graph)
    joyh_ir = _run_joy_opt(joyl_ir,
                           ["--codegen-rms-norm", "--lower-joyl-to-joyh"])
    assert joyh_ir is not None, "[Lesson10] Test 8 failed: backend pipeline failed"

    if print_ir:
        print("\n--- Joyh IR (codegen + customcall together) ---")
        print(joyh_ir)
        print("--- end ---")

    checks = [
        # Codegen side
        ("func.func private @joy_fuse_add_rmsnorm_kernel" in joyh_ir,
         "Codegen: kernel function @joy_fuse_add_rmsnorm_kernel exists"),
        ("call @joy_fuse_add_rmsnorm_kernel" in joyh_ir,
         "Codegen: kernel func.call issued"),
        ("gpu_kernel" in joyh_ir,
         "Codegen: gpu_kernel attribute on kernel function"),
        ("math.rsqrt" in joyh_ir and "scf.for" in joyh_ir,
         "Codegen: math.rsqrt + scf.for proves computation is in IR"),
        # CustomCall side
        ('call_target_name = "joy_gpu_linear"' in joyh_ir,
         "CustomCall: joyl.linear → joyh.custom_call{joy_gpu_linear}"),
        ('backend = "gpu"' in joyh_ir,
         "CustomCall: backend attribute present"),
        ("num_inputs = 2" in joyh_ir,
         "CustomCall: num_inputs=2 for linear"),
        # And: every joyl op has been retired
        ('"joyl.' not in joyh_ir,
         "no joyl.* ops remain in final joyh IR"),
        # rms_norm/fuse_add_rmsnorm are NOT routed to custom_call
        ('call_target_name = "joy_gpu_rms_norm"' not in joyh_ir,
         "joyl.rms_norm is NOT routed to joyh.custom_call (codegen has priority)"),
        ('call_target_name = "joy_gpu_fuse_add_rmsnorm"' not in joyh_ir,
         "joyl.fuse_add_rmsnorm is NOT routed to joyh.custom_call"),
    ]
    assert _report(checks), "[Lesson10] Test 8 failed"
    print("\n[Lesson10]: ================== Test 8 PASSED ==================")


# Test 9: codegen pass is a no-op when no rms_norm-family op is present
# ----------------------------------------------------------------------------
def test_codegen_pass_is_noop_when_inapplicable(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 9: --codegen-rms-norm is a no-op when no target op exists")
    print("=" * 60)

    graph = Graph(name="codegen_noop_test")
    x = graph.input([1, 64, 1024], "f16")
    w = graph.input([512, 1024], "f16")
    y = ops.linear(x, w)
    graph.set_outputs([y])

    joy_ir = graph.get_ir()
    joyl_ir = _run_joy_opt(joy_ir, ["--lower-joy-to-joyl"])
    assert joyl_ir is not None, "[Lesson10] Test 9: lowering failed"

    codegen_ir = _run_joy_opt(joyl_ir, ["--codegen-rms-norm"])
    assert codegen_ir is not None, "[Lesson10] Test 9: codegen failed"

    if print_ir:
        print("\n--- IR after codegen pass (should be a no-op) ---")
        print(codegen_ir)
        print("--- end ---")

    checks = [
        ("@joy_rms_norm_kernel" not in codegen_ir,
         "no @joy_rms_norm_kernel inserted (no rms_norm to codegen)"),
        ("@joy_fuse_add_rmsnorm_kernel" not in codegen_ir,
         "no @joy_fuse_add_rmsnorm_kernel inserted"),
        ("math.rsqrt" not in codegen_ir,
         "no math.rsqrt op produced by no-op pass"),
        ('"joyl.linear"(' in codegen_ir,
         "joyl.linear preserved untouched by codegen pass"),
    ]
    assert _report(checks), "[Lesson10] Test 9 failed"
    print("\n[Lesson10]: ================== Test 9 PASSED ==================")


# Test 10: running the full backend twice is idempotent
# ----------------------------------------------------------------------------
def test_backend_pipeline_idempotent(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 10: codegen + customcall is idempotent on its own output")
    print("=" * 60)

    H = 1024
    graph = Graph(name="idempotent_test")
    hidden = graph.input([1, 64, H], "f16")
    residual = graph.input([1, 64, H], "f16")
    ln_w = graph.input([H], "f16")
    w = graph.input([512, H], "f16")

    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=1e-6)
    out = ops.linear(normed, w)
    graph.set_outputs([out])

    joyl_ir = _build_joy_ir_and_lower(graph)

    # First run
    ir1 = _run_joy_opt(joyl_ir,
                       ["--codegen-rms-norm", "--lower-joyl-to-joyh"])
    assert ir1 is not None, "[Lesson10] Test 10: first run failed"

    # Second run (feed IR1 back through the same passes)
    ir2 = _run_joy_opt(ir1,
                       ["--codegen-rms-norm", "--lower-joyl-to-joyh"])
    assert ir2 is not None, "[Lesson10] Test 10: second run failed"

    if print_ir:
        print("\n--- Difference between run1 and run2 (should be empty) ---")
        if ir1 != ir2:
            import difflib
            for line in difflib.unified_diff(
                    ir1.splitlines(), ir2.splitlines(),
                    lineterm="", fromfile="run1", tofile="run2"):
                print(line)
        print("--- end ---")

    def count_signature(ir):
        return (
            _count(ir, "func.func private @joy_rms_norm_kernel"),
            _count(ir, "func.func private @joy_fuse_add_rmsnorm_kernel"),
            _count(ir, "call @joy_rms_norm_kernel"),
            _count(ir, "call @joy_fuse_add_rmsnorm_kernel"),
            _count(ir, '"joyh.custom_call"('),
            _count(ir, '"joyl.'),
        )

    sig1, sig2 = count_signature(ir1), count_signature(ir2)
    print(f"\n  signature run1 = {sig1}")
    print(f"  signature run2 = {sig2}")

    # NOTE: this subgraph was already through --joy-optimization-pipeline,
    # so add+rms_norm have been fused into fuse_add_rmsnorm — there is no
    # standalone rms_norm kernel in this run.  We therefore expect exactly
    # one fuse_add_rmsnorm kernel definition, one of its calls, and one
    # custom_call for the linear op.
    (rms_def, fuse_def, rms_call, fuse_call, cc_count, joyl_left) = sig2
    checks = [
        (sig1 == sig2,
         "kernel/call/custom_call counts unchanged after second pass run"),
        (fuse_def == 1,
         f"exactly 1 fuse_add_rmsnorm kernel function definition "
         f"(got {fuse_def})"),
        (fuse_call >= 1,
         f"fuse_add_rmsnorm kernel func.call present (got {fuse_call})"),
        (cc_count >= 1,
         f"joyh.custom_call still present (got {cc_count})"),
        (joyl_left == 0,
         "still no joyl.* ops (no regressions)"),
        # Stronger guarantee: textual IR is identical run-to-run.
        (ir1 == ir2,
         "byte-identical IR across two runs (pure idempotence)"),
    ]
    assert _report(checks), "[Lesson10] Test 10 failed"
    print("\n[Lesson10]: ================== Test 10 PASSED =================")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Lesson 10: Auto operator codegen vs external library call")
    parser.add_argument("--print-ir-all", action="store_true",
                        help="Print IR at every stage")
    args = parser.parse_args()
    p = args.print_ir_all

    print("=" * 60)
    print("  Lesson 10: Auto Codegen vs External Library CustomCall")
    print("=" * 60)

    print("\n" + "-" * 60)
    print("  Part A: Auto Codegen path (CodegenRMSNormPass.cpp)")
    print("-" * 60)
    test_rms_norm_codegen(print_ir=p)
    test_fuse_add_rmsnorm_codegen(print_ir=p)
    test_kernel_body_structure(print_ir=p)
    test_kernel_function_is_shared(print_ir=p)

    print("\n" + "-" * 60)
    print("  Part B: External-library CustomCall path "
          "(LowerJoylToJoyhPass.cpp)")
    print("-" * 60)
    test_linear_to_custom_call(print_ir=p)
    test_attribute_forwarding(print_ir=p)
    test_mlp_subgraph_custom_calls(print_ir=p)

    print("\n" + "-" * 60)
    print("  Part C: Coexistence & workflow")
    print("-" * 60)
    test_full_backend_pipeline(print_ir=p)
    test_codegen_pass_is_noop_when_inapplicable(print_ir=p)
    test_backend_pipeline_idempotent(print_ir=p)

    print("\n" + "=" * 60)
    print("  ALL LESSON 10 TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
