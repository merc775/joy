#!/usr/bin/env python3
"""Lesson 8: AI Compiler Mid-end — Custom Op Fusion.

Validates the OpFusionPass in joy/lib/optimizer/OpFusionPass.cpp:

  Pattern  : AddRMSNormFusionPattern
  Driver   : applyPatternsAndFoldGreedily (with useTopDownTraversal = true)
  Match    : OpRewritePattern<joy::RMSNormOp>
  Rewrite  : Three-step replace:
               1. replaceOp(normOp, fuseOp->getResult(1))
               2. replaceAllUsesWith(addOp.getResult(0), fuseOp->getResult(0))
               3. eraseOp(addOp)

Tests:
  Test 1: basic fusion — add+rms_norm → fuse_add_rmsnorm
  Test 2: multi-output semantics — fuse op has 2 results
  Test 3: standalone rms_norm (input not from add) is NOT fused
  Test 4: standalone add (output not consumed by rms_norm) is NOT fused
  Test 5: epsilon attribute is preserved on the fused op
  Test 6: shape/dtype match — multi-dim tensor still fuses
  Test 7: multi-layer decoder — multiple add+rms_norm pairs all fuse
  Test 8: end-to-end full pipeline drives fusion + CF + CSE cleanup
  Test 9: fusion is idempotent — re-running the Pass does not blow up
  Test 10: fused IR still lowers cleanly to joyl via --lower-joy-to-joyl

Usage:
    python3 joy/tests/python_tests/test_lesson8.py
    python3 joy/tests/python_tests/test_lesson8.py --print-ir-all
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


def _require_joy_opt():
    if not os.path.exists(JOY_OPT):
        print(f"  SKIP: joy-opt not found at {JOY_OPT}. "
              f"Run scripts/build.sh first.")
        return False
    return True


def _run_joy_opt(input_ir, passes, timeout=30):
    """Run joy-opt with the given passes; return (stdout, stderr, rc)."""
    if not os.path.exists(JOY_OPT):
        return None, "joy-opt not built", -1
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mlir")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(input_ir)
        cmd = [JOY_OPT] + passes + [tmp_path]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout)
        return result.stdout, result.stderr, result.returncode
    finally:
        os.unlink(tmp_path)


def _count_op(ir, op_name):
    return ir.count(f'"{op_name}"(')


# ============================================================================
# Test 1: basic fusion
# ============================================================================
def test_basic_fusion(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 1: basic add + rms_norm → fuse_add_rmsnorm")
    print("=" * 60)

    H = 1024
    g = Graph(name="basic_fusion")
    hidden = g.input([1, 64, H], "f16")
    residual = g.input([1, 64, H], "f16")
    ln_w = g.input([H], "f16")
    proj_w = g.input([512, H], "f16")

    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=1e-6)
    out = ops.linear(normed, proj_w)
    g.set_outputs([out])
    ir_before = g.get_ir()

    add_before = _count_op(ir_before, "joy.add")
    rms_before = _count_op(ir_before, "joy.rms_norm")

    stdout, _, rc = _run_joy_opt(ir_before, ["--joy-op-fusion"])
    if print_ir:
        print(stdout)

    add_after = _count_op(stdout, "joy.add")
    rms_after = _count_op(stdout, "joy.rms_norm")
    fuse_after = _count_op(stdout, "joy.fuse_add_rmsnorm")

    print(f"\n  Before: add={add_before}, rms_norm={rms_before}")
    print(f"  After:  add={add_after}, rms_norm={rms_after}, "
          f"fuse={fuse_after}")

    checks = [
        (rc == 0, "joy-opt --joy-op-fusion returned 0"),
        (add_before == 1, "before: 1 add"),
        (rms_before == 1, "before: 1 rms_norm"),
        (fuse_after == 1, f"after: 1 fuse_add_rmsnorm (got {fuse_after})"),
        (add_after == 0, f"after: add consumed (got {add_after})"),
        (rms_after == 0, f"after: rms_norm consumed (got {rms_after})"),
        ('"joy.linear"' in stdout, "downstream joy.linear preserved"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson8] Test 1 failed"
    print(f"\n[Lesson8]: ================== Test 1 PASSED ==================")


# ============================================================================
# Test 2: multi-output semantics
# ============================================================================
def test_multi_output_semantics(print_ir=False):
    """The fused op must produce 2 results (add_out, norm_out) so the add
    result is still available for the next residual connection."""
    print("\n" + "=" * 60)
    print("  Test 2: fused op has 2 results (add_out + norm_out)")
    print("=" * 60)

    H = 1024
    g = Graph(name="multi_output")
    hidden = g.input([1, 64, H], "f16")
    residual = g.input([1, 64, H], "f16")
    ln_w = g.input([H], "f16")
    w2 = g.input([H, H], "f16")
    ln2_w = g.input([H], "f16")
    final_w = g.input([512, H], "f16")

    # First residual block: add -> rms_norm -> linear
    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=1e-6)
    proj = ops.linear(normed, w2)

    # Second residual block: ADD uses the result of the first ADD
    # (i.e. `added`) directly as the residual.  This is exactly the
    # Transformer pattern where the add result feeds the *next* layer.
    added2 = ops.add(added, proj)
    normed2 = ops.rms_norm(added2, ln2_w, epsilon=1e-6)
    out = ops.linear(normed2, final_w)
    g.set_outputs([out])
    ir_before = g.get_ir()

    stdout, _, rc = _run_joy_opt(ir_before, ["--joy-op-fusion"])
    if print_ir:
        print(stdout)

    fuse_count = _count_op(stdout, "joy.fuse_add_rmsnorm")
    add_count = _count_op(stdout, "joy.add")
    rms_count = _count_op(stdout, "joy.rms_norm")

    # The fused op should have a result of (T, T) — two tensor results.
    # Detect by counting the comma-style result of the op.
    has_two_results = re.search(
        r'"joy\.fuse_add_rmsnorm"\([^)]+\)\s*'
        r'(?:\{[^}]*\})?\s*'
        r':\s*\([^)]+\)\s*->\s*\(\s*tensor[^,]+,\s*tensor[^)]+\)',
        stdout) is not None

    print(f"\n  fuse_add_rmsnorm = {fuse_count}, "
          f"add = {add_count}, rms_norm = {rms_count}")

    checks = [
        (rc == 0, "joy-opt --joy-op-fusion returned 0"),
        (fuse_count == 2, f"2 fuse_add_rmsnorm created (got {fuse_count})"),
        (add_count == 0, f"all add consumed (got {add_count})"),
        (rms_count == 0, f"all rms_norm consumed (got {rms_count})"),
        (has_two_results,
         "fused op signature shows 2 tensor results in IR text"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson8] Test 2 failed"
    print(f"\n[Lesson8]: ================== Test 2 PASSED ==================")


# ============================================================================
# Test 3: standalone rms_norm not fused
# ============================================================================
def test_standalone_rms_not_fused(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 3: standalone rms_norm (input is NOT an add) is not fused")
    print("=" * 60)

    H = 1024
    g = Graph(name="standalone_rms")
    x = g.input([1, 64, H], "f16")
    scale = g.input([H], "f16")
    w = g.input([512, H], "f16")
    n = ops.rms_norm(x, scale, epsilon=1e-6)
    y = ops.linear(n, w)
    g.set_outputs([y])
    ir_before = g.get_ir()

    stdout, _, rc = _run_joy_opt(ir_before, ["--joy-op-fusion"])
    if print_ir:
        print(stdout)

    rms_after = _count_op(stdout, "joy.rms_norm")
    fuse_after = _count_op(stdout, "joy.fuse_add_rmsnorm")

    checks = [
        (rc == 0, "joy-opt --joy-op-fusion returned 0"),
        (rms_after == 1, f"rms_norm preserved (got {rms_after})"),
        (fuse_after == 0, f"no fuse_add_rmsnorm created (got {fuse_after})"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson8] Test 3 failed"
    print(f"\n[Lesson8]: ================== Test 3 PASSED ==================")


# ============================================================================
# Test 4: standalone add (no following rms_norm) is not fused
# ============================================================================
def test_standalone_add_not_fused(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 4: standalone add (no consumer rms_norm) is not fused")
    print("=" * 60)

    H = 1024
    g = Graph(name="standalone_add")
    a = g.input([1, 64, H], "f16")
    b = g.input([1, 64, H], "f16")
    w = g.input([512, H], "f16")
    s = ops.add(a, b)               # produces an add but…
    y = ops.linear(s, w)            # …its consumer is a linear, not rms_norm
    g.set_outputs([y])
    ir_before = g.get_ir()

    stdout, _, rc = _run_joy_opt(ir_before, ["--joy-op-fusion"])
    if print_ir:
        print(stdout)

    add_after = _count_op(stdout, "joy.add")
    fuse_after = _count_op(stdout, "joy.fuse_add_rmsnorm")

    checks = [
        (rc == 0, "joy-opt --joy-op-fusion returned 0"),
        (add_after == 1, f"add preserved (got {add_after})"),
        (fuse_after == 0, f"no fuse_add_rmsnorm created (got {fuse_after})"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson8] Test 4 failed"
    print(f"\n[Lesson8]: ================== Test 4 PASSED ==================")


# ============================================================================
# Test 5: epsilon attribute preserved
# ============================================================================
def test_epsilon_preserved(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 5: epsilon attribute transferred to fused op")
    print("=" * 60)

    H = 1024
    eps = 1.23e-5  # an unusual epsilon to make detection easy
    g = Graph(name="eps_test")
    hidden = g.input([1, 64, H], "f16")
    residual = g.input([1, 64, H], "f16")
    ln_w = g.input([H], "f16")
    proj_w = g.input([512, H], "f16")
    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=eps)
    out = ops.linear(normed, proj_w)
    g.set_outputs([out])
    ir_before = g.get_ir()

    stdout, _, rc = _run_joy_opt(ir_before, ["--joy-op-fusion"])
    if print_ir:
        print(stdout)

    has_eps = "1.230000e-05" in stdout
    fuse_line = ""
    for line in stdout.split("\n"):
        if "joy.fuse_add_rmsnorm" in line:
            fuse_line = line
            break

    checks = [
        (rc == 0, "joy-opt --joy-op-fusion returned 0"),
        (_count_op(stdout, "joy.fuse_add_rmsnorm") == 1,
         "exactly 1 fuse_add_rmsnorm"),
        (has_eps, "epsilon = 1.230000e-05 present in fused IR"),
        ("epsilon" in fuse_line,
         "epsilon attribute on fuse_add_rmsnorm op"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson8] Test 5 failed"
    print(f"\n[Lesson8]: ================== Test 5 PASSED ==================")


# ============================================================================
# Test 6: multi-dim tensor still fuses
# ============================================================================
def test_multi_dim_tensor(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 6: multi-dim tensor (B,S,D) still fuses correctly")
    print("=" * 60)

    B, S, H = 4, 128, 512
    g = Graph(name="multi_dim")
    hidden = g.input([B, S, H], "f16")
    residual = g.input([B, S, H], "f16")
    ln_w = g.input([H], "f16")
    proj_w = g.input([256, H], "f16")

    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=1e-6)
    out = ops.linear(normed, proj_w)
    g.set_outputs([out])
    ir_before = g.get_ir()

    stdout, _, rc = _run_joy_opt(ir_before, ["--joy-op-fusion"])
    if print_ir:
        print(stdout)

    fuse_after = _count_op(stdout, "joy.fuse_add_rmsnorm")
    has_shape = f"tensor<{B}x{S}x{H}xf16>" in stdout
    checks = [
        (rc == 0, "joy-opt --joy-op-fusion returned 0"),
        (fuse_after == 1, f"1 fuse_add_rmsnorm created (got {fuse_after})"),
        (has_shape, f"shape tensor<{B}x{S}x{H}xf16> preserved"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson8] Test 6 failed"
    print(f"\n[Lesson8]: ================== Test 6 PASSED ==================")


# ============================================================================
# Test 7: multi-layer decoder — many fusions
# ============================================================================
def test_multi_layer_fusion(print_ir=False):
    """Build a 2-layer mini-decoder, expect 4 fusions (2 layers × 2 residual
    blocks each).  Mirrors the structure of Qwen3-0.6B decoder layers."""
    print("\n" + "=" * 60)
    print("  Test 7: multi-layer decoder — 4 fusions expected")
    print("=" * 60)

    H, INTER = 1024, 3072
    g = Graph(name="multi_layer")
    hidden = g.input([1, 64, H], "f16")

    for li in range(2):
        residual = hidden

        # Block 1: attn-like (just a proj for brevity)
        ln1_w = g.input([H], "f16", name=f"l{li}.ln1.weight")
        norm1 = ops.rms_norm(hidden, ln1_w, epsilon=1e-6)
        attn_w = g.input([H, H], "f16", name=f"l{li}.attn.weight")
        attn_out = ops.linear(norm1, attn_w)
        hidden = ops.add(residual, attn_out)

        # Block 2: MLP
        residual = hidden
        ln2_w = g.input([H], "f16", name=f"l{li}.ln2.weight")
        norm2 = ops.rms_norm(hidden, ln2_w, epsilon=1e-6)
        gate_w = g.input([INTER, H], "f16", name=f"l{li}.gate.weight")
        up_w = g.input([INTER, H], "f16", name=f"l{li}.up.weight")
        down_w = g.input([H, INTER], "f16", name=f"l{li}.down.weight")
        gate = ops.silu(ops.linear(norm2, gate_w))
        up = ops.linear(norm2, up_w)
        mlp_out = ops.linear(ops.mul(gate, up), down_w)
        hidden = ops.add(residual, mlp_out)

    final_ln = g.input([H], "f16", name="final_ln.weight")
    hidden = ops.rms_norm(hidden, final_ln, epsilon=1e-6)
    g.set_outputs([hidden])
    ir_before = g.get_ir()

    add_before = _count_op(ir_before, "joy.add")
    rms_before = _count_op(ir_before, "joy.rms_norm")

    stdout, _, rc = _run_joy_opt(ir_before, ["--joy-op-fusion"])
    if print_ir:
        print(stdout)

    fuse_after = _count_op(stdout, "joy.fuse_add_rmsnorm")
    add_after = _count_op(stdout, "joy.add")
    rms_after = _count_op(stdout, "joy.rms_norm")

    print(f"\n  Before: add={add_before}, rms_norm={rms_before}")
    print(f"  After:  fuse={fuse_after}, add={add_after}, "
          f"rms_norm={rms_after}")

    # 2 layers × (post-attn fusion + post-MLP fusion) = 4 fusions.
    # Remaining rms_norm:
    #   - layer0 input_ln  (input is not add)
    #   - layer1 input_ln  (input is the *result* of the first MLP add ->
    #                       FUSED on top of it, so this becomes part of the
    #                       fusion not standalone)
    #   - actually layer1.ln1 IS preceded by an add (the MLP residual),
    #     so it WILL be fused.  So only layer0.ln1 and final_ln remain.
    checks = [
        (rc == 0, "joy-opt --joy-op-fusion returned 0"),
        (add_before == 4, f"before: 4 add (2 layers × 2) (got {add_before})"),
        (rms_before == 5,
         f"before: 5 rms_norm (2 layers × 2 + final) (got {rms_before})"),
        (fuse_after == 4, f"after: 4 fusions (got {fuse_after})"),
        (add_after == 0, f"all add consumed (got {add_after})"),
        (rms_after == 1,
         f"1 standalone rms_norm (layer0 input_ln; final got fused) "
         f"(got {rms_after})"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson8] Test 7 failed"
    print(f"\n[Lesson8]: ================== Test 7 PASSED ==================")


# ============================================================================
# Test 8: end-to-end pipeline
# ============================================================================
def test_full_pipeline(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 8: --joy-optimization-pipeline drives fusion + CF + CSE")
    print("=" * 60)

    H = 1024
    g = Graph(name="pipeline_e2e")
    hidden = g.input([1, 64, H], "f16")
    residual = g.input([1, 64, H], "f16")
    ln_w = g.input([H], "f16")
    proj_w = g.input([512, H], "f16")

    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=1e-6)
    out = ops.linear(normed, proj_w)
    g.set_outputs([out])
    ir_before = g.get_ir()

    stdout, _, rc = _run_joy_opt(ir_before, ["--joy-optimization-pipeline"])
    if print_ir:
        print(stdout)

    fuse_after = _count_op(stdout, "joy.fuse_add_rmsnorm")
    add_after = _count_op(stdout, "joy.add")
    rms_after = _count_op(stdout, "joy.rms_norm")
    linear_after = _count_op(stdout, "joy.linear")

    checks = [
        (rc == 0, "joy-opt --joy-optimization-pipeline returned 0"),
        (fuse_after == 1, f"fuse_add_rmsnorm = 1 (got {fuse_after})"),
        (add_after == 0, f"no add (got {add_after})"),
        (rms_after == 0, f"no rms_norm (got {rms_after})"),
        (linear_after == 1, "linear preserved"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson8] Test 8 failed"
    print(f"\n[Lesson8]: ================== Test 8 PASSED ==================")


# ============================================================================
# Test 9: fusion is idempotent
# ============================================================================
def test_fusion_idempotent(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 9: running fusion twice produces the same IR (no diverge)")
    print("=" * 60)

    H = 1024
    g = Graph(name="idempotent")
    hidden = g.input([1, 64, H], "f16")
    residual = g.input([1, 64, H], "f16")
    ln_w = g.input([H], "f16")
    proj_w = g.input([512, H], "f16")
    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=1e-6)
    out = ops.linear(normed, proj_w)
    g.set_outputs([out])
    ir_before = g.get_ir()

    once, _, rc1 = _run_joy_opt(ir_before, ["--joy-op-fusion"])
    twice, _, rc2 = _run_joy_opt(once, ["--joy-op-fusion"])
    if print_ir:
        print("---- after 1st fusion ----")
        print(once)
        print("---- after 2nd fusion ----")
        print(twice)

    # Normalize whitespace differences from the two emissions.
    def _strip(text):
        return re.sub(r"\s+", " ", text or "").strip()

    fuse_once = _count_op(once, "joy.fuse_add_rmsnorm")
    fuse_twice = _count_op(twice, "joy.fuse_add_rmsnorm")
    checks = [
        (rc1 == 0 and rc2 == 0, "both runs returned 0"),
        (fuse_once == 1, f"1st run: 1 fuse_add_rmsnorm (got {fuse_once})"),
        (fuse_twice == 1, f"2nd run: still 1 fuse_add_rmsnorm (got {fuse_twice})"),
        (_strip(once) == _strip(twice),
         "IR is byte-identical after the 2nd pass (idempotence)"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson8] Test 9 failed"
    print(f"\n[Lesson8]: ================== Test 9 PASSED ==================")


# ============================================================================
# Test 10: fused IR lowers cleanly to joyl
# ============================================================================
def test_fused_lowers_to_joyl(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 10: fused IR survives --lower-joy-to-joyl")
    print("=" * 60)

    H = 1024
    g = Graph(name="fuse_then_lower")
    hidden = g.input([1, 64, H], "f16")
    residual = g.input([1, 64, H], "f16")
    ln_w = g.input([H], "f16")
    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=1e-6)
    g.set_outputs([normed])
    ir = g.get_ir()

    stdout, stderr, rc = _run_joy_opt(
        ir, ["--joy-op-fusion", "--lower-joy-to-joyl"])
    if print_ir:
        print(stdout)

    has_joyl_fuse = '"joyl.fuse_add_rmsnorm"' in stdout
    has_memref = "memref<" in stdout
    has_tensor = "tensor<" in stdout
    checks = [
        (rc == 0, f"joy-opt --joy-op-fusion --lower-joy-to-joyl returned 0 "
                  f"(stderr: {stderr.strip()[:200] if stderr else ''})"),
        (has_joyl_fuse, "joyl.fuse_add_rmsnorm appears in lowered IR"),
        (has_memref, "memref<...> types appear after lowering"),
        (not has_tensor, "no tensor<...> types remain in joyl IR"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson8] Test 10 failed"
    print(f"\n[Lesson8]: ================== Test 10 PASSED =================")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Lesson 8: Custom Op Fusion — Test")
    parser.add_argument("--print-ir-all", action="store_true",
                        help="Print IR at every stage")
    args = parser.parse_args()
    p = args.print_ir_all

    print("=" * 60)
    print("  Lesson 8: AI Compiler Mid-end — Custom Op Fusion")
    print("=" * 60)

    if not _require_joy_opt():
        sys.exit(0)

    test_basic_fusion(print_ir=p)
    test_multi_output_semantics(print_ir=p)
    test_standalone_rms_not_fused(print_ir=p)
    test_standalone_add_not_fused(print_ir=p)
    test_epsilon_preserved(print_ir=p)
    test_multi_dim_tensor(print_ir=p)
    test_multi_layer_fusion(print_ir=p)
    test_full_pipeline(print_ir=p)
    test_fusion_idempotent(print_ir=p)
    test_fused_lowers_to_joyl(print_ir=p)

    print("\n" + "=" * 60)
    print("  ALL LESSON 8 TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
