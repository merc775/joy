#!/usr/bin/env python3
"""Lesson 7: AI Compiler Mid-end — Pass & Pipeline.

Validates the infrastructure shared by every Joy optimization pass:
    joy/lib/optimizer/ConstantFolding.cpp
    joy/lib/optimizer/CSE.cpp
    joy/lib/optimizer/Pipeline.cpp

The tests deliberately *do not* exercise OpFusion-specific behaviour
(that is the focus of test_lesson8.py).  Here we focus on:

  Test 1: --joy-constant-folding survives an IR with no foldable constants
  Test 2: --joy-cse eliminates exact duplicates (same op, operands, attrs)
  Test 3: --joy-cse keeps ops apart when attributes differ
  Test 4: --joy-cse keeps ops apart when operands differ
  Test 5: --joy-cse keeps ops apart when result types differ
  Test 6: --joy-cse cleans up dead pure ops (no user)
  Test 7: --joy-optimization-pipeline applies CF + CSE + Fusion + CF + CSE
  Test 8: joy-opt --help registers exactly 4 joy-* pass / pipeline names
  Test 9: joy-opt rejects an unknown pass argument
  Test 10: --mlir-print-ir-after-all output covers every stage of the pipeline

Prerequisites:
    joy/build/bin/joy-opt must exist (run scripts/build.sh first)

Usage:
    python3 joy/tests/python_tests/test_lesson7.py
    python3 joy/tests/python_tests/test_lesson7.py --print-ir-all
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


def _run_joy_opt(input_ir, passes, timeout=30, extra_args=None):
    """Run joy-opt with the given passes on input_ir text, return (stdout, stderr, rc)."""
    if not os.path.exists(JOY_OPT):
        return None, "joy-opt not built", -1
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mlir")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(input_ir)
        cmd = [JOY_OPT] + (extra_args or []) + passes + [tmp_path]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout)
        return result.stdout, result.stderr, result.returncode
    finally:
        os.unlink(tmp_path)


def _count_op(ir, op_name):
    """Count occurrences of a Joy/Joyl/Joyh op in IR text."""
    return ir.count(f'"{op_name}"(')


# ============================================================================
# Test 1: ConstantFolding survives an IR with no foldable constants
# ============================================================================
def test_constant_folding_noop(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 1: --joy-constant-folding is a noop on Joy-only IR")
    print("=" * 60)

    g = Graph(name="cf_noop")
    x = g.input([1, 64, 1024], "f16")
    w = g.input([512, 1024], "f16")
    y = ops.linear(x, w)
    g.set_outputs([y])
    ir = g.get_ir()

    stdout, stderr, rc = _run_joy_opt(ir, ["--joy-constant-folding"])
    if print_ir:
        print(stdout)

    ok_rc = rc == 0
    ok_linear = '"joy.linear"' in stdout
    ok_func = "func.func @cf_noop" in stdout
    ok_no_added_constants = stdout.count("arith.constant") == ir.count("arith.constant")

    checks = [
        (ok_rc, "joy-opt --joy-constant-folding returned 0"),
        (ok_linear, "joy.linear preserved (nothing to fold)"),
        (ok_func, "function name preserved"),
        (ok_no_added_constants, "no spurious arith.constant added"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson7] Test 1 failed"
    print(f"\n[Lesson7]: ================== Test 1 PASSED ==================")


# ============================================================================
# Test 2: CSE eliminates exact duplicates
# ============================================================================
def test_cse_eliminates_duplicates(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 2: --joy-cse eliminates exact duplicate operations")
    print("=" * 60)

    g = Graph(name="cse_dup")
    x = g.input([1, 64, 1024], "f16")
    scale = g.input([1024], "f16")

    n1 = ops.rms_norm(x, scale, epsilon=1e-6)
    n2 = ops.rms_norm(x, scale, epsilon=1e-6)  # identical to n1
    out = ops.add(n1, n2)
    g.set_outputs([out])
    ir_before = g.get_ir()

    stdout, _, rc = _run_joy_opt(ir_before, ["--joy-cse"])
    if print_ir:
        print(stdout)

    before = _count_op(ir_before, "joy.rms_norm")
    after = _count_op(stdout, "joy.rms_norm")
    add_after = _count_op(stdout, "joy.add")

    checks = [
        (rc == 0, "joy-opt --joy-cse returned 0"),
        (before == 2, f"before: 2 rms_norm (got {before})"),
        (after == 1, f"after:  1 rms_norm (duplicate removed) (got {after})"),
        (add_after == 1, "joy.add preserved"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson7] Test 2 failed"
    print(f"\n[Lesson7]: ================== Test 2 PASSED ==================")


# ============================================================================
# Test 3: CSE preserves ops with different attrs (epsilon)
# ============================================================================
def test_cse_different_attrs(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 3: --joy-cse preserves ops with different attributes")
    print("=" * 60)

    g = Graph(name="cse_attrs")
    x = g.input([1, 64, 1024], "f16")
    scale = g.input([1024], "f16")

    # Use values that are exactly representable in f32 so they survive
    # the MLIR text round-trip with a predictable string form.
    eps_small = 0.25
    eps_large = 0.5
    n1 = ops.rms_norm(x, scale, epsilon=eps_small)
    n2 = ops.rms_norm(x, scale, epsilon=eps_large)  # different epsilon
    out = ops.add(n1, n2)
    g.set_outputs([out])
    ir_before = g.get_ir()

    stdout, _, rc = _run_joy_opt(ir_before, ["--joy-cse"])
    if print_ir:
        print(stdout)

    after = _count_op(stdout, "joy.rms_norm")
    # Count distinct epsilon attribute values seen in the output.
    eps_values = set(re.findall(r"epsilon\s*=\s*([0-9eE\.\+\-]+)\s*:\s*f32",
                                stdout))
    has_two_eps = len(eps_values) == 2

    checks = [
        (rc == 0, "joy-opt --joy-cse returned 0"),
        (after == 2, f"both rms_norm ops preserved (got {after}, expected 2)"),
        (has_two_eps,
         f"two distinct epsilon attr values in IR (got {sorted(eps_values)})"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson7] Test 3 failed"
    print(f"\n[Lesson7]: ================== Test 3 PASSED ==================")


# ============================================================================
# Test 4: CSE preserves ops with different operand SSA values
# ============================================================================
def test_cse_different_operands(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 4: --joy-cse preserves ops with different operand values")
    print("=" * 60)

    g = Graph(name="cse_ops")
    x = g.input([1, 64, 1024], "f16")
    s1 = g.input([1024], "f16")   # two different SSA values
    s2 = g.input([1024], "f16")   # same shape, different value

    n1 = ops.rms_norm(x, s1, epsilon=1e-6)
    n2 = ops.rms_norm(x, s2, epsilon=1e-6)
    out = ops.add(n1, n2)
    g.set_outputs([out])
    ir_before = g.get_ir()

    stdout, _, rc = _run_joy_opt(ir_before, ["--joy-cse"])
    if print_ir:
        print(stdout)

    after = _count_op(stdout, "joy.rms_norm")
    checks = [
        (rc == 0, "joy-opt --joy-cse returned 0"),
        (after == 2, f"both rms_norm preserved (different scale operands) (got {after})"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson7] Test 4 failed"
    print(f"\n[Lesson7]: ================== Test 4 PASSED ==================")


# ============================================================================
# Test 5: CSE preserves ops with different result types
# ============================================================================
def test_cse_different_result_types(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 5: --joy-cse preserves ops with different result types")
    print("=" * 60)

    g = Graph(name="cse_types")
    x = g.input([1, 64, 1024], "f16")
    w1 = g.input([512, 1024], "f16")   # produces [1,64,512]
    w2 = g.input([256, 1024], "f16")   # produces [1,64,256]

    y1 = ops.linear(x, w1)
    y2 = ops.linear(x, w2)              # same input, different out_features
    g.set_outputs([y1, y2])
    ir_before = g.get_ir()

    stdout, _, rc = _run_joy_opt(ir_before, ["--joy-cse"])
    if print_ir:
        print(stdout)

    after = _count_op(stdout, "joy.linear")
    checks = [
        (rc == 0, "joy-opt --joy-cse returned 0"),
        (after == 2, f"both linear preserved (different result types) (got {after})"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson7] Test 5 failed"
    print(f"\n[Lesson7]: ================== Test 5 PASSED ==================")


# ============================================================================
# Test 6: CSE also cleans up dead pure ops (no user)
# ============================================================================
def test_cse_dce(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 6: --joy-cse cleans up dead Pure ops with no users")
    print("=" * 60)

    g = Graph(name="cse_dce")
    x = g.input([1, 64, 1024], "f16")
    scale = g.input([1024], "f16")
    w = g.input([512, 1024], "f16")

    # The first rms_norm result is never used by an output -> DCE candidate.
    dead = ops.rms_norm(x, scale, epsilon=1e-6)
    # The linear is the actual output.
    out = ops.linear(x, w)
    g.set_outputs([out])
    ir_before = g.get_ir()

    rms_before = _count_op(ir_before, "joy.rms_norm")
    stdout, _, rc = _run_joy_opt(ir_before, ["--joy-cse"])
    if print_ir:
        print(stdout)

    rms_after = _count_op(stdout, "joy.rms_norm")
    linear_after = _count_op(stdout, "joy.linear")
    checks = [
        (rc == 0, "joy-opt --joy-cse returned 0"),
        (rms_before == 1, f"before: 1 dead rms_norm (got {rms_before})"),
        (rms_after == 0,
         f"after: dead rms_norm removed by CSE/DCE (got {rms_after})"),
        (linear_after == 1, "live joy.linear preserved"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson7] Test 6 failed"
    print(f"\n[Lesson7]: ================== Test 6 PASSED ==================")


# ============================================================================
# Test 7: full pipeline applies all steps
# ============================================================================
def test_full_pipeline(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 7: --joy-optimization-pipeline = CF + CSE + Fusion + CF + CSE")
    print("=" * 60)

    H = 1024
    g = Graph(name="pipeline_test")
    hidden = g.input([1, 64, H], "f16")
    residual = g.input([1, 64, H], "f16")
    ln_w = g.input([H], "f16")
    proj_w = g.input([512, H], "f16")

    # 1) fusable: residual + hidden -> rms_norm
    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=1e-6)

    # 2) duplicate of normed (same operands, attrs) -> CSE should kill it
    dup = ops.rms_norm(added, ln_w, epsilon=1e-6)

    # 3) live use of `normed` (not `dup`) - dup has no real user
    out = ops.linear(normed, proj_w)
    # Add dup to outputs so it's referenced and not DCE'd before fusion;
    # the optimization pipeline should still be able to merge dup with normed
    # via CSE (since they are structurally identical and pure).
    g.set_outputs([out, dup])
    ir_before = g.get_ir()

    stdout, _, rc = _run_joy_opt(ir_before, ["--joy-optimization-pipeline"])
    if print_ir:
        print(stdout)

    add_after = _count_op(stdout, "joy.add")
    rms_after = _count_op(stdout, "joy.rms_norm")
    fuse_after = _count_op(stdout, "joy.fuse_add_rmsnorm")
    linear_after = _count_op(stdout, "joy.linear")

    checks = [
        (rc == 0, "joy-opt --joy-optimization-pipeline returned 0"),
        (fuse_after >= 1, f"at least 1 fuse_add_rmsnorm created (got {fuse_after})"),
        (add_after == 0, f"all add ops consumed by fusion (got {add_after})"),
        (rms_after == 0,
         f"duplicate rms_norm merged by CSE before/after fusion (got {rms_after})"),
        (linear_after == 1, "joy.linear preserved"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson7] Test 7 failed"
    print(f"\n[Lesson7]: ================== Test 7 PASSED ==================")


# ============================================================================
# Test 8: joy-opt --help registers the 4 joy-* pass / pipeline names
# ============================================================================
def test_help_registers_passes(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 8: joy-opt --help lists all four joy-* pass / pipeline names")
    print("=" * 60)

    if not _require_joy_opt():
        return

    result = subprocess.run([JOY_OPT, "--help"], capture_output=True, text=True)
    out = result.stdout + result.stderr

    expected = [
        "--joy-constant-folding",
        "--joy-cse",
        "--joy-op-fusion",
        "--joy-optimization-pipeline",
    ]
    all_pass = True
    for name in expected:
        ok = name in out
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {name} registered in --help")

    # Also check the descriptions are present (not just placeholders).
    descriptions = [
        "Perform constant folding on Joy dialect operations",
        "Eliminate common sub-expressions in Joy dialect",
        "Fuse add + rms_norm into fuse_add_rmsnorm in Joy dialect",
        "Joy compiler optimization pipeline",
    ]
    for desc in descriptions:
        ok = desc in out
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  description {desc!r}")

    assert all_pass, "[Lesson7] Test 8 failed"
    print(f"\n[Lesson7]: ================== Test 8 PASSED ==================")


# ============================================================================
# Test 9: joy-opt rejects an unknown pass argument
# ============================================================================
def test_unknown_pass_rejected(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 9: joy-opt rejects an unknown pass argument")
    print("=" * 60)

    if not _require_joy_opt():
        return

    g = Graph(name="reject_test")
    x = g.input([8], "f16")
    g.set_outputs([x])
    ir = g.get_ir()

    stdout, stderr, rc = _run_joy_opt(ir, ["--joy-nonexistent-pass"])

    # MLIR's MlirOptMain returns non-zero rc *and* prints a diagnostic to stderr.
    ok_rc = rc != 0
    ok_msg = (
        "unknown" in (stderr or "").lower() or
        "no such" in (stderr or "").lower() or
        "could not find" in (stderr or "").lower() or
        "--joy-nonexistent-pass" in (stderr or "")
    )

    checks = [
        (ok_rc, f"joy-opt exited non-zero on unknown pass (rc={rc})"),
        (ok_msg, "stderr mentions the unknown pass / unrecognised flag"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson7] Test 9 failed"
    print(f"\n[Lesson7]: ================== Test 9 PASSED ==================")


# ============================================================================
# Test 10: --mlir-print-ir-after-all covers every stage of the pipeline
# ============================================================================
def test_print_ir_after_all(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 10: --mlir-print-ir-after-all prints every Pass stage")
    print("=" * 60)

    if not _require_joy_opt():
        return

    H = 1024
    g = Graph(name="pipeline_dump")
    hidden = g.input([1, 64, H], "f16")
    residual = g.input([1, 64, H], "f16")
    ln_w = g.input([H], "f16")
    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=1e-6)
    g.set_outputs([normed])
    ir = g.get_ir()

    stdout, stderr, rc = _run_joy_opt(
        ir, ["--joy-optimization-pipeline"],
        extra_args=["--mlir-print-ir-after-all"])

    if print_ir:
        print("---- joy-opt stderr (IR dumps go here) ----")
        print(stderr)
        print("---- end ----")

    # MLIR prints IR dumps to stderr after each pass.
    dump = stderr or ""
    expected_substrs = [
        "IR Dump After",
        "ConstantFolding",
        "OpFusion",
    ]
    all_pass = rc == 0
    for sub in expected_substrs:
        ok = sub in dump
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  IR-after-all output mentions "
              f"{sub!r}")

    # Count distinct "IR Dump After" headers — should be 5 for the pipeline.
    n_dumps = len(re.findall(r"IR Dump After", dump))
    ok_count = n_dumps >= 5
    print(f"  {'PASS' if ok_count else 'FAIL'}  "
          f"saw {n_dumps} IR dumps (expected >= 5 for the 5-stage pipeline)")
    if not ok_count:
        all_pass = False

    assert all_pass, "[Lesson7] Test 10 failed"
    print(f"\n[Lesson7]: ================== Test 10 PASSED =================")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Lesson 7: Pass & Pipeline — Test")
    parser.add_argument("--print-ir-all", action="store_true",
                        help="Print IR fragments where useful")
    args = parser.parse_args()
    p = args.print_ir_all

    print("=" * 60)
    print("  Lesson 7: AI Compiler Mid-end — Pass & Pipeline")
    print("=" * 60)

    if not _require_joy_opt():
        sys.exit(0)

    test_constant_folding_noop(print_ir=p)
    test_cse_eliminates_duplicates(print_ir=p)
    test_cse_different_attrs(print_ir=p)
    test_cse_different_operands(print_ir=p)
    test_cse_different_result_types(print_ir=p)
    test_cse_dce(print_ir=p)
    test_full_pipeline(print_ir=p)
    test_help_registers_passes(print_ir=p)
    test_unknown_pass_rejected(print_ir=p)
    test_print_ir_after_all(print_ir=p)

    print("\n" + "=" * 60)
    print("  ALL LESSON 7 TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
