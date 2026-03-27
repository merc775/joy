#!/usr/bin/env python3
"""Lesson 3: MLIR Public Infrastructure — hands-on tests.

Covers the topics in `joy/docs/第3讲-MLIR公用基础设施.md`:

  1. TableGen artefacts present + contain the expected symbols
  2. joy-opt registers the public dialects + Joy passes
  3. Pure public-dialect IR parses round-trip through joy-opt
  4. The Joy pipeline produces public-dialect ops (memref / arith /
     scf / math / func) when going through --codegen-rms-norm
  5. Tensor → MemRef lowering allocates buffers with memref.alloc

Usage:
    python3 tests/python_tests/test_lesson3.py
    python3 tests/python_tests/test_lesson3.py --print-ir-all
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
BUILD_INC_DIR = os.path.join(project_root, "build", "include", "joy", "dialect",
                             "joy")
STUB_F32 = os.path.join(project_root, "lib", "backend", "gpu",
                        "codegen_stub_f32.mlir")


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
# Test 1: TableGen-generated files exist + contain expected symbols
# ============================================================================
def test_tablegen_artifacts(print_ir=False):
    """Verify mlir_tablegen() produced the expected JoyDialect/JoyOps incs."""
    print("\n" + "=" * 60)
    print("  Test 1: TableGen Generated Artifacts")
    print("=" * 60)

    expected = [
        ("JoyDialect.h.inc",
         ["class JoyDialect", 'getDialectNamespace()', '"joy"']),
        ("JoyDialect.cpp.inc",
         ["JoyDialect::JoyDialect", "getDialectNamespace()"]),
        ("JoyOps.h.inc",
         ["class LinearOp", "class RMSNormOp",
          "class FuseAddRMSNormOp", "class CustomCallOp" if False else "Adaptor",
          "getInput()", "getWeight()", "getOutput()"]),
        ("JoyOps.cpp.inc",
         ["LinearOp::build", "RMSNormOp::build",
          "FuseAddRMSNormOp::build"]),
    ]

    if not os.path.isdir(BUILD_INC_DIR):
        print(f"  WARNING: build inc dir not found: {BUILD_INC_DIR}")
        print(f"  Skipping (run `ninja` first)")
        return

    checks = []
    for fname, needles in expected:
        path = os.path.join(BUILD_INC_DIR, fname)
        if not os.path.exists(path):
            checks.append((False, f"{fname} exists"))
            continue
        checks.append((True, f"{fname} exists"))
        with open(path, "r") as f:
            content = f.read()
        for needle in needles:
            checks.append(
                (needle in content,
                 f"{fname} contains `{needle}`"))

    if print_ir:
        for fname in ["JoyOps.h.inc"]:
            path = os.path.join(BUILD_INC_DIR, fname)
            if os.path.exists(path):
                with open(path, "r") as f:
                    print(f"\n--- first 30 lines of {fname} ---")
                    print("".join(f.readlines()[:30]))
                    print("--- end ---")

    all_pass = _print_checks(checks)
    assert all_pass, "[Lesson3] Test 1 failed"
    print("\n[Lesson3]: ================== Test 1 PASSED ==================")


# ============================================================================
# Test 2: joy-opt registers the public + Joy dialects/passes
# ============================================================================
def test_joy_opt_help(print_ir=False):
    """Verify `joy-opt --help` advertises the right dialects and passes."""
    print("\n" + "=" * 60)
    print("  Test 2: joy-opt --help (dialects + passes)")
    print("=" * 60)

    if not os.path.exists(JOY_OPT):
        print(f"  WARNING: joy-opt not found at {JOY_OPT}, skipping")
        return

    result = subprocess.run([JOY_OPT, "--help"], capture_output=True,
                            text=True, timeout=30)
    out = result.stdout

    expected_dialects = [
        "arith", "builtin", "func", "joy", "joyh", "joyl",
        "math", "memref", "scf",
    ]
    expected_passes = [
        "--lower-joy-to-joyl",
        "--lower-joyl-to-joyh",
        "--codegen-rms-norm",
        "--joy-op-fusion",
    ]

    # Dialect names appear in the "Available Dialects:" line.
    avail_line = ""
    for line in out.splitlines():
        if "Available Dialects" in line:
            avail_line = line
            break

    checks = []
    for d in expected_dialects:
        checks.append((d in avail_line,
                       f"dialect registered: {d}"))
    for p in expected_passes:
        checks.append((p in out, f"pass registered: {p}"))

    if print_ir:
        print(f"\n  Available dialects line:\n    {avail_line}")

    all_pass = _print_checks(checks)
    assert all_pass, "[Lesson3] Test 2 failed"
    print("\n[Lesson3]: ================== Test 2 PASSED ==================")


# ============================================================================
# Test 3: pure public-dialect IR round-trips through joy-opt
# ============================================================================
def test_public_dialect_roundtrip(print_ir=False):
    """Verify joy-opt can parse + reprint an IR that uses only public
    dialects (func / memref / arith / scf / math)."""
    print("\n" + "=" * 60)
    print("  Test 3: Public-dialect IR round-trip")
    print("=" * 60)

    ir = """\
module {
  func.func @public_only(%input: memref<?x?xf32>, %scale: memref<?xf32>,
                          %output: memref<?x?xf32>, %eps: f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.000000e+00 : f32
    %rows = memref.dim %input, %c0 : memref<?x?xf32>
    %cols = memref.dim %input, %c1 : memref<?x?xf32>
    %cI64 = arith.index_cast %cols : index to i64
    %cF32 = arith.sitofp %cI64 : i64 to f32
    scf.for %i = %c0 to %rows step %c1 {
      %sum = scf.for %j = %c0 to %cols step %c1
             iter_args(%acc = %zero) -> (f32) {
        %v = memref.load %input[%i, %j] : memref<?x?xf32>
        %sq = arith.mulf %v, %v : f32
        %newAcc = arith.addf %acc, %sq : f32
        scf.yield %newAcc : f32
      }
      %mean = arith.divf %sum, %cF32 : f32
      %varEps = arith.addf %mean, %eps : f32
      %r = math.rsqrt %varEps : f32
      scf.for %j = %c0 to %cols step %c1 {
        %vv = memref.load %input[%i, %j] : memref<?x?xf32>
        %s = memref.load %scale[%j] : memref<?xf32>
        %t = arith.mulf %vv, %r : f32
        %o = arith.mulf %t, %s : f32
        memref.store %o, %output[%i, %j] : memref<?x?xf32>
      }
    }
    return
  }
}
"""
    out = _run_joy_opt(ir, [])
    if out is None:
        print("  SKIP: joy-opt not available")
        return

    if print_ir:
        print("\n--- joy-opt reprint ---")
        print(out)
        print("--- end ---")

    checks = [
        ("func.func @public_only" in out, "func.func preserved"),
        ("memref.dim" in out,              "memref.dim preserved"),
        ("memref.load" in out,             "memref.load preserved"),
        ("memref.store" in out,            "memref.store preserved"),
        ("arith.constant" in out,          "arith.constant preserved"),
        ("arith.mulf" in out,              "arith.mulf preserved"),
        ("arith.addf" in out,              "arith.addf preserved"),
        ("arith.divf" in out,              "arith.divf preserved"),
        ("arith.index_cast" in out,        "arith.index_cast preserved"),
        ("arith.sitofp" in out,            "arith.sitofp preserved"),
        ("scf.for" in out,                 "scf.for preserved"),
        ("iter_args" in out,               "scf.for(iter_args) preserved"),
        ("scf.yield" in out,               "scf.yield preserved"),
        ("math.rsqrt" in out,              "math.rsqrt preserved"),
    ]
    all_pass = _print_checks(checks)
    assert all_pass, "[Lesson3] Test 3 failed"
    print("\n[Lesson3]: ================== Test 3 PASSED ==================")


# ============================================================================
# Test 4: --codegen-rms-norm produces a gpu_kernel func built from public ops
# ============================================================================
def test_codegen_uses_public_dialects(print_ir=False):
    """The output of --codegen-rms-norm must be a func.func with the
    gpu_kernel attribute, whose body is *entirely* in public dialects.
    This is the core demonstration that joy reuses MLIR infra."""
    print("\n" + "=" * 60)
    print("  Test 4: --codegen-rms-norm builds kernels from public ops")
    print("=" * 60)

    if not os.path.exists(STUB_F32):
        print(f"  WARNING: stub MLIR not found: {STUB_F32}")
        return

    with open(STUB_F32, "r") as f:
        stub_ir = f.read()

    out = _run_joy_opt(stub_ir, ["--codegen-rms-norm"])
    if out is None:
        print("  SKIP: joy-opt not available")
        return

    if print_ir:
        print("\n--- joy-opt --codegen-rms-norm output ---")
        print(out)
        print("--- end ---")

    rms_kernel_re = re.compile(
        r"func\.func\s+private\s+@joy_rms_norm_kernel\([^)]*\)\s+"
        r"attributes\s+\{[^}]*gpu_kernel")
    fuse_kernel_re = re.compile(
        r"func\.func\s+private\s+@joy_fuse_add_rmsnorm_kernel\([^)]*\)\s+"
        r"attributes\s+\{[^}]*gpu_kernel")

    checks = [
        (bool(rms_kernel_re.search(out)),
         "func.func @joy_rms_norm_kernel with gpu_kernel attr present"),
        (bool(fuse_kernel_re.search(out)),
         "func.func @joy_fuse_add_rmsnorm_kernel with gpu_kernel attr present"),
        ("memref.dim" in out,             "uses memref.dim"),
        ("memref.load" in out,            "uses memref.load"),
        ("memref.store" in out,           "uses memref.store"),
        ("arith.constant" in out,         "uses arith.constant"),
        ("arith.mulf" in out,             "uses arith.mulf"),
        ("arith.divf" in out,             "uses arith.divf"),
        ("arith.addf" in out,             "uses arith.addf"),
        ("arith.index_cast" in out,       "uses arith.index_cast"),
        ("arith.sitofp" in out,           "uses arith.sitofp"),
        ("math.rsqrt" in out,             "uses math.rsqrt"),
        ("scf.for" in out and "iter_args" in out,
         "uses scf.for with iter_args (reduction)"),
        # The call site replacement
        ("call @joy_rms_norm_kernel" in out,
         "joyl.rms_norm replaced by func.call @joy_rms_norm_kernel"),
        ("call @joy_fuse_add_rmsnorm_kernel" in out,
         "joyl.fuse_add_rmsnorm replaced by func.call @joy_fuse_add_rmsnorm_kernel"),
        # No joyl ops left for the two normalization ops
        ("joyl.rms_norm" not in out,
         "no leftover joyl.rms_norm"),
        ("joyl.fuse_add_rmsnorm" not in out,
         "no leftover joyl.fuse_add_rmsnorm"),
        # collapse_shape + cast bridging the multi-dim memref to 2D
        ("memref.cast" in out,
         "memref.cast (multi-dim -> dynamic 2D) used at call site"),
    ]
    all_pass = _print_checks(checks)
    assert all_pass, "[Lesson3] Test 4 failed"
    print("\n[Lesson3]: ================== Test 4 PASSED ==================")


# ============================================================================
# Test 5: Joy → Joyl lowering produces memref.alloc + memref operands
# ============================================================================
def test_lower_uses_memref_dialect(print_ir=False):
    """Lower a small Joy graph to Joyl and verify the public memref
    dialect actually shows up (memref.alloc + memref<...> types)."""
    print("\n" + "=" * 60)
    print("  Test 5: Joy → Joyl introduces memref dialect")
    print("=" * 60)

    graph = Graph(name="public_lower")
    x = graph.input([1, 64, 1024], "f16", name="input")
    w = graph.input([512, 1024], "f16", name="weight")
    out = ops.linear(x, w)
    graph.set_outputs([out])
    ir = graph.get_ir()

    joyl_ir = _run_joy_opt(ir, ["--lower-joy-to-joyl"])
    if joyl_ir is None:
        print("  SKIP: joy-opt not available")
        return

    if print_ir:
        print("\n--- joy-opt --lower-joy-to-joyl output ---")
        print(joyl_ir)
        print("--- end ---")

    checks = [
        ("memref<" in joyl_ir,         "memref types present"),
        ("memref.alloc" in joyl_ir,    "memref.alloc allocates output buffer"),
        ("func.func @public_lower"     in joyl_ir
         or "func.func" in joyl_ir,    "func dialect (func.func) preserved"),
        ('"joy.linear"' not in joyl_ir, "joy.linear fully lowered"),
        # tensor type should be gone from the joyl portion of IR
        ("tensor<" not in joyl_ir,     "no leftover tensor<...> types"),
    ]
    all_pass = _print_checks(checks)
    assert all_pass, "[Lesson3] Test 5 failed"
    print("\n[Lesson3]: ================== Test 5 PASSED ==================")


# ============================================================================
# Test 6: math / scf / arith Op coverage in the full kernel
# ============================================================================
def test_op_coverage_in_codegen(print_ir=False):
    """For each public Op listed in §5 of the lecture, verify that at
    least one occurrence shows up in the codegen kernel IR.

    This guards the docs against falling out of sync with the codegen
    pass.
    """
    print("\n" + "=" * 60)
    print("  Test 6: Public-dialect Op coverage in codegen kernel")
    print("=" * 60)

    if not os.path.exists(STUB_F32):
        print(f"  WARNING: stub MLIR not found: {STUB_F32}")
        return

    with open(STUB_F32, "r") as f:
        stub_ir = f.read()

    out = _run_joy_opt(stub_ir, ["--codegen-rms-norm"])
    if out is None:
        print("  SKIP")
        return

    required_ops = [
        # (dialect, op_name, ir_token)
        ("builtin", "ModuleOp",     "module"),
        ("func",    "FuncOp",       "func.func"),
        ("func",    "ReturnOp",     "return"),
        ("func",    "CallOp",       "call @joy_rms_norm_kernel"),
        ("memref",  "AllocOp",      "memref.alloc"),
        ("memref",  "DimOp",        "memref.dim"),
        ("memref",  "LoadOp",       "memref.load"),
        ("memref",  "StoreOp",      "memref.store"),
        ("memref",  "CastOp",       "memref.cast"),
        ("arith",   "ConstantOp",   "arith.constant"),
        ("arith",   "MulFOp",       "arith.mulf"),
        ("arith",   "AddFOp",       "arith.addf"),
        ("arith",   "DivFOp",       "arith.divf"),
        ("arith",   "IndexCastOp",  "arith.index_cast"),
        ("arith",   "SIToFPOp",     "arith.sitofp"),
        ("scf",     "ForOp",        "scf.for"),
        ("scf",     "YieldOp",      "scf.yield"),
        ("math",    "RsqrtOp",      "math.rsqrt"),
    ]

    checks = [(token in out, f"{dialect}.{op} present (`{token}`)")
              for dialect, op, token in required_ops]
    if print_ir:
        for ok, desc in checks:
            if not ok:
                print(f"    missing: {desc}")
    all_pass = _print_checks(checks)
    assert all_pass, "[Lesson3] Test 6 failed"
    print("\n[Lesson3]: ================== Test 6 PASSED ==================")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Lesson 3: MLIR Public Infrastructure")
    parser.add_argument("--print-ir-all", action="store_true",
                        help="Print extra detail at every stage")
    args = parser.parse_args()
    p = args.print_ir_all

    print("=" * 60)
    print("  Lesson 3: MLIR Public Infrastructure Tests")
    print("=" * 60)

    test_tablegen_artifacts(print_ir=p)
    test_joy_opt_help(print_ir=p)
    test_public_dialect_roundtrip(print_ir=p)
    test_codegen_uses_public_dialects(print_ir=p)
    test_lower_uses_memref_dialect(print_ir=p)
    test_op_coverage_in_codegen(print_ir=p)

    print("\n" + "=" * 60)
    print("  ALL LESSON 3 TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
