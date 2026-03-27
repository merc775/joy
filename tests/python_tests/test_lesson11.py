#!/usr/bin/env python3
"""Lesson 11: CUDA code generation — hands-on tests.

This test suite mirrors the FIRST compile-time stage of the joy GPU backend:
turning gpu_kernel-tagged MLIR funcs into a self-contained CUDA C source file
(``codegen_kernel.cu``).  The follow-on stages (nvcc / ar / ld) live in
Lesson 12.

  Part A — MLIR -> CUDA C (joy/lib/optimizer/EmitCudaC.cpp,
                            joy/tools/joy-emit-cuda.cpp)
           T1: joy-emit-cuda accepts post-codegen IR and emits a complete
               CUDA C translation unit (header banner, includes, __global__,
               extern "C" launcher).
           T2: The emitted CUDA C contains every key CUDA construct we
               expect from the reduction+normalize pattern (blockIdx.x,
               threadIdx.x, shared memory, __syncthreads, rsqrtf, half-f32
               casts, etc.).
           T3: --source-tag is preserved as a // Source: comment in the
               emitted text.

  Part B — codegen_kernel.cu (build-time artifact)
           T4: The build-time artifact build/lib/backend/gpu/codegen_kernel.cu
               actually exists and contains both f32 and f16 versions of
               rms_norm + fuse_add_rmsnorm kernels and their launchers.

  Part C — joy-emit-cuda CLI behaviour
           T5: joy-emit-cuda refuses kernel-less modules with a clear
               error, and the joy-opt | joy-emit-cuda pipeline used by
               scripts/regen_codegen_kernel.sh produces a complete CUDA C
               translation unit from raw joy IR.

Usage:
    python3 tests/python_tests/test_lesson11.py
    python3 tests/python_tests/test_lesson11.py --print-ir-all
"""

import argparse
import os
import subprocess
import sys
import tempfile

cur_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(cur_path, "../.."))
sys.path.insert(0, os.path.join(project_root, "python"))


# ============================================================================
# Paths & helpers
# ============================================================================
BUILD_DIR = os.path.join(project_root, "build")
JOY_OPT = os.path.join(BUILD_DIR, "bin", "joy-opt")
JOY_EMIT_CUDA = os.path.join(BUILD_DIR, "bin", "joy-emit-cuda")

STUB_F32 = os.path.join(project_root, "lib", "backend", "gpu",
                        "codegen_stub_f32.mlir")
STUB_F16 = os.path.join(project_root, "lib", "backend", "gpu",
                        "codegen_stub_f16.mlir")

CODEGEN_CU = os.path.join(BUILD_DIR, "lib", "backend", "gpu",
                          "codegen_kernel.cu")


def _check_required():
    """Skip the suite if mandatory binaries / artifacts are missing."""
    missing = []
    for path in [JOY_OPT, JOY_EMIT_CUDA, STUB_F32, STUB_F16]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print("Lesson 11 prerequisites missing — please build the project first:")
        for p in missing:
            print(f"  missing: {p}")
        sys.exit(2)


def _run(cmd, *, input_text=None, timeout=60):
    """Run a subprocess, return (rc, stdout, stderr)."""
    res = subprocess.run(cmd, input=input_text, capture_output=True,
                          text=True, timeout=timeout)
    return res.returncode, res.stdout, res.stderr


def _emit_cuda_from_stub(stub_path, source_tag=None):
    """joy-opt --codegen-rms-norm STUB | joy-emit-cuda - -> stdout string."""
    rc1, ir, err1 = _run([JOY_OPT, "--codegen-rms-norm", stub_path])
    assert rc1 == 0, f"joy-opt failed on {stub_path}:\n{err1}"
    cmd = [JOY_EMIT_CUDA]
    if source_tag is not None:
        cmd.append(f"--source-tag={source_tag}")
    cmd.append("-")
    rc2, cu_text, err2 = _run(cmd, input_text=ir)
    assert rc2 == 0, f"joy-emit-cuda failed on {stub_path}:\n{err2}"
    return cu_text


def _report(checks):
    all_ok = True
    for ok, desc in checks:
        prefix = "  [PASS]" if ok else "  [FAIL]"
        print(f"{prefix} {desc}")
        if not ok:
            all_ok = False
    return all_ok


# ============================================================================
# Part A — MLIR -> CUDA C
# ============================================================================

def test_emit_cuda_basic(print_ir=False):
    """T1: joy-emit-cuda emits a self-contained CUDA C TU from f32 stub."""
    print("\n" + "=" * 60)
    print("  T1: joy-emit-cuda produces a complete CUDA C translation unit")
    print("=" * 60)

    cu = _emit_cuda_from_stub(STUB_F32)
    if print_ir:
        print(cu)

    checks = [
        ("#include <cuda_fp16.h>" in cu, "CUDA fp16 header included"),
        ("#include <cuda_runtime.h>" in cu, "CUDA runtime header included"),
        ("#include <cstdint>" in cu, "cstdint header included"),
        ("THIS FILE IS AUTO-GENERATED FROM MLIR." in cu,
         "auto-generated banner present"),
        ("__global__ void joy_codegen_rms_norm_f32_kernel(" in cu,
         "__global__ rms_norm kernel emitted"),
        ("__global__ void joy_codegen_fuse_add_rms_norm_f32_kernel(" in cu,
         "__global__ fuse_add_rmsnorm kernel emitted"),
        ('extern "C" void joy_codegen_rms_norm_f32(' in cu,
         "extern \"C\" rms_norm launcher emitted"),
        ('extern "C" void joy_codegen_fuse_add_rms_norm_f32(' in cu,
         "extern \"C\" fuse_add_rmsnorm launcher emitted"),
        # exactly the two kernels + two launchers.
        (cu.count("__global__ void ") == 2,
         "emits exactly 2 __global__ kernels (rms + fuse) for f32 stub"),
        (cu.count('extern "C" void ') == 2,
         "emits exactly 2 extern \"C\" launchers"),
    ]
    assert _report(checks), "[Lesson11] T1 failed"
    print("\n[Lesson11]: ================== T1 PASSED ==================")


def test_emit_cuda_body_constructs(print_ir=False):
    """T2: emitted CUDA C contains every key CUDA construct we expect."""
    print("\n" + "=" * 60)
    print("  T2: emitted CUDA C uses the expected CUDA constructs")
    print("=" * 60)

    cu_f32 = _emit_cuda_from_stub(STUB_F32)
    cu_f16 = _emit_cuda_from_stub(STUB_F16)
    if print_ir:
        print("--- F32 ---")
        print(cu_f32)
        print("--- F16 ---")
        print(cu_f16)

    # Reduction pattern: "1 block per row" + thread-strided + shared-mem tree.
    f32_checks = [
        ("int64_t i = blockIdx.x;" in cu_f32,
         "row dim mapped to blockIdx.x"),
        ("if (i >= rows) return;" in cu_f32,
         "out-of-range early return guard"),
        ("extern __shared__ float sdata[]" in cu_f32,
         "extern dynamic shared memory declared"),
        ("for (int64_t j = threadIdx.x; j < cols; j += blockDim.x)" in cu_f32,
         "inner loop is thread-strided over cols"),
        ("sdata[threadIdx.x]" in cu_f32, "shared mem reduction buffer used"),
        ("__syncthreads();" in cu_f32, "__syncthreads() barrier emitted"),
        ("for (int s = blockDim.x / 2; s > 0; s >>= 1)" in cu_f32,
         "tree-reduction sweep emitted"),
        ("rsqrtf(" in cu_f32, "math.rsqrt translated to rsqrtf()"),
        ("kBlockSize = 256" in cu_f32, "launcher uses 256-thread block"),
        ("reinterpret_cast<cudaStream_t>(stream)" in cu_f32,
         "stream argument forwarded to launch config"),
    ]

    # F16 path also exercises the half<->float casts.
    f16_checks = [
        ("__half2float(" in cu_f16,
         "arith.extf f16->f32 emitted as __half2float()"),
        ("__float2half(" in cu_f16,
         "arith.truncf f32->f16 emitted as __float2half()"),
        # f16 kernel still does accumulation in float
        ("float acc0 = 0.0f;" in cu_f16,
         "accumulator stays in f32 even with f16 IO"),
        ("const __half *" in cu_f16, "f16 inputs typed as __half*"),
    ]

    assert _report(f32_checks + f16_checks), "[Lesson11] T2 failed"
    print("\n[Lesson11]: ================== T2 PASSED ==================")


def test_source_tag(print_ir=False):
    """T3: --source-tag is forwarded into the // Source: comment."""
    print("\n" + "=" * 60)
    print("  T3: --source-tag option is respected")
    print("=" * 60)

    tag = "lesson11_unit_test_tag"
    cu = _emit_cuda_from_stub(STUB_F32, source_tag=tag)
    if print_ir:
        print(cu[:200])

    checks = [
        (f"// Source: {tag}" in cu,
         "emitted file contains '// Source: <tag>' line"),
        # the second internal "Source: CodegenRMSNormPass output ..." should also
        # still be there.
        ("// Source: CodegenRMSNormPass output -> joy::emitCudaC." in cu,
         "internal pipeline source comment preserved"),
    ]
    assert _report(checks), "[Lesson11] T3 failed"
    print("\n[Lesson11]: ================== T3 PASSED ==================")


# ============================================================================
# Part B — codegen_kernel.cu build-time artifact
# ============================================================================

def test_codegen_kernel_artifact(print_ir=False):
    """T4: build/lib/backend/gpu/codegen_kernel.cu exists and is complete."""
    print("\n" + "=" * 60)
    print("  T4: build/lib/backend/gpu/codegen_kernel.cu contains all 4 kernels")
    print("=" * 60)

    if not os.path.exists(CODEGEN_CU):
        print(f"  WARNING: {CODEGEN_CU} missing; build the project first")
        # Fail loudly: this is exactly the artifact this lesson teaches about.
        assert False, "[Lesson11] T4: codegen_kernel.cu does not exist"

    with open(CODEGEN_CU, "r") as f:
        cu = f.read()
    if print_ir:
        print(cu[:400])

    expected_globals = [
        "joy_codegen_rms_norm_f32_kernel",
        "joy_codegen_rms_norm_f16_kernel",
        "joy_codegen_fuse_add_rms_norm_f32_kernel",
        "joy_codegen_fuse_add_rms_norm_f16_kernel",
    ]
    expected_launchers = [
        "joy_codegen_rms_norm_f32",
        "joy_codegen_rms_norm_f16",
        "joy_codegen_fuse_add_rms_norm_f32",
        "joy_codegen_fuse_add_rms_norm_f16",
    ]

    checks = []
    for k in expected_globals:
        checks.append((f"__global__ void {k}(" in cu,
                        f"{k} kernel definition present"))
    for L in expected_launchers:
        checks.append((f'extern "C" void {L}(' in cu,
                        f"{L} extern \"C\" launcher present"))
    checks.append((cu.count("__global__ void ") == 4,
                    "exactly 4 __global__ definitions in codegen_kernel.cu"))
    checks.append((cu.count('extern "C" void ') == 4,
                    "exactly 4 extern \"C\" launchers in codegen_kernel.cu"))
    checks.append(("Auto-generated by joy/scripts/regen_codegen_kernel.sh" in cu,
                    "regen_codegen_kernel.sh banner present"))

    assert _report(checks), "[Lesson11] T4 failed"
    print("\n[Lesson11]: ================== T4 PASSED ==================")


# ============================================================================
# Part C — joy-emit-cuda CLI behaviour
# ============================================================================

def test_emit_cuda_error_and_pipeline(print_ir=False):
    """T5: joy-emit-cuda refuses kernel-less modules and works with the real
    joy-opt | joy-emit-cuda pipeline used by scripts/regen_codegen_kernel.sh."""
    print("\n" + "=" * 60)
    print("  T5: joy-emit-cuda CLI: error reporting + pipe-based pipeline")
    print("=" * 60)

    # ---- 5.a: refuse kernel-less input (no func.func {gpu_kernel}) ----
    kernel_less = """\
module {
  func.func @noop() -> () {
    return
  }
}
"""
    rc, out, err = _run([JOY_EMIT_CUDA, "-"], input_text=kernel_less)
    refused = (rc != 0) and ("gpu_kernel" in err or "no func.func" in err)
    if print_ir:
        print(f"  refuse-no-kernel: rc={rc} stderr={err!r}")
    assert refused, ("[Lesson11] T5a: emitter should refuse kernel-less input "
                     f"(got rc={rc}, stderr={err!r})")
    print("  [PASS] joy-emit-cuda rejects kernel-less modules with a clear error")

    # ---- 5.b: emitter works through the production pipeline
    # joy-opt --lower-joy-to-joyl --codegen-rms-norm | joy-emit-cuda - ----
    raw_joy_ir = """\
module {
  func.func @rms(%x: tensor<4x16xf32>, %s: tensor<16xf32>) -> tensor<4x16xf32> {
    %0 = "joy.rms_norm"(%x, %s) {epsilon = 1.000000e-06 : f32}
        : (tensor<4x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }
}
"""
    # Step 1: run lower-joy-to-joyl + codegen-rms-norm with joy-opt.
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as tf:
        tf.write(raw_joy_ir)
        tf_path = tf.name
    try:
        rc, joyl_ir, err = _run(
            [JOY_OPT, "--lower-joy-to-joyl", "--codegen-rms-norm", tf_path])
        assert rc == 0, f"[Lesson11] T5b: joy-opt failed:\n{err}"
        # Step 2: pipe to joy-emit-cuda (this is what regen_codegen_kernel.sh does).
        rc, cu, err = _run([JOY_EMIT_CUDA, "-"], input_text=joyl_ir)
        if print_ir:
            print("  joy-opt | joy-emit-cuda stderr:", err)
            print("  emitted CUDA (truncated):", cu[:300])
        assert rc == 0, ("[Lesson11] T5b: piped joy-opt | joy-emit-cuda failed:\n"
                         f"  stderr={err!r}")
    finally:
        os.unlink(tf_path)

    checks = [
        ("__global__ void joy_codegen_rms_norm_f32_kernel(" in cu,
         "rms_norm kernel emitted through joy-opt | joy-emit-cuda pipeline"),
        ('extern "C" void joy_codegen_rms_norm_f32(' in cu,
         "rms_norm launcher emitted through joy-opt | joy-emit-cuda pipeline"),
        ("__half2float(" not in cu,
         "f32 input does not introduce half<->float casts"),
    ]
    assert _report(checks), "[Lesson11] T5b failed"
    print("\n[Lesson11]: ================== T5 PASSED ==================")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Lesson 11: CUDA code generation tests")
    parser.add_argument("--print-ir-all", action="store_true",
                        help="Print emitted CUDA / IR for each test")
    args = parser.parse_args()
    p = args.print_ir_all

    _check_required()

    print("=" * 60)
    print("  Lesson 11: CUDA code generation")
    print("=" * 60)

    print("\n" + "-" * 60)
    print("  Part A: MLIR -> CUDA C (joy-emit-cuda / EmitCudaC.cpp)")
    print("-" * 60)
    test_emit_cuda_basic(print_ir=p)
    test_emit_cuda_body_constructs(print_ir=p)
    test_source_tag(print_ir=p)

    print("\n" + "-" * 60)
    print("  Part B: codegen_kernel.cu build-time artifact")
    print("-" * 60)
    test_codegen_kernel_artifact(print_ir=p)

    print("\n" + "-" * 60)
    print("  Part C: joy-emit-cuda CLI behaviour")
    print("-" * 60)
    test_emit_cuda_error_and_pipeline(print_ir=p)

    print("\n" + "=" * 60)
    print("  ALL LESSON 11 TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
