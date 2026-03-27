#!/usr/bin/env python3
"""Lesson 12: Binary compilation (.cu -> .o -> .a / .so) — hands-on tests.

This test suite mirrors the SECOND compile-time stage of the joy GPU backend:
turning the CUDA C source files (``gpu_kernels.cu`` + ``codegen_kernel.cu``)
into the final fat-binary-bearing static library and the loadable shared
library.  The first stage (MLIR -> CUDA C) is covered by Lesson 11.

  Part A — nvcc compiles .cu files into objects + fatbins
           T1: The CUDA C emitted in Lesson 11 is independently compilable
               by ``nvcc -c`` into a real object file.
           T2: ``nvcc --ptx`` on the emitted CUDA C produces real PTX
               for at least one SM target (.entry joy_codegen_*_kernel,
               .target sm_*, with float arithmetic instructions).

  Part B — ar / archive into a static library
           T3: build/lib/backend/gpu/libJOYGpuKernels.a contains BOTH
               joy_codegen_* (auto codegen) and joy_kernel_* (hand-written)
               text symbols.
           T4: The fat binary embedded in libJOYGpuKernels.a covers the SM
               architectures listed in CMAKE_CUDA_ARCHITECTURES (sm_70..90)
               for both SASS and PTX.

  Part C — g++ -shared into the loadable runtime library
           T5: build/lib/libjoy_gpu_runtime.so exports the joy_gpu_* C ABI
               surface, plus all the kernel launchers it depends on, and
               links against the expected CUDA system libraries
               (cudart / cublas / cudnn).

Usage:
    python3 tests/python_tests/test_lesson12.py
    python3 tests/python_tests/test_lesson12.py --print-ir-all
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

cur_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(cur_path, "../.."))


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
JOY_GPU_KERNELS_A = os.path.join(BUILD_DIR, "lib", "backend", "gpu",
                                  "libJOYGpuKernels.a")
JOY_GPU_RUNTIME_SO = os.path.join(BUILD_DIR, "lib", "libjoy_gpu_runtime.so")

NVCC = "/usr/local/cuda/bin/nvcc"
CUDART_INCLUDE = "/usr/local/cuda/include"
CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"

# Architectures that joy/lib/backend/gpu/CMakeLists.txt asks for by default.
EXPECTED_SM_ARCHS = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"]


def _check_required():
    """Skip the suite if mandatory binaries / artifacts are missing."""
    missing = []
    for path in [JOY_OPT, JOY_EMIT_CUDA, STUB_F32, STUB_F16,
                 JOY_GPU_KERNELS_A, JOY_GPU_RUNTIME_SO]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print("Lesson 12 prerequisites missing — please build the project first:")
        for p in missing:
            print(f"  missing: {p}")
        sys.exit(2)


def _run(cmd, *, input_text=None, timeout=120):
    """Run a subprocess, return (rc, stdout, stderr)."""
    res = subprocess.run(cmd, input=input_text, capture_output=True,
                          text=True, timeout=timeout)
    return res.returncode, res.stdout, res.stderr


def _emit_cuda_from_stub(stub_path):
    """joy-opt --codegen-rms-norm STUB | joy-emit-cuda - -> stdout string."""
    rc1, ir, err1 = _run([JOY_OPT, "--codegen-rms-norm", stub_path])
    assert rc1 == 0, f"joy-opt failed on {stub_path}:\n{err1}"
    rc2, cu_text, err2 = _run([JOY_EMIT_CUDA, "-"], input_text=ir)
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


def _nm_text_symbols(path):
    """Return the set of `T` (text) symbols in path (demangled)."""
    rc, out, err = _run(["nm", "-C", "--defined-only", path], timeout=30)
    if rc != 0:
        # nm exits non-zero on some archives without index; retry without -C.
        rc, out, err = _run(["nm", "--defined-only", path], timeout=30)
    syms = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-2] in ("T", "t", "W", "w"):
            syms.add(parts[-1])
    return syms


# ============================================================================
# Part A — nvcc compiles .cu into objects
# ============================================================================

def test_nvcc_can_compile_emitted_cu(print_ir=False):
    """T1: emitted CUDA C can be compiled by nvcc into an object file."""
    print("\n" + "=" * 60)
    print("  T1: emitted CUDA C is compilable by `nvcc -c`")
    print("=" * 60)

    if not os.path.exists(NVCC):
        print(f"  WARNING: nvcc not at {NVCC}, skipping T1")
        return

    cu = _emit_cuda_from_stub(STUB_F32)

    with tempfile.TemporaryDirectory() as td:
        cu_path = os.path.join(td, "emit.cu")
        obj_path = os.path.join(td, "emit.o")
        with open(cu_path, "w") as f:
            f.write(cu)
        # We compile for one (the lowest) arch only to keep the test fast.
        cmd = [NVCC, "-c", "-arch=sm_70",
                "-I", CUDART_INCLUDE,
                "-o", obj_path, cu_path]
        rc, out, err = _run(cmd, timeout=120)
        if print_ir or rc != 0:
            print("  nvcc stdout:", out)
            print("  nvcc stderr:", err)
        assert rc == 0, f"[Lesson12] T1: nvcc failed (rc={rc}):\n{err}"
        assert os.path.exists(obj_path), \
            f"[Lesson12] T1: nvcc reported success but produced no .o"
        size = os.path.getsize(obj_path)
        assert size > 1024, f"[Lesson12] T1: object file too small ({size} B)"
        print(f"  [PASS] nvcc emit-cuda → emit.o ({size} bytes)")
    print("\n[Lesson12]: ================== T1 PASSED ==================")


def test_nvcc_ptx_output(print_ir=False):
    """T2: nvcc --ptx on the emitted CUDA C produces real PTX."""
    print("\n" + "=" * 60)
    print("  T2: nvcc can emit PTX for the codegen'd kernel")
    print("=" * 60)

    if not os.path.exists(NVCC):
        print(f"  WARNING: nvcc not at {NVCC}, skipping T2")
        return

    cu = _emit_cuda_from_stub(STUB_F32)

    with tempfile.TemporaryDirectory() as td:
        cu_path = os.path.join(td, "emit.cu")
        ptx_path = os.path.join(td, "emit.ptx")
        with open(cu_path, "w") as f:
            f.write(cu)
        cmd = [NVCC, "--ptx", "-arch=sm_70",
                "-I", CUDART_INCLUDE,
                "-o", ptx_path, cu_path]
        rc, out, err = _run(cmd, timeout=120)
        if print_ir or rc != 0:
            print("  nvcc stdout:", out)
            print("  nvcc stderr:", err)
        assert rc == 0, f"[Lesson12] T2: nvcc --ptx failed:\n{err}"
        assert os.path.exists(ptx_path), \
            f"[Lesson12] T2: nvcc reported success but produced no .ptx"
        with open(ptx_path, "r") as f:
            ptx = f.read()
        if print_ir:
            print(ptx[:600])

    checks = [
        (".version" in ptx, "PTX directive .version present"),
        (".target sm_70" in ptx, "PTX targets sm_70 (as requested)"),
        (".entry" in ptx, "PTX defines kernel entry point"),
        # We at least expect one of the codegen kernels to appear in the
        # entry name.  PTX mangles the C names so just substring-match.
        ("joy_codegen_rms_norm_f32_kernel" in ptx
            or "joy_codegen_fuse_add_rms_norm_f32_kernel" in ptx,
         "PTX entry references one of the codegen kernel names"),
        # Look for at least one fp32 math instruction we expect from the
        # rms_norm body.
        ("mul.f32" in ptx or "fma.rn.f32" in ptx or "mad" in ptx,
         "PTX contains float math instructions"),
    ]
    assert _report(checks), "[Lesson12] T2 failed"
    print("\n[Lesson12]: ================== T2 PASSED ==================")


# ============================================================================
# Part B — ar / archive into a static library
# ============================================================================

def test_static_library_symbols(print_ir=False):
    """T3: libJOYGpuKernels.a exposes hand-written + codegen launcher symbols."""
    print("\n" + "=" * 60)
    print("  T3: libJOYGpuKernels.a contains both hand-written + codegen")
    print("       kernel launcher symbols")
    print("=" * 60)

    if not os.path.exists(JOY_GPU_KERNELS_A):
        print(f"  WARNING: {JOY_GPU_KERNELS_A} missing; build the project first")
        assert False, "[Lesson12] T3: libJOYGpuKernels.a not built"

    syms = _nm_text_symbols(JOY_GPU_KERNELS_A)
    if print_ir:
        print("  hand-written + codegen symbols (sample):")
        for s in sorted(syms):
            if s.startswith("joy_kernel_") or s.startswith("joy_codegen_"):
                print(f"    {s}")

    must_have_codegen = [
        "joy_codegen_rms_norm_f32",
        "joy_codegen_rms_norm_f16",
        "joy_codegen_fuse_add_rms_norm_f32",
        "joy_codegen_fuse_add_rms_norm_f16",
    ]
    must_have_handwritten = [
        "joy_kernel_add_f32",
        "joy_kernel_add_f16",
        "joy_kernel_mul_f32",
        "joy_kernel_silu_f32",
        "joy_kernel_silu_f16",
        "joy_kernel_apply_rotary_emb_f32",
        "joy_kernel_repeat_kv_f16",
        "joy_kernel_transpose",
    ]
    checks = []
    for s in must_have_codegen:
        checks.append((s in syms,
                        f"codegen launcher {s} present in static lib"))
    for s in must_have_handwritten:
        checks.append((s in syms,
                        f"hand-written launcher {s} present in static lib"))
    assert _report(checks), "[Lesson12] T3 failed"
    print("\n[Lesson12]: ================== T3 PASSED ==================")


def test_fatbin_archs(print_ir=False):
    """T4: fat binary embedded in libJOYGpuKernels.a covers expected SM archs."""
    print("\n" + "=" * 60)
    print("  T4: fat binary covers all CMAKE_CUDA_ARCHITECTURES targets")
    print("=" * 60)

    if not os.path.exists(JOY_GPU_KERNELS_A):
        assert False, "[Lesson12] T4: libJOYGpuKernels.a not built"

    if not os.path.exists(CUOBJDUMP):
        print(f"  WARNING: cuobjdump not at {CUOBJDUMP}, skipping T4")
        return

    rc, out, err = _run([CUOBJDUMP, "--list-text", JOY_GPU_KERNELS_A],
                         timeout=30)
    assert rc == 0, f"[Lesson12] T4: cuobjdump failed:\n{err}"
    archs_found = set(re.findall(r"sm_\d+", out))
    if print_ir:
        print(f"  archs found in fatbin: {sorted(archs_found)}")

    checks = []
    for arch in EXPECTED_SM_ARCHS:
        checks.append((arch in archs_found,
                        f"fatbin contains SASS for {arch}"))
    # PTX should also be present for every arch (cuobjdump --list-ptx).
    rc2, out2, err2 = _run([CUOBJDUMP, "--list-ptx", JOY_GPU_KERNELS_A],
                             timeout=30)
    assert rc2 == 0, f"[Lesson12] T4: cuobjdump --list-ptx failed:\n{err2}"
    ptx_archs = set(re.findall(r"sm_\d+", out2))
    for arch in EXPECTED_SM_ARCHS:
        checks.append((arch in ptx_archs,
                        f"fatbin contains PTX for {arch}"))

    assert _report(checks), "[Lesson12] T4 failed"
    print("\n[Lesson12]: ================== T4 PASSED ==================")


# ============================================================================
# Part C — g++ -shared into the loadable runtime library
# ============================================================================

def test_shared_library_symbols(print_ir=False):
    """T5: libjoy_gpu_runtime.so exports the full joy_gpu_* C ABI plus
    kernel launchers and depends on the right system libraries."""
    print("\n" + "=" * 60)
    print("  T5: libjoy_gpu_runtime.so exposes runtime + kernel symbols and")
    print("       links the expected CUDA system libraries")
    print("=" * 60)

    if not os.path.exists(JOY_GPU_RUNTIME_SO):
        assert False, "[Lesson12] T5: libjoy_gpu_runtime.so not built"

    syms = _nm_text_symbols(JOY_GPU_RUNTIME_SO)
    if print_ir:
        for s in sorted(syms):
            if s.startswith("joy_gpu_") or s.startswith("joy_codegen_") \
                    or s.startswith("joy_kernel_"):
                print(f"    {s}")

    must_have_runtime = [
        "joy_gpu_linear",
        "joy_gpu_matmul",
        "joy_gpu_softmax",
        "joy_gpu_rms_norm",
        "joy_gpu_fuse_add_rmsnorm",
        "joy_gpu_silu",
        "joy_gpu_add",
        "joy_gpu_mul",
        "joy_gpu_embedding",
        "joy_gpu_reshape",
        "joy_gpu_transpose",
        "joy_gpu_apply_rotary_emb",
        "joy_gpu_repeat_kv",
    ]
    must_have_codegen = [
        "joy_codegen_rms_norm_f32",
        "joy_codegen_fuse_add_rms_norm_f32",
    ]
    must_have_handwritten = [
        "joy_kernel_silu_f32",
        "joy_kernel_add_f16",
    ]

    checks = []
    for s in must_have_runtime:
        checks.append((s in syms,
                        f"runtime C ABI entry {s} exported by .so"))
    for s in must_have_codegen:
        checks.append((s in syms,
                        f"codegen launcher {s} exported by .so"))
    for s in must_have_handwritten:
        checks.append((s in syms,
                        f"hand-written launcher {s} exported by .so"))

    # ldd: confirm runtime dependencies (cudart / cublas / cudnn).
    rc, out, err = _run(["ldd", JOY_GPU_RUNTIME_SO], timeout=10)
    if rc == 0:
        checks.append(("libcudart.so" in out,
                        "shared lib links against libcudart.so"))
        checks.append(("libcublas.so" in out,
                        "shared lib links against libcublas.so"))
        checks.append(("libcudnn.so" in out,
                        "shared lib links against libcudnn.so"))
    else:
        # Don't outright fail if ldd is not available; just warn.
        print(f"  WARNING: ldd failed (rc={rc}); skipping dependency checks")

    assert _report(checks), "[Lesson12] T5 failed"
    print("\n[Lesson12]: ================== T5 PASSED ==================")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Lesson 12: Binary compilation tests")
    parser.add_argument("--print-ir-all", action="store_true",
                        help="Print emitted CUDA / PTX / symbols for each test")
    args = parser.parse_args()
    p = args.print_ir_all

    _check_required()

    print("=" * 60)
    print("  Lesson 12: Binary compilation (.cu -> .o -> .a / .so)")
    print("=" * 60)

    print("\n" + "-" * 60)
    print("  Part A: nvcc compiles .cu -> .o / PTX")
    print("-" * 60)
    test_nvcc_can_compile_emitted_cu(print_ir=p)
    test_nvcc_ptx_output(print_ir=p)

    print("\n" + "-" * 60)
    print("  Part B: ar packages .o into libJOYGpuKernels.a (fatbinary)")
    print("-" * 60)
    test_static_library_symbols(print_ir=p)
    test_fatbin_archs(print_ir=p)

    print("\n" + "-" * 60)
    print("  Part C: g++ -shared into libjoy_gpu_runtime.so")
    print("-" * 60)
    test_shared_library_symbols(print_ir=p)

    print("\n" + "=" * 60)
    print("  ALL LESSON 12 TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
