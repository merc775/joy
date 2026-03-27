#!/usr/bin/env python3
"""Lesson 14: Joy full-stack build & artifact audit.

This suite is the engineering-side complement to Lesson 13 (the runtime
mechanics) and Lesson 15 (the actual Qwen3-0.6B inference): it verifies
that a fresh ``./scripts/build.sh`` has materialised every artifact the
later lessons require, and that the source tree itself has the expected
top-level layout / CMake structure.

Eight self-contained checks:

  T1: top-level source layout (CMakeLists.txt + 7 standard dirs)
  T2: every key CMakeLists.txt referenced by the lesson is on disk
  T3: TableGen output (``*.h.inc`` + ``*.cpp.inc`` for joy/joyl/joyh)
  T4: static libraries (MLIR dialect + JOYOptimizer + JOYFrontend + GPU)
  T5: command-line tools (``joy-opt`` / ``joy-emit-cuda``) exist and
      respond to ``--help``
  T6: shared library exposes the 13 ``joy_gpu_*`` + 12 ``joy_test_*``
      entry points
  T7: build scripts are executable and ``build/env.sh`` is present (or
      regenerable from ``init.sh``)
  T8: build-time codegen pipeline produced ``codegen_kernel.cu`` and
      it contains all four kernels (f32/f16 × rms_norm/fuse_add_rmsnorm)
      with the expected auto-generation banner

Usage:
    python3 tests/python_tests/test_lesson14.py
    python3 tests/python_tests/test_lesson14.py --print-info
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
BUILD_BIN = os.path.join(BUILD_DIR, "bin")
BUILD_LIB = os.path.join(BUILD_DIR, "lib")
BUILD_INC = os.path.join(BUILD_DIR, "include")


def _exists_path(p: str) -> bool:
    return os.path.exists(p)


def _executable(p: str) -> bool:
    return os.path.isfile(p) and os.access(p, os.X_OK)


def _banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _report(checks) -> bool:
    all_ok = True
    for ok, desc in checks:
        prefix = "  [PASS]" if ok else "  [FAIL]"
        print(f"{prefix} {desc}")
        if not ok:
            all_ok = False
    return all_ok


# ----------------------------------------------------------------------
# T1: Top-level source layout
# ----------------------------------------------------------------------
def test_source_layout(*, print_info: bool = False) -> None:
    _banner("T1: top-level source layout (joy/)")
    layout = [
        ("CMakeLists.txt",         "file"),
        ("README.md",              "file"),
        ("include",                "dir"),
        ("lib",                    "dir"),
        ("tools",                  "dir"),
        ("scripts",                "dir"),
        ("python",                 "dir"),
        ("tests",                  "dir"),
        ("docs",                   "dir"),
    ]
    checks = []
    for name, kind in layout:
        p = os.path.join(PROJECT_ROOT, name)
        if kind == "file":
            ok = os.path.isfile(p)
        else:
            ok = os.path.isdir(p)
        checks.append((ok, f"{name:24s} present  ({kind})"))
        if print_info:
            print(f"  {name:24s}-> {p}")
    assert _report(checks), "[Lesson14] T1 failed"
    print("\n[Lesson14]: ================== T1 PASSED ==================")


# ----------------------------------------------------------------------
# T2: Key CMakeLists.txt files
# ----------------------------------------------------------------------
def test_cmake_files(*, print_info: bool = False) -> None:
    _banner("T2: key CMakeLists.txt files")
    cmakes = [
        "CMakeLists.txt",
        "include/CMakeLists.txt",
        "include/joy/dialect/joy/CMakeLists.txt",
        "include/joy/dialect/joyl/CMakeLists.txt",
        "include/joy/dialect/joyh/CMakeLists.txt",
        "lib/CMakeLists.txt",
        "lib/dialect/CMakeLists.txt",
        "lib/dialect/joy/CMakeLists.txt",
        "lib/dialect/joyl/CMakeLists.txt",
        "lib/dialect/joyh/CMakeLists.txt",
        "lib/frontend/CMakeLists.txt",
        "lib/optimizer/CMakeLists.txt",
        "lib/backend/CMakeLists.txt",
        "lib/backend/gpu/CMakeLists.txt",
        "tools/CMakeLists.txt",
    ]
    checks = []
    for rel in cmakes:
        p = os.path.join(PROJECT_ROOT, rel)
        ok = os.path.isfile(p)
        checks.append((ok, f"{rel}"))
        if print_info and ok:
            sz = os.path.getsize(p)
            print(f"    {rel}: {sz} bytes")
    assert _report(checks), "[Lesson14] T2 failed"
    print("\n[Lesson14]: ================== T2 PASSED ==================")


# ----------------------------------------------------------------------
# T3: TableGen output
# ----------------------------------------------------------------------
def test_tablegen_output(*, print_info: bool = False) -> None:
    _banner("T3: TableGen .inc outputs in build/include/")
    checks = []
    for dialect in ("joy", "joyl", "joyh"):
        # naming: JoyOps / JoylOps / JoyhOps + JoyDialect / JoylDialect / JoyhDialect
        prefix_ops = "JoyOps" if dialect == "joy" else (
            "JoylOps" if dialect == "joyl" else "JoyhOps")
        prefix_dialect = "JoyDialect" if dialect == "joy" else (
            "JoylDialect" if dialect == "joyl" else "JoyhDialect")
        base = os.path.join(BUILD_INC, "joy", "dialect", dialect)
        for fname in (f"{prefix_ops}.h.inc",
                      f"{prefix_ops}.cpp.inc",
                      f"{prefix_dialect}.h.inc",
                      f"{prefix_dialect}.cpp.inc"):
            p = os.path.join(base, fname)
            ok = os.path.isfile(p) and os.path.getsize(p) > 0
            checks.append((ok, f"{dialect}/{fname}"))
            if print_info and ok:
                print(f"    {dialect}/{fname}: {os.path.getsize(p)} bytes")
    assert _report(checks), "[Lesson14] T3 failed"
    print("\n[Lesson14]: ================== T3 PASSED ==================")


# ----------------------------------------------------------------------
# T4: Static libraries
# ----------------------------------------------------------------------
def _find_static_lib(name: str) -> str:
    """Search build/ for ``lib<name>.a`` regardless of subdirectory."""
    target = f"lib{name}.a"
    # First try the canonical location.
    candidates = [
        os.path.join(BUILD_LIB, target),
        os.path.join(BUILD_LIB, "backend", "gpu", target),
        os.path.join(BUILD_LIB, "runtime", "gpu", target),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Fallback: full walk.
    for root, _dirs, files in os.walk(BUILD_DIR):
        if target in files:
            return os.path.join(root, target)
    return ""


def test_static_libs(*, print_info: bool = False) -> None:
    _banner("T4: static libraries in build/")
    expected = [
        "MLIRJoyDialect",
        "MLIRJoylDialect",
        "MLIRJoyhDialect",
        "JOYOptimizer",
        "JOYFrontend",
        "JOYGpuKernels",
        "JOYGpuBackend",
    ]
    checks = []
    for name in expected:
        p = _find_static_lib(name)
        ok = bool(p) and os.path.isfile(p) and os.path.getsize(p) > 0
        if ok and print_info:
            print(f"    lib{name}.a -> {p}  ({os.path.getsize(p)} bytes)")
        checks.append((ok, f"lib{name}.a"))
    assert _report(checks), "[Lesson14] T4 failed"
    print("\n[Lesson14]: ================== T4 PASSED ==================")


# ----------------------------------------------------------------------
# T5: Command-line tools
# ----------------------------------------------------------------------
def test_cli_tools(*, print_info: bool = False) -> None:
    _banner("T5: command-line tools (joy-opt / joy-emit-cuda)")
    tools = [
        ("joy-opt",       ["--help"]),
        ("joy-emit-cuda", ["--help"]),
    ]
    checks = []
    for name, args in tools:
        binp = os.path.join(BUILD_BIN, name)
        is_exe = _executable(binp)
        checks.append((is_exe, f"{name} exists & executable  ({binp})"))
        if is_exe:
            try:
                p = subprocess.run([binp] + args,
                                   capture_output=True,
                                   text=True, timeout=10)
                # llvm tools sometimes print --help to stderr; consider both
                blob = (p.stdout or "") + (p.stderr or "")
                blob_short = blob[:200].replace("\n", " | ")
                ok_help = len(blob) > 0
                checks.append((ok_help,
                               f"{name} --help produces output "
                               f"(len={len(blob)})"))
                if print_info and ok_help:
                    print(f"    {name} --help[:200]: {blob_short}")
            except subprocess.TimeoutExpired:
                checks.append((False, f"{name} --help timed out"))
    assert _report(checks), "[Lesson14] T5 failed"
    print("\n[Lesson14]: ================== T5 PASSED ==================")


# ----------------------------------------------------------------------
# T6: shared library exports the C ABI
# ----------------------------------------------------------------------
EXPECTED_OP_SYMBOLS = [
    "joy_gpu_embedding",
    "joy_gpu_rms_norm",
    "joy_gpu_linear",
    "joy_gpu_matmul",
    "joy_gpu_softmax",
    "joy_gpu_silu",
    "joy_gpu_add",
    "joy_gpu_mul",
    "joy_gpu_reshape",
    "joy_gpu_transpose",
    "joy_gpu_apply_rotary_emb",
    "joy_gpu_repeat_kv",
    "joy_gpu_fuse_add_rmsnorm",
]
EXPECTED_TEST_SYMBOLS = [
    "joy_test_device_alloc",
    "joy_test_device_free",
    "joy_test_memcpy_h2d",
    "joy_test_memcpy_d2h",
    "joy_test_memset_zero",
    "joy_test_device_synchronize",
    "joy_test_create_context",
    "joy_test_destroy_context",
    "joy_test_stream_synchronize",
    "joy_test_runtime_signature",
    "joy_test_cuda_runtime_version",
    "joy_test_cudnn_version",
]


def test_shared_library_symbols(*, print_info: bool = False) -> None:
    _banner("T6: libjoy_gpu_runtime.so + exported symbols")
    so = os.path.join(BUILD_LIB, "libjoy_gpu_runtime.so")
    checks = [(os.path.isfile(so), f"libjoy_gpu_runtime.so present  ({so})")]
    if os.path.isfile(so):
        try:
            p = subprocess.run(
                ["nm", "-D", "--defined-only", so],
                capture_output=True, text=True, timeout=30)
            symbols = set()
            for line in (p.stdout or "").splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "T":
                    symbols.add(parts[2])
            for s in EXPECTED_OP_SYMBOLS:
                checks.append((s in symbols, f"symbol  {s}"))
            for s in EXPECTED_TEST_SYMBOLS:
                checks.append((s in symbols, f"symbol  {s}"))
            if print_info:
                print(f"    total exported text symbols: {len(symbols)}")
        except FileNotFoundError:
            checks.append((False, "nm tool not on PATH"))
    assert _report(checks), "[Lesson14] T6 failed"
    print("\n[Lesson14]: ================== T6 PASSED ==================")


# ----------------------------------------------------------------------
# T7: Build scripts and env file
# ----------------------------------------------------------------------
def test_build_scripts(*, print_info: bool = False) -> None:
    _banner("T7: build scripts under scripts/")
    scripts = [
        "scripts/init.sh",
        "scripts/build.sh",
        "scripts/regen_codegen_kernel.sh",
    ]
    checks = []
    for rel in scripts:
        p = os.path.join(PROJECT_ROOT, rel)
        ok = _executable(p) or (os.path.isfile(p) and os.access(p, os.R_OK))
        checks.append((ok, f"{rel} exists"))
        if print_info and ok:
            print(f"    {rel}  size={os.path.getsize(p)}")
    # env.sh is optional, but if absent, init.sh must be there.
    env_sh = os.path.join(BUILD_DIR, "env.sh")
    init_sh = os.path.join(PROJECT_ROOT, "scripts", "init.sh")
    ok_env = os.path.isfile(env_sh) or os.path.isfile(init_sh)
    checks.append((ok_env,
                   f"build/env.sh present or regenerable from init.sh"))
    assert _report(checks), "[Lesson14] T7 failed"
    print("\n[Lesson14]: ================== T7 PASSED ==================")


# ----------------------------------------------------------------------
# T8: build-time codegen produced codegen_kernel.cu
# ----------------------------------------------------------------------
EXPECTED_KERNEL_LAUNCHERS = [
    "joy_codegen_rms_norm_f32",
    "joy_codegen_rms_norm_f16",
    "joy_codegen_fuse_add_rms_norm_f32",
    "joy_codegen_fuse_add_rms_norm_f16",
]


def test_codegen_kernel_cu(*, print_info: bool = False) -> None:
    _banner("T8: build/lib/backend/gpu/codegen_kernel.cu")
    cu = os.path.join(BUILD_LIB, "backend", "gpu", "codegen_kernel.cu")
    checks = [(os.path.isfile(cu), f"codegen_kernel.cu present  ({cu})")]
    if not os.path.isfile(cu):
        assert _report(checks), "[Lesson14] T8 failed"
        return

    with open(cu, "r") as f:
        text = f.read()
    if print_info:
        print(f"    codegen_kernel.cu size: {len(text)} bytes")
        # show first 6 banner lines
        for line in text.splitlines()[:6]:
            print(f"    | {line}")

    # banner
    checks.append((
        "regen_codegen_kernel.sh" in text,
        "header banner mentions regen_codegen_kernel.sh"))
    checks.append((
        "joy-opt --codegen-rms-norm | joy-emit-cuda" in text,
        "banner records the codegen pipeline"))
    checks.append((
        "joy::emitCudaC" in text,
        "banner mentions joy::emitCudaC as emitter"))

    # 4 host launchers (extern "C" entry points)
    for sym in EXPECTED_KERNEL_LAUNCHERS:
        # match e.g. `extern "C" void joy_codegen_rms_norm_f32(...)`
        pat = re.compile(rf'extern\s+"C"\s+void\s+{sym}\s*\(')
        ok = pat.search(text) is not None
        checks.append((ok, f"extern \"C\" launcher  {sym}"))

    # 4 corresponding __global__ kernels
    for sym in EXPECTED_KERNEL_LAUNCHERS:
        kernel = f"{sym}_kernel"
        pat = re.compile(rf'__global__\s+void\s+{kernel}\s*\(')
        ok = pat.search(text) is not None
        checks.append((ok, f"__global__ kernel    {kernel}"))

    # Some MLIR-y artefacts that prove this came out of the emitter.
    checks.append((
        "rsqrtf" in text or "hrsqrt" in text or "rsqrt" in text,
        "kernel contains rsqrt-flavored RMSNorm primitive"))
    checks.append((
        "blockIdx.x" in text and "threadIdx.x" in text,
        "kernel uses block/thread indexing"))

    assert _report(checks), "[Lesson14] T8 failed"
    print("\n[Lesson14]: ================== T8 PASSED ==================")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lesson 14: joy full-stack build audit")
    parser.add_argument("--print-info", action="store_true",
                        help="Print discovered paths and sizes")
    args = parser.parse_args()
    p = args.print_info

    print("=" * 70)
    print("  Lesson 14: joy project full-stack build audit")
    print("=" * 70)
    print(f"  PROJECT_ROOT  : {PROJECT_ROOT}")
    print(f"  BUILD_DIR     : {BUILD_DIR}")

    test_source_layout(print_info=p)
    test_cmake_files(print_info=p)
    test_tablegen_output(print_info=p)
    test_static_libs(print_info=p)
    test_cli_tools(print_info=p)
    test_shared_library_symbols(print_info=p)
    test_build_scripts(print_info=p)
    test_codegen_kernel_cu(print_info=p)

    print("\n" + "=" * 70)
    print("  ALL LESSON 14 TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
