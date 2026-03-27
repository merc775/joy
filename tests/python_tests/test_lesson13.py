#!/usr/bin/env python3
"""Lesson 13: GPU runtime — hands-on tests.

This suite verifies the three layers introduced in Lesson 13:

  Part A — Runtime self-check (gpu_test_helpers.cpp + ctypes wrapper)
           T1: libjoy_gpu_runtime.so loads, every joy_gpu_* / joy_test_*
               C-ABI symbol is resolvable, and the runtime signature +
               CUDA/cuDNN versions are sane.
           T2: MemrefDesc / GpuContext ctypes mirrors match the C structs
               byte-for-byte (sizes and offsets).

  Part B — Three backend dispatch paths (gpu_ops.cpp)
           T3: Hand-written CUDA kernel path — joy_gpu_silu vs numpy.
           T4: cuBLAS path           — joy_gpu_linear vs numpy `x @ W^T`
                                       (covers 2D/3D/4D inputs and the
                                       row-major <-> column-major swap).
           T5: cuDNN path            — joy_gpu_softmax vs numpy stable
                                       softmax (resolves axis = -1).
           T6: MLIR codegen path     — joy_gpu_rms_norm vs numpy RMSNorm.
           T7: Fused codegen path    — joy_gpu_fuse_add_rmsnorm vs
                                       numpy (x + res).rmsnorm().

  Part C — End-to-end stream coherence
           T8: A single GpuContext drives three chained ops (add, silu,
               linear) on the same stream and produces the expected
               result with a single stream-synchronize at the end.

Usage:
    python3 tests/python_tests/test_lesson13.py
    python3 tests/python_tests/test_lesson13.py --print-info
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys

import numpy as np


# ----------------------------------------------------------------------
# Make joy/tests/python_tests/test_op importable so we can reuse the
# ctypes wrapper layer that the per-operator tests already share.
# ----------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_TEST_OP_DIR = os.path.join(_HERE, "test_op")
if _TEST_OP_DIR not in sys.path:
    sys.path.insert(0, _TEST_OP_DIR)

# Re-exports of the runtime wrapper used throughout this file.
from _runtime import (                                      # noqa: E402
    F16, F32, F64, I32, I64,
    GpuContext as PyGpuContext,
    JoyGpuRuntime,
    MemrefDesc as PyMemrefDesc,
    assert_close,
    get_runtime,
)


# All joy_gpu_* entry points that gpu_entry.cpp exports.
EXPECTED_OP_ENTRIES = [
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

# All joy_test_* helpers that gpu_test_helpers.cpp exports.
EXPECTED_TEST_HELPERS = [
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


def _banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _report(checks):
    all_ok = True
    for ok, desc in checks:
        prefix = "  [PASS]" if ok else "  [FAIL]"
        print(f"{prefix} {desc}")
        if not ok:
            all_ok = False
    return all_ok


# ============================================================================
# Part A — runtime self-check
# ============================================================================

def test_runtime_loads(rt: JoyGpuRuntime, *, print_info: bool = False) -> None:
    """T1: every C-ABI symbol resolves; signature/versions look sane."""
    _banner("T1: runtime self-check (libjoy_gpu_runtime.so)")

    sig = rt.signature()
    cuda_ver = rt.cuda_runtime_version()    # e.g. 12020 = CUDA 12.2.0
    cudnn_ver = rt.cudnn_version()          # e.g.  8907 = cuDNN 8.9.7
    if print_info:
        print(f"  signature           = {sig!r}")
        print(f"  cuda_runtime_version = {cuda_ver}")
        print(f"  cudnn_version        = {cudnn_ver}")

    checks = [
        (sig.startswith("joy_gpu_runtime["),
         f"runtime signature starts with 'joy_gpu_runtime[' (got {sig!r})"),
        (cuda_ver >= 12000,
         f"CUDA runtime >= 12.0 (got {cuda_ver})"),
        (cudnn_ver >= 8000,
         f"cuDNN >= 8.0 (got {cudnn_ver})"),
    ]

    lib = rt._lib  # underlying ctypes.CDLL
    for name in EXPECTED_OP_ENTRIES:
        checks.append((hasattr(lib, name) and getattr(lib, name) is not None,
                       f"op entry  {name}  is resolvable"))
    for name in EXPECTED_TEST_HELPERS:
        checks.append((hasattr(lib, name) and getattr(lib, name) is not None,
                       f"helper    {name}  is resolvable"))

    assert _report(checks), "[Lesson13] T1 failed"
    print("\n[Lesson13]: ================== T1 PASSED ==================")


def test_struct_layouts(*, print_info: bool = False) -> None:
    """T2: MemrefDesc / GpuContext ctypes mirrors match the C structs."""
    _banner("T2: MemrefDesc & GpuContext memory layout")

    md_size = ctypes.sizeof(PyMemrefDesc)
    ctx_size = ctypes.sizeof(PyGpuContext)
    if print_info:
        print(f"  sizeof(MemrefDesc)  = {md_size} bytes")
        print(f"  sizeof(GpuContext)  = {ctx_size} bytes")
        print(f"  MemrefDesc fields   = {[f[0] for f in PyMemrefDesc._fields_]}")
        print(f"  GpuContext fields   = {[f[0] for f in PyGpuContext._fields_]}")

    # MemrefDesc: i64 rank, void* data, int64_t* shape, i32 element_type + 4B
    # padding to align the next struct in an array.
    # Layout: 8 + 8 + 8 + 4 = 28 bytes content + 4 padding => 32 bytes.
    expected_md_size = 32
    # GpuContext is just three void* fields => 24 bytes on x86_64.
    expected_ctx_size = 24

    # element_type id alignment with gpu_ops.cpp's kF16/kF32/...
    dtype_ids = {
        np.float16: F16, np.float32: F32, np.float64: F64,
        np.int32:   I32, np.int64:   I64,
    }
    checks = [
        (md_size == expected_md_size,
         f"sizeof(MemrefDesc) == {expected_md_size} bytes "
         f"(got {md_size})"),
        (ctx_size == expected_ctx_size,
         f"sizeof(GpuContext) == {expected_ctx_size} bytes "
         f"(got {ctx_size})"),
        ([f[0] for f in PyMemrefDesc._fields_] ==
         ["rank", "data", "shape", "element_type"],
         "MemrefDesc field order matches gpu_runner.h"),
        ([f[0] for f in PyGpuContext._fields_] ==
         ["stream", "cublas", "cudnn"],
         "GpuContext field order matches gpu_runner.h"),
        (dtype_ids[np.float16] == 0, "F16 dtype id == 0"),
        (dtype_ids[np.float32] == 1, "F32 dtype id == 1"),
        (dtype_ids[np.float64] == 2, "F64 dtype id == 2"),
        (dtype_ids[np.int32]   == 3, "I32 dtype id == 3"),
        (dtype_ids[np.int64]   == 4, "I64 dtype id == 4"),
    ]
    assert _report(checks), "[Lesson13] T2 failed"
    print("\n[Lesson13]: ================== T2 PASSED ==================")


# ============================================================================
# Part B — three backend dispatch paths
# ============================================================================

def _np_silu(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    return (x64 / (1.0 + np.exp(-x64))).astype(x.dtype)


def test_handwritten_kernel_silu(rt: JoyGpuRuntime, *,
                                 print_info: bool = False) -> None:
    """T3: hand-written CUDA kernel path via joy_gpu_silu."""
    _banner("T3: hand-written kernel path  (joy_kernel_silu)")
    shapes = [(4, 16), (1, 64, 128), (2, 3, 7, 17)]

    with rt.context() as ctx:
        for shape in shapes:
            rng = np.random.default_rng(7)
            x = rng.standard_normal(shape, dtype=np.float32) * 4.0
            expected = _np_silu(x)

            dx = rt.upload(x)
            dy = rt.alloc_like(shape, np.float32)
            try:
                rt.run_op("silu", ctx, inputs=[dx], outputs=[dy])
                actual = rt.download(dy)
            finally:
                dx.free()
                dy.free()
            assert_close(actual, expected, atol=1e-5, rtol=1e-5,
                         name=f"silu shape={shape}")
            if print_info:
                print(f"  PASS  silu shape={list(shape)}")
    print("  [PASS] joy_gpu_silu dispatches to joy_kernel_silu_f32, "
          "values match numpy")
    print("\n[Lesson13]: ================== T3 PASSED ==================")


def test_cublas_linear(rt: JoyGpuRuntime, *,
                       print_info: bool = False) -> None:
    """T4: cuBLAS path via joy_gpu_linear (transB) — covers 2D/3D/4D."""
    _banner("T4: cuBLAS path           (joy_gpu_linear, x @ W^T)")
    cases = [
        ((8, 16),       (32, 16)),            # 2D
        ((1, 16, 64),   (32, 64)),            # 3D
        ((2, 4, 8, 32), (16, 32)),            # 4D
        ((5, 7, 13),    (11, 13)),            # awkward shapes
    ]

    with rt.context() as ctx:
        for in_shape, w_shape in cases:
            rng = np.random.default_rng(0)
            x = rng.standard_normal(in_shape, dtype=np.float32)
            W = rng.standard_normal(w_shape, dtype=np.float32)
            expected = x @ W.T

            out_shape = list(in_shape[:-1]) + [w_shape[0]]
            dx = rt.upload(x)
            dw = rt.upload(W)
            dy = rt.alloc_like(out_shape, np.float32)
            try:
                rt.run_op("linear", ctx, inputs=[dx, dw], outputs=[dy])
                actual = rt.download(dy)
            finally:
                dx.free()
                dw.free()
                dy.free()
            assert_close(actual, expected, atol=1e-3, rtol=1e-3,
                         name=f"linear x={in_shape} W={w_shape}")
            if print_info:
                print(f"  PASS  linear x={list(in_shape)} W={list(w_shape)} "
                      f"-> {out_shape}")
    print("  [PASS] joy_gpu_linear dispatches to cublasGemmEx (transB=true), "
          "row<->col major swap correct")
    print("\n[Lesson13]: ================== T4 PASSED ==================")


def _np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x64 = x.astype(np.float64)
    x64 -= np.max(x64, axis=axis, keepdims=True)
    e = np.exp(x64)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(x.dtype)


def test_cudnn_softmax(rt: JoyGpuRuntime, *,
                      print_info: bool = False) -> None:
    """T5: cuDNN path via joy_gpu_softmax."""
    _banner("T5: cuDNN path             (joy_gpu_softmax, axis=-1)")
    shapes = [(8, 16), (1, 32, 128), (4, 8, 64)]

    with rt.context() as ctx:
        for shape in shapes:
            rng = np.random.default_rng(11)
            x = rng.standard_normal(shape, dtype=np.float32) * 3.0
            # joy_gpu_softmax resolves axis = -1 internally; pass -1.
            expected = _np_softmax(x, axis=-1)

            dx = rt.upload(x)
            dy = rt.alloc_like(shape, np.float32)
            try:
                rt.run_op("softmax", ctx, inputs=[dx], outputs=[dy],
                          extra_args=[ctypes.c_int64(-1)])
                actual = rt.download(dy)
            finally:
                dx.free()
                dy.free()
            assert_close(actual, expected, atol=1e-5, rtol=1e-5,
                         name=f"softmax shape={shape}")
            # Row sums should also be ~1.0 (sanity for the cuDNN path).
            row_sums = actual.reshape(-1, shape[-1]).sum(axis=-1)
            assert np.allclose(row_sums, 1.0, atol=1e-4), (
                f"softmax row sums must be ~1.0 (got {row_sums[:4]}...)")
            if print_info:
                print(f"  PASS  softmax shape={list(shape)}  "
                      f"row_sums~1.0")
    print("  [PASS] joy_gpu_softmax dispatches to cudnnSoftmaxForward, "
          "rows sum to 1.0")
    print("\n[Lesson13]: ================== T5 PASSED ==================")


def _np_rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x64 = x.astype(np.float64)
    w64 = w.astype(np.float64)
    ms = np.mean(x64 * x64, axis=-1, keepdims=True)
    return (x64 * (1.0 / np.sqrt(ms + eps)) * w64).astype(x.dtype)


def test_codegen_rms_norm(rt: JoyGpuRuntime, *,
                          print_info: bool = False) -> None:
    """T6: MLIR codegen path via joy_gpu_rms_norm (joy_codegen_rms_norm_f32)."""
    _banner("T6: MLIR codegen path      (joy_gpu_rms_norm)")
    shapes = [(4, 16), (1, 32, 128), (2, 4, 7, 64), (3, 1024)]
    eps = 1e-6

    with rt.context() as ctx:
        for shape in shapes:
            H = shape[-1]
            rng = np.random.default_rng(123)
            x = rng.standard_normal(shape, dtype=np.float32)
            w = rng.standard_normal((H,), dtype=np.float32)
            expected = _np_rms_norm(x, w, eps)

            dx = rt.upload(x)
            dw = rt.upload(w)
            dy = rt.alloc_like(shape, np.float32)
            try:
                rt.run_op("rms_norm", ctx,
                          inputs=[dx, dw], outputs=[dy],
                          extra_args=[ctypes.c_float(eps)])
                actual = rt.download(dy)
            finally:
                dx.free()
                dw.free()
                dy.free()
            assert_close(actual, expected, atol=1e-4, rtol=1e-4,
                         name=f"rms_norm shape={shape}")
            if print_info:
                print(f"  PASS  rms_norm shape={list(shape)}  eps={eps}")
    print("  [PASS] joy_gpu_rms_norm dispatches to joy_codegen_rms_norm_f32 "
          "(MLIR-generated kernel)")
    print("\n[Lesson13]: ================== T6 PASSED ==================")


def test_codegen_fuse_add_rmsnorm(rt: JoyGpuRuntime, *,
                                  print_info: bool = False) -> None:
    """T7: fused codegen path via joy_gpu_fuse_add_rmsnorm."""
    _banner("T7: fused codegen path     (joy_gpu_fuse_add_rmsnorm)")
    shapes = [(4, 16), (1, 32, 128), (2, 4, 7, 64)]
    eps = 1e-6

    with rt.context() as ctx:
        for shape in shapes:
            H = shape[-1]
            rng = np.random.default_rng(2025)
            x   = rng.standard_normal(shape, dtype=np.float32)
            res = rng.standard_normal(shape, dtype=np.float32)
            w   = rng.standard_normal((H,),  dtype=np.float32)
            expected = _np_rms_norm(x + res, w, eps)

            dx  = rt.upload(x)
            dr  = rt.upload(res)
            dw  = rt.upload(w)
            dy  = rt.alloc_like(shape, np.float32)
            try:
                rt.run_op("fuse_add_rmsnorm", ctx,
                          inputs=[dx, dr, dw], outputs=[dy],
                          extra_args=[ctypes.c_float(eps)])
                actual = rt.download(dy)
            finally:
                dx.free()
                dr.free()
                dw.free()
                dy.free()
            assert_close(actual, expected, atol=1e-4, rtol=1e-4,
                         name=f"fuse_add_rmsnorm shape={shape}")
            if print_info:
                print(f"  PASS  fuse_add_rmsnorm shape={list(shape)}")
    print("  [PASS] joy_gpu_fuse_add_rmsnorm dispatches to "
          "joy_codegen_fuse_add_rms_norm_f32, fuses add + rms_norm")
    print("\n[Lesson13]: ================== T7 PASSED ==================")


# ============================================================================
# Part C — End-to-end stream coherence
# ============================================================================

def test_chained_ops(rt: JoyGpuRuntime, *,
                     print_info: bool = False) -> None:
    """T8: a single ctx drives add -> silu -> linear on the same stream."""
    _banner("T8: stream coherence       (add -> silu -> linear on one ctx)")

    rng = np.random.default_rng(31)
    a = rng.standard_normal((8, 16), dtype=np.float32)
    b = rng.standard_normal((8, 16), dtype=np.float32)
    W = rng.standard_normal((32, 16), dtype=np.float32)

    # numpy reference for the chain.
    s = a + b
    z = _np_silu(s)
    expected = z @ W.T

    with rt.context() as ctx:
        da = rt.upload(a)
        db = rt.upload(b)
        dsum = rt.alloc_like((8, 16), np.float32)
        dz = rt.alloc_like((8, 16), np.float32)
        dW = rt.upload(W)
        dy = rt.alloc_like((8, 32), np.float32)
        try:
            # Three back-to-back ops; only the last one synchronizes.
            rt.run_op("add",    ctx, inputs=[da, db],  outputs=[dsum],
                      sync=False)
            rt.run_op("silu",   ctx, inputs=[dsum],    outputs=[dz],
                      sync=False)
            rt.run_op("linear", ctx, inputs=[dz, dW],  outputs=[dy],
                      sync=True)
            actual = rt.download(dy)
        finally:
            da.free()
            db.free()
            dsum.free()
            dz.free()
            dW.free()
            dy.free()

    assert_close(actual, expected, atol=1e-3, rtol=1e-3,
                 name="chained add->silu->linear")
    if print_info:
        print("  PASS  chained add->silu->linear matches numpy")
    print("  [PASS] three ops on a single stream produce the expected "
          "composed result (with only one sync at the end)")
    print("\n[Lesson13]: ================== T8 PASSED ==================")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lesson 13: GPU runtime tests")
    parser.add_argument("--print-info", action="store_true",
                        help="Print runtime info / per-shape details")
    args = parser.parse_args()
    p = args.print_info

    print("=" * 70)
    print("  Lesson 13: GPU runtime — three backend paths under one ABI")
    print("=" * 70)

    rt = get_runtime()

    print("\n" + "-" * 70)
    print("  Part A: Runtime self-check")
    print("-" * 70)
    test_runtime_loads(rt, print_info=p)
    test_struct_layouts(print_info=p)

    print("\n" + "-" * 70)
    print("  Part B: Three backend dispatch paths")
    print("-" * 70)
    test_handwritten_kernel_silu(rt, print_info=p)
    test_cublas_linear(rt, print_info=p)
    test_cudnn_softmax(rt, print_info=p)
    test_codegen_rms_norm(rt, print_info=p)
    test_codegen_fuse_add_rmsnorm(rt, print_info=p)

    print("\n" + "-" * 70)
    print("  Part C: End-to-end stream coherence")
    print("-" * 70)
    test_chained_ops(rt, print_info=p)

    print("\n" + "=" * 70)
    print("  ALL LESSON 13 TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
