"""ctypes wrapper around ``libjoy_gpu_runtime.so``.

This module hides the boring details of:

  * Locating and loading the shared library produced by the JOY build.
  * Declaring the C ABI of every ``joy_gpu_*`` entry point and of the
    test helpers (``joy_test_device_alloc``, ``joy_test_create_context`` ...).
  * Marshalling NumPy arrays through GPU memory and through ``MemrefDesc``
    structures so individual tests can stay short and readable.

The public surface used by the per-operator tests is small:

    rt = JoyGpuRuntime()           # find + load the .so

    with rt.context() as ctx:      # cuBLAS/cuDNN handles bound to a stream
        x_dev = rt.upload(host_x)  # numpy -> device buffer
        y_dev = rt.alloc_like(out_shape, np.float32)

        rt.run_op(
            "matmul",
            ctx,
            inputs=[x_dev, w_dev],
            outputs=[y_dev],
        )

        result = rt.download(y_dev, out_shape, np.float32)
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# Element-type ids must match MemrefDesc::element_type in gpu_runner.h
F16 = 0
F32 = 1
F64 = 2
I32 = 3
I64 = 4

_DTYPE_TO_ID = {
    np.dtype(np.float16): F16,
    np.dtype(np.float32): F32,
    np.dtype(np.float64): F64,
    np.dtype(np.int32):   I32,
    np.dtype(np.int64):   I64,
}

_ID_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ID.items()}


def dtype_id(dt: np.dtype) -> int:
    dt = np.dtype(dt)
    if dt not in _DTYPE_TO_ID:
        raise ValueError(f"unsupported dtype: {dt}")
    return _DTYPE_TO_ID[dt]


# ---------------------------------------------------------------------------
# C structures (must match gpu_runner.h)
# ---------------------------------------------------------------------------
class MemrefDesc(ctypes.Structure):
    _fields_ = [
        ("rank",         ctypes.c_int64),
        ("data",         ctypes.c_void_p),
        ("shape",        ctypes.POINTER(ctypes.c_int64)),
        ("element_type", ctypes.c_int32),
    ]


class GpuContext(ctypes.Structure):
    _fields_ = [
        ("stream", ctypes.c_void_p),
        ("cublas", ctypes.c_void_p),
        ("cudnn",  ctypes.c_void_p),
    ]


# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------
def _candidate_lib_paths() -> List[str]:
    """Heuristically locate libjoy_gpu_runtime.so.

    Order of preference:
      1. $JOY_GPU_RUNTIME explicitly set
      2. <project_root>/build/lib/libjoy_gpu_runtime.so
      3. ./build/lib/libjoy_gpu_runtime.so
      4. Anywhere on LD_LIBRARY_PATH / system path (handled by ctypes itself)
    """
    paths: List[str] = []
    env = os.environ.get("JOY_GPU_RUNTIME")
    if env:
        paths.append(env)

    here = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    paths.append(os.path.join(project_root, "build", "lib",
                              "libjoy_gpu_runtime.so"))
    paths.append(os.path.join(os.getcwd(), "build", "lib",
                              "libjoy_gpu_runtime.so"))
    return paths


@dataclass
class DeviceBuffer:
    """An owned GPU memory allocation."""
    ptr: int
    nbytes: int
    shape: Tuple[int, ...]
    dtype: np.dtype
    _runtime: "JoyGpuRuntime"

    def free(self) -> None:
        if self.ptr:
            self._runtime._lib.joy_test_device_free(ctypes.c_void_p(self.ptr))
            self.ptr = 0


class JoyGpuRuntime:
    """Loaded handle to libjoy_gpu_runtime.so."""

    def __init__(self, lib_path: Optional[str] = None) -> None:
        if lib_path is None:
            for cand in _candidate_lib_paths():
                if os.path.exists(cand):
                    lib_path = cand
                    break
        if lib_path is None:
            raise FileNotFoundError(
                "libjoy_gpu_runtime.so not found.  Build the JOY project first "
                "(scripts/build.sh) or set JOY_GPU_RUNTIME=/path/to/libjoy_gpu_runtime.so"
            )
        self.lib_path = lib_path
        self._lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        self._declare()

    # ----- declarations -----
    def _declare(self) -> None:
        L = self._lib

        # Test helpers.
        L.joy_test_device_alloc.argtypes = [ctypes.c_size_t]
        L.joy_test_device_alloc.restype = ctypes.c_void_p

        L.joy_test_device_free.argtypes = [ctypes.c_void_p]
        L.joy_test_device_free.restype = None

        L.joy_test_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_size_t]
        L.joy_test_memcpy_h2d.restype = ctypes.c_int
        L.joy_test_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_size_t]
        L.joy_test_memcpy_d2h.restype = ctypes.c_int

        L.joy_test_memset_zero.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        L.joy_test_memset_zero.restype = ctypes.c_int

        L.joy_test_device_synchronize.argtypes = []
        L.joy_test_device_synchronize.restype = ctypes.c_int

        L.joy_test_create_context.argtypes = []
        L.joy_test_create_context.restype = ctypes.POINTER(GpuContext)

        L.joy_test_destroy_context.argtypes = [ctypes.POINTER(GpuContext)]
        L.joy_test_destroy_context.restype = None

        L.joy_test_stream_synchronize.argtypes = [ctypes.POINTER(GpuContext)]
        L.joy_test_stream_synchronize.restype = ctypes.c_int

        L.joy_test_runtime_signature.argtypes = []
        L.joy_test_runtime_signature.restype = ctypes.c_char_p
        L.joy_test_cuda_runtime_version.argtypes = []
        L.joy_test_cuda_runtime_version.restype = ctypes.c_int
        L.joy_test_cudnn_version.argtypes = []
        L.joy_test_cudnn_version.restype = ctypes.c_int

        # Operator entry points.
        ctx_t = ctypes.POINTER(GpuContext)
        operands_t = ctypes.POINTER(MemrefDesc)

        def _decl(fn, extra_args=()):
            fn.argtypes = [ctx_t, ctypes.c_int64, ctypes.c_int64, operands_t,
                           *extra_args]
            fn.restype = None

        _decl(L.joy_gpu_embedding)
        _decl(L.joy_gpu_rms_norm,           [ctypes.c_float])
        _decl(L.joy_gpu_linear)
        _decl(L.joy_gpu_matmul)
        _decl(L.joy_gpu_softmax,            [ctypes.c_int64])
        _decl(L.joy_gpu_silu)
        _decl(L.joy_gpu_add)
        _decl(L.joy_gpu_mul)
        _decl(L.joy_gpu_reshape)
        _decl(L.joy_gpu_transpose,
              [ctypes.POINTER(ctypes.c_int64), ctypes.c_int64])
        _decl(L.joy_gpu_apply_rotary_emb)
        _decl(L.joy_gpu_repeat_kv,          [ctypes.c_int64])
        _decl(L.joy_gpu_fuse_add_rmsnorm,   [ctypes.c_float])

    # ----- info -----
    def signature(self) -> str:
        return self._lib.joy_test_runtime_signature().decode("utf-8")

    def cuda_runtime_version(self) -> int:
        return int(self._lib.joy_test_cuda_runtime_version())

    def cudnn_version(self) -> int:
        return int(self._lib.joy_test_cudnn_version())

    # ----- context -----
    @contextlib.contextmanager
    def context(self):
        ctx = self._lib.joy_test_create_context()
        if not ctx:
            raise RuntimeError("joy_test_create_context failed")
        try:
            yield ctx
            # Make sure all pending work on the test stream has finished
            # before the context (and its handles) are torn down.
            rc = self._lib.joy_test_stream_synchronize(ctx)
            if rc != 0:
                raise RuntimeError(f"stream sync failed (rc={rc})")
        finally:
            self._lib.joy_test_destroy_context(ctx)

    # ----- buffers -----
    def alloc(self, nbytes: int, shape: Sequence[int],
              dtype: np.dtype) -> DeviceBuffer:
        if nbytes <= 0:
            raise ValueError("alloc nbytes must be > 0")
        ptr = self._lib.joy_test_device_alloc(ctypes.c_size_t(nbytes))
        if not ptr:
            raise MemoryError(f"joy_test_device_alloc({nbytes}) failed")
        return DeviceBuffer(ptr=int(ptr), nbytes=nbytes,
                            shape=tuple(shape), dtype=np.dtype(dtype),
                            _runtime=self)

    def alloc_like(self, shape: Sequence[int],
                   dtype: np.dtype) -> DeviceBuffer:
        dtype = np.dtype(dtype)
        n = 1
        for s in shape:
            n *= int(s)
        return self.alloc(n * dtype.itemsize, shape, dtype)

    def upload(self, host: np.ndarray) -> DeviceBuffer:
        host = np.ascontiguousarray(host)
        buf = self.alloc(host.nbytes, host.shape, host.dtype)
        rc = self._lib.joy_test_memcpy_h2d(
            ctypes.c_void_p(buf.ptr),
            host.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(host.nbytes))
        if rc != 0:
            buf.free()
            raise RuntimeError(f"H2D memcpy failed (rc={rc})")
        return buf

    def download(self, buf: DeviceBuffer,
                 shape: Optional[Sequence[int]] = None,
                 dtype: Optional[np.dtype] = None) -> np.ndarray:
        shape = tuple(shape) if shape is not None else buf.shape
        dtype = np.dtype(dtype) if dtype is not None else buf.dtype
        host = np.empty(shape, dtype=dtype)
        rc = self._lib.joy_test_memcpy_d2h(
            host.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(buf.ptr),
            ctypes.c_size_t(host.nbytes))
        if rc != 0:
            raise RuntimeError(f"D2H memcpy failed (rc={rc})")
        return host

    def memset_zero(self, buf: DeviceBuffer) -> None:
        rc = self._lib.joy_test_memset_zero(ctypes.c_void_p(buf.ptr),
                                            ctypes.c_size_t(buf.nbytes))
        if rc != 0:
            raise RuntimeError(f"memset failed (rc={rc})")

    # ----- memref descriptor builder -----
    def make_operands(self,
                      tensors: Sequence[DeviceBuffer],
                      ) -> Tuple[ctypes.Array, List[ctypes.Array]]:
        """Build a contiguous MemrefDesc array and pin shape arrays.

        Returns ``(operands, shape_arrays)``.  The caller MUST keep
        ``shape_arrays`` alive while the C call is in flight, otherwise
        the ``shape*`` pointers inside ``operands`` are dangling.
        """
        n = len(tensors)
        operands = (MemrefDesc * n)()
        shape_arrays: List[ctypes.Array] = []
        for i, t in enumerate(tensors):
            shp = (ctypes.c_int64 * len(t.shape))(*t.shape)
            shape_arrays.append(shp)
            operands[i].rank = len(t.shape)
            operands[i].data = ctypes.c_void_p(t.ptr)
            operands[i].shape = ctypes.cast(shp,
                                            ctypes.POINTER(ctypes.c_int64))
            operands[i].element_type = dtype_id(t.dtype)
        return operands, shape_arrays

    # ----- generic op runner -----
    def run_op(self,
               name: str,
               ctx,
               inputs: Sequence[DeviceBuffer],
               outputs: Sequence[DeviceBuffer],
               extra_args: Iterable = (),
               sync: bool = True,
               ) -> None:
        """Dispatch one ``joy_gpu_<name>`` extern "C" entry point.

        When chaining many ops in a forward pass, set ``sync=False`` on
        intermediate calls and synchronize once at the end via
        ``joy_test_stream_synchronize`` -- this avoids the per-op sync
        overhead while still preserving correctness.
        """
        all_bufs = list(inputs) + list(outputs)
        operands, _keep = self.make_operands(all_bufs)
        fn = getattr(self._lib, f"joy_gpu_{name}")
        fn(ctx, ctypes.c_int64(len(inputs)), ctypes.c_int64(len(outputs)),
           operands, *extra_args)
        if sync:
            rc = self._lib.joy_test_stream_synchronize(ctx)
            if rc != 0:
                raise RuntimeError(
                    f"stream sync after joy_gpu_{name} failed (rc={rc})")

    def stream_synchronize(self, ctx) -> None:
        rc = self._lib.joy_test_stream_synchronize(ctx)
        if rc != 0:
            raise RuntimeError(f"stream sync failed (rc={rc})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def assert_close(actual: np.ndarray, expected: np.ndarray, *,
                 atol: float = 1e-5, rtol: float = 1e-5,
                 name: str = "tensor") -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"[{name}] shape mismatch: actual={actual.shape} "
            f"expected={expected.shape}")
    if actual.dtype != expected.dtype:
        actual = actual.astype(expected.dtype)
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    base = np.abs(expected.astype(np.float64))
    tol = atol + rtol * base
    bad = diff > tol
    if np.any(bad):
        n_bad = int(bad.sum())
        max_diff = float(diff.max())
        max_idx = np.unravel_index(int(diff.argmax()), diff.shape)
        raise AssertionError(
            f"[{name}] mismatch: {n_bad}/{actual.size} elements above tolerance.\n"
            f"  max |diff|={max_diff:.6g} at {max_idx}\n"
            f"  actual[{max_idx}]={actual[max_idx]}, "
            f"expected[{max_idx}]={expected[max_idx]}, "
            f"atol={atol}, rtol={rtol}")


def banner(title: str) -> None:
    bar = "=" * 70
    print(bar)
    print(f"  {title}")
    print(bar)


# Singleton accessor used by test modules.
_RUNTIME: Optional[JoyGpuRuntime] = None


def get_runtime() -> JoyGpuRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = JoyGpuRuntime()
    return _RUNTIME
