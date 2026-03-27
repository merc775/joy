"""Unit test: ``GpuRMSNormOp``."""
from __future__ import annotations

import ctypes

import numpy as np

from _runtime import assert_close, banner, get_runtime


def _np_rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Reference RMSNorm along the last axis."""
    x64 = x.astype(np.float64)
    w64 = w.astype(np.float64)
    ms = np.mean(x64 * x64, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(ms + eps)
    return (x64 * inv_std * w64).astype(x.dtype)


def _run(rt, shape, eps: float = 1e-6) -> None:
    H = shape[-1]
    rng = np.random.default_rng(123)
    x = rng.standard_normal(shape, dtype=np.float32)
    w = rng.standard_normal((H,), dtype=np.float32)
    expected = _np_rms_norm(x, w, eps)

    with rt.context() as ctx:
        dx = rt.upload(x)
        dw = rt.upload(w)
        dy = rt.alloc_like(shape, np.float32)
        try:
            rt.run_op("rms_norm", ctx, inputs=[dx, dw], outputs=[dy],
                      extra_args=[ctypes.c_float(eps)])
            actual = rt.download(dy)
        finally:
            dx.free(); dw.free(); dy.free()

    assert_close(actual, expected, atol=1e-4, rtol=1e-4,
                 name=f"rms_norm shape={shape}")
    print(f"  PASS  rms_norm shape={list(shape)} eps={eps}")


def main() -> bool:
    banner("test_rms_norm: GpuRMSNormOp")
    rt = get_runtime()
    _run(rt, (4, 16))
    _run(rt, (1, 32, 128))
    _run(rt, (2, 4, 7, 64))
    _run(rt, (3, 1024))     # large H -- exercise reduction loop
    print("test_rms_norm: ALL PASSED")
    return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
