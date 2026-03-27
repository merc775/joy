"""Unit test: ``GpuFuseAddRMSNormOp`` -- (x + residual) -> RMSNorm fused."""
from __future__ import annotations

import ctypes

import numpy as np

from _runtime import assert_close, banner, get_runtime


def _np_rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x64 = x.astype(np.float64)
    w64 = w.astype(np.float64)
    ms = np.mean(x64 * x64, axis=-1, keepdims=True)
    return (x64 * (1.0 / np.sqrt(ms + eps)) * w64).astype(x.dtype)


def _run(rt, shape, eps: float = 1e-6) -> None:
    H = shape[-1]
    rng = np.random.default_rng(2025)
    x   = rng.standard_normal(shape, dtype=np.float32)
    res = rng.standard_normal(shape, dtype=np.float32)
    w   = rng.standard_normal((H,),  dtype=np.float32)
    summed = x + res
    expected = _np_rms_norm(summed, w, eps)

    with rt.context() as ctx:
        dx  = rt.upload(x)
        dres = rt.upload(res)
        dw  = rt.upload(w)
        dy  = rt.alloc_like(shape, np.float32)
        try:
            rt.run_op("fuse_add_rmsnorm", ctx,
                      inputs=[dx, dres, dw], outputs=[dy],
                      extra_args=[ctypes.c_float(eps)])
            actual = rt.download(dy)
        finally:
            dx.free(); dres.free(); dw.free(); dy.free()

    assert_close(actual, expected, atol=1e-4, rtol=1e-4,
                 name=f"fuse_add_rmsnorm shape={shape}")
    print(f"  PASS  fuse_add_rmsnorm shape={list(shape)} eps={eps}")


def main() -> bool:
    banner("test_fuse_add_rmsnorm: GpuFuseAddRMSNormOp")
    rt = get_runtime()
    _run(rt, (4, 16))
    _run(rt, (1, 32, 128))
    _run(rt, (2, 4, 7, 64))
    print("test_fuse_add_rmsnorm: ALL PASSED")
    return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
