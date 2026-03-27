"""Unit test: ``GpuSoftmaxOp`` along last axis (cuDNN softmax)."""
from __future__ import annotations

import numpy as np
import ctypes

from _runtime import assert_close, banner, get_runtime


def _np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def _run(rt, shape, axis: int) -> None:
    rng = np.random.default_rng(2024)
    x = rng.standard_normal(shape, dtype=np.float32) * 3.0
    expected = _np_softmax(x, axis=axis if axis >= 0 else axis + len(shape))

    with rt.context() as ctx:
        dx = rt.upload(x)
        dy = rt.alloc_like(shape, np.float32)
        try:
            rt.run_op("softmax", ctx, inputs=[dx], outputs=[dy],
                      extra_args=[ctypes.c_int64(axis)])
            actual = rt.download(dy)
        finally:
            dx.free(); dy.free()

    assert_close(actual, expected, atol=1e-5, rtol=1e-4,
                 name=f"softmax shape={shape} axis={axis}")
    # Each row should sum to 1.
    sums = actual.sum(axis=axis if axis >= 0 else axis + len(shape))
    assert np.allclose(sums, 1.0, atol=1e-4), \
        f"softmax rows do not sum to 1 (max-deviation={np.max(np.abs(sums-1.0)):.3g})"
    print(f"  PASS  softmax shape={list(shape)} axis={axis}")


def main() -> bool:
    banner("test_softmax: GpuSoftmaxOp (cuDNN)")
    rt = get_runtime()

    _run(rt, (4, 16),       axis=-1)
    _run(rt, (2, 8, 64),    axis=-1)
    _run(rt, (1, 1, 32),    axis=2)
    _run(rt, (3, 5, 7, 11), axis=-1)

    print("test_softmax: ALL PASSED")
    return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
