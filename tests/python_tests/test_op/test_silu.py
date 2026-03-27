"""Unit test: ``GpuSiLUOp`` -- y = x * sigmoid(x)."""
from __future__ import annotations

import numpy as np

from _runtime import assert_close, banner, get_runtime


def _np_silu(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float64) / (1.0 + np.exp(-x.astype(np.float64)))


def _run(rt, shape) -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(shape, dtype=np.float32) * 4.0
    expected = _np_silu(x).astype(np.float32)

    with rt.context() as ctx:
        dx = rt.upload(x)
        dy = rt.alloc_like(shape, np.float32)
        try:
            rt.run_op("silu", ctx, inputs=[dx], outputs=[dy])
            actual = rt.download(dy)
        finally:
            dx.free(); dy.free()

    assert_close(actual, expected, atol=1e-5, rtol=1e-5,
                 name=f"silu shape={shape}")
    print(f"  PASS  silu shape={list(shape)}")


def main() -> bool:
    banner("test_silu: GpuSiLUOp (custom CUDA kernel)")
    rt = get_runtime()
    _run(rt, (4, 16))
    _run(rt, (1, 64, 128))
    _run(rt, (2, 3, 7, 17))
    print("test_silu: ALL PASSED")
    return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
