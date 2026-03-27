"""Unit test: ``GpuLinearOp`` -- y = x @ W^T (PyTorch nn.Linear convention)."""
from __future__ import annotations

import numpy as np

from _runtime import assert_close, banner, get_runtime


def _run(rt, in_shape, w_shape, *, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(in_shape, dtype=np.float32)
    W = rng.standard_normal(w_shape, dtype=np.float32)  # [out, in]
    expected = x @ W.T

    out_shape = list(in_shape[:-1]) + [w_shape[0]]

    with rt.context() as ctx:
        dx = rt.upload(x)
        dw = rt.upload(W)
        dy = rt.alloc_like(out_shape, np.float32)
        try:
            rt.run_op("linear", ctx, inputs=[dx, dw], outputs=[dy])
            actual = rt.download(dy)
        finally:
            dx.free(); dw.free(); dy.free()

    assert_close(actual, expected, atol=1e-3, rtol=1e-3,
                 name=f"linear x={in_shape} W={w_shape}")
    print(f"  PASS  linear x={list(in_shape)} W={list(w_shape)} "
          f"-> {out_shape}")


def main() -> bool:
    banner("test_linear: GpuLinearOp (cuBLAS GEMM, transB)")
    rt = get_runtime()

    _run(rt, (8, 16), (32, 16))                # 2D
    _run(rt, (1, 16, 64), (32, 64))            # 3D
    _run(rt, (2, 4, 8, 32), (16, 32))          # 4D
    _run(rt, (5, 7, 13), (11, 13))             # awkward sizes

    print("test_linear: ALL PASSED")
    return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
