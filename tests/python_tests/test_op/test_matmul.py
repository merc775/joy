"""Unit test: ``GpuMatMulOp`` (cuBLAS GEMM / strided-batched GEMM)."""
from __future__ import annotations

import numpy as np

from _runtime import F32, assert_close, banner, get_runtime


def _run_2d(rt, M: int, K: int, N: int, *, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((M, K), dtype=np.float32)
    b = rng.standard_normal((K, N), dtype=np.float32)
    expected = a @ b

    with rt.context() as ctx:
        da = rt.upload(a)
        db = rt.upload(b)
        dc = rt.alloc_like((M, N), np.float32)
        try:
            rt.run_op("matmul", ctx, inputs=[da, db], outputs=[dc])
            actual = rt.download(dc)
        finally:
            da.free(); db.free(); dc.free()

    assert_close(actual, expected, atol=1e-3, rtol=1e-3,
                 name=f"matmul 2D [{M},{K}]x[{K},{N}]")
    print(f"  PASS  matmul 2D  [{M},{K}] x [{K},{N}] -> [{M},{N}]")


def _run_batched(rt, B, M, K, N, *, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((B, M, K), dtype=np.float32)
    b = rng.standard_normal((B, K, N), dtype=np.float32)
    expected = np.matmul(a, b)

    with rt.context() as ctx:
        da = rt.upload(a)
        db = rt.upload(b)
        dc = rt.alloc_like((B, M, N), np.float32)
        try:
            rt.run_op("matmul", ctx, inputs=[da, db], outputs=[dc])
            actual = rt.download(dc)
        finally:
            da.free(); db.free(); dc.free()

    assert_close(actual, expected, atol=1e-3, rtol=1e-3,
                 name=f"matmul 3D [{B},{M},{K}]x[{B},{K},{N}]")
    print(f"  PASS  matmul 3D  [{B},{M},{K}] x [{B},{K},{N}] -> [{B},{M},{N}]")


def main() -> bool:
    banner("test_matmul: GpuMatMulOp (cuBLAS GEMM)")
    rt = get_runtime()

    _run_2d(rt, 2, 2, 2)             # parity with the cuBLAS sample
    _run_2d(rt, 32, 64, 16)
    _run_2d(rt, 1, 128, 1)
    _run_2d(rt, 65, 33, 17)          # awkward sizes
    _run_batched(rt, 4, 8, 16, 24)
    _run_batched(rt, 2, 1, 32, 32)

    print("test_matmul: ALL PASSED")
    return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
