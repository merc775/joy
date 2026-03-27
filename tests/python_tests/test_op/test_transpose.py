"""Unit test: ``GpuTransposeOp`` for arbitrary permutations."""
from __future__ import annotations

import ctypes

import numpy as np

from _runtime import assert_close, banner, get_runtime


def _run(rt, shape, perm) -> None:
    rng = np.random.default_rng(0xBEEF)
    x = rng.standard_normal(shape, dtype=np.float32)
    expected = np.transpose(x, perm)

    perm_arr = (ctypes.c_int64 * len(perm))(*perm)

    with rt.context() as ctx:
        dx = rt.upload(x)
        dy = rt.alloc_like(expected.shape, np.float32)
        try:
            rt.run_op("transpose", ctx, inputs=[dx], outputs=[dy],
                      extra_args=[perm_arr, ctypes.c_int64(len(perm))])
            actual = rt.download(dy)
        finally:
            dx.free(); dy.free()

    assert_close(actual, expected, atol=0, rtol=0,
                 name=f"transpose shape={shape} perm={perm}")
    print(f"  PASS  transpose {list(shape)} perm={list(perm)} "
          f"-> {list(expected.shape)}")


def main() -> bool:
    banner("test_transpose: GpuTransposeOp")
    rt = get_runtime()
    _run(rt, (3, 5),                 (1, 0))
    _run(rt, (2, 3, 4),               (2, 0, 1))
    _run(rt, (1, 8, 16, 32),          (0, 2, 1, 3))     # B,H,S,D -> B,S,H,D
    _run(rt, (2, 4, 8, 16),           (3, 1, 2, 0))
    _run(rt, (2, 3),                  (0, 1))           # identity
    print("test_transpose: ALL PASSED")
    return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
