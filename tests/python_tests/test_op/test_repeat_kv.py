"""Unit test: ``GpuRepeatKVOp`` -- expand KV heads for grouped-query attn."""
from __future__ import annotations

import ctypes

import numpy as np

from _runtime import assert_close, banner, get_runtime


def _np_repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """Reference: HF's repeat_kv.

    x: [B, H_kv, S, D] -> [B, H_kv * n_rep, S, D]
    Each KV head is replicated n_rep times consecutively.
    """
    if n_rep == 1:
        return x.copy()
    B, H_kv, S, D = x.shape
    return np.repeat(x, n_rep, axis=1)


def _run(rt, B, H_kv, S, D, n_rep) -> None:
    rng = np.random.default_rng(0xCAFE)
    x = rng.standard_normal((B, H_kv, S, D), dtype=np.float32)
    expected = _np_repeat_kv(x, n_rep)

    with rt.context() as ctx:
        dx = rt.upload(x)
        dy = rt.alloc_like(expected.shape, np.float32)
        try:
            rt.run_op("repeat_kv", ctx, inputs=[dx], outputs=[dy],
                      extra_args=[ctypes.c_int64(n_rep)])
            actual = rt.download(dy)
        finally:
            dx.free(); dy.free()

    assert_close(actual, expected, atol=0, rtol=0,
                 name=f"repeat_kv x=[{B},{H_kv},{S},{D}] n_rep={n_rep}")
    print(f"  PASS  repeat_kv [B={B},H_kv={H_kv},S={S},D={D}] n_rep={n_rep} "
          f"-> [{B},{H_kv*n_rep},{S},{D}]")


def main() -> bool:
    banner("test_repeat_kv: GpuRepeatKVOp")
    rt = get_runtime()
    _run(rt, 1, 2, 4, 8,    n_rep=1)
    _run(rt, 1, 2, 4, 8,    n_rep=4)
    _run(rt, 2, 8, 16, 64,  n_rep=2)
    _run(rt, 1, 1, 32, 128, n_rep=8)
    print("test_repeat_kv: ALL PASSED")
    return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
