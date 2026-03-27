"""Unit test: ``GpuApplyRotaryEmbOp`` -- HF-style rotate_half RoPE."""
from __future__ import annotations

import numpy as np

from _runtime import assert_close, banner, get_runtime


def _np_rotary(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Reference matching HF's apply_rotary_pos_emb / rotate_half.

    x  : [B, H, S, D]
    cos: [S, D]
    sin: [S, D]
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    cos_b = cos[None, None, :, :]   # broadcast across (B, H)
    sin_b = sin[None, None, :, :]
    return x * cos_b + rotated * sin_b


def _run(rt, B, H, S, D) -> None:
    rng = np.random.default_rng(0xC0DE)
    x   = rng.standard_normal((B, H, S, D), dtype=np.float32)
    cos = rng.standard_normal((S, D),       dtype=np.float32)
    sin = rng.standard_normal((S, D),       dtype=np.float32)
    expected = _np_rotary(x, cos, sin)

    with rt.context() as ctx:
        dx = rt.upload(x); dc = rt.upload(cos); ds = rt.upload(sin)
        dy = rt.alloc_like((B, H, S, D), np.float32)
        try:
            rt.run_op("apply_rotary_emb", ctx, inputs=[dx, dc, ds],
                      outputs=[dy])
            actual = rt.download(dy)
        finally:
            dx.free(); dc.free(); ds.free(); dy.free()

    assert_close(actual, expected, atol=1e-5, rtol=1e-5,
                 name=f"apply_rotary_emb [{B},{H},{S},{D}]")
    print(f"  PASS  apply_rotary_emb [B={B},H={H},S={S},D={D}]")


def main() -> bool:
    banner("test_apply_rotary_emb: GpuApplyRotaryEmbOp")
    rt = get_runtime()
    _run(rt, 1, 2, 4, 8)
    _run(rt, 2, 4, 16, 64)
    _run(rt, 1, 1, 32, 128)
    print("test_apply_rotary_emb: ALL PASSED")
    return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
