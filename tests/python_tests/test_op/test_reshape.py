"""Unit test: ``GpuReshapeOp`` -- buffer-to-buffer copy preserving values."""
from __future__ import annotations

import numpy as np

from _runtime import assert_close, banner, get_runtime


def _run(rt, in_shape, out_shape) -> None:
    rng = np.random.default_rng(11)
    x = rng.standard_normal(in_shape, dtype=np.float32)

    with rt.context() as ctx:
        dx = rt.upload(x)
        dy = rt.alloc_like(out_shape, np.float32)
        try:
            rt.run_op("reshape", ctx, inputs=[dx], outputs=[dy])
            actual = rt.download(dy)
        finally:
            dx.free(); dy.free()

    expected = x.reshape(out_shape)
    assert_close(actual, expected, atol=0, rtol=0,
                 name=f"reshape {in_shape}->{out_shape}")
    print(f"  PASS  reshape {list(in_shape)} -> {list(out_shape)}")


def main() -> bool:
    banner("test_reshape: GpuReshapeOp")
    rt = get_runtime()
    _run(rt, (4, 16),     (64,))
    _run(rt, (2, 3, 8),   (6, 8))
    _run(rt, (1, 64, 64), (4, 16, 64))
    print("test_reshape: ALL PASSED")
    return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
