"""Unit test: ``GpuMulOp`` -- element-wise multiply."""
from __future__ import annotations

import numpy as np

from _runtime import assert_close, banner, get_runtime


def _run(rt, shape) -> None:
    rng = np.random.default_rng(0x6D)  # arbitrary fixed seed
    a = rng.standard_normal(shape, dtype=np.float32)
    b = rng.standard_normal(shape, dtype=np.float32)
    expected = a * b

    with rt.context() as ctx:
        da = rt.upload(a)
        db = rt.upload(b)
        dc = rt.alloc_like(shape, np.float32)
        try:
            rt.run_op("mul", ctx, inputs=[da, db], outputs=[dc])
            actual = rt.download(dc)
        finally:
            da.free(); db.free(); dc.free()

    assert_close(actual, expected, atol=1e-6, rtol=1e-6,
                 name=f"mul shape={shape}")
    print(f"  PASS  mul shape={list(shape)}")


def main() -> bool:
    banner("test_mul: GpuMulOp")
    rt = get_runtime()
    _run(rt, (4, 8))
    _run(rt, (3, 5, 17))
    _run(rt, (2, 1, 64, 64))
    print("test_mul: ALL PASSED")
    return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
