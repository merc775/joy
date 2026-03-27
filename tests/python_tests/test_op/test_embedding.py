"""Unit test: ``GpuEmbeddingOp`` -- table lookup."""
from __future__ import annotations

import numpy as np

from _runtime import assert_close, banner, get_runtime


def _run(rt, ids_shape, vocab: int, hidden: int,
         ids_dtype: np.dtype = np.int32) -> None:
    rng = np.random.default_rng(42)
    ids = rng.integers(0, vocab, size=ids_shape, dtype=ids_dtype)
    table = rng.standard_normal((vocab, hidden), dtype=np.float32)
    expected = table[ids]   # numpy gather

    out_shape = list(ids_shape) + [hidden]

    with rt.context() as ctx:
        d_ids = rt.upload(ids)
        d_tbl = rt.upload(table)
        d_out = rt.alloc_like(out_shape, np.float32)
        try:
            rt.run_op("embedding", ctx, inputs=[d_ids, d_tbl], outputs=[d_out])
            actual = rt.download(d_out)
        finally:
            d_ids.free(); d_tbl.free(); d_out.free()

    assert_close(actual, expected, atol=0.0, rtol=0.0,
                 name=f"embedding ids={ids_shape} V={vocab} H={hidden} dtype={ids_dtype}")
    print(f"  PASS  embedding ids_shape={list(ids_shape)} vocab={vocab} "
          f"hidden={hidden} idx_dtype={np.dtype(ids_dtype).name}")


def main() -> bool:
    banner("test_embedding: GpuEmbeddingOp")
    rt = get_runtime()
    _run(rt, (4,),     vocab=100, hidden=8,   ids_dtype=np.int32)
    _run(rt, (2, 8),   vocab=257, hidden=64,  ids_dtype=np.int32)
    _run(rt, (3, 5),   vocab=1000, hidden=32, ids_dtype=np.int64)
    print("test_embedding: ALL PASSED")
    return True


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
