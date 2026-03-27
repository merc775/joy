#!/usr/bin/env python3
"""Run every single-operator unit test for joy/lib/backend/gpu.

Each ``test_*.py`` module owns one operator and exposes a ``main()`` that
returns ``True`` on success.  We import them in-process so any failure
shows a real Python traceback (instead of an opaque subprocess exit code).

Usage:
    cd joy/tests/python_tests/test_op
    python3 run_all.py
    # or
    python3 -m run_all
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import traceback
from typing import List, Tuple

# Make the test modules importable when running this file directly.
HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _runtime import banner, get_runtime  # noqa: E402


TEST_MODULES: List[str] = [
    "test_add",
    "test_mul",
    "test_silu",
    "test_rms_norm",
    "test_embedding",
    "test_reshape",
    "test_transpose",
    "test_repeat_kv",
    "test_apply_rotary_emb",
    "test_softmax",         # uses cuDNN
    "test_matmul",          # uses cuBLAS
    "test_linear",          # uses cuBLAS
    "test_fuse_add_rmsnorm",
]


def _check_environment() -> None:
    rt = get_runtime()
    sig = rt.signature()
    cuda_v = rt.cuda_runtime_version()
    cudnn_v = rt.cudnn_version()
    print(f"  runtime signature : {sig}")
    print(f"  CUDA runtime ver  : {cuda_v // 1000}.{(cuda_v % 1000) // 10}")
    print(f"  cuDNN version     : {cudnn_v // 1000}.{(cudnn_v % 1000) // 100}."
          f"{cudnn_v % 100}")
    print(f"  shared library    : {rt.lib_path}")


def main(argv: List[str]) -> int:
    banner("Joy GPU backend - single-operator unit tests")
    try:
        _check_environment()
    except Exception as e:
        print(f"ERROR: failed to load runtime: {e}")
        return 1

    selected = TEST_MODULES
    if len(argv) > 1:
        # Allow running a subset, e.g. `python3 run_all.py matmul softmax`.
        wanted = set(argv[1:])
        selected = [m for m in TEST_MODULES
                    if any(w in m for w in wanted)]
        if not selected:
            print(f"No test modules match {sorted(wanted)}")
            return 1

    results: List[Tuple[str, bool, str, float]] = []
    for mod_name in selected:
        print()
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(mod_name)
            ok = bool(mod.main())
            err = ""
        except Exception:
            ok = False
            err = traceback.format_exc()
            print(err)
        dt = time.perf_counter() - t0
        results.append((mod_name, ok, err, dt))

    print()
    banner("Summary")
    width = max(len(m) for m, *_ in results)
    n_pass = 0
    for name, ok, _err, dt in results:
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag:>4}  {name:<{width}}  ({dt:6.2f} s)")
        if ok:
            n_pass += 1

    print()
    print(f"  {n_pass}/{len(results)} tests passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
