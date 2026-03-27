#!/usr/bin/env python3
"""Lesson 9: AI Compiler Mid-end — Dialect Lowering in depth.

Validates the two Lowering passes:

  Stage 1: joy/lib/optimizer/LowerJoyToJoylPass.cpp
           tensor → memref via Dialect Conversion + applyFullConversion
  Stage 2: joy/lib/optimizer/LowerJoylToJoyhPass.cpp
           joyl.* → joyh.custom_call via GreedyDriver + MatchAnyOpTypeTag

Tests:
  Test 1: single joy.linear lowers to joyl.linear with output memref alloc
  Test 2: function signature switches from tensor to memref
  Test 3: epsilon / axis / permutation / n_rep attributes survive lowering
  Test 4: multi-result op (joy.fuse_add_rmsnorm) → 2 output buffers
  Test 5: full MLP block (linear+silu+mul+linear+linear) lowers cleanly
  Test 6: completeness — no joy.* op survives --lower-joy-to-joyl
  Test 7: joyl → joyh per-op naming convention call_target_name = "joy_gpu_X"
  Test 8: num_inputs attribute correctness on joyh.custom_call
  Test 9: joyl.rms_norm / joyl.fuse_add_rmsnorm are skipped by lower-joyl-to-joyh
  Test 10: end-to-end pipeline: optimize → lower-joy-to-joyl
           → codegen-rms-norm → lower-joyl-to-joyh

Usage:
    python3 joy/tests/python_tests/test_lesson9.py
    python3 joy/tests/python_tests/test_lesson9.py --print-ir-all
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

cur_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(cur_path, "../.."))
sys.path.insert(0, os.path.join(project_root, "python"))

from joy.builder import Graph
from joy.builder import ops


JOY_OPT = os.path.join(project_root, "build", "bin", "joy-opt")


def _require_joy_opt():
    if not os.path.exists(JOY_OPT):
        print(f"  SKIP: joy-opt not found at {JOY_OPT}. "
              f"Run scripts/build.sh first.")
        return False
    return True


def _run_joy_opt(input_ir, passes, timeout=30):
    """Run joy-opt with the given passes; return (stdout, stderr, rc)."""
    if not os.path.exists(JOY_OPT):
        return None, "joy-opt not built", -1
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mlir")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(input_ir)
        cmd = [JOY_OPT] + passes + [tmp_path]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout)
        return result.stdout, result.stderr, result.returncode
    finally:
        os.unlink(tmp_path)


def _count_op(ir, op_name):
    return ir.count(f'"{op_name}"(')


def _find_ops(ir, dialect_prefix):
    """Return mnemonics of all ops in the given dialect (e.g. 'joy.')."""
    return re.findall(r'"(' + re.escape(dialect_prefix) + r'[a-zA-Z_]+)"\(', ir)


# ============================================================================
# Test 1: single joy.linear lowers to joyl.linear with alloc
# ============================================================================
def test_single_linear_lowering(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 1: joy.linear → joyl.linear (with memref.alloc)")
    print("=" * 60)

    g = Graph(name="linear_lower")
    x = g.input([1, 64, 1024], "f16")
    w = g.input([512, 1024], "f16")
    y = ops.linear(x, w)
    g.set_outputs([y])

    joy_ir = g.get_ir()
    joyl_ir, stderr, rc = _run_joy_opt(joy_ir, ["--lower-joy-to-joyl"])
    if print_ir:
        print("--- Joy IR ---"); print(joy_ir)
        print("--- Joyl IR ---"); print(joyl_ir)

    checks = [
        (rc == 0, f"--lower-joy-to-joyl returned 0 (stderr={stderr.strip()[:120]})"),
        (_count_op(joyl_ir, "joyl.linear") == 1, "joyl.linear created"),
        (_count_op(joyl_ir, "joy.linear") == 0, "joy.linear eliminated"),
        ("memref.alloc" in joyl_ir, "memref.alloc inserted for output"),
        ("memref<1x64x512xf16>" in joyl_ir, "output memref<1x64x512xf16> present"),
        ("memref<1x64x1024xf16>" in joyl_ir, "input memref<1x64x1024xf16> present"),
        ("memref<512x1024xf16>" in joyl_ir, "weight memref<512x1024xf16> present"),
        ("tensor<" not in joyl_ir, "no tensor types remain in Joyl IR"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson9] Test 1 failed"
    print(f"\n[Lesson9]: ================== Test 1 PASSED ==================")


# ============================================================================
# Test 2: function signature is rewritten from tensor to memref
# ============================================================================
def test_func_signature_conversion(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 2: func.func signature tensor→memref")
    print("=" * 60)

    g = Graph(name="sig_test")
    x = g.input([1, 64, 1024], "f16")
    w = g.input([512, 1024], "f16")
    y = ops.linear(x, w)
    g.set_outputs([y])
    joy_ir = g.get_ir()

    joyl_ir, _, rc = _run_joy_opt(joy_ir, ["--lower-joy-to-joyl"])
    if print_ir:
        print(joyl_ir)

    sig_line = ""
    return_line = ""
    for line in joyl_ir.split("\n"):
        if "func.func @sig_test" in line:
            sig_line = line
        if "return " in line and "memref" in line:
            return_line = line

    checks = [
        (rc == 0, "rc == 0"),
        ("func.func @sig_test(" in joyl_ir, "function name preserved"),
        ("memref<1x64x1024xf16>" in sig_line, "arg0 type is memref<1x64x1024xf16>"),
        ("memref<512x1024xf16>" in sig_line, "arg1 type is memref<512x1024xf16>"),
        ("memref<1x64x512xf16>" in sig_line, "return type is memref<1x64x512xf16>"),
        ("tensor<" not in sig_line, "no tensor in signature line"),
        ("memref" in return_line and "tensor" not in return_line,
         "func.return uses memref"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson9] Test 2 failed"
    print(f"\n[Lesson9]: ================== Test 2 PASSED ==================")


# ============================================================================
# Test 3: multiple attributes survive lowering (epsilon/axis/permutation/n_rep)
# ============================================================================
def test_attribute_preservation(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 3: attributes preserved (epsilon, axis, permutation, n_rep)")
    print("=" * 60)

    H, NH, HD = 1024, 16, 128
    B, S = 1, 64
    g = Graph(name="attr_test")
    x = g.input([B, S, H], "f16")
    scale = g.input([H], "f16")
    w = g.input([NH * HD, H], "f16")

    n = ops.rms_norm(x, scale, epsilon=1e-6)
    q = ops.linear(n, w)
    q = ops.reshape(q, [B, S, NH, HD])
    q = ops.transpose(q, [0, 2, 1, 3])
    q_repeat = ops.repeat_kv(q, n_rep=4)
    k_t = ops.transpose(q_repeat, [0, 1, 3, 2])
    attn = ops.matmul(q_repeat, k_t)
    sm = ops.softmax(attn, axis=-1)
    g.set_outputs([sm])

    joy_ir = g.get_ir()
    joyl_ir, _, rc = _run_joy_opt(joy_ir, ["--lower-joy-to-joyl"])
    if print_ir:
        print(joyl_ir)

    checks = [
        (rc == 0, "rc == 0"),
        ("epsilon" in joyl_ir and "joyl.rms_norm" in joyl_ir,
         "epsilon preserved on joyl.rms_norm"),
        ("axis" in joyl_ir and "joyl.softmax" in joyl_ir,
         "axis preserved on joyl.softmax"),
        ("permutation" in joyl_ir and "joyl.transpose" in joyl_ir,
         "permutation preserved on joyl.transpose"),
        ("n_rep = 4" in joyl_ir and "joyl.repeat_kv" in joyl_ir,
         "n_rep = 4 preserved on joyl.repeat_kv"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson9] Test 3 failed"
    print(f"\n[Lesson9]: ================== Test 3 PASSED ==================")


# ============================================================================
# Test 4: multi-result op lowering (joy.fuse_add_rmsnorm → 2 buffers)
# ============================================================================
def test_multi_result_lowering(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 4: multi-result op (fuse_add_rmsnorm) → 2 output buffers")
    print("=" * 60)

    H = 1024
    g = Graph(name="multi_res")
    hidden = g.input([1, 64, H], "f16")
    residual = g.input([1, 64, H], "f16")
    ln_w = g.input([H], "f16")
    proj_w = g.input([512, H], "f16")
    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=1e-6)
    out = ops.linear(normed, proj_w)
    g.set_outputs([out])
    joy_ir = g.get_ir()

    # Apply fusion first, then lower
    fused_ir, _, rc1 = _run_joy_opt(joy_ir, ["--joy-op-fusion"])
    assert rc1 == 0, "fusion failed"
    joyl_ir, _, rc2 = _run_joy_opt(fused_ir, ["--lower-joy-to-joyl"])
    if print_ir:
        print(joyl_ir)

    alloc_count = joyl_ir.count("memref.alloc()")

    # Inspect the joyl.fuse_add_rmsnorm: it should have 5 operands
    # (3 inputs + 2 outputs) — the textual form must contain 5 memref args.
    fuse_match = re.search(
        r'"joyl\.fuse_add_rmsnorm"\(([^)]+)\)', joyl_ir)
    fuse_arg_count = (
        len(fuse_match.group(1).split(",")) if fuse_match else 0)

    checks = [
        (rc2 == 0, "lower-joy-to-joyl returned 0"),
        (_count_op(joyl_ir, "joyl.fuse_add_rmsnorm") == 1,
         "joyl.fuse_add_rmsnorm created"),
        (_count_op(joyl_ir, "joyl.linear") == 1, "joyl.linear created"),
        (alloc_count >= 3,
         f"at least 3 memref.alloc (2 for fuse + 1 for linear) got {alloc_count}"),
        (fuse_arg_count == 5,
         f"joyl.fuse_add_rmsnorm has 5 operands (3 in + 2 out), got {fuse_arg_count}"),
        ("joy.add" not in joyl_ir and "joy.rms_norm" not in joyl_ir
         and "joy.fuse_add_rmsnorm" not in joyl_ir,
         "all joy.* eliminated"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson9] Test 4 failed"
    print(f"\n[Lesson9]: ================== Test 4 PASSED ==================")


# ============================================================================
# Test 5: full MLP block lowers cleanly
# ============================================================================
def test_multi_op_graph_lowering(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 5: MLP block (linear+silu+mul+linear+linear) lowering")
    print("=" * 60)

    H, INTER = 1024, 3072
    g = Graph(name="mlp_test")
    x = g.input([1, 64, H], "f16")
    gate_w = g.input([INTER, H], "f16")
    up_w = g.input([INTER, H], "f16")
    down_w = g.input([H, INTER], "f16")

    gate = ops.linear(x, gate_w)
    gate = ops.silu(gate)
    up = ops.linear(x, up_w)
    gate_up = ops.mul(gate, up)
    out = ops.linear(gate_up, down_w)
    g.set_outputs([out])

    joy_ir = g.get_ir()
    joyl_ir, _, rc = _run_joy_opt(joy_ir, ["--lower-joy-to-joyl"])
    if print_ir:
        print(joyl_ir)

    expected = {"joyl.linear": 3, "joyl.silu": 1, "joyl.mul": 1}
    alloc_count = joyl_ir.count("memref.alloc()")

    checks = [(rc == 0, "rc == 0")]
    for name, n in expected.items():
        actual = _count_op(joyl_ir, name)
        checks.append((actual == n, f"{name}: expected {n}, got {actual}"))
    checks.extend([
        (alloc_count == 5, f"5 memref.alloc for 5 op results (got {alloc_count})"),
        ("joy.linear" not in joyl_ir and "joy.silu" not in joyl_ir
         and "joy.mul" not in joyl_ir, "all joy.* ops eliminated"),
        ("tensor<" not in joyl_ir, "no tensor types remain"),
    ])

    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson9] Test 5 failed"
    print(f"\n[Lesson9]: ================== Test 5 PASSED ==================")


# ============================================================================
# Test 6: completeness — every joy.* op eliminated
# ============================================================================
def test_lowering_completeness(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 6: completeness — addIllegalDialect<joy> enforces all joy.* go")
    print("=" * 60)

    H, NH, HD = 1024, 16, 128
    B, S = 1, 64

    g = Graph(name="completeness_test")
    x = g.input([B, S], "i64")
    cos = g.input([B, S, HD], "f16")
    sin = g.input([B, S, HD], "f16")
    scale = g.input([H], "f16")
    q_w = g.input([H, H], "f16")
    emb_w = g.input([32000, H], "f16")

    emb = ops.embedding(x, emb_w)
    normed = ops.rms_norm(emb, scale, epsilon=1e-6)
    q = ops.linear(normed, q_w)
    q = ops.reshape(q, [B, S, NH, HD])
    q = ops.transpose(q, [0, 2, 1, 3])
    q = ops.apply_rotary_emb(q, cos, sin)
    k = ops.repeat_kv(q, n_rep=2)
    k_t = ops.transpose(k, [0, 1, 3, 2])
    attn = ops.matmul(q, k_t)
    attn = ops.softmax(attn, axis=-1)
    out = ops.reshape(attn, [B, S, NH * HD])
    g.set_outputs([out])

    joy_ir = g.get_ir()
    joyl_ir, stderr, rc = _run_joy_opt(joy_ir, ["--lower-joy-to-joyl"])
    if print_ir:
        print(joyl_ir)

    joy_ops_remaining = _find_ops(joyl_ir, "joy.")
    expected_joyl_ops = [
        "joyl.embedding", "joyl.rms_norm", "joyl.linear", "joyl.reshape",
        "joyl.transpose", "joyl.apply_rotary_emb", "joyl.repeat_kv",
        "joyl.matmul", "joyl.softmax",
    ]

    checks = [(rc == 0, f"rc == 0 (stderr={stderr.strip()[:120]})")]
    for name in expected_joyl_ops:
        ok = _count_op(joyl_ir, name) > 0
        checks.append((ok, f"{name} present"))
    checks.append((len(joy_ops_remaining) == 0,
                   f"no joy.* op remains (saw {joy_ops_remaining[:5]})"))

    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson9] Test 6 failed"
    print(f"\n[Lesson9]: ================== Test 6 PASSED ==================")


# ============================================================================
# Test 7: joyl→joyh naming convention "joy_gpu_X"
# ============================================================================
def test_joyl_to_joyh_naming(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 7: joyl.X → joyh.custom_call {call_target_name = \"joy_gpu_X\"}")
    print("=" * 60)

    H, INTER = 1024, 3072
    g = Graph(name="naming_test")
    x = g.input([1, 64, H], "f16")
    gate_w = g.input([INTER, H], "f16")
    up_w = g.input([INTER, H], "f16")
    down_w = g.input([H, INTER], "f16")
    gate = ops.silu(ops.linear(x, gate_w))
    up = ops.linear(x, up_w)
    out = ops.linear(ops.mul(gate, up), down_w)
    g.set_outputs([out])
    joy_ir = g.get_ir()

    joyh_ir, stderr, rc = _run_joy_opt(joy_ir,
                                        ["--lower-joy-to-joyl",
                                         "--lower-joyl-to-joyh"])
    if print_ir:
        print(joyh_ir)

    expected_names = {
        "joy_gpu_linear",
        "joy_gpu_silu",
        "joy_gpu_mul",
    }
    found_names = set(re.findall(r'call_target_name = "([^"]+)"', joyh_ir))

    checks = [
        (rc == 0, f"--lower-joyl-to-joyh returned 0 (stderr={stderr.strip()[:120]})"),
        ('"joyh.custom_call"' in joyh_ir, "joyh.custom_call present"),
        (expected_names.issubset(found_names),
         f"all expected targets present: missing={expected_names - found_names}"),
        ('backend = "gpu"' in joyh_ir, 'backend = "gpu" present'),
        ("joyl." not in joyh_ir
         or ("joyl.rms_norm" in joyh_ir or "joyl.fuse_add_rmsnorm" in joyh_ir),
         "no leftover joyl.* (rms_norm/fuse_add_rmsnorm allowed)"),
    ]
    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    print(f"  INFO  call_target_names found: {sorted(found_names)}")
    assert all_pass, "[Lesson9] Test 7 failed"
    print(f"\n[Lesson9]: ================== Test 7 PASSED ==================")


# ============================================================================
# Test 8: num_inputs attribute correctness
# ============================================================================
def test_num_inputs_correctness(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 8: num_inputs computed correctly on joyh.custom_call")
    print("=" * 60)

    H = 1024
    g = Graph(name="num_inputs_test")
    x = g.input([1, 64, H], "f16")
    a = g.input([1, 64, H], "f16")
    b = g.input([1, 64, H], "f16")
    w = g.input([512, H], "f16")
    sum_ab = ops.add(a, b)        # 2 inputs → 1 output  ⇒ num_inputs=2
    mul_ab = ops.mul(sum_ab, x)   # 2 inputs → 1 output  ⇒ num_inputs=2
    lin = ops.linear(mul_ab, w)   # 2 inputs → 1 output  ⇒ num_inputs=2
    sil = ops.silu(lin)           # 1 input  → 1 output  ⇒ num_inputs=1
    g.set_outputs([sil])
    joy_ir = g.get_ir()

    joyh_ir, _, rc = _run_joy_opt(joy_ir,
                                   ["--lower-joy-to-joyl",
                                    "--lower-joyl-to-joyh"])
    if print_ir:
        print(joyh_ir)

    # Each custom_call line: extract its call_target_name + num_inputs
    pattern = re.compile(
        r'"joyh\.custom_call"\(([^)]+)\)\s*<?\{[^}]*'
        r'call_target_name\s*=\s*"([^"]+)"[^}]*'
        r'num_inputs\s*=\s*(\d+)\s*:\s*i64',
        re.DOTALL)
    triples = pattern.findall(joyh_ir)

    # Sometimes the attribute order differs; fall back to looser match.
    if not triples:
        target_re = re.compile(r'call_target_name = "([^"]+)"')
        ni_re = re.compile(r'num_inputs = (\d+)')
        ops_re = re.compile(r'"joyh\.custom_call"\(([^)]+)\)')
        targets = target_re.findall(joyh_ir)
        nis = ni_re.findall(joyh_ir)
        opss = ops_re.findall(joyh_ir)
        triples = list(zip(opss, targets, nis))

    expected_ni = {
        "joy_gpu_add": 2,
        "joy_gpu_mul": 2,
        "joy_gpu_linear": 2,
        "joy_gpu_silu": 1,
    }

    seen = {}
    for operands, tgt, ni in triples:
        n_ops = len(operands.split(","))
        seen[tgt] = int(ni)
        expected = expected_ni.get(tgt)
        if expected is not None:
            ok = int(ni) == expected
            print(
                f"  {'PASS' if ok else 'FAIL'}  {tgt}: "
                f"num_inputs={ni} (expected {expected}, n_operands={n_ops})")

    all_pass = (rc == 0 and all(
        seen.get(name) == expected_ni[name] for name in expected_ni))

    if not all_pass:
        print(f"  seen        : {seen}")
        print(f"  expected    : {expected_ni}")

    assert all_pass, "[Lesson9] Test 8 failed"
    print(f"\n[Lesson9]: ================== Test 8 PASSED ==================")


# ============================================================================
# Test 9: joyl.rms_norm / joyl.fuse_add_rmsnorm are skipped by lower-joyl-to-joyh
# ============================================================================
def test_rms_norm_skipped(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 9: joyl.rms_norm / joyl.fuse_add_rmsnorm are skipped")
    print("=" * 60)

    H = 1024
    g = Graph(name="skip_test")
    x = g.input([1, 64, H], "f16")
    scale = g.input([H], "f16")
    n = ops.rms_norm(x, scale, epsilon=1e-6)
    w = g.input([512, H], "f16")
    out = ops.linear(n, w)
    g.set_outputs([out])
    joy_ir = g.get_ir()

    joyh_ir, _, rc = _run_joy_opt(
        joy_ir, ["--lower-joy-to-joyl", "--lower-joyl-to-joyh"])
    if print_ir:
        print(joyh_ir)

    checks = [
        (rc == 0, "rc == 0"),
        (_count_op(joyh_ir, "joyl.rms_norm") == 1,
         "joyl.rms_norm preserved (Pass skipped it)"),
        ('call_target_name = "joy_gpu_linear"' in joyh_ir,
         "joyl.linear was converted to joyh.custom_call"),
        ('call_target_name = "joy_gpu_rms_norm"' not in joyh_ir,
         "no joy_gpu_rms_norm custom_call (codegen handles it)"),
    ]

    # Same for fused op: fuse_add+rmsnorm path
    g2 = Graph(name="skip_test_fused")
    hidden = g2.input([1, 64, H], "f16")
    residual = g2.input([1, 64, H], "f16")
    ln_w = g2.input([H], "f16")
    proj_w = g2.input([512, H], "f16")
    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=1e-6)
    out2 = ops.linear(normed, proj_w)
    g2.set_outputs([out2])
    joy_ir2 = g2.get_ir()

    joyh_ir2, _, rc2 = _run_joy_opt(
        joy_ir2, ["--joy-op-fusion", "--lower-joy-to-joyl",
                  "--lower-joyl-to-joyh"])
    if print_ir:
        print(joyh_ir2)

    checks.extend([
        (rc2 == 0, "fusion+lower+joyh returned 0"),
        (_count_op(joyh_ir2, "joyl.fuse_add_rmsnorm") == 1,
         "joyl.fuse_add_rmsnorm preserved (Pass skipped it)"),
        ('call_target_name = "joy_gpu_fuse_add_rmsnorm"' not in joyh_ir2,
         "no joy_gpu_fuse_add_rmsnorm custom_call"),
    ])

    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson9] Test 9 failed"
    print(f"\n[Lesson9]: ================== Test 9 PASSED ==================")


# ============================================================================
# Test 10: end-to-end pipeline optimize → lower → codegen → joyh
# ============================================================================
def test_full_compile_pipeline(print_ir=False):
    print("\n" + "=" * 60)
    print("  Test 10: end-to-end optimize → lower → codegen → joyh")
    print("=" * 60)

    H = 1024
    g = Graph(name="full_pipeline")
    hidden = g.input([1, 64, H], "f16")
    residual = g.input([1, 64, H], "f16")
    ln_w = g.input([H], "f16")
    proj_w = g.input([512, H], "f16")
    added = ops.add(residual, hidden)
    normed = ops.rms_norm(added, ln_w, epsilon=1e-6)
    out = ops.linear(normed, proj_w)
    g.set_outputs([out])
    joy_ir = g.get_ir()

    joyh_ir, _, rc = _run_joy_opt(
        joy_ir,
        ["--joy-optimization-pipeline",
         "--lower-joy-to-joyl",
         "--codegen-rms-norm",
         "--lower-joyl-to-joyh"])
    if print_ir:
        print(joyh_ir)

    has_linear_call = 'call_target_name = "joy_gpu_linear"' in joyh_ir
    has_fuse_kernel = "@joy_fuse_add_rmsnorm_kernel" in joyh_ir
    # MLIR prints func.call in shortened form: `call @kernel(...)`.
    has_func_call = (
        "call @joy_fuse_add_rmsnorm_kernel" in joyh_ir or
        "func.call @joy_fuse_add_rmsnorm_kernel" in joyh_ir)
    no_joy = len(_find_ops(joyh_ir, "joy.")) == 0
    no_joyl_other = all(
        x not in joyh_ir for x in ['"joyl.linear"(', '"joyl.add"(',
                                    '"joyl.mul"(', '"joyl.silu"('])

    checks = [
        (rc == 0, "rc == 0"),
        (has_linear_call, "linear emitted as joyh.custom_call"),
        (has_fuse_kernel, "fuse_add_rmsnorm GPU kernel function codegen'd"),
        (has_func_call, "fuse_add_rmsnorm replaced with func.call"),
        (no_joy, "no joy.* remains"),
        (no_joyl_other, "no joyl.linear/add/mul/silu remains"),
    ]

    all_pass = True
    for ok, desc in checks:
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    assert all_pass, "[Lesson9] Test 10 failed"
    print(f"\n[Lesson9]: ================== Test 10 PASSED =================")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Lesson 9: Dialect Lowering in depth — Test")
    parser.add_argument("--print-ir-all", action="store_true",
                        help="Print IR at every stage")
    args = parser.parse_args()
    p = args.print_ir_all

    print("=" * 60)
    print("  Lesson 9: AI Compiler Mid-end — Dialect Lowering")
    print("=" * 60)

    if not _require_joy_opt():
        sys.exit(0)

    test_single_linear_lowering(print_ir=p)
    test_func_signature_conversion(print_ir=p)
    test_attribute_preservation(print_ir=p)
    test_multi_result_lowering(print_ir=p)
    test_multi_op_graph_lowering(print_ir=p)
    test_lowering_completeness(print_ir=p)
    test_joyl_to_joyh_naming(print_ir=p)
    test_num_inputs_correctness(print_ir=p)
    test_rms_norm_skipped(print_ir=p)
    test_full_compile_pipeline(print_ir=p)

    print("\n" + "=" * 60)
    print("  ALL LESSON 9 TESTS PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
