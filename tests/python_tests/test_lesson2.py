#!/usr/bin/env python3
"""Lesson 2: Environment audit for the joy compiler.

Before any of the later lessons (3..15) can pass, the host machine
must have a coherent stack of OS / toolchain / LLVM-MLIR 18 /
CUDA 12.x / cuDNN 8.x / Python 3.x + pip packages installed.  This
file performs ten independent checks and prints a concise version
report, so newcomers can spot a misconfigured environment in one
shot.

Checks (see joy/docs/第2讲-AI编译器概述-环境安装.md):

  T1  OS & system toolchain   (uname, glibc, /etc/os-release)
  T2  gcc / g++ / cmake / ninja      versions on PATH
  T3  LLVM / MLIR 18                 mlir-tblgen --version + cmake module
  T4  CUDA Toolkit 12.x              nvcc + libcudart/libcublas via ctypes
  T5  cuDNN 8.x                      cudnn_version.h + libcudnn via ctypes
  T6  NVIDIA Driver / GPU            nvidia-smi + compute capability
  T7  Python interpreter + pip pkgs  python>=3.8 + 6 core pip packages
  T8  torch <-> CUDA integration     torch.cuda.is_available + cuDNN ver
  T9  joy repository layout          critical source files present
  T10 Qwen3-0.6B weights (optional)  config.json + model.safetensors

Usage:
    python3 tests/python_tests/test_lesson2.py
    python3 tests/python_tests/test_lesson2.py --print-info
    python3 tests/python_tests/test_lesson2.py --skip-weights
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


# ----------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------
@dataclass
class Report:
    name: str
    checks: List[Tuple[bool, str]] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    def add(self, ok: bool, desc: str) -> None:
        self.checks.append((bool(ok), desc))

    def skip(self, reason: str) -> None:
        self.skipped = True
        self.skip_reason = reason

    def passed(self) -> bool:
        if self.skipped:
            return True
        return all(ok for ok, _ in self.checks)

    def dump(self) -> None:
        print()
        print("=" * 70)
        print(f"  {self.name}")
        print("=" * 70)
        if self.skipped:
            print(f"  [SKIP] {self.skip_reason}")
            return
        for ok, desc in self.checks:
            prefix = "  [PASS]" if ok else "  [FAIL]"
            print(f"{prefix} {desc}")


def _run(cmd: List[str], timeout: float = 10) -> Tuple[int, str]:
    """Run cmd, return (returncode, stdout+stderr)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def _version_tuple(text: str) -> Tuple[int, ...]:
    """Pull the first `x.y(.z)?` substring out of `text` as a tuple."""
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not m:
        return ()
    return tuple(int(x) for x in m.groups() if x is not None)


def _ge(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
    """Compare version tuples padded with zeros."""
    n = max(len(a), len(b))
    a2 = a + (0,) * (n - len(a))
    b2 = b + (0,) * (n - len(b))
    return a2 >= b2


# ----------------------------------------------------------------------
# T1: OS & system toolchain
# ----------------------------------------------------------------------
def check_os(print_info: bool) -> Report:
    r = Report("T1: OS & system toolchain")
    machine = platform.machine()
    system = platform.system()
    kernel = platform.release()
    r.add(system == "Linux", f"system={system} (must be Linux)")
    r.add(machine in ("x86_64", "amd64"),
          f"arch={machine} (must be x86_64)")

    glibc_ok = False
    glibc_ver = ""
    try:
        glibc_ver = platform.libc_ver()[1]
        glibc_ok = bool(glibc_ver)
    except Exception:
        pass
    r.add(glibc_ok, f"glibc version  = {glibc_ver or 'unknown'}")

    os_release = ""
    try:
        with open("/etc/os-release") as f:
            os_release = f.read()
    except Exception:
        pass
    pretty = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', os_release, re.M)
    pretty_name = pretty.group(1) if pretty else "unknown"
    r.add(bool(pretty_name and pretty_name != "unknown"),
          f"distro         = {pretty_name}")

    if print_info:
        print(f"    uname            : {platform.uname()}")
        print(f"    kernel           : {kernel}")
    return r


# ----------------------------------------------------------------------
# T2: gcc/g++/cmake/ninja
# ----------------------------------------------------------------------
def check_toolchain(print_info: bool) -> Report:
    r = Report("T2: gcc / g++ / cmake / ninja / binutils")

    tools = [
        ("gcc",   ["gcc", "--version"],   (9, 0),  True),
        ("g++",   ["g++", "--version"],   (9, 0),  True),
        ("cmake", ["cmake", "--version"], (3, 15), True),
        ("ninja", ["ninja", "--version"], (1, 10), True),
        ("nm",     ["nm", "--version"],     (2, 0), False),
        ("ar",     ["ar", "--version"],     (2, 0), False),
        ("objdump",["objdump", "--version"],(2, 0), False),
    ]
    for name, cmd, minver, version_required in tools:
        path = shutil.which(cmd[0])
        if not path:
            r.add(False, f"{name} not found on PATH")
            continue
        rc, blob = _run(cmd)
        if rc != 0:
            r.add(False, f"{name} -> rc={rc}")
            continue
        ver = _version_tuple(blob)
        ok_ver = (not version_required) or _ge(ver, minver)
        ver_str = ".".join(str(v) for v in ver) if ver else "?"
        if version_required:
            r.add(ok_ver,
                  f"{name:8s} {ver_str:12s} (>= "
                  f"{'.'.join(str(v) for v in minver)})  "
                  f"-> {path}")
        else:
            r.add(True, f"{name:8s} {ver_str:12s}  -> {path}")

    return r


# ----------------------------------------------------------------------
# T3: LLVM/MLIR 18
# ----------------------------------------------------------------------
# Standard install locations of LLVM/MLIR 18 binaries shipped by the
# upstream packages (apt.llvm.org on Debian/Ubuntu, dnf llvm18 on
# RHEL/Rocky/Alma, manual builds under /opt/llvm-18, ...).  The user
# can additionally point LLVM18_HOME at an arbitrary install prefix.
LLVM18_BIN_CANDIDATES: List[str] = [
    "/usr/lib/llvm-18/bin",
    "/usr/bin",                       # if llvm18 packages dropped binaries here
    "/usr/local/bin",
    "/opt/llvm-18/bin",
]
LLVM18_CMAKE_CANDIDATES: List[str] = [
    "/usr/lib/llvm-18/lib/cmake",
    "/usr/lib64/cmake",
    "/usr/lib/cmake",
    "/opt/llvm-18/lib/cmake",
    "/usr/local/lib/cmake",
]
if os.environ.get("LLVM18_HOME"):
    _home = os.environ["LLVM18_HOME"]
    LLVM18_BIN_CANDIDATES.insert(0, os.path.join(_home, "bin"))
    LLVM18_CMAKE_CANDIDATES.insert(0, os.path.join(_home, "lib", "cmake"))
    LLVM18_CMAKE_CANDIDATES.insert(1, os.path.join(_home, "lib64", "cmake"))


def _which_with_fallback(tool: str) -> str:
    """Find ``tool`` on PATH first, otherwise scan the LLVM18 candidates.

    Also accepts version-suffixed names like ``llvm-config-18``.
    """
    for name in (tool, f"{tool}-18"):
        p = shutil.which(name)
        if p:
            return p
    for d in LLVM18_BIN_CANDIDATES:
        for name in (tool, f"{tool}-18"):
            c = os.path.join(d, name)
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
    return ""


def _find_versioned_llvm_config_18() -> str:
    """Find any llvm-config that reports version 18.x.

    On hosts where multiple LLVM versions coexist (for example one from
    the distro and one shipped by an LLVM upstream package), joy only
    needs MLIRConfig.cmake from LLVM 18 -- any LLVM 18 ``llvm-config``
    on disk satisfies T3, regardless of what ``llvm-config`` resolves
    to on PATH.
    """
    seen = set()
    candidates: List[str] = []
    for d in LLVM18_BIN_CANDIDATES:
        for name in ("llvm-config", "llvm-config-18"):
            candidates.append(os.path.join(d, name))
    for name in ("llvm-config-18", "llvm-config"):
        c = shutil.which(name)
        if c:
            candidates.append(c)
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        if not (os.path.isfile(c) and os.access(c, os.X_OK)):
            continue
        rc, blob = _run([c, "--version"])
        ver = _version_tuple(blob)
        if _ge(ver, (18, 0)) and not _ge(ver, (19, 0)):
            return c
    return ""


def check_llvm_mlir(print_info: bool) -> Report:
    r = Report("T3: LLVM / MLIR 18")
    needed = ["mlir-tblgen", "mlir-opt"]
    paths = {}
    for tool in needed:
        p = _which_with_fallback(tool)
        ok = bool(p) and os.access(p, os.X_OK)
        paths[tool] = p
        r.add(ok, f"{tool:14s} -> {p or 'NOT FOUND'}")

    if paths["mlir-tblgen"]:
        rc, blob = _run([paths["mlir-tblgen"], "--version"])
        ver = _version_tuple(blob)
        ok = _ge(ver, (18, 0)) and not _ge(ver, (19, 0))
        r.add(ok, f"mlir-tblgen version = "
                  f"{'.'.join(str(v) for v in ver) if ver else '?'}  "
                  f"(must be 18.x)")
        if print_info:
            print(f"    mlir-tblgen output:")
            for line in blob.splitlines()[:5]:
                print(f"      | {line}")

    # llvm-config: accept ANY copy that reports 18.x.  joy's build only
    # consumes the LLVM 18 cmake module + headers; a leftover 17.x
    # /usr/bin/llvm-config on PATH does not impact the build.
    llvm18 = _find_versioned_llvm_config_18()
    r.add(bool(llvm18),
          f"LLVM-18 llvm-config -> {llvm18 or 'NOT FOUND'}")
    if llvm18:
        path_first = shutil.which("llvm-config") or ""
        if path_first and path_first != llvm18:
            rc, blob = _run([path_first, "--version"])
            print(f"  [INFO] llvm-config on PATH first  : "
                  f"{path_first}  -> {blob.strip()}")
            print(f"         (joy uses MLIRConfig.cmake & LLVMConfig.cmake "
                  f"explicitly; PATH version does not block the build)")

    # CMake config files (mirror the search list joy/CMakeLists.txt uses).
    mlir_cmake = ""
    for base in LLVM18_CMAKE_CANDIDATES:
        c = os.path.join(base, "mlir", "MLIRConfig.cmake")
        if os.path.isfile(c):
            mlir_cmake = c
            break
    r.add(bool(mlir_cmake), f"MLIRConfig.cmake -> {mlir_cmake or 'NOT FOUND'}")

    # LLVMConfig.cmake: on multi-LLVM hosts the user may need to set
    # LLVM_DIR explicitly; here we accept any plausible copy on disk.
    llvm_cmake = ""
    for base in LLVM18_CMAKE_CANDIDATES:
        c = os.path.join(base, "llvm", "LLVMConfig.cmake")
        if os.path.isfile(c):
            llvm_cmake = c
            break
    r.add(bool(llvm_cmake), f"LLVMConfig.cmake -> {llvm_cmake or 'NOT FOUND'}")
    return r


# ----------------------------------------------------------------------
# T4: CUDA Toolkit
# ----------------------------------------------------------------------
def check_cuda(print_info: bool) -> Report:
    r = Report("T4: CUDA Toolkit 12.x (nvcc + cuBLAS + cudart)")
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    has_nvcc = os.path.isfile(nvcc) and os.access(nvcc, os.X_OK)
    r.add(has_nvcc, f"nvcc -> {nvcc if has_nvcc else 'NOT FOUND'}")

    nvcc_ver = ()
    if has_nvcc:
        rc, blob = _run([nvcc, "--version"])
        m = re.search(r"release\s+(\d+)\.(\d+)", blob)
        if m:
            nvcc_ver = (int(m.group(1)), int(m.group(2)))
        ok = nvcc_ver and _ge(nvcc_ver, (12, 0)) and not _ge(nvcc_ver, (13, 0))
        r.add(ok, f"nvcc release = "
                  f"{'.'.join(str(v) for v in nvcc_ver) if nvcc_ver else '?'} "
                  f"(must be 12.x)")
        if print_info:
            for line in blob.splitlines()[-5:]:
                print(f"    nvcc: {line}")

    # Library probes via ctypes
    for libname in ("libcudart.so", "libcublas.so"):
        try:
            h = ctypes.CDLL(libname)
            r.add(True, f"ctypes.CDLL('{libname}') ok  (handle={hex(h._handle)})")
        except OSError as e:
            r.add(False, f"ctypes.CDLL('{libname}') failed: {e}")

    return r


# ----------------------------------------------------------------------
# T5: cuDNN 8.x
# ----------------------------------------------------------------------
def check_cudnn(print_info: bool) -> Report:
    r = Report("T5: cuDNN 8.x (header + shared library)")

    header_candidates = [
        "/usr/include/cudnn_version.h",
        "/usr/local/include/cudnn_version.h",
        "/usr/local/include/cudnn/cudnn_version.h",
        "/usr/local/cuda/include/cudnn_version.h",
    ]
    if os.environ.get("CUDNN_HOME"):
        header_candidates.insert(
            0, os.path.join(os.environ["CUDNN_HOME"], "include",
                            "cudnn_version.h"))
        header_candidates.insert(
            1, os.path.join(os.environ["CUDNN_HOME"], "include", "cudnn",
                            "cudnn_version.h"))
    header = next((p for p in header_candidates if os.path.isfile(p)), "")
    r.add(bool(header),
          f"cudnn_version.h -> {header or 'NOT FOUND'}")

    cudnn_ver = (0, 0, 0)
    if header:
        try:
            with open(header) as f:
                txt = f.read()
            maj = re.search(r"#define\s+CUDNN_MAJOR\s+(\d+)", txt)
            mnr = re.search(r"#define\s+CUDNN_MINOR\s+(\d+)", txt)
            patch = re.search(r"#define\s+CUDNN_PATCHLEVEL\s+(\d+)", txt)
            cudnn_ver = (int(maj.group(1)),
                          int(mnr.group(1)),
                          int(patch.group(1)))
        except Exception as e:
            r.add(False, f"failed to parse cudnn_version.h: {e}")
    ok_v = _ge(cudnn_ver, (8, 0)) and not _ge(cudnn_ver, (9, 0))
    r.add(ok_v, f"cuDNN version = "
                f"{'.'.join(str(v) for v in cudnn_ver)}  (must be 8.x)")

    try:
        h = ctypes.CDLL("libcudnn.so")
        r.add(True, f"ctypes.CDLL('libcudnn.so') ok  (handle={hex(h._handle)})")
    except OSError as e:
        r.add(False, f"ctypes.CDLL('libcudnn.so') failed: {e}\n"
              "        Hint: echo /usr/local/lib64 | "
              "sudo tee /etc/ld.so.conf.d/cudnn.conf && sudo ldconfig")

    return r


# ----------------------------------------------------------------------
# T6: NVIDIA Driver / GPU
# ----------------------------------------------------------------------
DEFAULT_CUDA_ARCHS = {70, 75, 80, 86, 89, 90}


def check_driver(print_info: bool) -> Report:
    r = Report("T6: NVIDIA Driver + GPU device")
    has_smi = bool(shutil.which("nvidia-smi"))
    r.add(has_smi, "nvidia-smi found on PATH")
    if not has_smi:
        return r

    rc, blob = _run(["nvidia-smi",
                     "--query-gpu=name,driver_version,compute_cap,"
                     "memory.total",
                     "--format=csv,noheader,nounits"])
    if rc != 0:
        r.add(False, f"nvidia-smi -q failed: {blob.strip()}")
        return r
    rows = [l for l in blob.strip().splitlines() if l.strip()]
    r.add(len(rows) >= 1,
          f"at least one CUDA-capable GPU found  ({len(rows)})")
    for i, row in enumerate(rows):
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 4:
            continue
        name, drv, cap, mem = parts[:4]
        try:
            sm = int(round(float(cap) * 10))     # e.g. 8.9 -> 89
        except ValueError:
            sm = 0
        in_default = sm in DEFAULT_CUDA_ARCHS
        r.add(True, f"GPU[{i}]  name={name}  driver={drv}  "
                    f"SM={cap}({sm})  mem={mem} MiB  "
                    f"in joy default archs: {'YES' if in_default else 'NO (set CMAKE_CUDA_ARCHITECTURES)'}")

    if print_info:
        for line in blob.splitlines():
            print(f"    {line}")

    return r


# ----------------------------------------------------------------------
# T7: Python interpreter + pip packages
# ----------------------------------------------------------------------
PY_PACKAGES = [
    ("numpy",            (1, 23), None),
    ("torch",            (2, 0),  None),
    ("safetensors",      (0, 4),  None),
    ("transformers",     (4, 40), None),
    ("huggingface_hub",  (0, 17), None),
    ("onnx",             (1, 14), None),
]


def check_python(print_info: bool) -> Report:
    r = Report("T7: Python interpreter + pip packages")
    py = sys.version_info
    r.add((py.major, py.minor) >= (3, 8),
          f"python version = {py.major}.{py.minor}.{py.micro} (>= 3.8)")

    for pkg, minver, _ in PY_PACKAGES:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            ver_t = _version_tuple(ver)
            ok = _ge(ver_t, minver)
            r.add(ok, f"{pkg:18s} {ver:12s} (>= "
                      f"{'.'.join(str(v) for v in minver)})")
        except ImportError as e:
            r.add(False, f"{pkg:18s} NOT INSTALLED  ({e})")
    return r


# ----------------------------------------------------------------------
# T8: torch <-> CUDA
# ----------------------------------------------------------------------
def check_torch_cuda(print_info: bool) -> Report:
    r = Report("T8: torch <-> CUDA integration")
    try:
        import torch
    except ImportError as e:
        r.skip(f"torch not importable: {e}")
        return r

    cuda_avail = torch.cuda.is_available()
    r.add(cuda_avail, f"torch.cuda.is_available() = {cuda_avail}")
    if not cuda_avail:
        return r

    r.add(torch.version.cuda is not None,
          f"torch.version.cuda    = {torch.version.cuda}")
    r.add(torch.cuda.device_count() >= 1,
          f"torch.cuda.device_count() = {torch.cuda.device_count()}")

    cudnn_v = torch.backends.cudnn.version()
    r.add(cudnn_v is not None and cudnn_v >= 8000,
          f"torch.backends.cudnn.version() = {cudnn_v} (>= 8000)")

    if print_info:
        print(f"    torch                 : {torch.__version__}")
        print(f"    torch.version.cuda    : {torch.version.cuda}")
        print(f"    torch.cuda.device_count(): {torch.cuda.device_count()}")
        for i in range(min(torch.cuda.device_count(), 4)):
            props = torch.cuda.get_device_properties(i)
            print(f"    torch.cuda[{i}]: "
                  f"name={props.name} cap={props.major}.{props.minor} "
                  f"mem={props.total_memory // 1024**2} MiB")
    return r


# ----------------------------------------------------------------------
# T9: joy repository layout
# ----------------------------------------------------------------------
JOY_KEY_PATHS = [
    "CMakeLists.txt",
    "scripts/init.sh",
    "scripts/build.sh",
    "scripts/regen_codegen_kernel.sh",
    "include/joy/dialect/joy/JoyOps.td",
    "include/joy/dialect/joyl/JoylOps.td",
    "include/joy/dialect/joyh/JoyhOps.td",
    "lib/CMakeLists.txt",
    "lib/optimizer/CMakeLists.txt",
    "lib/backend/gpu/CMakeLists.txt",
    "lib/backend/gpu/gpu_kernels.cu",
    "lib/runtime/gpu/gpu_ops.cpp",
    "lib/runtime/gpu/gpu_entry.cpp",
    "tools/joy-opt.cpp",
    "tools/joy-emit-cuda.cpp",
    "python/joy/builder/graph.py",
    "tests/python_tests/qwen3_gpu_runner.py",
    "tests/python_tests/test_lesson15.py",
]


def check_joy_repo(print_info: bool) -> Report:
    r = Report("T9: joy repository layout")
    r.add(os.path.isdir(PROJECT_ROOT), f"project root: {PROJECT_ROOT}")
    for rel in JOY_KEY_PATHS:
        p = os.path.join(PROJECT_ROOT, rel)
        ok = os.path.isfile(p)
        r.add(ok, f"{rel}")
        if print_info and ok:
            print(f"    {rel}  ({os.path.getsize(p)} bytes)")
    return r


# ----------------------------------------------------------------------
# T10: Qwen3-0.6B weights (optional)
# ----------------------------------------------------------------------
DEFAULT_MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca")


def _resolve_model_path() -> Optional[str]:
    env = os.environ.get("QWEN3_MODEL_PATH")
    if env and os.path.isdir(env):
        return env
    if os.path.isdir(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        base = os.path.join(HF_HUB_CACHE, "models--Qwen--Qwen3-0.6B",
                            "snapshots")
        if os.path.isdir(base):
            for snap in sorted(os.listdir(base)):
                cand = os.path.join(base, snap)
                if os.path.isfile(os.path.join(cand, "config.json")):
                    return cand
    except Exception:
        pass
    return None


def check_qwen3_weights(print_info: bool) -> Report:
    r = Report("T10: Qwen3-0.6B weights (optional)")
    path = _resolve_model_path()
    if path is None:
        r.skip("Qwen3-0.6B model not found; "
               "set QWEN3_MODEL_PATH or download via huggingface_hub")
        return r
    r.add(True, f"resolved path: {path}")

    cfg = os.path.join(path, "config.json")
    st = os.path.join(path, "model.safetensors")
    r.add(os.path.isfile(cfg), f"config.json present")
    r.add(os.path.isfile(st),  f"model.safetensors present")

    if os.path.isfile(cfg):
        try:
            import json
            with open(cfg) as f:
                conf = json.load(f)
            ok_hs = conf.get("hidden_size") == 1024
            ok_nl = conf.get("num_hidden_layers") == 28
            ok_kv = conf.get("num_key_value_heads") == 8
            ok_hd = conf.get("head_dim") == 128
            r.add(ok_hs, f"hidden_size = {conf.get('hidden_size')}  (expect 1024)")
            r.add(ok_nl, f"num_hidden_layers = {conf.get('num_hidden_layers')}  (expect 28)")
            r.add(ok_kv, f"num_key_value_heads = {conf.get('num_key_value_heads')}  (expect 8)")
            r.add(ok_hd, f"head_dim = {conf.get('head_dim')}  (expect 128)")
            if print_info:
                for k in ("hidden_size", "num_attention_heads",
                          "num_hidden_layers", "vocab_size",
                          "rope_theta", "rms_norm_eps"):
                    print(f"    {k:25s} = {conf.get(k)}")
        except Exception as e:
            r.add(False, f"failed to parse config.json: {e}")
    return r


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Lesson 2: environment audit for the joy compiler")
    parser.add_argument("--print-info", action="store_true",
                        help="Print discovered versions in addition to "
                             "PASS/FAIL")
    parser.add_argument("--skip-weights", action="store_true",
                        help="Skip the optional Qwen3-0.6B weight check")
    args = parser.parse_args(argv)

    print("=" * 70)
    print("  Lesson 2: joy compiler environment audit")
    print("=" * 70)
    print(f"  Python      : {sys.executable}")
    print(f"  Project root: {PROJECT_ROOT}")

    reports: List[Report] = []
    reports.append(check_os(args.print_info))
    reports.append(check_toolchain(args.print_info))
    reports.append(check_llvm_mlir(args.print_info))
    reports.append(check_cuda(args.print_info))
    reports.append(check_cudnn(args.print_info))
    reports.append(check_driver(args.print_info))
    reports.append(check_python(args.print_info))
    reports.append(check_torch_cuda(args.print_info))
    reports.append(check_joy_repo(args.print_info))

    if not args.skip_weights:
        reports.append(check_qwen3_weights(args.print_info))

    # Print each report
    failed = []
    for rep in reports:
        rep.dump()
        if not rep.passed():
            failed.append(rep.name)

    print("\n" + "=" * 70)
    if failed:
        print(f"  FAIL  ({len(failed)} of {len(reports)} test groups failed):")
        for name in failed:
            print(f"    - {name}")
        print("=" * 70 + "\n")
        return 1
    print(f"  ALL LESSON 2 CHECKS PASSED  ({len(reports)} test groups)")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
