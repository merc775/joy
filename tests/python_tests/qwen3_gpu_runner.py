"""End-to-end Qwen3-0.6B forward executor on the joy GPU backend.

This module dispatches every operator in a Qwen3 forward pass through
the ``joy_gpu_*`` C ABI exported by ``libjoy_gpu_runtime.so`` (built
from ``joy/lib/backend/gpu``).  cuBLAS handles the matmul / linear
projections, cuDNN handles softmax, and the remaining ops use the
custom CUDA kernels in ``gpu_kernels.cu``.

The same C entry points are what the joyh dialect lowers to via
``joyh.custom_call``, so successfully running this executor end-to-end
is equivalent to executing the lowered Joyh IR with these custom calls.

Public API:

    runner = Qwen3GpuRunner.from_pretrained("/path/to/Qwen3-0.6B")
    output_ids = runner.generate(input_ids,
                                 max_new_tokens=10,
                                 eos_token_id=151645)

The forward pass is recompute-mode (no KV cache): each decode step
re-runs the full network on the growing prompt.  This keeps the
executor small and correct without touching the model graph.
"""
from __future__ import annotations

import ctypes
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Make tests/python_tests/test_op importable as a package.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from test_op._runtime import (  # noqa: E402
    DeviceBuffer,
    JoyGpuRuntime,
    get_runtime,
)


# ---------------------------------------------------------------------------
# Qwen3-0.6B configuration (must match HF config.json)
# ---------------------------------------------------------------------------
@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    num_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    num_layers: int = 28
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @classmethod
    def from_json(cls, path: str) -> "Qwen3Config":
        import json
        with open(path) as f:
            cfg = json.load(f)
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            intermediate_size=cfg["intermediate_size"],
            num_layers=cfg["num_hidden_layers"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg["rope_theta"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
        )


# ---------------------------------------------------------------------------
# RoPE table computation (matches HF Qwen3 modeling_qwen3.py)
# ---------------------------------------------------------------------------
def compute_rope_tables(positions: np.ndarray,
                        head_dim: int,
                        theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return cos/sin tables for the given absolute positions.

    positions: ``[S]`` integer or float.
    Returns ``(cos, sin)`` with shape ``[S, head_dim]`` in float32.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) /
                                float(half)))
    freqs = positions.astype(np.float32)[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


# ---------------------------------------------------------------------------
# Helper: small RAII container for intermediate buffers
# ---------------------------------------------------------------------------
class _BufferScope:
    """Tracks DeviceBuffers and frees them in bulk on exit."""

    def __init__(self, rt: JoyGpuRuntime):
        self.rt = rt
        self._owned: List[DeviceBuffer] = []

    def adopt(self, buf: DeviceBuffer) -> DeviceBuffer:
        self._owned.append(buf)
        return buf

    def alloc(self, shape: Sequence[int], dtype: np.dtype) -> DeviceBuffer:
        return self.adopt(self.rt.alloc_like(shape, dtype))

    def upload(self, host: np.ndarray) -> DeviceBuffer:
        return self.adopt(self.rt.upload(host))

    def free(self, buf: DeviceBuffer) -> None:
        if buf in self._owned:
            self._owned.remove(buf)
        buf.free()

    def free_all(self) -> None:
        for b in self._owned:
            b.free()
        self._owned.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.free_all()
        return False


# ---------------------------------------------------------------------------
# Qwen3 GPU executor
# ---------------------------------------------------------------------------
class Qwen3GpuRunner:
    """Executes Qwen3-0.6B end-to-end on the joy GPU backend.

    All weights are uploaded once to GPU at construction time.  Each
    ``forward()`` allocates per-call activation buffers, dispatches the
    layer ops via ``joy_gpu_*``, downloads logits to host, and frees the
    activation buffers.
    """

    def __init__(self, weights: Dict[str, np.ndarray],
                 config: Qwen3Config,
                 dtype: np.dtype = np.float32) -> None:
        self.cfg = config
        self.np_dtype = np.dtype(dtype)
        if self.np_dtype != np.float32:
            raise NotImplementedError(
                "Only float32 execution is currently supported; the "
                "f16 path of GpuFuseAddRMSNormOp/RMSNorm has not been "
                "validated end-to-end.")

        self.rt: JoyGpuRuntime = get_runtime()

        # Upload all weights to GPU.  We bake the 1/sqrt(head_dim) attention
        # scaling factor into q_norm.weight, since the Joy graph definition
        # does NOT include an explicit scaling op between Q and K.
        scale = 1.0 / math.sqrt(self.cfg.head_dim)
        self.weights: Dict[str, DeviceBuffer] = {}
        t0 = time.perf_counter()
        total_bytes = 0
        for name, arr in weights.items():
            arr = np.ascontiguousarray(arr, dtype=self.np_dtype)
            if name.endswith(".q_norm.weight"):
                arr = arr * scale
            self.weights[name] = self.rt.upload(arr)
            total_bytes += arr.nbytes
        print(f"  uploaded {len(self.weights)} tensors ("
              f"{total_bytes / 1e6:.1f} MB) to GPU in "
              f"{time.perf_counter() - t0:.2f} s")

    # ----- factory -----
    @classmethod
    def from_pretrained(cls, model_path: str,
                        dtype: np.dtype = np.float32) -> "Qwen3GpuRunner":
        # The Qwen3 safetensors file uses bfloat16, which numpy cannot
        # represent natively, so we route through torch for the dtype cast.
        import torch
        from safetensors.torch import load_file

        cfg = Qwen3Config.from_json(os.path.join(model_path, "config.json"))
        st = load_file(os.path.join(model_path, "model.safetensors"))

        def _to_f32(t: "torch.Tensor") -> np.ndarray:
            return t.detach().to(torch.float32).contiguous().cpu().numpy()

        wanted: List[str] = ["model.embed_tokens.weight",
                             "model.norm.weight"]
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}"
            wanted += [
                f"{p}.input_layernorm.weight",
                f"{p}.post_attention_layernorm.weight",
                f"{p}.self_attn.q_proj.weight",
                f"{p}.self_attn.k_proj.weight",
                f"{p}.self_attn.v_proj.weight",
                f"{p}.self_attn.o_proj.weight",
                f"{p}.self_attn.q_norm.weight",
                f"{p}.self_attn.k_norm.weight",
                f"{p}.mlp.gate_proj.weight",
                f"{p}.mlp.up_proj.weight",
                f"{p}.mlp.down_proj.weight",
            ]

        weights: Dict[str, np.ndarray] = {}
        for k in wanted:
            if k not in st:
                raise KeyError(f"missing weight: {k}")
            weights[k] = _to_f32(st[k])

        # lm_head: prefer file's own copy if present (Qwen3-0.6B does keep
        # one in addition to the tied embedding); otherwise reuse the
        # embedding table.
        if "lm_head.weight" in st:
            weights["lm_head.weight"] = _to_f32(st["lm_head.weight"])
        else:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

        return cls(weights, config=cfg, dtype=dtype)

    # ===================================================================
    # Per-op helpers (each allocates a fresh output buffer)
    # ===================================================================
    def _embed(self, ctx, scope: _BufferScope, ids: DeviceBuffer,
               table: DeviceBuffer) -> DeviceBuffer:
        out_shape = list(ids.shape) + [self.cfg.hidden_size]
        out = scope.alloc(out_shape, self.np_dtype)
        self.rt.run_op("embedding", ctx, [ids, table], [out], sync=False)
        return out

    def _rms_norm(self, ctx, scope: _BufferScope, x: DeviceBuffer,
                  weight: DeviceBuffer, eps: float) -> DeviceBuffer:
        out = scope.alloc(x.shape, self.np_dtype)
        self.rt.run_op("rms_norm", ctx, [x, weight], [out],
                       extra_args=[ctypes.c_float(eps)], sync=False)
        return out

    def _linear(self, ctx, scope: _BufferScope, x: DeviceBuffer,
                weight: DeviceBuffer) -> DeviceBuffer:
        # weight is [out_features, in_features], output last dim = out_features
        out_shape = list(x.shape[:-1]) + [weight.shape[0]]
        out = scope.alloc(out_shape, self.np_dtype)
        self.rt.run_op("linear", ctx, [x, weight], [out], sync=False)
        return out

    def _matmul(self, ctx, scope: _BufferScope, a: DeviceBuffer,
                b: DeviceBuffer) -> DeviceBuffer:
        out_shape = list(a.shape[:-1]) + [b.shape[-1]]
        out = scope.alloc(out_shape, self.np_dtype)
        self.rt.run_op("matmul", ctx, [a, b], [out], sync=False)
        return out

    def _reshape(self, ctx, scope: _BufferScope, x: DeviceBuffer,
                 new_shape: Sequence[int]) -> DeviceBuffer:
        out = scope.alloc(new_shape, self.np_dtype)
        self.rt.run_op("reshape", ctx, [x], [out], sync=False)
        return out

    def _transpose(self, ctx, scope: _BufferScope, x: DeviceBuffer,
                   perm: Sequence[int]) -> DeviceBuffer:
        new_shape = [x.shape[p] for p in perm]
        out = scope.alloc(new_shape, self.np_dtype)
        perm_arr = (ctypes.c_int64 * len(perm))(*perm)
        self.rt.run_op("transpose", ctx, [x], [out],
                       extra_args=[perm_arr, ctypes.c_int64(len(perm))],
                       sync=False)
        return out

    def _apply_rotary(self, ctx, scope: _BufferScope, x: DeviceBuffer,
                      cos: DeviceBuffer, sin: DeviceBuffer) -> DeviceBuffer:
        out = scope.alloc(x.shape, self.np_dtype)
        self.rt.run_op("apply_rotary_emb", ctx, [x, cos, sin], [out],
                       sync=False)
        return out

    def _repeat_kv(self, ctx, scope: _BufferScope, x: DeviceBuffer,
                   n_rep: int) -> DeviceBuffer:
        b, h, s, d = x.shape
        out = scope.alloc((b, h * n_rep, s, d), self.np_dtype)
        self.rt.run_op("repeat_kv", ctx, [x], [out],
                       extra_args=[ctypes.c_int64(n_rep)], sync=False)
        return out

    def _softmax(self, ctx, scope: _BufferScope, x: DeviceBuffer,
                 axis: int = -1) -> DeviceBuffer:
        out = scope.alloc(x.shape, self.np_dtype)
        self.rt.run_op("softmax", ctx, [x], [out],
                       extra_args=[ctypes.c_int64(axis)], sync=False)
        return out

    def _silu(self, ctx, scope: _BufferScope, x: DeviceBuffer) -> DeviceBuffer:
        out = scope.alloc(x.shape, self.np_dtype)
        self.rt.run_op("silu", ctx, [x], [out], sync=False)
        return out

    def _add(self, ctx, scope: _BufferScope, a: DeviceBuffer,
             b: DeviceBuffer) -> DeviceBuffer:
        out = scope.alloc(a.shape, self.np_dtype)
        self.rt.run_op("add", ctx, [a, b], [out], sync=False)
        return out

    def _mul(self, ctx, scope: _BufferScope, a: DeviceBuffer,
             b: DeviceBuffer) -> DeviceBuffer:
        out = scope.alloc(a.shape, self.np_dtype)
        self.rt.run_op("mul", ctx, [a, b], [out], sync=False)
        return out

    # ===================================================================
    # Self-attention block
    # ===================================================================
    def _self_attention(self, ctx, scope: _BufferScope,
                        hidden_norm: DeviceBuffer, layer_idx: int,
                        cos: DeviceBuffer, sin: DeviceBuffer,
                        causal_mask: Optional[np.ndarray] = None,
                        ) -> DeviceBuffer:
        cfg = self.cfg
        B, S, _ = hidden_norm.shape
        p = f"model.layers.{layer_idx}.self_attn"

        # 1) Q/K/V projections
        q_w = self.weights[f"{p}.q_proj.weight"]
        k_w = self.weights[f"{p}.k_proj.weight"]
        v_w = self.weights[f"{p}.v_proj.weight"]
        q = self._linear(ctx, scope, hidden_norm, q_w)
        k = self._linear(ctx, scope, hidden_norm, k_w)
        v = self._linear(ctx, scope, hidden_norm, v_w)

        # 2) Reshape to [B, S, H*, D]
        q = self._reshape(ctx, scope, q,
                          (B, S, cfg.num_heads, cfg.head_dim))
        k = self._reshape(ctx, scope, k,
                          (B, S, cfg.num_kv_heads, cfg.head_dim))
        v = self._reshape(ctx, scope, v,
                          (B, S, cfg.num_kv_heads, cfg.head_dim))

        # 3) per-head RMSNorm (q_norm.weight already absorbs 1/sqrt(D))
        q_norm_w = self.weights[f"{p}.q_norm.weight"]
        k_norm_w = self.weights[f"{p}.k_norm.weight"]
        q = self._rms_norm(ctx, scope, q, q_norm_w, cfg.rms_norm_eps)
        k = self._rms_norm(ctx, scope, k, k_norm_w, cfg.rms_norm_eps)

        # 4) Transpose to [B, H*, S, D]
        q = self._transpose(ctx, scope, q, (0, 2, 1, 3))
        k = self._transpose(ctx, scope, k, (0, 2, 1, 3))
        v = self._transpose(ctx, scope, v, (0, 2, 1, 3))

        # 5) RoPE on Q and K
        q = self._apply_rotary(ctx, scope, q, cos, sin)
        k = self._apply_rotary(ctx, scope, k, cos, sin)

        # 6) Repeat KV (GQA)
        if cfg.num_kv_groups > 1:
            k = self._repeat_kv(ctx, scope, k, cfg.num_kv_groups)
            v = self._repeat_kv(ctx, scope, v, cfg.num_kv_groups)

        # 7) Attention scores: Q @ K^T
        k_t = self._transpose(ctx, scope, k, (0, 1, 3, 2))
        attn = self._matmul(ctx, scope, q, k_t)  # [B, H, S, S]

        # 8) Apply causal mask additively (mask is uploaded once per forward).
        if causal_mask is not None:
            # mask shape [1, 1, S, S], broadcast-add via element-wise add
            # We build a [B, H, S, S] copy on host then upload — simple and
            # only happens once per forward pass.
            bhss_mask = np.broadcast_to(causal_mask,
                                        (B, cfg.num_heads, S, S)
                                        ).astype(self.np_dtype)
            mask_buf = scope.upload(np.ascontiguousarray(bhss_mask))
            attn = self._add(ctx, scope, attn, mask_buf)

        # 9) Softmax over last dim
        attn = self._softmax(ctx, scope, attn, axis=-1)

        # 10) attn @ V -> [B, H, S, D]
        attn_out = self._matmul(ctx, scope, attn, v)

        # 11) [B, H, S, D] -> [B, S, H*D]
        attn_out = self._transpose(ctx, scope, attn_out, (0, 2, 1, 3))
        attn_out = self._reshape(ctx, scope, attn_out,
                                 (B, S, cfg.num_heads * cfg.head_dim))

        # 12) Output projection
        o_w = self.weights[f"{p}.o_proj.weight"]
        return self._linear(ctx, scope, attn_out, o_w)

    # ===================================================================
    # MLP
    # ===================================================================
    def _mlp(self, ctx, scope: _BufferScope, x: DeviceBuffer,
             layer_idx: int) -> DeviceBuffer:
        p = f"model.layers.{layer_idx}.mlp"
        gate = self._linear(ctx, scope, x, self.weights[f"{p}.gate_proj.weight"])
        gate = self._silu(ctx, scope, gate)
        up = self._linear(ctx, scope, x, self.weights[f"{p}.up_proj.weight"])
        gate_up = self._mul(ctx, scope, gate, up)
        return self._linear(ctx, scope, gate_up,
                            self.weights[f"{p}.down_proj.weight"])

    # ===================================================================
    # Decoder layer
    # ===================================================================
    def _decoder_layer(self, ctx, scope: _BufferScope, hidden: DeviceBuffer,
                       layer_idx: int, cos: DeviceBuffer, sin: DeviceBuffer,
                       causal_mask: Optional[np.ndarray]) -> DeviceBuffer:
        p = f"model.layers.{layer_idx}"
        # Pre-attention RMSNorm
        normed = self._rms_norm(
            ctx, scope, hidden,
            self.weights[f"{p}.input_layernorm.weight"], self.cfg.rms_norm_eps)
        attn_out = self._self_attention(ctx, scope, normed, layer_idx,
                                        cos, sin, causal_mask)
        h2 = self._add(ctx, scope, hidden, attn_out)

        # Post-attention RMSNorm
        normed2 = self._rms_norm(
            ctx, scope, h2,
            self.weights[f"{p}.post_attention_layernorm.weight"],
            self.cfg.rms_norm_eps)
        mlp_out = self._mlp(ctx, scope, normed2, layer_idx)
        return self._add(ctx, scope, h2, mlp_out)

    # ===================================================================
    # Forward
    # ===================================================================
    def forward(self, input_ids: np.ndarray,
                position_ids: Optional[np.ndarray] = None,
                use_causal_mask: bool = True) -> np.ndarray:
        """One forward pass.  Returns logits ``[B, S, V]`` on host."""
        if input_ids.dtype != np.int64:
            input_ids = input_ids.astype(np.int64)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        B, S = input_ids.shape

        if position_ids is None:
            position_ids = np.arange(S, dtype=np.int64)

        cos_host, sin_host = compute_rope_tables(
            position_ids.reshape(-1).astype(np.int64),
            self.cfg.head_dim, self.cfg.rope_theta)

        causal_mask: Optional[np.ndarray] = None
        if use_causal_mask and S > 1:
            mask = np.zeros((1, 1, S, S), dtype=np.float32)
            # additive: 0 for kept, -inf for masked future positions
            tri = np.triu(np.ones((S, S), dtype=np.bool_), k=1)
            mask[0, 0, tri] = -1e9
            causal_mask = mask

        with self.rt.context() as ctx, _BufferScope(self.rt) as scope:
            ids_buf = scope.upload(input_ids)
            cos_buf = scope.upload(cos_host)
            sin_buf = scope.upload(sin_host)

            hidden = self._embed(
                ctx, scope, ids_buf,
                self.weights["model.embed_tokens.weight"])

            for i in range(self.cfg.num_layers):
                hidden = self._decoder_layer(ctx, scope, hidden, i,
                                             cos_buf, sin_buf, causal_mask)

            hidden = self._rms_norm(
                ctx, scope, hidden,
                self.weights["model.norm.weight"], self.cfg.rms_norm_eps)

            logits = self._linear(
                ctx, scope, hidden, self.weights["lm_head.weight"])

            self.rt.stream_synchronize(ctx)
            host_logits = self.rt.download(logits)

        return host_logits

    # ===================================================================
    # Greedy generation (recompute mode -- no KV cache)
    # ===================================================================
    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 8,
                 eos_token_id: Optional[int] = None,
                 verbose: bool = False) -> np.ndarray:
        if input_ids.dtype != np.int64:
            input_ids = input_ids.astype(np.int64)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        B, prompt_len = input_ids.shape
        if B != 1:
            raise NotImplementedError("Only batch=1 is supported.")

        all_ids = input_ids.copy()
        new_tokens: List[int] = []

        # ----- Prefill -----
        t0 = time.perf_counter()
        logits = self.forward(all_ids)
        next_token = int(np.argmax(logits[0, -1]))
        prefill_dt = time.perf_counter() - t0
        new_tokens.append(next_token)
        if verbose:
            print(f"  prefill ({prompt_len} tokens) -> "
                  f"first_token={next_token} "
                  f"({prefill_dt:.2f} s)")
        if eos_token_id is not None and next_token == eos_token_id:
            return np.array([new_tokens], dtype=np.int64)

        # ----- Decode loop (recompute mode) -----
        for step in range(max_new_tokens - 1):
            all_ids = np.concatenate(
                [all_ids, np.array([[next_token]], dtype=np.int64)], axis=1)
            t0 = time.perf_counter()
            logits = self.forward(all_ids)
            next_token = int(np.argmax(logits[0, -1]))
            dt = time.perf_counter() - t0
            new_tokens.append(next_token)
            if verbose:
                print(f"  decode step {step + 1} ({all_ids.shape[1]} tokens) -> "
                      f"token={next_token} ({dt:.2f} s)")
            if eos_token_id is not None and next_token == eos_token_id:
                break

        return np.array([new_tokens], dtype=np.int64)
