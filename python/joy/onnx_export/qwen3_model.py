"""Qwen3-0.6B PyTorch reference implementation suitable for ONNX export.

The model is intentionally written to mirror ``transformers/modeling_qwen3.py``
exactly on the *compute* side (RMSNorm, GQA + Q/K RMSNorm, RoPE, SwiGLU,
tied embedding), while omitting everything that ``torch.onnx.export`` would
choke on (attention masks, KV cache, generation utilities, model parallel,
flash-attention dispatch, etc.).  See Lesson 5 in ``joy/docs``.
"""

from dataclasses import dataclass, field
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Qwen3Config:
    """Subset of the HuggingFace Qwen3 config used by Lesson 5/6."""
    vocab_size: int = 151936
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = True


# Real Qwen3-0.6B hyper-parameters (matches huggingface.co/Qwen/Qwen3-0.6B).
REAL_QWEN3_06B_CFG = Qwen3Config()

# Teaching-scale model — same *structure* as Qwen3-0.6B (GQA, Q/K norm,
# RoPE, SwiGLU, tied embedding) but small enough to export & store under
# ``joy/tests/onnx_model/``.
TINY_QWEN3_CFG = Qwen3Config(
    vocab_size=2048,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    intermediate_size=256,
    rms_norm_eps=1e-6,
    rope_theta=1e6,
    max_position_embeddings=128,
    tie_word_embeddings=True,
)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class Qwen3RMSNorm(nn.Module):
    """RMSNorm matching transformers/modeling_qwen3.py::Qwen3RMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)
        return (x_fp32.to(in_dtype) * self.weight)


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------
def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 1e6,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute (cos, sin) tables of shape ``[seq_len, head_dim]``.

    Matches Qwen3RotaryEmbedding from transformers.  The output tensors are
    in ``dtype`` (cast outside if needed) and *not* batched — the caller is
    expected to unsqueeze the batch dimension before feeding them into the
    model.
    """
    half = head_dim // 2
    freqs = theta ** (-torch.arange(0, half, dtype=dtype, device=device) / half)
    t = torch.arange(seq_len, dtype=dtype, device=device)
    angles = torch.outer(t, freqs)                       # [S, D/2]
    emb = torch.cat([angles, angles], dim=-1)            # [S, D]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to ``x`` shaped ``[B, H, S, D]``.

    ``cos`` and ``sin`` have shape ``[B, S, D]`` (matching the layout the
    Joy frontend expects).  We broadcast over the head axis explicitly via
    ``unsqueeze(1)``.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA: ``[B, KVH, S, D] -> [B, KVH*n_rep, S, D]``."""
    if n_rep == 1:
        return hidden_states
    B, kv_heads, S, D = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        B, kv_heads, n_rep, S, D)
    return hidden_states.reshape(B, kv_heads * n_rep, S, D)


# ---------------------------------------------------------------------------
# MLP (SwiGLU)
# ---------------------------------------------------------------------------
class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size,
                                   bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Attention (GQA + Q/K RMSNorm + RoPE)
# ---------------------------------------------------------------------------
class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        H = cfg.hidden_size
        self.q_proj = nn.Linear(H, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(H, self.num_kv_heads * self.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(H, self.num_kv_heads * self.head_dim,
                                bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, H, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        weights = scores.softmax(dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(B, S, self.num_heads * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Decoder Layer & Model
# ---------------------------------------------------------------------------
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg)
        self.post_attention_layernorm = Qwen3RMSNorm(cfg.hidden_size,
                                                     cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Model(nn.Module):
    """End-to-end Qwen3 forward without KV cache / attention mask.

    Inputs:
      input_ids: ``[B, S]`` (long)
      cos, sin:  ``[B, S, head_dim]`` (float32) — pre-computed by the caller
                                                  (see ``build_rope_cache``).

    Output:
      logits:    ``[B, S, vocab_size]`` (float32)
    """

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, cos, sin)
        h = self.norm(h)
        return self.lm_head(h)
