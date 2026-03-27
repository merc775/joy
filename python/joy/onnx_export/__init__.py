"""Qwen3-0.6B PyTorch reference implementation and ONNX export utilities.

The PyTorch module here is structurally equivalent to the official
``transformers.modeling_qwen3.Qwen3Model`` but stripped of attention mask /
KV cache / model parallelism plumbing so it traces cleanly through
``torch.onnx.export``.  It is used by Lesson 5 (model construction & ONNX
export) and Lesson 6 (ONNX → Joy frontend parsing) as the canonical
PyTorch reference.
"""

from .qwen3_model import (
    Qwen3Config,
    Qwen3Model,
    Qwen3DecoderLayer,
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
    build_rope_cache,
    TINY_QWEN3_CFG,
    REAL_QWEN3_06B_CFG,
)

from .export import export_qwen3_to_onnx

__all__ = [
    "Qwen3Config",
    "Qwen3Model",
    "Qwen3DecoderLayer",
    "Qwen3Attention",
    "Qwen3MLP",
    "Qwen3RMSNorm",
    "apply_rotary_pos_emb",
    "repeat_kv",
    "build_rope_cache",
    "TINY_QWEN3_CFG",
    "REAL_QWEN3_06B_CFG",
    "export_qwen3_to_onnx",
]
