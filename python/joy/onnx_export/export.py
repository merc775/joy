"""Helpers for exporting Qwen3 PyTorch models to ONNX."""

import os
from typing import Optional, Tuple

import torch

from .qwen3_model import Qwen3Config, Qwen3Model, build_rope_cache


def export_qwen3_to_onnx(
    cfg: Qwen3Config,
    output_path: str,
    *,
    batch_size: int = 1,
    seq_len: int = 32,
    opset_version: int = 17,
    use_external_data: bool = False,
    model: Optional[Qwen3Model] = None,
    dtype: torch.dtype = torch.float32,
    deterministic_init: bool = True,
) -> Tuple[str, Qwen3Model]:
    """Export a Qwen3 model to an ONNX file.

    Parameters
    ----------
    cfg : Qwen3Config
        Hyper-parameters of the model.  If ``model`` is provided, ``cfg`` is
        only used for the dummy-input shapes (``head_dim``).
    output_path : str
        Where to write the ONNX file.  Parent directory is created if
        needed.
    batch_size, seq_len : int
        Dummy-input shapes used to trace the model.  ``dynamic_axes`` is
        always set so the exported ONNX accepts arbitrary B / S at
        inference time.
    opset_version : int
        Default 17 (introduces SimplifiedLayerNormalization / Pow support).
    use_external_data : bool
        If True, weights are saved to a sibling ``.weights`` file via
        ``save_as_external_data``.  Required for >2GB models.
    model : Optional[Qwen3Model]
        Pre-built model.  If ``None``, a fresh ``Qwen3Model(cfg)`` is
        created (with deterministic random init when
        ``deterministic_init=True``).
    dtype : torch.dtype
        Dtype of the dummy inputs (parameters stay at fp32 by default).

    Returns
    -------
    (path, model) : str, Qwen3Model
        The path of the written ONNX file and the (possibly newly built)
        PyTorch model.  The model is returned so callers can run
        side-by-side PyTorch / ONNX-RT comparisons.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".",
                exist_ok=True)

    if model is None:
        if deterministic_init:
            torch.manual_seed(0)
        model = Qwen3Model(cfg)
    model.eval()

    dummy_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    cos, sin = build_rope_cache(seq_len, cfg.head_dim,
                                theta=cfg.rope_theta, dtype=dtype)
    cos = cos.unsqueeze(0).expand(batch_size, seq_len, cfg.head_dim).contiguous()
    sin = sin.unsqueeze(0).expand(batch_size, seq_len, cfg.head_dim).contiguous()

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "cos":       {0: "batch", 1: "seq"},
        "sin":       {0: "batch", 1: "seq"},
        "logits":    {0: "batch", 1: "seq"},
    }

    # ``torch.onnx.export`` writes to ``output_path`` directly; for huge
    # models we need a two-step procedure (export → onnx.save with
    # save_as_external_data).
    if not use_external_data:
        torch.onnx.export(
            model,
            (dummy_ids, cos, sin),
            output_path,
            input_names=["input_ids", "cos", "sin"],
            output_names=["logits"],
            opset_version=opset_version,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )
    else:
        # Export to a tmp buffer first, then save_model with external data.
        import io
        import onnx
        buf = io.BytesIO()
        torch.onnx.export(
            model,
            (dummy_ids, cos, sin),
            buf,
            input_names=["input_ids", "cos", "sin"],
            output_names=["logits"],
            opset_version=opset_version,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )
        buf.seek(0)
        onnx_model = onnx.load_model_from_string(buf.read())
        onnx.save_model(
            onnx_model, output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(output_path) + ".weights",
            size_threshold=1024,
        )
    return output_path, model
