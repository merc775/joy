"""Normalization operations for Joy dialect."""


def rms_norm(input_op, scale, epsilon=1e-6):
    """RMS normalization: joy.rms_norm(input, scale) {epsilon}."""
    return input_op.graph._create_op(
        "joy.rms_norm", [input_op, scale],
        list(input_op.shape), input_op.dtype,
        {"epsilon": float(epsilon)})


def fused_add_rms_norm(input_op, residual, scale, epsilon=1e-6):
    """Fused add + RMS normalization: joy.fused_add_rms_norm."""
    return input_op.graph._create_multi_result_op(
        "joy.fused_add_rms_norm", [input_op, residual, scale],
        [list(input_op.shape), list(residual.shape)],
        [input_op.dtype, residual.dtype],
        {"epsilon": float(epsilon)})
