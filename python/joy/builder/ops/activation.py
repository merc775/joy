"""Activation operations for Joy dialect."""


def sigmoid(input_op):
    """Sigmoid activation: joy.sigmoid(input)."""
    return input_op.graph._create_op(
        "joy.sigmoid", [input_op],
        list(input_op.shape), input_op.dtype)


def silu(input_op):
    """SiLU (Swish) activation: joy.silu(input).

    SiLU(x) = x * sigmoid(x)
    """
    return input_op.graph._create_op(
        "joy.silu", [input_op],
        list(input_op.shape), input_op.dtype)
