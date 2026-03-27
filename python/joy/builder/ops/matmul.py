"""Matrix multiplication and linear operations for Joy dialect."""

from .eltwise import _broadcast_shape


def matmul(lhs, rhs):
    """Matrix multiply: joy.matmul(lhs, rhs)."""
    lhs_shape = list(lhs.shape)
    rhs_shape = list(rhs.shape)

    if len(lhs_shape) >= 2 and len(rhs_shape) >= 2:
        lhs_batch = lhs_shape[:-2]
        rhs_batch = rhs_shape[:-2]
        if lhs_batch and rhs_batch:
            batch = _broadcast_shape(lhs_batch, rhs_batch)
        else:
            batch = lhs_batch or rhs_batch
        m = lhs_shape[-2]
        n = rhs_shape[-1]
        result_shape = batch + [m, n]
    elif len(rhs_shape) == 1:
        result_shape = lhs_shape[:-1]
    else:
        result_shape = lhs_shape[:-1] + [rhs_shape[-1]]

    return lhs.graph._create_op("joy.matmul", [lhs, rhs],
                                result_shape, lhs.dtype)


def linear(input_op, weight):
    """Linear projection matching PyTorch nn.Linear convention.

    PyTorch nn.Linear(in_features, out_features):
      weight shape: [out_features, in_features]
      forward:      output = input @ weight^T + bias

    So output dim = weight.shape[0] (out_features).

    input: [..., in_features]
    weight: [out_features, in_features]
    output: [..., out_features]
    """
    input_shape = list(input_op.shape)
    weight_shape = list(weight.shape)
    result_shape = input_shape[:-1] + [weight_shape[0]]
    return input_op.graph._create_op("joy.linear", [input_op, weight],
                                     result_shape, input_op.dtype)
