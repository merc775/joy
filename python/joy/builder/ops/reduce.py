"""Reduction operations for Joy dialect."""


def softmax(input_op, axis=-1):
    """Softmax: joy.softmax(input) {axis}."""
    if axis < 0:
        axis = len(input_op.shape) + axis
    return input_op.graph._create_op(
        "joy.softmax", [input_op],
        list(input_op.shape), input_op.dtype,
        {"axis": axis})
