"""View / shape operations for Joy dialect."""


def reshape(input_op, new_shape):
    """Reshape tensor: joy.reshape(input) -> new_shape."""
    return input_op.graph._create_op(
        "joy.reshape", [input_op],
        list(new_shape), input_op.dtype)


def transpose(input_op, perm):
    """Transpose tensor: joy.transpose(input) {permutation}."""
    old_shape = list(input_op.shape)
    new_shape = [old_shape[p] for p in perm]
    return input_op.graph._create_op(
        "joy.transpose", [input_op],
        new_shape, input_op.dtype,
        {"permutation": list(perm)})


def unsqueeze(input_op, axis):
    """Unsqueeze: joy.unsqueeze(input) — insert dim at axis."""
    new_shape = list(input_op.shape)
    new_shape.insert(axis, 1)
    return input_op.graph._create_op(
        "joy.unsqueeze", [input_op],
        new_shape, input_op.dtype)


def squeeze(input_op, axis):
    """Squeeze: joy.squeeze(input) — remove dim at axis."""
    new_shape = list(input_op.shape)
    if 0 <= axis < len(new_shape) and new_shape[axis] == 1:
        new_shape.pop(axis)
    return input_op.graph._create_op(
        "joy.squeeze", [input_op],
        new_shape, input_op.dtype)


def repeat_kv(input_op, n_rep):
    """GQA KV head expansion: joy.repeat_kv(input) {n_rep}.

    Mirrors repeat_kv() in modeling_qwen3.py:
      [B, kv_heads, S, D] -> [B, kv_heads*n_rep, S, D]

    Equivalent to:
      hidden[:, :, None, :, :].expand(B, kv_heads, n_rep, S, D)
                              .reshape(B, kv_heads*n_rep, S, D)
    """
    shape = list(input_op.shape)
    result_shape = [shape[0], shape[1] * n_rep, shape[2], shape[3]]
    return input_op.graph._create_op(
        "joy.repeat_kv", [input_op], result_shape, input_op.dtype,
        {"n_rep": n_rep})
