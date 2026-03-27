"""Lookup / embedding / positional encoding operations for Joy dialect."""


def embedding(input_op, weight):
    """Token embedding: joy.embedding(input, weight).

    input: [B, S] int64, weight: [V, H] -> [B, S, H]
    """
    result_shape = list(input_op.shape) + [weight.shape[-1]]
    return input_op.graph._create_op(
        "joy.embedding", [input_op, weight], result_shape, weight.dtype)


def apply_rotary_emb(input_op, cos, sin):
    """Apply rotary positional embedding: joy.apply_rotary_emb(input, cos, sin).

    Mirrors apply_rotary_pos_emb() in modeling_qwen3.py:
      out = input * cos + rotate_half(input) * sin

    input: [B, H, S, D]
    cos/sin: [B, S, D] (broadcast over H via unsqueeze_dim=1)
    output: same shape as input
    """
    return input_op.graph._create_op(
        "joy.apply_rotary_emb", [input_op, cos, sin],
        list(input_op.shape), input_op.dtype)


def rotary_embedding(q, k, cos, sin):
    """Apply RoPE to Q and K separately (helper over apply_rotary_emb).

    Matches the call pattern in Qwen3Attention.forward():
      query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin)

    Returns (q_rotated, k_rotated).
    """
    q_rot = apply_rotary_emb(q, cos, sin)
    k_rot = apply_rotary_emb(k, cos, sin)
    return q_rot, k_rot


def gather(data, indices, axis=0):
    """Gather operation: joy.gather(data, indices) {axis}."""
    data_shape = list(data.shape)
    indices_shape = list(indices.shape)
    result_shape = data_shape[:axis] + indices_shape + data_shape[axis + 1:]
    return data.graph._create_op(
        "joy.gather", [data, indices], result_shape, data.dtype,
        {"axis": axis})
