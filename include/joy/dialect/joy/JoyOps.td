//===- JoyOps.td - Joy operation definitions --------------*- tablegen -*-===//
//
// Joy Compiler - Operations for Qwen 0.6B
//
//===----------------------------------------------------------------------===//

#ifndef JOY_OPS
#define JOY_OPS

include "JoyDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def Joy_EmbeddingOp: Joy_Op<"embedding", [Pure]> {
  let summary = "Joy embedding operator";
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight
  );
  let results = (outs AnyTensor:$output);
}

def Joy_AddOp: Joy_Op<"add", [Pure]> {
  let summary = "Joy add operator";
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  let results = (outs AnyTensor:$output);
}

def Joy_RMSNormOp: Joy_Op<"rms_norm", [Pure]> {
  let summary = "Joy RMSNorm operator";
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$scale,
    F32Attr:$epsilon
  );
  let results = (outs AnyTensor:$output);
}

def Joy_FuseAddRMSNormOp: Joy_Op<"fuse_add_rmsnorm", [Pure]> {
  let summary = "Fused residual-add + RMS normalization";
  let description = [{
    Fused residual add + RMS normalization. Equivalent to:
      add_result = lhs + rhs
      norm_result = rms_norm(add_result, scale, epsilon)
    Produces both the add and norm results so the add can still
    feed the next residual connection.
  }];
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    AnyTensor:$scale,
    F32Attr:$epsilon
  );
  let results = (outs AnyTensor:$add_result, AnyTensor:$norm_result);
}

def Joy_MulOp: Joy_Op<"mul", [Pure]> {
  let summary = "Joy element-wise multiply operator";
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  let results = (outs AnyTensor:$output);
}

def Joy_MatMulOp: Joy_Op<"matmul", [Pure]> {
  let summary = "Joy matrix multiply operator";
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  let results = (outs AnyTensor:$output);
}

def Joy_LinearOp: Joy_Op<"linear", [Pure]> {
  let summary = "Joy linear projection operator";
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight
  );
  let results = (outs AnyTensor:$output);
}

def Joy_SoftmaxOp: Joy_Op<"softmax", [Pure]> {
  let summary = "Joy softmax operator";
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$axis
  );
  let results = (outs AnyTensor:$output);
}

def Joy_ReshapeOp: Joy_Op<"reshape", [Pure]> {
  let summary = "Joy reshape operator";
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Joy_TransposeOp: Joy_Op<"transpose", [Pure]> {
  let summary = "Joy transpose operator";
  let arguments = (ins
    AnyTensor:$input,
    I64ElementsAttr:$permutation
  );
  let results = (outs AnyTensor:$output);
}

def Joy_SiLUOp: Joy_Op<"silu", [Pure]> {
  let summary = "Joy SiLU activation operator";
  let arguments = (ins
    AnyTensor:$input
  );
  let results = (outs AnyTensor:$output);
}

def Joy_ApplyRotaryEmbOp: Joy_Op<"apply_rotary_emb", [Pure]> {
  let summary = "Apply rotary positional embedding with pre-computed cos/sin";
  let description = [{
    Applies rotary positional embedding to a single query or key tensor.
    Mirrors apply_rotary_pos_emb() in modeling_qwen3.py:
      out = input * cos + rotate_half(input) * sin
    cos and sin are pre-computed by Qwen3RotaryEmbedding.
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$cos,
    AnyTensor:$sin
  );
  let results = (outs AnyTensor:$output);
}

def Joy_RepeatKVOp: Joy_Op<"repeat_kv", [Pure]> {
  let summary = "Repeat KV heads for grouped query attention (GQA)";
  let description = [{
    Expands key or value tensors from num_kv_heads to num_attention_heads.
    Mirrors repeat_kv() in modeling_qwen3.py:
      [B, kv_heads, S, D] -> [B, kv_heads*n_rep, S, D]
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$n_rep
  );
  let results = (outs AnyTensor:$output);
}

#endif // JOY_OPS
