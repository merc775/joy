//===- JoylDialect.td - Joyl dialect definition ------------*- tablegen -*-===//
//
// Joy Compiler - Low-level memory-based operations dialect
//
//===----------------------------------------------------------------------===//

#ifndef JOYL_DIALECT
#define JOYL_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Joyl Dialect Definition
//===----------------------------------------------------------------------===//

def Joyl_Dialect : Dialect {
  let name = "joyl";
  let summary = "Joyl low-level memory operations dialect";
  let description = [{
    The Joyl dialect contains low-level memory-based operations.
  }];
  let cppNamespace = "::mlir::joyl";
  let usePropertiesForAttributes = 0;
}

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

def JOYL_Buffer : MemRefOf<[AnyFloat, AnyInteger, AnyComplex, Index]>;

//===----------------------------------------------------------------------===//
// Joyl Operation Base Class
//===----------------------------------------------------------------------===//

class Joyl_Op<string mnemonic, list<Trait> traits = []> :
    Op<Joyl_Dialect, mnemonic, 
       !listconcat([MemoryEffects<[MemRead, MemWrite]>], traits)>;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def Joyl_EmbeddingOp : Joyl_Op<"embedding", []> {
  let summary = "Joyl embedding operator";
  let arguments = (ins
    Arg<JOYL_Buffer, "", [MemRead]>:$input,
    Arg<JOYL_Buffer, "", [MemRead]>:$weight,
    Arg<JOYL_Buffer, "", [MemWrite]>:$output
  );
}

def Joyl_AddOp : Joyl_Op<"add", []> {
  let summary = "Joyl add operator";
  let arguments = (ins
    Arg<JOYL_Buffer, "", [MemRead]>:$lhs,
    Arg<JOYL_Buffer, "", [MemRead]>:$rhs,
    Arg<JOYL_Buffer, "", [MemWrite]>:$output
  );
}

def Joyl_RMSNormOp : Joyl_Op<"rms_norm", []> {
  let summary = "Joyl RMSNorm operator";
  let arguments = (ins
    Arg<JOYL_Buffer, "", [MemRead]>:$input,
    Arg<JOYL_Buffer, "", [MemRead]>:$scale,
    Arg<JOYL_Buffer, "", [MemWrite]>:$output,
    F32Attr:$epsilon
  );
}

def Joyl_FuseAddRMSNormOp : Joyl_Op<"fuse_add_rmsnorm", []> {
  let summary = "Fused residual-add + RMS normalization";
  let arguments = (ins
    Arg<JOYL_Buffer, "", [MemRead]>:$lhs,
    Arg<JOYL_Buffer, "", [MemRead]>:$rhs,
    Arg<JOYL_Buffer, "", [MemRead]>:$scale,
    Arg<JOYL_Buffer, "", [MemWrite]>:$add_output,
    Arg<JOYL_Buffer, "", [MemWrite]>:$norm_output,
    F32Attr:$epsilon
  );
}

def Joyl_MulOp : Joyl_Op<"mul", []> {
  let summary = "Joyl element-wise multiply operator";
  let arguments = (ins
    Arg<JOYL_Buffer, "", [MemRead]>:$lhs,
    Arg<JOYL_Buffer, "", [MemRead]>:$rhs,
    Arg<JOYL_Buffer, "", [MemWrite]>:$output
  );
}

def Joyl_MatMulOp : Joyl_Op<"matmul", []> {
  let summary = "Joyl matrix multiply operator";
  let arguments = (ins
    Arg<JOYL_Buffer, "", [MemRead]>:$lhs,
    Arg<JOYL_Buffer, "", [MemRead]>:$rhs,
    Arg<JOYL_Buffer, "", [MemWrite]>:$output
  );
}

def Joyl_LinearOp : Joyl_Op<"linear", []> {
  let summary = "Joyl linear projection operator";
  let arguments = (ins
    Arg<JOYL_Buffer, "", [MemRead]>:$input,
    Arg<JOYL_Buffer, "", [MemRead]>:$weight,
    Arg<JOYL_Buffer, "", [MemWrite]>:$output
  );
}

def Joyl_SoftmaxOp : Joyl_Op<"softmax", []> {
  let summary = "Joyl softmax operator";
  let arguments = (ins
    Arg<JOYL_Buffer, "", [MemRead]>:$input,
    Arg<JOYL_Buffer, "", [MemWrite]>:$output,
    I64Attr:$axis
  );
}

def Joyl_SiLUOp : Joyl_Op<"silu", []> {
  let summary = "Joyl SiLU activation operator";
  let arguments = (ins
    Arg<JOYL_Buffer, "", [MemRead]>:$input,
    Arg<JOYL_Buffer, "", [MemWrite]>:$output
  );
}

def Joyl_ApplyRotaryEmbOp : Joyl_Op<"apply_rotary_emb", []> {
  let summary = "Joyl apply rotary positional embedding with pre-computed cos/sin";
  let arguments = (ins
    Arg<JOYL_Buffer, "", [MemRead]>:$input,
    Arg<JOYL_Buffer, "", [MemRead]>:$cos,
    Arg<JOYL_Buffer, "", [MemRead]>:$sin,
    Arg<JOYL_Buffer, "", [MemWrite]>:$output
  );
}

def Joyl_RepeatKVOp : Joyl_Op<"repeat_kv", []> {
  let summary = "Joyl GQA KV head repetition";
  let arguments = (ins
    Arg<JOYL_Buffer, "", [MemRead]>:$input,
    Arg<JOYL_Buffer, "", [MemWrite]>:$output,
    I64Attr:$n_rep
  );
}

def Joyl_ReshapeOp : Joyl_Op<"reshape", []> {
  let summary = "Joyl reshape operator";
  let arguments = (ins
    Arg<JOYL_Buffer, "", [MemRead]>:$input,
    Arg<JOYL_Buffer, "", [MemWrite]>:$output
  );
}

def Joyl_TransposeOp : Joyl_Op<"transpose", []> {
  let summary = "Joyl transpose operator";
  let arguments = (ins
    Arg<JOYL_Buffer, "", [MemRead]>:$input,
    Arg<JOYL_Buffer, "", [MemWrite]>:$output,
    I64ElementsAttr:$permutation
  );
}

#endif // JOYL_DIALECT
