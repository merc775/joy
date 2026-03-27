//===- JoyhOps.td - Joyh operation definitions -------------*- tablegen -*-===//
//
// Joy Compiler - HAL operations for GPU backend
//
// Joy Compiler - HAL custom call operations
//
//===----------------------------------------------------------------------===//

#ifndef JOYH_OPS
#define JOYH_OPS

include "JoyhDialect.td"

//===----------------------------------------------------------------------===//
// CustomCallOp — unified entry point for GPU kernel dispatch
//
// CustomCallOp layout:
//   - All memref operands (inputs first, then outputs) in a flat list
//   - call_target_name: the GPU kernel entry point (e.g. "joy_gpu_embedding")
//   - backend: target device ("gpu")
//   - num_inputs: how many leading operands are inputs (rest are outputs)
//   - Extra attributes from the original op (epsilon, axis, etc.) are
//     preserved as additional named attributes on this op.
//===----------------------------------------------------------------------===//

def Joyh_CustomCallOp : Joyh_Op<"custom_call",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "GPU custom call operation";
  let description = [{
    Represents a call to a GPU backend operator.
    Operands are memref buffers: the first `num_inputs` are read-only inputs,
    the remaining are write-only outputs.
    The `call_target_name` identifies the GPU kernel entry point
    (e.g. "joy_gpu_embedding", "joy_gpu_rms_norm").
  }];
  let arguments = (ins
    Variadic<AnyMemRef>:$operands,
    StrAttr:$call_target_name,
    StrAttr:$backend,
    I64Attr:$num_inputs
  );
}

#endif // JOYH_OPS
