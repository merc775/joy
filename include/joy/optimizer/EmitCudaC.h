//===- EmitCudaC.h - MLIR -> CUDA C source emitter ------------------------===//
//
// Joy Compiler - "EmitC-style" translation from MLIR func.func bodies
// produced by CodegenRMSNormPass into a self-contained CUDA C/C++ source
// file.
//
// The emitter walks the IR (it does NOT pattern-match a fixed text
// template).  It supports the subset of ops the codegen pass actually
// generates today:
//
//   * func.func / func.return
//   * scf.for (with and without iter_args; the iter_args case is
//     mapped to a thread-strided + shared-memory reduction)
//   * memref.dim / memref.load / memref.store
//   * arith.constant / addf / mulf / divf / sitofp / extf / truncf /
//     index_cast
//   * math.rsqrt
//
// The output is one self-contained translation unit per call.  See
// joy/lib/optimizer/EmitCudaC.cpp for the precise launch wrapper layout.
//
//===----------------------------------------------------------------------===//

#ifndef JOY_OPTIMIZER_EMITCUDAC_H
#define JOY_OPTIMIZER_EMITCUDAC_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace joy {

/// Emit a CUDA C/C++ source file derived from every `func.func`
/// in `module` that carries the unit attribute "gpu_kernel".
///
/// The function name and element type read from the func signature
/// determine the launcher that gets emitted.  Currently supported
/// kernel names (case-sensitive):
///
///   * `joy_rms_norm_kernel`           ->  `joy_codegen_rms_norm_<dt>`
///   * `joy_fuse_add_rmsnorm_kernel`   ->  `joy_codegen_fuse_add_rms_norm_<dt>`
///
/// where `<dt>` is `f32` or `f16` depending on the element type of the
/// kernel's first memref argument.
///
/// `os` receives the generated CUDA source.  `headerComment` is
/// prepended verbatim (use it to embed a "do not edit" banner with
/// the source MLIR file path).
mlir::LogicalResult emitCudaC(mlir::ModuleOp module,
                              llvm::raw_ostream &os,
                              llvm::StringRef headerComment = "");

} // namespace joy

#endif // JOY_OPTIMIZER_EMITCUDAC_H
