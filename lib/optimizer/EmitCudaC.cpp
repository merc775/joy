//===- EmitCudaC.cpp - MLIR -> CUDA C source emitter ----------------------===//
//
// Walks every gpu_kernel-tagged func.func produced by CodegenRMSNormPass
// and emits an equivalent CUDA C/C++ translation unit.
//
// The emitter is intentionally NOT a fixed-string template: it walks the
// IR and translates each op into a C expression / statement.  Two
// constructs that don't have a one-to-one C equivalent get explicit
// special cases:
//
//   * outer scf.for over the row count is mapped to "one CUDA block per
//     row" (i.e. `int64_t i = blockIdx.x; if (i >= rows) return;`).  We
//     detect it by comparing its upper bound to memref.dim(input, 0).
//
//   * inner scf.for(with iter_args=zero) is mapped to a thread-strided
//     reduction over the column dimension, using a shared-memory tree
//     reduction.  The body is emitted inside the thread loop; whatever
//     value its scf.yield produces becomes the per-thread contribution.
//
//   * inner scf.for(no iter_args) is a thread-strided write loop --
//     each thread takes a stride of blockDim.x.
//
// All scalar f32 computation maps to `float`, all f16 to `__half`
// (with explicit `__half2float` / `__float2half` conversions when
// arith.extf / arith.truncf are encountered).
//
//===----------------------------------------------------------------------===//

#include "joy/optimizer/EmitCudaC.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace joy {

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static bool isF32(Type t) { return t.isF32(); }
static bool isF16(Type t) { return t.isF16(); }

/// Element-type tag used in launcher / kernel names: "f32" or "f16".
static std::string elementTag(Type t) {
  if (isF32(t)) return "f32";
  if (isF16(t)) return "f16";
  return "unknown";
}

/// Map an MLIR scalar type to its CUDA C scalar name used in
/// per-thread variables.  All compute is performed in f32; this is
/// only used for constants and load/store types.
static std::string ctypeOf(Type t) {
  if (t.isF32()) return "float";
  if (t.isF16()) return "__half";
  if (t.isF64()) return "double";
  if (t.isIndex() || t.isSignlessInteger(64)) return "int64_t";
  if (t.isSignlessInteger(32)) return "int32_t";
  return "auto";
}

//===----------------------------------------------------------------------===//
// FunctionEmitter
//
// Translates one `gpu_kernel` func.func into a CUDA `__global__`
// definition plus an `extern "C"` host launcher.
//===----------------------------------------------------------------------===//
class FunctionEmitter {
public:
  FunctionEmitter(func::FuncOp funcOp, llvm::raw_ostream &os)
      : funcOp_(funcOp), os_(os) {}

  LogicalResult emit();

private:
  // ---- name / SSA bookkeeping ----
  std::string nameFor(Value v) {
    auto it = names_.find(v);
    if (it != names_.end()) return it->second;
    std::string n = "v" + std::to_string(nextId_++);
    names_[v] = n;
    return n;
  }

  bool hasName(Value v) const { return names_.count(v) != 0; }

  void bindName(Value v, std::string name) { names_[v] = std::move(name); }

  // ---- emit helpers ----
  void emitLine(unsigned indent, const std::string &line) {
    for (unsigned i = 0; i < indent; ++i) os_ << "  ";
    os_ << line << "\n";
  }

  /// Emit a per-op assignment that produces `lhs = rhs` and binds the
  /// op's result to `lhs`.  Result type is inferred from the op result.
  void emitAssign(unsigned indent, Operation *op, const std::string &rhs) {
    Value res = op->getResult(0);
    std::string n = nameFor(res);
    std::string ty = ctypeOf(res.getType());
    emitLine(indent, ty + " " + n + " = " + rhs + ";");
  }

  // ---- IR translation ----

  /// Translate a region known to be either:
  ///   * straight-line (no scf), or
  ///   * a single scf.for inside a row-loop (reduce / write).
  /// `phase` describes whether we are inside the "reduction" or
  /// "normalize" inner loop, used to pick parallelization shape.
  ///
  /// Returns the per-thread "contribution" value name when we're
  /// inside a reduction loop, otherwise an empty string.
  std::string translateBlock(Block &block, unsigned indent);

  /// Special handling for the outer per-row scf.for.
  LogicalResult translateOuterRowLoop(scf::ForOp outer, unsigned indent);

  /// Translate an arithmetic / cast / math op into a C expression and
  /// bind the result name.
  LogicalResult translateScalarOp(Operation *op, unsigned indent);

  /// Translate a reduction inner scf.for(with iter_args) into a
  /// strided thread loop + shared-memory tree reduction.
  /// `outVar` receives the C name of the final reduced (per-block)
  /// scalar value.
  LogicalResult translateReductionLoop(scf::ForOp loop, unsigned indent,
                                        std::string &outVar);

  /// Translate an inner scf.for(no iter_args) into a strided-thread
  /// write loop.
  LogicalResult translateWriteLoop(scf::ForOp loop, unsigned indent);

  // ---- launcher ----
  void emitLauncher(StringRef baseName, StringRef dt);

private:
  func::FuncOp funcOp_;
  llvm::raw_ostream &os_;
  llvm::DenseMap<Value, std::string> names_;
  unsigned nextId_ = 0;

  // Names of the launcher-supplied scalars (from kernel arguments).
  std::string rowsName_ = "rows";
  std::string colsName_ = "cols";
  std::string epsilonName_ = "epsilon";
  // Names of memref arguments, in declaration order.
  llvm::SmallVector<std::string> memrefArgNames_;
  // Element type for the memref arguments (we currently expect a
  // single element type across all operands).
  Type elementType_;
  // Whether memref accesses need explicit __half2float / __float2half
  // wrappers (true for f16 element type).
  bool needsCast_ = false;
};

//===----------------------------------------------------------------------===//
// FunctionEmitter::emit
//===----------------------------------------------------------------------===//

LogicalResult FunctionEmitter::emit() {
  StringRef funcName = funcOp_.getName();
  if (funcName != "joy_rms_norm_kernel" &&
      funcName != "joy_fuse_add_rmsnorm_kernel") {
    return funcOp_.emitOpError(
        "unsupported gpu_kernel function name (expected "
        "joy_rms_norm_kernel or joy_fuse_add_rmsnorm_kernel)");
  }

  // ---- inspect signature ----
  FunctionType ft = funcOp_.getFunctionType();
  // The codegen pass guarantees:
  //   rms_norm:        (input2d, scale1d, output2d, eps)
  //   fuse_add_rmsnorm: (lhs2d, rhs2d, scale1d, addOut2d, normOut2d, eps)
  unsigned nMemref = 0;
  for (Type t : ft.getInputs()) {
    if (isa<MemRefType>(t)) ++nMemref;
  }
  unsigned expectedMemrefs = (funcName == "joy_rms_norm_kernel") ? 3 : 5;
  if (nMemref != expectedMemrefs) {
    return funcOp_.emitOpError(
        "unexpected number of memref operands for kernel ");
  }

  // First memref's element type drives the launcher tag.
  auto firstMemref = cast<MemRefType>(ft.getInput(0));
  elementType_ = firstMemref.getElementType();
  needsCast_ = isF16(elementType_);
  if (!isF32(elementType_) && !isF16(elementType_)) {
    return funcOp_.emitOpError("unsupported element type (expected f16/f32)");
  }
  std::string dt = elementTag(elementType_);

  // ---- compute kernel + launcher names ----
  std::string baseName =
      (funcName == "joy_rms_norm_kernel")
          ? std::string("joy_codegen_rms_norm_") + dt
          : std::string("joy_codegen_fuse_add_rms_norm_") + dt;
  std::string globalKernelName = baseName + "_kernel";

  // ---- bind names for kernel arguments ----
  Block &entry = funcOp_.getBody().front();
  // Memref args -> "input"/"scale"/"output" or "lhs"/"rhs"/etc.
  static const char *kRmsArgs[] = {"input", "scale", "output"};
  static const char *kFuseArgs[] = {"lhs", "rhs", "scale", "add_out",
                                    "norm_out"};
  const char **argNames =
      (funcName == "joy_rms_norm_kernel") ? kRmsArgs : kFuseArgs;
  for (unsigned i = 0; i < expectedMemrefs; ++i) {
    bindName(entry.getArgument(i), argNames[i]);
    memrefArgNames_.push_back(argNames[i]);
  }
  bindName(entry.getArgument(expectedMemrefs), epsilonName_);

  // ---- emit __global__ kernel signature ----
  std::string eltTy = ctypeOf(elementType_);
  std::string sig;
  llvm::raw_string_ostream so(sig);
  so << "__global__ void " << globalKernelName << "(";
  for (unsigned i = 0; i < expectedMemrefs; ++i) {
    so << (i == 0 ? "" : ", ");
    // outputs are mutable; everything is pointer to element type.
    bool isOutput = (funcName == "joy_rms_norm_kernel" && i == 2) ||
                    (funcName == "joy_fuse_add_rmsnorm_kernel" &&
                     (i == 3 || i == 4));
    so << (isOutput ? "" : "const ") << eltTy << " *" << argNames[i];
  }
  so << ", int64_t " << rowsName_ << ", int64_t " << colsName_
     << ", float " << epsilonName_ << ") ";
  os_ << "// ---- generated from MLIR func @" << funcName.str() << " ("
      << dt << ") ----\n";
  os_ << so.str() << "{\n";

  // ---- emit body ----
  // The body of the func (after CodegenRMSNormPass) consists of:
  //   c0 / c1 / zeroF32 constants
  //   rows = memref.dim(input, c0)
  //   cols = memref.dim(input, c1)
  //   colsI64 = arith.index_cast(cols)
  //   colsF32 = arith.sitofp(colsI64)
  //   scf.for outer (rows) { ... }
  //   func.return
  //
  // We treat constant + dim + index_cast + sitofp specially here because
  // their results need predictable C names (`rows`, `cols`, `colsF32`,
  // `0.0f`, etc.) used by the rest of the body.
  for (Operation &op : entry.getOperations()) {
    if (auto fOp = dyn_cast<func::ReturnOp>(op)) {
      // implicit fall-through; CUDA __global__ returns void.
      continue;
    }
    if (auto cstOp = dyn_cast<arith::ConstantOp>(op)) {
      // Track index 0/1 and zero-f32 constants without emitting them
      // (the body knows row index `i`, col index `j` and the constant
      // 0.0f directly).
      Attribute val = cstOp.getValue();
      if (auto ia = dyn_cast<IntegerAttr>(val)) {
        bindName(cstOp.getResult(),
                 std::to_string(ia.getInt()));
      } else if (auto fa = dyn_cast<FloatAttr>(val)) {
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << llvm::format("%.9gf", fa.getValueAsDouble());
        bindName(cstOp.getResult(), ss.str());
      } else {
        bindName(cstOp.getResult(), "0");
      }
      continue;
    }
    if (auto dimOp = dyn_cast<memref::DimOp>(op)) {
      // memref.dim(input, 0) -> rows, memref.dim(input, 1) -> cols.
      auto idxAttr = dimOp.getConstantIndex();
      if (!idxAttr.has_value()) {
        return op.emitOpError("non-constant dim index");
      }
      bindName(dimOp.getResult(), *idxAttr == 0 ? rowsName_ : colsName_);
      continue;
    }
    if (auto castOp = dyn_cast<arith::IndexCastOp>(op)) {
      bindName(castOp.getResult(), nameFor(castOp.getIn()));
      continue;
    }
    if (auto sitofp = dyn_cast<arith::SIToFPOp>(op)) {
      // expected: cols (int64) -> colsF32
      std::string in = nameFor(sitofp.getIn());
      bindName(sitofp.getResult(),
               std::string("(float)(") + in + ")");
      continue;
    }
    if (auto outer = dyn_cast<scf::ForOp>(op)) {
      if (failed(translateOuterRowLoop(outer, /*indent=*/1)))
        return failure();
      continue;
    }
    return op.emitOpError("unsupported top-level op in gpu_kernel function");
  }

  os_ << "}\n\n";

  // ---- emit launcher ----
  emitLauncher(baseName, dt);

  return success();
}

//===----------------------------------------------------------------------===//
// translateOuterRowLoop
//
// scf.for %i = c0 to %rows step c1 {
//   <reduce + post + write>
// }
//===----------------------------------------------------------------------===//
LogicalResult FunctionEmitter::translateOuterRowLoop(scf::ForOp outer,
                                                      unsigned indent) {
  emitLine(indent, "int64_t i = blockIdx.x;");
  emitLine(indent, "if (i >= " + rowsName_ + ") return;");
  emitLine(indent, "extern __shared__ float sdata[];");

  bindName(outer.getInductionVar(), "i");

  Block &body = outer.getRegion().front();
  for (Operation &op : body.getOperations()) {
    if (isa<scf::YieldOp>(op)) continue;

    if (auto inner = dyn_cast<scf::ForOp>(op)) {
      bool isReduce = !inner.getInitArgs().empty();
      if (isReduce) {
        std::string outName;
        if (failed(translateReductionLoop(inner, indent, outName)))
          return failure();
        // The result of a reduction loop is exactly its first iter result.
        bindName(inner.getResult(0), outName);
      } else {
        if (failed(translateWriteLoop(inner, indent)))
          return failure();
      }
      continue;
    }

    if (failed(translateScalarOp(&op, indent))) return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// translateScalarOp
//===----------------------------------------------------------------------===//
LogicalResult FunctionEmitter::translateScalarOp(Operation *op,
                                                  unsigned indent) {
  // arith.constant inside loop bodies (e.g. zeroF32 inherited from
  // outer scope -- but constants are usually hoisted, so we still
  // handle them).
  if (auto cstOp = dyn_cast<arith::ConstantOp>(op)) {
    Attribute val = cstOp.getValue();
    if (auto fa = dyn_cast<FloatAttr>(val)) {
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << llvm::format("%.9gf", fa.getValueAsDouble());
      bindName(cstOp.getResult(), ss.str());
    } else if (auto ia = dyn_cast<IntegerAttr>(val)) {
      bindName(cstOp.getResult(), std::to_string(ia.getInt()));
    } else {
      bindName(cstOp.getResult(), "0");
    }
    return success();
  }
  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    Value mem = load.getMemref();
    if (!hasName(mem)) {
      return op->emitOpError("memref operand has no bound name");
    }
    std::string memName = nameFor(mem);
    auto indices = load.getIndices();
    std::string offset;
    if (indices.size() == 1) {
      offset = nameFor(indices[0]);
    } else if (indices.size() == 2) {
      offset = nameFor(indices[0]) + " * " + colsName_ + " + " +
               nameFor(indices[1]);
    } else {
      return op->emitOpError("unsupported memref rank");
    }
    std::string raw = memName + "[" + offset + "]";
    if (needsCast_) {
      // f16 -> implicit f32 promotion happens via arith.extf, so
      // here we keep the raw element type.
      emitAssign(indent, op, raw);
    } else {
      emitAssign(indent, op, raw);
    }
    return success();
  }
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    Value mem = store.getMemref();
    Value val = store.getValueToStore();
    std::string memName = nameFor(mem);
    auto indices = store.getIndices();
    std::string offset;
    if (indices.size() == 1) {
      offset = nameFor(indices[0]);
    } else if (indices.size() == 2) {
      offset = nameFor(indices[0]) + " * " + colsName_ + " + " +
               nameFor(indices[1]);
    } else {
      return op->emitOpError("unsupported memref rank");
    }
    std::string rhs = nameFor(val);
    emitLine(indent, memName + "[" + offset + "] = " + rhs + ";");
    return success();
  }
  if (auto extf = dyn_cast<arith::ExtFOp>(op)) {
    std::string in = nameFor(extf.getIn());
    Type srcTy = extf.getIn().getType();
    if (isF16(srcTy)) {
      emitAssign(indent, op, "__half2float(" + in + ")");
    } else {
      emitAssign(indent, op, "(float)(" + in + ")");
    }
    return success();
  }
  if (auto truncf = dyn_cast<arith::TruncFOp>(op)) {
    std::string in = nameFor(truncf.getIn());
    Type dstTy = truncf.getResult().getType();
    if (isF16(dstTy)) {
      emitAssign(indent, op, "__float2half(" + in + ")");
    } else {
      emitAssign(indent, op,
                 "(" + ctypeOf(dstTy) + ")(" + in + ")");
    }
    return success();
  }
  if (auto add = dyn_cast<arith::AddFOp>(op)) {
    emitAssign(indent, op, nameFor(add.getLhs()) + " + " + nameFor(add.getRhs()));
    return success();
  }
  if (auto mul = dyn_cast<arith::MulFOp>(op)) {
    emitAssign(indent, op, nameFor(mul.getLhs()) + " * " + nameFor(mul.getRhs()));
    return success();
  }
  if (auto div = dyn_cast<arith::DivFOp>(op)) {
    emitAssign(indent, op, nameFor(div.getLhs()) + " / " + nameFor(div.getRhs()));
    return success();
  }
  if (auto rsq = dyn_cast<math::RsqrtOp>(op)) {
    emitAssign(indent, op, "rsqrtf(" + nameFor(rsq.getOperand()) + ")");
    return success();
  }
  if (auto sitofp = dyn_cast<arith::SIToFPOp>(op)) {
    emitAssign(indent, op, "(float)(" + nameFor(sitofp.getIn()) + ")");
    return success();
  }
  if (auto idxCast = dyn_cast<arith::IndexCastOp>(op)) {
    bindName(idxCast.getResult(), nameFor(idxCast.getIn()));
    return success();
  }
  if (auto dimOp = dyn_cast<memref::DimOp>(op)) {
    auto idxAttr = dimOp.getConstantIndex();
    if (!idxAttr.has_value()) {
      return op->emitOpError("non-constant dim index");
    }
    bindName(dimOp.getResult(),
             *idxAttr == 0 ? rowsName_ : colsName_);
    return success();
  }
  return op->emitOpError("unsupported op in gpu_kernel body");
}

//===----------------------------------------------------------------------===//
// translateReductionLoop
//
// scf.for %j = c0 to %cols step c1 iter_args(%acc = %zeroF32) -> f32 {
//   <body that produces %newAcc via arith.addf(%acc, %sq)>
//   scf.yield %newAcc
// }
//
// Lowered to:
//
//   float __acc = 0.f;
//   for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
//     <body without the final addf(acc, sq)>
//     __acc += sq;        // sq is whatever was being added to acc
//   }
//   sdata[threadIdx.x] = __acc;
//   __syncthreads();
//   for (int s = blockDim.x/2; s > 0; s >>= 1) {
//     if ((int)threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
//     __syncthreads();
//   }
//   float <out> = sdata[0];
//===----------------------------------------------------------------------===//
LogicalResult FunctionEmitter::translateReductionLoop(scf::ForOp loop,
                                                      unsigned indent,
                                                      std::string &outVar) {
  if (loop.getNumRegionIterArgs() != 1) {
    return loop.emitOpError("only single-iter_args reduction supported");
  }

  std::string accName = "acc" + std::to_string(nextId_++);
  emitLine(indent, "float " + accName + " = 0.0f;");

  // Bind the iter-arg to accName so any uses inside body refer to it.
  Value iterArg = loop.getRegionIterArg(0);
  bindName(iterArg, accName);
  bindName(loop.getInductionVar(), "j");

  emitLine(indent,
           "for (int64_t j = threadIdx.x; j < " + colsName_ +
               "; j += blockDim.x) {");

  // The codegen pass produces one of two shapes:
  //   shape A (rms_norm reduce):
  //     val   = load(input, [i, j])   // possibly with extf
  //     sq    = mulf(val, val)
  //     newAcc= addf(acc, sq)
  //     yield newAcc
  //
  //   shape B (fuse_add_rmsnorm reduce):
  //     a     = load(lhs, [i, j])
  //     b     = load(rhs, [i, j])
  //     t     = addf(a, b)
  //     store(addOut, [i, j], t/possibly truncf)
  //     sq    = mulf(t, t)
  //     newAcc= addf(acc, sq)
  //     yield newAcc
  //
  // We translate every body op normally except the final
  // arith.addf(acc, sq) + scf.yield -- those become "acc += sq;".
  Block &body = loop.getRegion().front();
  // Identify the trailing addf / yield pair.
  Operation *yieldOp = body.getTerminator();
  if (!isa<scf::YieldOp>(yieldOp)) {
    return loop.emitOpError("reduction body must end with scf.yield");
  }
  Value yielded = yieldOp->getOperand(0);
  auto trailingAdd = yielded.getDefiningOp<arith::AddFOp>();
  if (!trailingAdd) {
    return loop.emitOpError(
        "reduction body must yield result of arith.addf(acc, sq)");
  }
  // Confirm one of its operands is the iter_arg.
  Value rhsContrib;
  if (trailingAdd.getLhs() == iterArg) {
    rhsContrib = trailingAdd.getRhs();
  } else if (trailingAdd.getRhs() == iterArg) {
    rhsContrib = trailingAdd.getLhs();
  } else {
    return loop.emitOpError(
        "trailing addf does not reference iter_arg");
  }

  for (Operation &op : body.getOperations()) {
    if (&op == trailingAdd || &op == yieldOp) continue;
    if (failed(translateScalarOp(&op, indent + 1))) return failure();
  }
  emitLine(indent + 1, accName + " = " + accName + " + " +
                          nameFor(rhsContrib) + ";");
  emitLine(indent, "}");

  // Shared-memory tree reduction.
  emitLine(indent, "sdata[threadIdx.x] = " + accName + ";");
  emitLine(indent, "__syncthreads();");
  emitLine(indent, "for (int s = blockDim.x / 2; s > 0; s >>= 1) {");
  emitLine(indent + 1,
           "if ((int)threadIdx.x < s) sdata[threadIdx.x] += "
           "sdata[threadIdx.x + s];");
  emitLine(indent + 1, "__syncthreads();");
  emitLine(indent, "}");

  std::string sumName = "sum" + std::to_string(nextId_++);
  emitLine(indent, "float " + sumName + " = sdata[0];");
  outVar = sumName;
  return success();
}

//===----------------------------------------------------------------------===//
// translateWriteLoop
//
// scf.for %j = c0 to %cols step c1 {
//   <body with memref.store(out, [i, j], ...)>
// }
//===----------------------------------------------------------------------===//
LogicalResult FunctionEmitter::translateWriteLoop(scf::ForOp loop,
                                                   unsigned indent) {
  bindName(loop.getInductionVar(), "j");

  emitLine(indent,
           "for (int64_t j = threadIdx.x; j < " + colsName_ +
               "; j += blockDim.x) {");

  Block &body = loop.getRegion().front();
  for (Operation &op : body.getOperations()) {
    if (isa<scf::YieldOp>(op)) continue;
    if (failed(translateScalarOp(&op, indent + 1))) return failure();
  }
  emitLine(indent, "}");
  return success();
}

//===----------------------------------------------------------------------===//
// emitLauncher
//
// Emits an `extern "C"` host wrapper that invokes the __global__
// kernel.  Two flavors, keyed on the kernel base name.
//===----------------------------------------------------------------------===//
void FunctionEmitter::emitLauncher(StringRef baseName, StringRef dt) {
  std::string eltTy = ctypeOf(elementType_);
  std::string kernelName = std::string(baseName) + "_kernel";

  os_ << "extern \"C\" void " << baseName << "(";
  if (funcOp_.getName() == "joy_rms_norm_kernel") {
    os_ << "const " << eltTy << " *input, const " << eltTy
        << " *scale, " << eltTy << " *output, "
        << "int64_t rows, int64_t cols, float epsilon, void *stream) {\n";
    os_ << "  if (rows <= 0 || cols <= 0) return;\n";
    os_ << "  constexpr int kBlockSize = 256;\n";
    os_ << "  int grid = (rows < 65535) ? (int)rows : 65535;\n";
    os_ << "  size_t shm = sizeof(float) * kBlockSize;\n";
    os_ << "  " << kernelName
        << "<<<grid, kBlockSize, shm, "
           "reinterpret_cast<cudaStream_t>(stream)>>>(\n"
        << "      input, scale, output, rows, cols, epsilon);\n";
    os_ << "}\n\n";
  } else { // fuse_add_rmsnorm
    os_ << "const " << eltTy << " *lhs, const " << eltTy
        << " *rhs, const " << eltTy << " *scale, " << eltTy
        << " *add_out, " << eltTy << " *norm_out, "
        << "int64_t rows, int64_t cols, float epsilon, void *stream) {\n";
    os_ << "  if (rows <= 0 || cols <= 0) return;\n";
    os_ << "  constexpr int kBlockSize = 256;\n";
    os_ << "  int grid = (rows < 65535) ? (int)rows : 65535;\n";
    os_ << "  size_t shm = sizeof(float) * kBlockSize;\n";
    os_ << "  " << kernelName
        << "<<<grid, kBlockSize, shm, "
           "reinterpret_cast<cudaStream_t>(stream)>>>(\n"
        << "      lhs, rhs, scale, add_out, norm_out, rows, cols, "
           "epsilon);\n";
    os_ << "}\n\n";
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Public entry point
//===----------------------------------------------------------------------===//

LogicalResult emitCudaC(ModuleOp module, llvm::raw_ostream &os,
                         llvm::StringRef headerComment) {
  if (!headerComment.empty()) os << headerComment << "\n";
  os << "// THIS FILE IS AUTO-GENERATED FROM MLIR.\n";
  os << "// Source: CodegenRMSNormPass output -> joy::emitCudaC.\n";
  os << "// Do not edit by hand; rerun scripts/regen_codegen_kernel.sh.\n";
  os << "\n";
  os << "#include <cuda_fp16.h>\n";
  os << "#include <cuda_runtime.h>\n";
  os << "#include <cstdint>\n";
  os << "\n";

  bool found = false;
  LogicalResult result = success();
  module.walk([&](func::FuncOp funcOp) {
    if (!funcOp->hasAttr("gpu_kernel")) return;
    found = true;
    FunctionEmitter emitter(funcOp, os);
    if (failed(emitter.emit())) result = failure();
  });

  if (!found) {
    return module.emitError(
        "no func.func with `gpu_kernel` attribute found in module");
  }
  return result;
}

} // namespace joy
