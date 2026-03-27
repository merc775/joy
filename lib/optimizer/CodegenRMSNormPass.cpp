//===- CodegenRMSNormPass.cpp - RMS Norm GPU kernel codegen ---------------===//
//
// Joy Compiler - Generate standalone GPU kernels for normalization ops
//
// Handles two operations:
//
//   1. joyl.rms_norm → @joy_rms_norm_kernel
//      func.func private @joy_rms_norm_kernel(
//          %input: memref<?x?xELT>, %scale: memref<?xELT>,
//          %output: memref<?x?xELT>, %epsilon: f32)
//
//   2. joyl.fuse_add_rmsnorm → @joy_fuse_add_rmsnorm_kernel
//      func.func private @joy_fuse_add_rmsnorm_kernel(
//          %lhs: memref<?x?xELT>, %rhs: memref<?x?xELT>,
//          %scale: memref<?xELT>,
//          %add_out: memref<?x?xELT>, %norm_out: memref<?x?xELT>,
//          %epsilon: f32)
//
// At call sites, multi-dim memrefs are collapsed to 2D via
// memref.collapse_shape (all dims except last → row, last → col).
//
//===----------------------------------------------------------------------===//

#include "joy/dialect/joyl/JoylDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

static constexpr const char *kRMSNormKernel = "joy_rms_norm_kernel";
static constexpr const char *kFuseAddRMSNormKernel =
    "joy_fuse_add_rmsnorm_kernel";

// =========================================================================
// Kernel builders
// =========================================================================

/// Build the standalone RMS norm kernel.
/// All computation is done in f32; f16 values are extended/truncated.
static func::FuncOp createRMSNormKernel(ModuleOp module, OpBuilder &builder,
                                         Type eltType) {
  auto *ctx = builder.getContext();
  auto loc = builder.getUnknownLoc();
  auto f32Type = builder.getF32Type();

  auto dyn2d = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, eltType);
  auto dyn1d = MemRefType::get({ShapedType::kDynamic}, eltType);
  auto funcType =
      FunctionType::get(ctx, {dyn2d, dyn1d, dyn2d, f32Type}, {});

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto funcOp = builder.create<func::FuncOp>(loc, kRMSNormKernel, funcType);
  funcOp.setVisibility(SymbolTable::Visibility::Private);
  funcOp->setAttr("gpu_kernel", builder.getUnitAttr());
  funcOp->setAttr("kernel_name", builder.getStringAttr("rms_norm"));

  auto *entry = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  Value input = entry->getArgument(0);
  Value scale = entry->getArgument(1);
  Value output = entry->getArgument(2);
  Value epsilon = entry->getArgument(3);
  bool needsCast = (eltType != f32Type);

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value zeroF32 =
      builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(0.0f));
  Value rows = builder.create<memref::DimOp>(loc, input, c0);
  Value cols = builder.create<memref::DimOp>(loc, input, c1);
  Value colsI64 =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), cols);
  Value colsF32 = builder.create<arith::SIToFPOp>(loc, f32Type, colsI64);

  auto loadF32 = [&](OpBuilder &b, Location loc, Value memref,
                      ValueRange indices) -> Value {
    Value v = b.create<memref::LoadOp>(loc, memref, indices);
    if (needsCast)
      v = b.create<arith::ExtFOp>(loc, f32Type, v);
    return v;
  };

  auto storeFromF32 = [&](OpBuilder &b, Location loc, Value val,
                           Value memref, ValueRange indices) {
    Value v = val;
    if (needsCast)
      v = b.create<arith::TruncFOp>(loc, eltType, v);
    b.create<memref::StoreOp>(loc, v, memref, indices);
  };

  builder.create<scf::ForOp>(
      loc, c0, rows, c1, ValueRange{},
      [&](OpBuilder &ob, Location loc, Value i, ValueRange) {
        auto sumLoop = ob.create<scf::ForOp>(
            loc, c0, cols, c1, ValueRange{zeroF32},
            [&](OpBuilder &ib, Location loc, Value j, ValueRange iterArgs) {
              Value acc = iterArgs[0];
              Value val = loadF32(ib, loc, input, {i, j});
              Value sq = ib.create<arith::MulFOp>(loc, val, val);
              Value newAcc = ib.create<arith::AddFOp>(loc, acc, sq);
              ib.create<scf::YieldOp>(loc, ValueRange{newAcc});
            });

        Value sumSq = sumLoop.getResult(0);
        Value meanSq = ob.create<arith::DivFOp>(loc, sumSq, colsF32);
        Value varEps = ob.create<arith::AddFOp>(loc, meanSq, epsilon);
        Value rrms = ob.create<math::RsqrtOp>(loc, varEps);

        ob.create<scf::ForOp>(
            loc, c0, cols, c1, ValueRange{},
            [&](OpBuilder &ib, Location loc, Value j, ValueRange) {
              Value val = loadF32(ib, loc, input, {i, j});
              Value s = loadF32(ib, loc, scale, {j});
              Value normed = ib.create<arith::MulFOp>(loc, val, rrms);
              Value result = ib.create<arith::MulFOp>(loc, normed, s);
              storeFromF32(ib, loc, result, output, {i, j});
              ib.create<scf::YieldOp>(loc);
            });

        ob.create<scf::YieldOp>(loc);
      });

  builder.create<func::ReturnOp>(loc);
  return funcOp;
}

/// Build the fused add + RMS norm kernel.
///
/// For each row i:
///   Phase 1: add + accumulate squares
///     for j in 0..cols:
///       t = lhs[i,j] + rhs[i,j]
///       add_out[i,j] = t
///       sum_sq += t*t
///   Phase 2: rrms = rsqrt(sum_sq/cols + epsilon)
///   Phase 3: normalize and scale
///     for j in 0..cols:
///       norm_out[i,j] = add_out[i,j] * rrms * scale[j]
static func::FuncOp createFuseAddRMSNormKernel(ModuleOp module,
                                                 OpBuilder &builder,
                                                 Type eltType) {
  auto *ctx = builder.getContext();
  auto loc = builder.getUnknownLoc();
  auto f32Type = builder.getF32Type();

  auto dyn2d = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, eltType);
  auto dyn1d = MemRefType::get({ShapedType::kDynamic}, eltType);

  // (lhs, rhs, scale, add_out, norm_out, epsilon)
  auto funcType = FunctionType::get(
      ctx, {dyn2d, dyn2d, dyn1d, dyn2d, dyn2d, f32Type}, {});

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto funcOp =
      builder.create<func::FuncOp>(loc, kFuseAddRMSNormKernel, funcType);
  funcOp.setVisibility(SymbolTable::Visibility::Private);
  funcOp->setAttr("gpu_kernel", builder.getUnitAttr());
  funcOp->setAttr("kernel_name",
                   builder.getStringAttr("fuse_add_rmsnorm"));

  auto *entry = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  Value lhs = entry->getArgument(0);
  Value rhs = entry->getArgument(1);
  Value scale = entry->getArgument(2);
  Value addOut = entry->getArgument(3);
  Value normOut = entry->getArgument(4);
  Value epsilon = entry->getArgument(5);
  bool needsCast = (eltType != f32Type);

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value zeroF32 =
      builder.create<arith::ConstantOp>(loc, builder.getF32FloatAttr(0.0f));
  Value rows = builder.create<memref::DimOp>(loc, lhs, c0);
  Value cols = builder.create<memref::DimOp>(loc, lhs, c1);
  Value colsI64 =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), cols);
  Value colsF32 = builder.create<arith::SIToFPOp>(loc, f32Type, colsI64);

  auto loadF32 = [&](OpBuilder &b, Location loc, Value memref,
                      ValueRange indices) -> Value {
    Value v = b.create<memref::LoadOp>(loc, memref, indices);
    if (needsCast)
      v = b.create<arith::ExtFOp>(loc, f32Type, v);
    return v;
  };

  auto storeFromF32 = [&](OpBuilder &b, Location loc, Value val,
                           Value memref, ValueRange indices) {
    Value v = val;
    if (needsCast)
      v = b.create<arith::TruncFOp>(loc, eltType, v);
    b.create<memref::StoreOp>(loc, v, memref, indices);
  };

  builder.create<scf::ForOp>(
      loc, c0, rows, c1, ValueRange{},
      [&](OpBuilder &ob, Location loc, Value i, ValueRange) {
        // Phase 1: element-wise add + accumulate sum of squares
        auto sumLoop = ob.create<scf::ForOp>(
            loc, c0, cols, c1, ValueRange{zeroF32},
            [&](OpBuilder &ib, Location loc, Value j, ValueRange iterArgs) {
              Value acc = iterArgs[0];
              Value a = loadF32(ib, loc, lhs, {i, j});
              Value b = loadF32(ib, loc, rhs, {i, j});
              Value t = ib.create<arith::AddFOp>(loc, a, b);
              storeFromF32(ib, loc, t, addOut, {i, j});
              Value sq = ib.create<arith::MulFOp>(loc, t, t);
              Value newAcc = ib.create<arith::AddFOp>(loc, acc, sq);
              ib.create<scf::YieldOp>(loc, ValueRange{newAcc});
            });

        Value sumSq = sumLoop.getResult(0);

        // Phase 2: rrms = rsqrt(mean_sq + epsilon)
        Value meanSq = ob.create<arith::DivFOp>(loc, sumSq, colsF32);
        Value varEps = ob.create<arith::AddFOp>(loc, meanSq, epsilon);
        Value rrms = ob.create<math::RsqrtOp>(loc, varEps);

        // Phase 3: normalize, scale, and store
        ob.create<scf::ForOp>(
            loc, c0, cols, c1, ValueRange{},
            [&](OpBuilder &ib, Location loc, Value j, ValueRange) {
              Value val = loadF32(ib, loc, addOut, {i, j});
              Value s = loadF32(ib, loc, scale, {j});
              Value normed = ib.create<arith::MulFOp>(loc, val, rrms);
              Value result = ib.create<arith::MulFOp>(loc, normed, s);
              storeFromF32(ib, loc, result, normOut, {i, j});
              ib.create<scf::YieldOp>(loc);
            });

        ob.create<scf::YieldOp>(loc);
      });

  builder.create<func::ReturnOp>(loc);
  return funcOp;
}

// =========================================================================
// Call-site replacement helpers
// =========================================================================

/// Collapse an N-D memref to 2D (all dims except last → row, last → col)
/// and cast to a fully dynamic 2D memref.
static Value collapseAndCast2D(OpBuilder &builder, Location loc, Value memref,
                                MemRefType dyn2d) {
  auto inputType = cast<MemRefType>(memref.getType());
  int rank = inputType.getRank();
  if (rank == 2) {
    return builder.create<memref::CastOp>(loc, dyn2d, memref);
  }
  SmallVector<ReassociationIndices> reassoc(2);
  for (int d = 0; d < rank - 1; ++d)
    reassoc[0].push_back(d);
  reassoc[1].push_back(rank - 1);
  Value collapsed =
      builder.create<memref::CollapseShapeOp>(loc, memref, reassoc);
  return builder.create<memref::CastOp>(loc, dyn2d, collapsed);
}

/// Replace a joyl.rms_norm with a call to @joy_rms_norm_kernel.
static void replaceRMSNormWithCall(Operation *op, OpBuilder &builder,
                                    Type eltType) {
  auto loc = op->getLoc();
  builder.setInsertionPoint(op);

  Value input = op->getOperand(0);
  Value scale = op->getOperand(1);
  Value output = op->getOperand(2);
  auto epsilonAttr = op->getAttrOfType<FloatAttr>("epsilon");

  auto dyn2d = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, eltType);
  auto dyn1d = MemRefType::get({ShapedType::kDynamic}, eltType);

  Value inputDyn = collapseAndCast2D(builder, loc, input, dyn2d);
  Value outputDyn = collapseAndCast2D(builder, loc, output, dyn2d);
  Value scaleDyn = builder.create<memref::CastOp>(loc, dyn1d, scale);
  Value eps = builder.create<arith::ConstantOp>(loc, epsilonAttr);

  builder.create<func::CallOp>(
      loc, kRMSNormKernel, TypeRange{},
      ValueRange{inputDyn, scaleDyn, outputDyn, eps});

  op->erase();
}

/// Replace a joyl.fuse_add_rmsnorm with a call to
/// @joy_fuse_add_rmsnorm_kernel.
///
/// Operands: lhs(0), rhs(1), scale(2), add_out(3), norm_out(4)
static void replaceFuseAddRMSNormWithCall(Operation *op, OpBuilder &builder,
                                           Type eltType) {
  auto loc = op->getLoc();
  builder.setInsertionPoint(op);

  Value lhs = op->getOperand(0);
  Value rhsVal = op->getOperand(1);
  Value scale = op->getOperand(2);
  Value addOut = op->getOperand(3);
  Value normOut = op->getOperand(4);
  auto epsilonAttr = op->getAttrOfType<FloatAttr>("epsilon");

  auto dyn2d = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic}, eltType);
  auto dyn1d = MemRefType::get({ShapedType::kDynamic}, eltType);

  Value lhsDyn = collapseAndCast2D(builder, loc, lhs, dyn2d);
  Value rhsDyn = collapseAndCast2D(builder, loc, rhsVal, dyn2d);
  Value addOutDyn = collapseAndCast2D(builder, loc, addOut, dyn2d);
  Value normOutDyn = collapseAndCast2D(builder, loc, normOut, dyn2d);
  Value scaleDyn = builder.create<memref::CastOp>(loc, dyn1d, scale);
  Value eps = builder.create<arith::ConstantOp>(loc, epsilonAttr);

  builder.create<func::CallOp>(
      loc, kFuseAddRMSNormKernel, TypeRange{},
      ValueRange{lhsDyn, rhsDyn, scaleDyn, addOutDyn, normOutDyn, eps});

  op->erase();
}

// =========================================================================
// Pass
// =========================================================================

struct CodegenRMSNormPass
    : public PassWrapper<CodegenRMSNormPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CodegenRMSNormPass)

  StringRef getArgument() const override { return "codegen-rms-norm"; }
  StringRef getDescription() const override {
    return "Generate GPU kernels for joyl.rms_norm and "
           "joyl.fuse_add_rmsnorm via code generation";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, memref::MemRefDialect,
                    arith::ArithDialect, scf::SCFDialect,
                    math::MathDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(&getContext());

    SmallVector<Operation *> rmsNormOps;
    SmallVector<Operation *> fuseAddRMSNormOps;

    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "joyl.rms_norm")
        rmsNormOps.push_back(op);
      else if (name == "joyl.fuse_add_rmsnorm")
        fuseAddRMSNormOps.push_back(op);
    });

    // Determine element type from the first available op
    Type eltType;
    if (!rmsNormOps.empty()) {
      eltType =
          cast<MemRefType>(rmsNormOps[0]->getOperand(0).getType())
              .getElementType();
    } else if (!fuseAddRMSNormOps.empty()) {
      eltType =
          cast<MemRefType>(fuseAddRMSNormOps[0]->getOperand(0).getType())
              .getElementType();
    } else {
      return;
    }

    if (!rmsNormOps.empty())
      createRMSNormKernel(module, builder, eltType);

    if (!fuseAddRMSNormOps.empty())
      createFuseAddRMSNormKernel(module, builder, eltType);

    for (auto *op : rmsNormOps)
      replaceRMSNormWithCall(op, builder, eltType);

    for (auto *op : fuseAddRMSNormOps)
      replaceFuseAddRMSNormWithCall(op, builder, eltType);
  }
};

} // namespace

void registerCodegenRMSNormPass() {
  PassRegistration<CodegenRMSNormPass>();
}
