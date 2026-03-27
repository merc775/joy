//===- OpFusionPass.cpp - Joy Op Fusion Pass -----------------------------===//
//
// Joy Compiler - Operation fusion pass
//
// Fuses add + rms_norm into fuse_add_rmsnorm in the Joy dialect.
//
// Pattern matched (in transformer decoder layers):
//   %add = joy.add(%residual, %hidden)
//   %norm = joy.rms_norm(%add, %scale) {epsilon = ...}
// becomes:
//   %add, %norm = joy.fuse_add_rmsnorm(%residual, %hidden, %scale)
//                   {epsilon = ...}
//
// The fused op produces both the add result (needed as the next residual)
// and the normalized result, eliminating a redundant read of the
// intermediate tensor.
//
//===----------------------------------------------------------------------===//

#include "joy/dialect/joy/JoyDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

using namespace mlir;

namespace {

/// Fuse joy.add + joy.rms_norm → joy.fuse_add_rmsnorm.
///
/// We pattern-match on rms_norm whose input is produced by an add op.
/// The add result may have other users (typically the next residual
/// connection), so the fused op produces two results.
struct AddRMSNormFusionPattern
    : public OpRewritePattern<joy::RMSNormOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(joy::RMSNormOp normOp,
                                PatternRewriter &rewriter) const override {
    Value normInput = normOp->getOperand(0);  // input
    auto *defOp = normInput.getDefiningOp();
    if (!defOp || defOp->getName().getStringRef() != "joy.add")
      return failure();

    auto addOp = cast<joy::AddOp>(defOp);

    auto addResultType = addOp->getResult(0).getType();
    auto normResultType = normOp->getResult(0).getType();
    auto epsilonAttr = normOp->getAttrOfType<FloatAttr>("epsilon");

    auto fuseOp = rewriter.create<joy::FuseAddRMSNormOp>(
        normOp.getLoc(),
        TypeRange{addResultType, normResultType},
        addOp->getOperand(0),   // lhs
        addOp->getOperand(1),   // rhs
        normOp->getOperand(1),  // scale
        epsilonAttr);

    rewriter.replaceOp(normOp, fuseOp->getResult(1));  // norm result
    rewriter.replaceAllUsesWith(addOp->getResult(0), fuseOp->getResult(0));
    rewriter.eraseOp(addOp);

    return success();
  }
};

struct OpFusionPass : public PassWrapper<OpFusionPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpFusionPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AddRMSNormFusionPattern>(&getContext());

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;

    if (failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override { return "joy-op-fusion"; }
  StringRef getDescription() const override {
    return "Fuse add + rms_norm into fuse_add_rmsnorm in Joy dialect";
  }
};

} // namespace

void registerOpFusionPass() {
  PassRegistration<OpFusionPass>();
}

std::unique_ptr<Pass> createOpFusionPass() {
  return std::make_unique<OpFusionPass>();
}
