//===- LowerJoylToJoyhPass.cpp - Joyl to Joyh lowering -------------------===//
//
// Joy Compiler - Lower buffer-based Joyl ops to Joyh custom calls
//
// Converts joyl.* buffer ops into joyh.custom_call for GPU dispatch.
//
// Pattern:
//   "joyl.embedding"(%in0, %in1, %out) : (memref, memref, memref) -> ()
// becomes:
//   "joyh.custom_call"(%in0, %in1, %out) {
//     call_target_name = "joy_gpu_embedding",
//     backend = "gpu",
//     num_inputs = 2 : i64
//   } : (memref, memref, memref) -> ()
//
//===----------------------------------------------------------------------===//

#include "joy/dialect/joyl/JoylDialect.h"
#include "joy/dialect/joyh/JoyhDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Map from joyl op mnemonic to the number of output operands.
static int64_t getNumOutputs(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name == "joyl.fuse_add_rmsnorm")
    return 2;
  return 1;
}

/// Generic converter: any joyl.* op → joyh.custom_call
///
/// Generic converter for all joyl ops.
/// Instead of templating per-op, we match any operation in the joyl dialect
/// and produce a single joyh.custom_call with:
///   - The same memref operands
///   - call_target_name = "joy_gpu_<mnemonic>"
///   - All original attributes preserved
class JoylToJoyhGenericConverter : public RewritePattern {
public:
  JoylToJoyhGenericConverter(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!op->getDialect() || op->getDialect()->getNamespace() != "joyl")
      return failure();

    StringRef opName = op->getName().getStringRef();
    if (opName == "joyl.rms_norm" || opName == "joyl.fuse_add_rmsnorm")
      return failure();

    auto loc = op->getLoc();

    StringRef fullName = op->getName().getStringRef();
    size_t dotPos = fullName.find('.');
    std::string mnemonic = fullName.substr(dotPos + 1).str();
    std::string targetName = "joy_gpu_" + mnemonic;

    int64_t numOutputs = getNumOutputs(op);
    int64_t numInputs =
        static_cast<int64_t>(op->getNumOperands()) - numOutputs;

    SmallVector<Value> operands(op->getOperands());

    auto customCall = rewriter.create<joyh::CustomCallOp>(
        loc, TypeRange{}, operands,
        rewriter.getStringAttr(targetName),
        rewriter.getStringAttr("gpu"),
        rewriter.getI64IntegerAttr(numInputs));

    for (auto &attr : op->getAttrs()) {
      StringRef attrName = attr.getName();
      if (attrName == "operandSegmentSizes")
        continue;
      customCall->setAttr(attrName, attr.getValue());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// The lowering pass
struct LowerJoylToJoyhPass
    : public PassWrapper<LowerJoylToJoyhPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerJoylToJoyhPass)

  StringRef getArgument() const override { return "lower-joyl-to-joyh"; }
  StringRef getDescription() const override {
    return "Lower Joyl dialect (memref) to Joyh custom calls (GPU backend)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<joyh::JoyhDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<JoylToJoyhGenericConverter>(context);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns),
                                            config))) {
      module.emitError("joyl-to-joyh lowering did not converge");
      signalPassFailure();
    }
  }
};

} // namespace

void registerLowerJoylToJoyhPass() {
  PassRegistration<LowerJoylToJoyhPass>();
}
