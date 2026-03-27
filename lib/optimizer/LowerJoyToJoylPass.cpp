//===- LowerJoyToJoylPass.cpp - Joy to Joyl lowering ---------------------===//
//
// Joy Compiler - Lower tensor-based Joy ops to buffer-based Joyl ops
//
// Converts tensor-based Joy ops into buffer-based Joyl ops (bufferization).
//
//===----------------------------------------------------------------------===//

#include "joy/dialect/joy/JoyDialect.h"
#include "joy/dialect/joyl/JoylDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

// Allocate a memref buffer for a tensor-typed op result.
// Allocates a memref buffer for the given tensor result.
static Value allocOutputBuffer(Location loc, OpResult result,
                                ConversionPatternRewriter &rewriter) {
  auto tensorType = cast<RankedTensorType>(result.getType());
  auto memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(result.getDefiningOp());
  return rewriter.create<memref::AllocOp>(loc, memrefType);
}

// Generic converter: Joy tensor op → Joyl buffer op.
// Generic converter for Joy tensor ops to Joyl buffer ops.
//
// Pattern:
//   %out = joy.op(%in0, %in1) {attrs} : (tensor, tensor) -> tensor
// becomes:
//   %buf = memref.alloc() : memref
//   joyl.op(%in0_memref, %in1_memref, %buf) {attrs}
//   // %out's uses are replaced with %buf
template <typename JoyOp, typename JoylOp>
class JoyToJoylOpConverter : public OpConversionPattern<JoyOp> {
public:
  using OpConversionPattern<JoyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(JoyOp op, typename JoyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    SmallVector<Value, 6> bufferArgs(adaptor.getOperands());

    for (auto result : op->getResults()) {
      bufferArgs.push_back(allocOutputBuffer(loc, result, rewriter));
    }

    rewriter.create<JoylOp>(loc, TypeRange{}, bufferArgs, op->getAttrs());

    rewriter.replaceOp(
        op, ArrayRef<Value>(bufferArgs).slice(adaptor.getOperands().size()));
    return success();
  }
};

// TypeConverter: RankedTensorType → MemRefType
class JoyToJoylTypeConverter : public TypeConverter {
public:
  JoyToJoylTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](RankedTensorType type) -> Type {
      return MemRefType::get(type.getShape(), type.getElementType());
    });
  }
};

// The lowering pass
struct LowerJoyToJoylPass
    : public PassWrapper<LowerJoyToJoylPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerJoyToJoylPass)

  StringRef getArgument() const override { return "lower-joy-to-joyl"; }
  StringRef getDescription() const override {
    return "Lower Joy dialect (tensor) to Joyl dialect (memref)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<joyl::JoylDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto &context = getContext();

    JoyToJoylTypeConverter typeConverter;
    ConversionTarget target(context);

    target.addLegalDialect<joyl::JoylDialect, memref::MemRefDialect,
                           arith::ArithDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<joy::JoyDialect>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return llvm::all_of(op.getOperandTypes(),
                          [&](Type t) { return typeConverter.isLegal(t); });
    });

    RewritePatternSet patterns(&context);

    // Register conversion patterns for all joy ops used in the Qwen3 test
    patterns.add<
        JoyToJoylOpConverter<joy::EmbeddingOp, joyl::EmbeddingOp>,
        JoyToJoylOpConverter<joy::AddOp, joyl::AddOp>,
        JoyToJoylOpConverter<joy::MulOp, joyl::MulOp>,
        JoyToJoylOpConverter<joy::RMSNormOp, joyl::RMSNormOp>,
        JoyToJoylOpConverter<joy::FuseAddRMSNormOp, joyl::FuseAddRMSNormOp>,
        JoyToJoylOpConverter<joy::LinearOp, joyl::LinearOp>,
        JoyToJoylOpConverter<joy::MatMulOp, joyl::MatMulOp>,
        JoyToJoylOpConverter<joy::SoftmaxOp, joyl::SoftmaxOp>,
        JoyToJoylOpConverter<joy::SiLUOp, joyl::SiLUOp>,
        JoyToJoylOpConverter<joy::ReshapeOp, joyl::ReshapeOp>,
        JoyToJoylOpConverter<joy::TransposeOp, joyl::TransposeOp>,
        JoyToJoylOpConverter<joy::ApplyRotaryEmbOp, joyl::ApplyRotaryEmbOp>,
        JoyToJoylOpConverter<joy::RepeatKVOp, joyl::RepeatKVOp>
    >(typeConverter, &context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void registerLowerJoyToJoylPass() {
  PassRegistration<LowerJoyToJoylPass>();
}
