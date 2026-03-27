//===- JoyBuilder.cpp - Joy dialect builder implementation -----------------===//
//
// Joy Compiler - Frontend builder for creating Joy dialect operations
//
//===----------------------------------------------------------------------===//

#include "joy/frontend/JoyBuilder.h"

namespace mlir {
namespace joy {
namespace front {

JoyBuilder::JoyBuilder() = default;
JoyBuilder::~JoyBuilder() = default;

bool JoyBuilder::init() {
  context_ = std::make_unique<mlir::MLIRContext>();
  context_->getOrLoadDialect<mlir::joy::JoyDialect>();
  return true;
}

mlir::Value JoyBuilder::createEmbedding(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Value input,
                                        mlir::Value weight,
                                        mlir::Type resultType) {
  return builder.create<mlir::joy::EmbeddingOp>(loc, resultType, input, weight)
      .getResult();
}

mlir::Value JoyBuilder::createAdd(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value lhs, mlir::Value rhs,
                                  mlir::Type resultType) {
  return builder.create<mlir::joy::AddOp>(loc, resultType, lhs, rhs)
      .getResult();
}

mlir::Value JoyBuilder::createMul(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value lhs, mlir::Value rhs,
                                  mlir::Type resultType) {
  return builder.create<mlir::joy::MulOp>(loc, resultType, lhs, rhs)
      .getResult();
}

mlir::Value JoyBuilder::createMatMul(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value lhs,
                                     mlir::Value rhs, mlir::Type resultType) {
  return builder.create<mlir::joy::MatMulOp>(loc, resultType, lhs, rhs)
      .getResult();
}

mlir::Value JoyBuilder::createLinear(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value input,
                                     mlir::Value weight,
                                     mlir::Type resultType) {
  return builder.create<mlir::joy::LinearOp>(loc, resultType, input, weight)
      .getResult();
}

mlir::Value JoyBuilder::createRMSNorm(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Value input,
                                      mlir::Value scale, float epsilon,
                                      mlir::Type resultType) {
  auto epsAttr = builder.getF32FloatAttr(epsilon);
  return builder
      .create<mlir::joy::RMSNormOp>(loc, resultType, input, scale, epsAttr)
      .getResult();
}

mlir::Value JoyBuilder::createSoftmax(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Value input,
                                      int64_t axis, mlir::Type resultType) {
  auto axisAttr = builder.getI64IntegerAttr(axis);
  return builder
      .create<mlir::joy::SoftmaxOp>(loc, resultType, input, axisAttr)
      .getResult();
}

mlir::Value JoyBuilder::createSiLU(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value input, mlir::Type resultType) {
  return builder.create<mlir::joy::SiLUOp>(loc, resultType, input).getResult();
}

mlir::Value JoyBuilder::createReshape(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Value input,
                                      mlir::Type resultType) {
  return builder.create<mlir::joy::ReshapeOp>(loc, resultType, input)
      .getResult();
}

mlir::Value JoyBuilder::createTranspose(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Value input,
                                        llvm::ArrayRef<int64_t> permutation,
                                        mlir::Type resultType) {
  auto permAttr = builder.getI64TensorAttr(permutation);
  return builder
      .create<mlir::joy::TransposeOp>(loc, resultType, input, permAttr)
      .getResult();
}

mlir::Value JoyBuilder::createApplyRotaryEmb(mlir::OpBuilder &builder,
                                             mlir::Location loc,
                                             mlir::Value input, mlir::Value cos,
                                             mlir::Value sin,
                                             mlir::Type resultType) {
  return builder
      .create<mlir::joy::ApplyRotaryEmbOp>(loc, resultType, input, cos, sin)
      .getResult();
}

mlir::Value JoyBuilder::createRepeatKV(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value input,
                                       int64_t n_rep, mlir::Type resultType) {
  auto nRepAttr = builder.getI64IntegerAttr(n_rep);
  return builder
      .create<mlir::joy::RepeatKVOp>(loc, resultType, input, nRepAttr)
      .getResult();
}

} // namespace front
} // namespace joy
} // namespace mlir
