//===- JoyBuilder.h - Joy dialect builder API ----------------*- C++ -*-===//
//
// Joy Compiler - Frontend builder for creating Joy dialect operations
//
//===----------------------------------------------------------------------===//

#ifndef JOY_FRONTEND_JOYBUILDER_H
#define JOY_FRONTEND_JOYBUILDER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "joy/dialect/joy/JoyDialect.h"

namespace mlir {
namespace joy {
namespace front {

class JoyBuilder {
public:
  JoyBuilder();
  ~JoyBuilder();

  bool init();
  mlir::MLIRContext &getContext() { return *context_; }

  mlir::Value createEmbedding(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value input, mlir::Value weight,
                              mlir::Type resultType);

  mlir::Value createAdd(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value lhs, mlir::Value rhs,
                        mlir::Type resultType);

  mlir::Value createMul(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value lhs, mlir::Value rhs,
                        mlir::Type resultType);

  mlir::Value createMatMul(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value lhs, mlir::Value rhs,
                           mlir::Type resultType);

  mlir::Value createLinear(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value input, mlir::Value weight,
                           mlir::Type resultType);

  mlir::Value createRMSNorm(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value input, mlir::Value scale,
                            float epsilon, mlir::Type resultType);

  mlir::Value createSoftmax(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value input, int64_t axis,
                            mlir::Type resultType);

  mlir::Value createSiLU(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value input, mlir::Type resultType);

  mlir::Value createReshape(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value input, mlir::Type resultType);

  mlir::Value createTranspose(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value input,
                              llvm::ArrayRef<int64_t> permutation,
                              mlir::Type resultType);

  mlir::Value createApplyRotaryEmb(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value input, mlir::Value cos,
                                   mlir::Value sin, mlir::Type resultType);

  mlir::Value createRepeatKV(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value input, int64_t n_rep,
                             mlir::Type resultType);

private:
  std::unique_ptr<mlir::MLIRContext> context_;
};

} // namespace front
} // namespace joy
} // namespace mlir

#endif // JOY_FRONTEND_JOYBUILDER_H
