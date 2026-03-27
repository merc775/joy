//===- MhloStubs.h --*- C++ -*-===//
//
// Joy Compiler.
//
//===----------------------------------------------------------------------===//

#ifndef JOY_SUPPORT_MHLOSTUBS_H
#define JOY_SUPPORT_MHLOSTUBS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

// =============================================================================
// mhlo 命名空间 - 模拟 mhlo 方言中的操作
// =============================================================================
namespace mhlo {

// 二元算术操作的通用基类宏
#define DEFINE_MHLO_BINARY_OP(OpName, OpMnemonic)                              \
  class OpName                                                                 \
      : public mlir::Op<OpName, mlir::OpTrait::OneResult,                      \
                        mlir::OpTrait::NOperands<2>::Impl> {                   \
  public:                                                                      \
    using Op::Op;                                                              \
    static llvm::StringRef getOperationName() {                                \
      return "mhlo." OpMnemonic;                                               \
    }                                                                          \
    static void build(mlir::OpBuilder &builder, mlir::OperationState &state,   \
                      mlir::Type resultType, mlir::Value lhs,                  \
                      mlir::Value rhs) {                                       \
      state.addOperands({lhs, rhs});                                           \
      state.addTypes(resultType);                                              \
    }                                                                          \
    mlir::Value getResult() { return getOperation()->getResult(0); }           \
  };

DEFINE_MHLO_BINARY_OP(MulOp, "multiply")
DEFINE_MHLO_BINARY_OP(DivOp, "divide")
DEFINE_MHLO_BINARY_OP(AddOp, "add")
DEFINE_MHLO_BINARY_OP(PowOp, "power")

#undef DEFINE_MHLO_BINARY_OP

// 一元操作的通用基类宏
#define DEFINE_MHLO_UNARY_OP(OpName, OpMnemonic)                               \
  class OpName                                                                 \
      : public mlir::Op<OpName, mlir::OpTrait::OneResult,                      \
                        mlir::OpTrait::OneOperand> {                           \
  public:                                                                      \
    using Op::Op;                                                              \
    static llvm::StringRef getOperationName() {                                \
      return "mhlo." OpMnemonic;                                               \
    }                                                                          \
    mlir::Value getResult() { return getOperation()->getResult(0); }           \
  };

DEFINE_MHLO_UNARY_OP(SqrtOp, "sqrt")
DEFINE_MHLO_UNARY_OP(RsqrtOp, "rsqrt")
DEFINE_MHLO_UNARY_OP(ConvertOp, "convert")
DEFINE_MHLO_UNARY_OP(NegOp, "negate")

#undef DEFINE_MHLO_UNARY_OP

class ConstantOp
    : public mlir::Op<ConstantOp, mlir::OpTrait::ZeroOperands,
                      mlir::OpTrait::OneResult,
                      mlir::OpTrait::ConstantLike> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() { return "mhlo.constant"; }
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::ElementsAttr value) {
    state.addAttribute("value", value);
    state.addTypes(value.getType());
  }
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type type, mlir::ElementsAttr value) {
    state.addAttribute("value", value);
    state.addTypes(type);
  }
  mlir::Value getResult() { return getOperation()->getResult(0); }
  // 支持 ConstantLike trait 所需的 fold
  mlir::OpFoldResult fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
    return getOperation()->getAttr("value");
  }
};

class BroadcastInDimOp
    : public mlir::Op<BroadcastInDimOp, mlir::OpTrait::OneResult> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() { return "mhlo.broadcast_in_dim"; }
  mlir::Value getResult() { return getOperation()->getResult(0); }
};

class DynamicBroadcastInDimOp
    : public mlir::Op<DynamicBroadcastInDimOp, mlir::OpTrait::OneResult> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() {
    return "mhlo.dynamic_broadcast_in_dim";
  }
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type resultType, mlir::Value operand,
                    mlir::Value shape, mlir::DenseIntElementsAttr dims) {
    state.addOperands({operand, shape});
    state.addAttribute("broadcast_dimensions", dims);
    state.addTypes(resultType);
  }
  mlir::Value getResult() { return getOperation()->getResult(0); }
  // 隐式转换为 Value，方便赋值
  operator mlir::Value() { return getResult(); }
};

class ReshapeOp : public mlir::Op<ReshapeOp, mlir::OpTrait::OneResult,
                                   mlir::OpTrait::OneOperand> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() { return "mhlo.reshape"; }
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type resultType, mlir::Value operand) {
    state.addOperands(operand);
    state.addTypes(resultType);
  }
  mlir::Value getResult() { return getOperation()->getResult(0); }
};

class DynamicReshapeOp
    : public mlir::Op<DynamicReshapeOp, mlir::OpTrait::OneResult> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() {
    return "mhlo.dynamic_reshape";
  }
  mlir::Value getResult() { return getOperation()->getResult(0); }
};

class TransposeOp : public mlir::Op<TransposeOp, mlir::OpTrait::OneResult,
                                     mlir::OpTrait::OneOperand> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() { return "mhlo.transpose"; }
  mlir::Value getResult() { return getOperation()->getResult(0); }
};

class ClampOp : public mlir::Op<ClampOp, mlir::OpTrait::OneResult> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() { return "mhlo.clamp"; }
  mlir::Value getResult() { return getOperation()->getResult(0); }
};

class TanhOp : public mlir::Op<TanhOp, mlir::OpTrait::OneResult,
                                mlir::OpTrait::OneOperand> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() { return "mhlo.tanh"; }
  mlir::Value getResult() { return getOperation()->getResult(0); }
};

class GetTupleElementOp
    : public mlir::Op<GetTupleElementOp, mlir::OpTrait::OneResult,
                      mlir::OpTrait::OneOperand> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() {
    return "mhlo.get_tuple_element";
  }
  mlir::Value getResult() { return getOperation()->getResult(0); }
};

} // namespace mhlo

#endif // JOY_SUPPORT_MHLOSTUBS_H
