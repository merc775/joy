//===- Utils.h - Utility functions for Joy optimizer ---------*- C++ -*-===//
//
// Joy Compiler - Utility functions for optimizer
//
//===----------------------------------------------------------------------===//

#ifndef JOY_SUPPORT_UTILS_H
#define JOY_SUPPORT_UTILS_H

#include "joy/support/MhloStubs.h"
#include "joy/dialect/joy/JoyDialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace joy {

// =============================================================================
// 常量字符串
// =============================================================================
constexpr const char *kQuantizeType = "quantize_type";

// =============================================================================
// 辅助函数
// =============================================================================

/// 检查操作是否具有 F32 量化类型属性
static inline bool haveF32QuantizeTypeAttr(mlir::Operation &op) {
  auto quantizeType = op.getAttrOfType<mlir::StringAttr>(kQuantizeType);
  if (quantizeType && ("f32" == quantizeType.getValue().str()))
    return true;
  return false;
}

static inline bool haveF32QuantizeTypeAttr(mlir::Operation *op) {
  return haveF32QuantizeTypeAttr(*op);
}

/// 设置 F32 量化类型属性
static inline void setF32QuantizeTypeAttr(mlir::Operation &op) {
  auto f32TypeAttr = mlir::StringAttr::get(op.getContext(), "f32");
  op.setAttr(kQuantizeType, f32TypeAttr);
}

static inline void setF32QuantizeTypeAttr(mlir::Operation *op) {
  return setF32QuantizeTypeAttr(*op);
}

/// 替换操作 - 简化版本
static inline void replaceOp(mlir::PatternRewriter &rewriter,
                              mlir::Operation *opOld, mlir::Value newValue) {
  rewriter.replaceOp(opOld, newValue);
}

static inline void replaceOp(mlir::PatternRewriter &rewriter,
                              mlir::Operation *opOld,
                              mlir::ValueRange newValues) {
  rewriter.replaceOp(opOld, newValues);
}

/// 获取常量操作的数据
static inline std::shared_ptr<void> getConstOpData(mlir::Operation *op) {
  auto attr = op->getAttrOfType<mlir::DenseElementsAttr>("value");
  if (!attr)
    return nullptr;
  auto rawData = attr.getRawData();
  // 返回一个不拥有内存的 shared_ptr，指向原始数据
  return std::shared_ptr<void>(const_cast<char *>(rawData.data()),
                               [](void *) {});
}

/// 创建指定类型和值的 DenseElementsAttr
static inline mlir::DenseElementsAttr
getLiteralDenseElementAttr(mlir::ShapedType type, double value,
                           mlir::MLIRContext *context) {
  auto elementType = type.getElementType();
  if (elementType.isF32()) {
    auto floatVal = static_cast<float>(value);
    return mlir::DenseElementsAttr::get(type, floatVal);
  }
  if (elementType.isF64()) {
    return mlir::DenseElementsAttr::get(type, value);
  }
  if (elementType.isF16()) {
    llvm::APFloat apVal(value);
    bool losesInfo;
    apVal.convert(llvm::APFloat::IEEEhalf(), llvm::APFloat::rmNearestTiesToEven,
                  &losesInfo);
    return mlir::DenseElementsAttr::get(type, apVal);
  }
  // 默认用 float
  auto floatVal = static_cast<float>(value);
  return mlir::DenseElementsAttr::get(type, floatVal);
}

} // namespace joy
} // namespace mlir

// =============================================================================
// 全局辅助函数 (在匿名命名空间外，pattern 代码中直接使用)
// =============================================================================

/// 向前追踪 Value 到上一个操作的第0个操作数
static inline bool nextValue(mlir::Value &cur, bool /*topDown*/ = false) {
  auto *curOp = cur.getDefiningOp();
  if (curOp == nullptr)
    return false;
  cur = curOp->getOperand(0);
  return true;
}

/// 跳过 Reshape/DynamicReshape 操作
static inline void ignoreTransform(mlir::Value &cur,
                                   bool ignoreTrans = true) {
  auto *op = cur.getDefiningOp();
  while (llvm::isa_and_nonnull<mhlo::ReshapeOp, mhlo::DynamicReshapeOp,
                               mhlo::BroadcastInDimOp,
                               mhlo::DynamicBroadcastInDimOp>(op) ||
         (ignoreTrans && llvm::isa_and_nonnull<mhlo::TransposeOp>(op))) {
    if (nextValue(cur) == false)
      break;
    op = cur.getDefiningOp();
  }
}

/// 跳过 BroadcastInDim/DynamicBroadcastInDim 操作
static inline void ignoreBroadcastInDim(mlir::Value &cur) {
  auto *op = cur.getDefiningOp();
  while (llvm::isa_and_nonnull<mhlo::DynamicBroadcastInDimOp,
                               mhlo::BroadcastInDimOp>(op)) {
    if (nextValue(cur) == false)
      break;
    op = cur.getDefiningOp();
  }
}

/// 跳过 Reshape/DynamicReshape 操作
static inline void ignoreReshapeDim(mlir::Value &cur, int64_t rank = 0) {
  auto *op = cur.getDefiningOp();
  while (
      llvm::isa_and_nonnull<mhlo::ReshapeOp, mhlo::DynamicReshapeOp>(op)) {
    if (nextValue(cur) == false)
      break;
    auto curType = llvm::cast<mlir::ShapedType>(cur.getType());
    if (rank != 0 && curType.getRank() == rank)
      break;
    op = cur.getDefiningOp();
  }
}

/// 跳过 transform 操作并匹配指定类型的操作
template <typename OpType>
static mlir::Operation *getMetaOps(mlir::Value v) {
  ignoreTransform(v);
  auto *op = v.getDefiningOp();
  if (!llvm::isa_and_nonnull<OpType>(op)) {
    return nullptr;
  }
  return op;
}

#endif // JOY_SUPPORT_UTILS_H
