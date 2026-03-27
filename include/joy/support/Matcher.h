//===- Matcher.h - Pattern matcher framework for Joy --------*- C++ -*-===//
//
// Joy Compiler - Enhanced matcher utility
//
// Enhanced matcher utility function based on mlir/IR/Matchers.h
// Has enhanced ability as follows:
// 1, auto collection subgraph of patterns
// 2, mOptional, which can skip some op during recursive match
// 3, mOpWithCapturer, which can capture Op for a specific matcher.
// 4, mOpWithConstraint, apply a constrait function for a matcher.
// 5, mOpCommutative, handle operands matcher with any order.
//===----------------------------------------------------------------------===//
#ifndef JOY_SUPPORT_MATCHER_H
#define JOY_SUPPORT_MATCHER_H

#include "joy/support/MhloStubs.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include <type_traits>

#define DEBUG_TYPE "joy-matcher"

namespace mlir::joy::opt {

using OpList = llvm::SmallVector<mlir::Operation *, 4>;
using FuncT = std::function<bool(mlir::Operation *)>;

struct MatcherBase {
  virtual ~MatcherBase() = default;
  virtual bool match(mlir::Operation *op) = 0;
  virtual void mergeCollection(OpList &from) {
    // default version, not collect
  }
  mlir::Operation *curOp = nullptr;
  OpList collection;
};

/// The matcher that matches a certain kind of op.
struct OpMatcherBase : public MatcherBase {
  void collectOp(mlir::Operation *op) {
    curOp = op;
    collection.push_back(curOp);
  }

  void mergeCollection(OpList &from) final {
    collection.insert(collection.end(), from.begin(), from.end());
  }
};

/// The matcher that matches a certain kind of op.
template <typename OpType> struct OpMatcher : public OpMatcherBase {
  bool match(mlir::Operation *op) final {
    collectOp(op);
    return llvm::isa<OpType>(op);
  }
};

/// Trait to check whether T provides a 'match' method with type
/// `OperationOrValue`.
template <typename T, typename OperationOrValue>
using HasOperationOrValueMatcher =
    decltype(std::declval<T>().match(std::declval<OperationOrValue>()));

/// Statically switch to a Value matcher.
template <typename MatcherClass>
typename std::enable_if_t<llvm::is_detected<HasOperationOrValueMatcher,
                                            MatcherClass, mlir::Value>::value,
                          bool>
matchOperandOrValueAtIndex(mlir::Operation *op, unsigned idx,
                           MatcherClass &matcher) {
  return matcher.match(op->getOperand(idx));
}

/// Statically switch to an Operation matcher.
template <typename MatcherClass>
typename std::enable_if_t<
    llvm::is_detected<HasOperationOrValueMatcher, MatcherClass,
                      mlir::Operation *>::value,
    bool>
matchOperandOrValueAtIndex(mlir::Operation *op, unsigned idx,
                           MatcherClass &matcher) {
  if (auto *defOp = op->getOperand(idx).getDefiningOp())
    return matcher.match(defOp);
  return false;
}

template <typename TupleT, class CallbackT, std::size_t... Is>
constexpr void enumerateImpl(TupleT &&tuple, CallbackT &&callback,
                             std::index_sequence<Is...>) {
  (void)std::initializer_list<int>{
      0,
      (callback(std::integral_constant<std::size_t, Is>{}, std::get<Is>(tuple)),
       0)...};
}

template <typename... Tys, typename CallbackT>
constexpr void enumerate(std::tuple<Tys...> &tuple, CallbackT &&callback) {
  mlir::joy::opt::enumerateImpl(tuple, std::forward<CallbackT>(callback),
                           std::make_index_sequence<sizeof...(Tys)>{});
}

/// RecursivePatternMatcher that composes.
template <typename OpType, typename... OperandMatchers>
struct RecursivePatternMatcher : public OpMatcherBase {
  RecursivePatternMatcher(bool isBrc, bool isCommutative,
                          OperandMatchers... matchers)
      : commutative(isCommutative), operandMatchers(matchers...) {}
  RecursivePatternMatcher(bool isBrc, bool isCommutative, bool isDerived,
                          OperandMatchers... matchers)
      : commutative(isCommutative), derived(isDerived), broadcast(isBrc),
        operandMatchers(matchers...) {}
  bool match(mlir::Operation *op) override {
    LLVM_DEBUG(if (!derived) {
      llvm::dbgs() << "Executing mOp<" << OpType::getOperationName() << ">\n";
    });

    if (!llvm::isa<OpType>(op) ||
        op->getNumOperands() != sizeof...(OperandMatchers)) {
      LLVM_DEBUG(
          if (!llvm::isa<OpType>(op)) llvm::dbgs() << "  * mismatch OpType"
                                                   << "\n";
          if (op->getNumOperands() != sizeof...(OperandMatchers)) {
            llvm::dbgs() << "  * mismatch operands number and operandMatchers"
                         << " number"
                         << "\n";
          }

          if (!derived) {
            llvm::dbgs() << "  * Failed to match mOp<"
                         << OpType::getOperationName() << ">, current op is \n";
          } else {
            llvm::dbgs() << "  * Failed to match mOpWith***<"
                         << OpType::getOperationName() << ">, current op is \n";
          });
      LLVM_DEBUG(op->print(llvm::dbgs()); llvm::dbgs() << "\n";);

      return false;
    }

    bool res = true;
    mlir::joy::opt::enumerate(operandMatchers, [&](size_t index, auto &matcher) {
      auto *defOp = op->getOperand(index).getDefiningOp();
      if (broadcast &&
          llvm::isa_and_nonnull<::mhlo::DynamicBroadcastInDimOp,
                                ::mhlo::BroadcastInDimOp>(defOp))
        res &= mlir::joy::opt::matchOperandOrValueAtIndex(defOp, 0, matcher);
      else
        res &= mlir::joy::opt::matchOperandOrValueAtIndex(op, index, matcher);
    });
    if (res) {
      std::apply(
          [this](auto &&...matcher) {
            ((this->mergeCollection(matcher.collection)), ...);
          },
          operandMatchers);
    }

    if (commutative && !res) {
      res = true;
      mlir::joy::opt::enumerate(operandMatchers, [&](size_t index, auto &matcher) {
        assert(index < 2 && "only support commutative for binaryOp");
        auto *defOp = op->getOperand(1 - index).getDefiningOp();
        if (broadcast &&
            llvm::isa_and_nonnull<::mhlo::DynamicBroadcastInDimOp,
                                  ::mhlo::BroadcastInDimOp>(defOp))
          res &= mlir::joy::opt::matchOperandOrValueAtIndex(defOp, 0, matcher);
        else
          res &= mlir::joy::opt::matchOperandOrValueAtIndex(op, 1 - index, matcher);
      });
      if (res) {
        std::apply(
            [this](auto &&...matcher) {
              ((this->mergeCollection(matcher.collection)), ...);
            },
            operandMatchers);
      }
    }

    collectOp(op);
    return res;
  }
  bool commutative = false;
  bool derived = false;
  bool broadcast = false;
  std::tuple<OperandMatchers...> operandMatchers;
};

template <typename OpType, typename... OperandMatchers>
struct OptionalPatternMatcher : public OpMatcherBase {
  OptionalPatternMatcher(OperandMatchers... matchers)
      : operandMatchers(matchers...) {}
  bool match(mlir::Operation *op) final {
    LLVM_DEBUG(llvm::dbgs() << "Executing mOptional<"
                            << OpType::getOperationName() << ">\n");
    bool toMatchOperand = false;
    if (llvm::isa<OpType>(op)) {
      toMatchOperand = true;
      assert(1 == op->getNumOperands() &&
             "Only support 1 operand for optional op");
      assert(1 == op->getNumResults() &&
             "Only support 1 result for optional op");
      assert(1 == sizeof...(OperandMatchers) &&
             "Only support 1 operand for optional op");
    }

    bool res = true;
    if (toMatchOperand) {
      LLVM_DEBUG(llvm::dbgs() << "  * Matched mOptional<"
                              << OpType::getOperationName() << ">\n");
      mlir::joy::opt::enumerate(operandMatchers, [&](size_t index, auto &matcher) {
        res &= mlir::joy::opt::matchOperandOrValueAtIndex(op, index, matcher);
        mergeCollection(matcher.collection);
      });
      collectOp(op);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "  * Not match mOptional<"
                              << OpType::getOperationName() << ">\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "  * Skip it and go on to process it's operand.\n");
      auto matcher = std::get<0>(operandMatchers);
      res = matcher.match(op);
      mergeCollection(matcher.collection);
    }

    return res;
  }
  std::tuple<OperandMatchers...> operandMatchers;
};

template <typename OpType, typename... OperandMatchers>
struct ConstraintPatternMatcher
    : public RecursivePatternMatcher<OpType, OperandMatchers...> {
  ConstraintPatternMatcher(FuncT cstrt, OperandMatchers... matchers)
      : RecursivePatternMatcher<OpType, OperandMatchers...>(false, false, true,
                                                            matchers...),
        constraint(std::move(cstrt)) {}
  bool match(mlir::Operation *op) final {
    LLVM_DEBUG(llvm::dbgs() << "Executing mOpWithConstraint<"
                            << OpType::getOperationName() << ">\n");
    if (!llvm::isa<OpType>(op) ||
        op->getNumOperands() != sizeof...(OperandMatchers)) {
      return false;
    }

    if (!constraint(op)) {
      return false;
    }

    return RecursivePatternMatcher<OpType, OperandMatchers...>::match(op);
  }

  FuncT constraint = nullptr;
};

template <typename OpType, typename... OperandMatchers>
struct CapturerPatternMatcher
    : public RecursivePatternMatcher<OpType, OperandMatchers...> {
  CapturerPatternMatcher(mlir::Operation **cap, OperandMatchers... matchers)
      : RecursivePatternMatcher<OpType, OperandMatchers...>(false, false, true,
                                                            matchers...),
        capturer(cap) {}
  bool match(mlir::Operation *op) final {
    if (!llvm::isa<OpType>(op) ||
        op->getNumOperands() != sizeof...(OperandMatchers)) {
      return false;
    }

    *capturer = op;

    return RecursivePatternMatcher<OpType, OperandMatchers...>::match(op);
  }

  mlir::Operation **capturer = nullptr;
};

/// Check to see if the specified operation is ConstantLike.
static inline bool isConstantLike(mlir::Operation *op) {
  return op->getNumOperands() == 0 && op->getNumResults() == 1 &&
         op->hasTrait<mlir::OpTrait::ConstantLike>();
}

/// The matcher that matches operations that have the `ConstantLike` trait.
struct ConstantOpMatcher : public MatcherBase {
  bool match(mlir::Operation *op) final { return isConstantLike(op); }
};

template <typename T> struct ConstantOpBinder : public MatcherBase {
  T *value;
  explicit ConstantOpBinder(T *binderValue) : value(binderValue) {}
  bool match(mlir::Operation *op) final {
    if (!isConstantLike(op)) {
      return false;
    }
    llvm::SmallVector<mlir::OpFoldResult, 1> foldedOp;
    mlir::LogicalResult result =
        op->fold(/*operands=*/llvm::ArrayRef<mlir::Attribute>(), foldedOp);
    (void)result;
    assert(succeeded(result) && "expected ConstantLike op to be foldable");
    if (auto attr = llvm::cast<mlir::Attribute>(foldedOp.front())) {
      auto type = op->getResult(0).getType();
      if (llvm::isa<mlir::VectorType, mlir::RankedTensorType>(type)) {
        if (auto splatAttr = llvm::dyn_cast<mlir::SplatElementsAttr>(attr)) {
          attr = splatAttr.getSplatValue<mlir::Attribute>();
        }
      }
      if constexpr (std::is_same_v<T, llvm::APInt>) {
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
          *value = intAttr.getValue();
          return true;
        }
      }
      if constexpr (std::is_same_v<T, llvm::APFloat>) {
        if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
          *value = floatAttr.getValue();
          return true;
        }
      }
    }
    return false;
  }
};

/// Terminal matcher, always returns true.
struct AnyValueMatcher {
  bool match(mlir::Value op) { return true; }

  void mergeCollection(OpList &from) {
    // do nothing;
  }
  OpList collection;
};

/// Terminal matcher, always returns true.
struct AnyCapturedValueMatcher {
  mlir::Value *what;
  explicit AnyCapturedValueMatcher(mlir::Value *what) : what(what) {}
  bool match(mlir::Value op) const {
    *what = op;
    return true;
  }

  void mergeCollection(OpList &from) {
    // do nothing;
  }
  OpList collection;
};

/// Binds to a specific value and matches it.
struct PatternMatcherValue {
  explicit PatternMatcherValue(mlir::Value val) : value(val) {}
  bool match(mlir::Value val) const { return val == value; }
  mlir::Value value;

  void mergeCollection(OpList &from) {
    // do nothing;
  }
  OpList collection;
};

/// Entry point for matching a pattern over an Operation.
template <typename Pattern>
inline bool mPattern(mlir::Operation *op, const Pattern &pattern) {
  return const_cast<Pattern &>(pattern).match(op);
}

/// Matches the given OpType.
template <typename OpType> inline OpMatcher<OpType> mOp() {
  return OpMatcher<OpType>();
}

template <typename OpType, typename... Matchers>
auto mOp(Matchers... matchers) {
  return RecursivePatternMatcher<OpType, Matchers...>(false, false, false,
                                                      matchers...);
}

template <typename OpType, typename... Matchers>
auto mOpOmitBrc(Matchers... matchers) {
  return RecursivePatternMatcher<OpType, Matchers...>(true, false, false,
                                                      matchers...);
}

template <typename OpType, typename... Matchers>
auto mOpCommutative(Matchers... matchers) {
  static_assert(2 == sizeof...(Matchers),
                "only support commutative for binaryOp");
  return RecursivePatternMatcher<OpType, Matchers...>(false, true, false,
                                                      matchers...);
}

template <typename OpType, typename... Matchers>
auto mOpCommutativeOmitBrc(Matchers... matchers) {
  static_assert(2 == sizeof...(Matchers),
                "only support commutative for binaryOp");
  return RecursivePatternMatcher<OpType, Matchers...>(true, true, false,
                                                      matchers...);
}

template <typename OpType, typename... Matchers>
auto mOptional(Matchers... matchers) {
  return OptionalPatternMatcher<OpType, Matchers...>(matchers...);
}

template <typename OpType, typename... Matchers>
auto mOpWithConstraint(const FuncT &cstrt, Matchers... matchers) {
  return ConstraintPatternMatcher<OpType, Matchers...>(cstrt, matchers...);
}

template <typename OpType, typename... Matchers>
auto mOpWithCapturer(mlir::Operation **capturer, Matchers... matchers) {
  return CapturerPatternMatcher<OpType, Matchers...>(capturer, matchers...);
}

/// Matches a constant foldable operation.
inline ConstantOpMatcher mConstant() { return {}; }

template <typename T> inline ConstantOpBinder<T> mConstant(T *value) {
  return ConstantOpBinder<T>(value);
}

/// Terminal matcher - matches any value
inline auto mAny() { return AnyValueMatcher(); }
/// Terminal matcher - matches any value and captures it
inline auto mAny(mlir::Value *val) { return AnyCapturedValueMatcher(val); }
/// Terminal matcher - matches a specific value
inline auto mVal(mlir::Value v) { return PatternMatcherValue(v); }

} // namespace mlir::joy::opt

#undef DEBUG_TYPE

#endif // JOY_SUPPORT_MATCHER_H
