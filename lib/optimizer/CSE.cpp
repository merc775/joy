//===- CSE.cpp - Joy Common Sub-Expression Elimination --------------------===//
//
// Joy Compiler - CSE pass
//
// Eliminates redundant operations: if two side-effect-free operations
// in the same scope have identical op-name, operands, attributes and
// result types, the later one is replaced by the earlier one.
//
// The implementation delegates to MLIR's `eliminateCommonSubExpressions`
// utility which performs a dominance-aware, scoped hash-table walk.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CSE.h"

using namespace mlir;

namespace {

struct JoyCSEPass : public PassWrapper<JoyCSEPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(JoyCSEPass)

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    DominanceInfo domInfo(getOperation());
    bool changed = false;
    eliminateCommonSubExpressions(rewriter, domInfo, getOperation(), &changed);
  }

  StringRef getArgument() const override { return "joy-cse"; }
  StringRef getDescription() const override {
    return "Eliminate common sub-expressions in Joy dialect";
  }
};

} // namespace

void registerJoyCSEPass() {
  PassRegistration<JoyCSEPass>();
}

std::unique_ptr<Pass> createJoyCSEPass() {
  return std::make_unique<JoyCSEPass>();
}
