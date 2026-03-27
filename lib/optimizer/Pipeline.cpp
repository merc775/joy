//===- Pipeline.cpp - Joy Optimization Pipeline ---------------------------===//
//
// Joy Compiler - Optimization pass pipeline
//
// Registers --joy-optimization-pipeline which chains:
//   1. ConstantFolding  (clean up trivially foldable ops)
//   2. CSE              (eliminate duplicate computations)
//   3. OpFusion         (fuse add + rms_norm -> fuse_add_rmsnorm)
//   4. ConstantFolding  (fold again after fusion created new opportunities)
//   5. CSE              (eliminate any new duplicates after fusion)
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

std::unique_ptr<Pass> createConstantFoldingPass();
std::unique_ptr<Pass> createJoyCSEPass();
std::unique_ptr<Pass> createOpFusionPass();

void registerOptimizationPipeline() {
  static PassPipelineRegistration<> reg(
      "joy-optimization-pipeline",
      "Joy compiler optimization pipeline: "
      "ConstantFolding -> CSE -> OpFusion -> ConstantFolding -> CSE",
      [](OpPassManager &pm) {
        pm.addPass(createConstantFoldingPass());
        pm.addPass(createJoyCSEPass());
        pm.addPass(createOpFusionPass());
        pm.addPass(createConstantFoldingPass());
        pm.addPass(createJoyCSEPass());
      });
}
