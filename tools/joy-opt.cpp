//===- joy-opt.cpp - Joy Optimizer Driver --------------------------------===//
//
// Tool for running Joy dialect passes:
//   --lower-joy-to-joyl   (tensor → memref)
//   --codegen-rms-norm    (joyl.rms_norm → generated GPU kernel)
//   --lower-joyl-to-joyh  (memref → GPU custom calls)
//
//===----------------------------------------------------------------------===//

#include "joy/dialect/joy/JoyDialect.h"
#include "joy/dialect/joyl/JoylDialect.h"
#include "joy/dialect/joyh/JoyhDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

void registerConstantFoldingPass();
void registerJoyCSEPass();
void registerOpFusionPass();
void registerOptimizationPipeline();
void registerLowerJoyToJoylPass();
void registerLowerJoylToJoyhPass();
void registerCodegenRMSNormPass();

int main(int argc, char **argv) {
  registerConstantFoldingPass();
  registerJoyCSEPass();
  registerOpFusionPass();
  registerOptimizationPipeline();
  registerLowerJoyToJoylPass();
  registerLowerJoylToJoyhPass();
  registerCodegenRMSNormPass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::joy::JoyDialect>();
  registry.insert<mlir::joyl::JoylDialect>();
  registry.insert<mlir::joyh::JoyhDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::math::MathDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Joy Optimizer Driver\n", registry));
}
