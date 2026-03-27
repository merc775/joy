# Optimizer library for Joy compiler

add_mlir_library(JOYOptimizer
  ConstantFolding.cpp
  CSE.cpp
  OpFusionPass.cpp
  Pipeline.cpp
  LowerJoyToJoylPass.cpp
  LowerJoylToJoyhPass.cpp
  CodegenRMSNormPass.cpp
  EmitCudaC.cpp

  LINK_LIBS PUBLIC
  MLIRJoyDialect
  MLIRJoylDialect
  MLIRJoyhDialect
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
  MLIRPass
  MLIRRewrite
  MLIRTensorDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRMathDialect
)
