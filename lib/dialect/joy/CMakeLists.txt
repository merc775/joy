add_mlir_dialect_library(MLIRJoyDialect
  JoyDialect.cpp
  JoyOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/joy/dialect/joy

  DEPENDS
  MLIRJoyOpsIncGen
  MLIRJoyDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
