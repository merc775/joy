add_mlir_dialect_library(MLIRJoylDialect
  JoylDialect.cpp
  JoylOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/joy/dialect/joyl

  DEPENDS
  MLIRJoylOpsIncGen
  MLIRJoylDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
