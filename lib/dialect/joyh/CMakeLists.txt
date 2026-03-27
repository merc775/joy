add_mlir_dialect_library(MLIRJoyhDialect
  JoyhDialect.cpp
  JoyhOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/joy/dialect/joyh

  DEPENDS
  MLIRJoyhOpsIncGen
  MLIRJoyhDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
