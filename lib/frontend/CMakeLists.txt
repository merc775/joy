add_mlir_library(JOYFrontend
  JoyBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/joy/frontend

  DEPENDS
  MLIRJoyOpsIncGen
  MLIRJoyDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRJoyDialect
)
