set(LLVM_TARGET_DEFINITIONS JoyDialect.td)
mlir_tablegen(JoyDialect.h.inc -gen-dialect-decls -dialect=joy)
mlir_tablegen(JoyDialect.cpp.inc -gen-dialect-defs -dialect=joy)
add_public_tablegen_target(MLIRJoyDialectIncGen)

set(LLVM_TARGET_DEFINITIONS JoyOps.td)
mlir_tablegen(JoyOps.h.inc -gen-op-decls)
mlir_tablegen(JoyOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRJoyOpsIncGen)
