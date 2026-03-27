set(LLVM_TARGET_DEFINITIONS JoyhDialect.td)
mlir_tablegen(JoyhDialect.h.inc -gen-dialect-decls -dialect=joyh)
mlir_tablegen(JoyhDialect.cpp.inc -gen-dialect-defs -dialect=joyh)
add_public_tablegen_target(MLIRJoyhDialectIncGen)

set(LLVM_TARGET_DEFINITIONS JoyhOps.td)
mlir_tablegen(JoyhOps.h.inc -gen-op-decls)
mlir_tablegen(JoyhOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRJoyhOpsIncGen)
