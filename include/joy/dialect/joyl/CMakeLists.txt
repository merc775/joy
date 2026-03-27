set(LLVM_TARGET_DEFINITIONS JoylDialect.td)
mlir_tablegen(JoylDialect.h.inc -gen-dialect-decls -dialect=joyl)
mlir_tablegen(JoylDialect.cpp.inc -gen-dialect-defs -dialect=joyl)
add_public_tablegen_target(MLIRJoylDialectIncGen)

set(LLVM_TARGET_DEFINITIONS JoylOps.td)
mlir_tablegen(JoylOps.h.inc -gen-op-decls)
mlir_tablegen(JoylOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRJoylOpsIncGen)
