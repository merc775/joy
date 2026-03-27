get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(joy-opt
  PARTIAL_SOURCES_INTENDED
  joy-opt.cpp
)

llvm_update_compile_flags(joy-opt)

target_link_libraries(joy-opt PRIVATE
  MLIRJoyDialect
  MLIRJoylDialect
  MLIRJoyhDialect
  JOYOptimizer
  ${dialect_libs}
  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMathDialect
)

# joy-emit-cuda: walks gpu_kernel func.func ops produced by the
# CodegenRMSNormPass and prints CUDA C source.  Used by
# joy/lib/backend/gpu/CMakeLists.txt to materialise codegen_kernel.cu
# at build time.
add_llvm_executable(joy-emit-cuda
  PARTIAL_SOURCES_INTENDED
  joy-emit-cuda.cpp
)

llvm_update_compile_flags(joy-emit-cuda)

target_link_libraries(joy-emit-cuda PRIVATE
  MLIRJoyDialect
  MLIRJoylDialect
  MLIRJoyhDialect
  JOYOptimizer
  ${dialect_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRMathDialect
)
