# GPU backend operator library
#
# Builds two artifacts from the same sources:
#   1. JOYGpuBackend     - a static library used by the JOY compiler/runtime
#   2. joy_gpu_runtime   - a shared library (.so) loaded by the Python
#                          unit tests under tests/python_tests/test_op via
#                          ctypes.  This shared library also embeds the
#                          test-only helpers in gpu_test_helpers.cpp.

# Allow overriding via -DJOY_ENABLE_CUDA=OFF if the host is missing CUDA.
option(JOY_ENABLE_CUDA "Build the GPU backend with real CUDA kernels and cuBLAS/cuDNN" ON)

if(NOT JOY_ENABLE_CUDA)
  message(STATUS "JOY_ENABLE_CUDA=OFF — building GPU backend stubs only")
  add_library(JOYGpuBackend STATIC
    gpu_ops_stub.cpp
    gpu_entry.cpp
  )
  target_include_directories(JOYGpuBackend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(JOYGpuBackend PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
  )
  return()
endif()

# -------------------------------------------------------------------
# Real CUDA build path
# -------------------------------------------------------------------

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR
    "CUDA compiler not found.  Set CMAKE_CUDA_COMPILER, install nvcc, or "
    "configure with -DJOY_ENABLE_CUDA=OFF.")
endif()
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

# cuDNN: not exposed via CUDAToolkit, so locate it ourselves.
find_path(CUDNN_INCLUDE_DIR
  NAMES cudnn.h cudnn_version.h
  HINTS
    $ENV{CUDNN_HOME}/include
    /usr/local/cuda/include
    /usr/local/include
    /usr/local/include/cudnn
    /usr/include
  )
find_library(CUDNN_LIBRARY
  NAMES cudnn
  HINTS
    $ENV{CUDNN_HOME}/lib64
    $ENV{CUDNN_HOME}/lib
    /usr/local/cuda/lib64
    /usr/local/lib64
    /usr/local/lib
    /usr/lib64
    /usr/lib/x86_64-linux-gnu
  )
if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
  message(FATAL_ERROR
    "cuDNN not found.  Set CUDNN_HOME or pass -DCUDNN_INCLUDE_DIR=... "
    "-DCUDNN_LIBRARY=... or build with -DJOY_ENABLE_CUDA=OFF.")
endif()
message(STATUS "Found cuDNN: include=${CUDNN_INCLUDE_DIR} lib=${CUDNN_LIBRARY}")

# Detect GPU architecture(s).  Default targets common NVIDIA SMs; the user
# can override with -DCMAKE_CUDA_ARCHITECTURES=89 (L20 / Ada).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "")
endif()

# Project-wide flags (-fno-rtti -fno-exceptions) interact badly with cuDNN
# headers and the C++ STL bits used in our launchers.  Drop them for this
# subdirectory only -- this does not propagate up to other targets.
string(REPLACE "-fno-exceptions" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-fno-rtti"        "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# --------------------------------------------------------------------
# Auto-generated CUDA kernel for RMSNorm / FuseAddRMSNorm
#
# Drives joy-emit-cuda over the codegen_stub_*.mlir files at build time
# to produce a self-contained codegen_kernel.cu.  This file replaces
# the hand-written rms_norm / fuse_add_rmsnorm kernels that previously
# lived in gpu_kernels.cu.
# --------------------------------------------------------------------
set(_CODEGEN_KERNEL_CU "${CMAKE_CURRENT_BINARY_DIR}/codegen_kernel.cu")
set(_CODEGEN_STUB_F32 "${CMAKE_CURRENT_SOURCE_DIR}/codegen_stub_f32.mlir")
set(_CODEGEN_STUB_F16 "${CMAKE_CURRENT_SOURCE_DIR}/codegen_stub_f16.mlir")
set(_REGEN_SCRIPT "${PROJECT_SOURCE_DIR}/scripts/regen_codegen_kernel.sh")

add_custom_command(
  OUTPUT  "${_CODEGEN_KERNEL_CU}"
  COMMAND "${_REGEN_SCRIPT}"
          "$<TARGET_FILE:joy-opt>"
          "$<TARGET_FILE:joy-emit-cuda>"
          "${_CODEGEN_STUB_F32}"
          "${_CODEGEN_STUB_F16}"
          "${_CODEGEN_KERNEL_CU}"
  DEPENDS joy-opt joy-emit-cuda
          "${_CODEGEN_STUB_F32}"
          "${_CODEGEN_STUB_F16}"
          "${_REGEN_SCRIPT}"
  COMMENT "Generating codegen_kernel.cu from MLIR via joy-opt + joy-emit-cuda"
  VERBATIM
)
add_custom_target(joy_codegen_kernel_cu DEPENDS "${_CODEGEN_KERNEL_CU}")

set_source_files_properties("${_CODEGEN_KERNEL_CU}"
  PROPERTIES GENERATED TRUE LANGUAGE CUDA)

set(_JOY_GPU_SOURCES
  gpu_ops.cpp
  gpu_entry.cpp
  gpu_kernels.cu
  "${_CODEGEN_KERNEL_CU}"
)

# ----- Static library used by JOY tools/runtime -----
add_library(JOYGpuBackend STATIC ${_JOY_GPU_SOURCES})
add_dependencies(JOYGpuBackend joy_codegen_kernel_cu)
target_include_directories(JOYGpuBackend PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CUDNN_INCLUDE_DIR}
)
set_target_properties(JOYGpuBackend PROPERTIES
  CXX_STANDARD 17
  CUDA_STANDARD 17
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF
)
target_link_libraries(JOYGpuBackend PUBLIC
  CUDA::cudart
  CUDA::cublas
  ${CUDNN_LIBRARY}
)

# ----- Shared library used by Python ctypes tests -----
add_library(joy_gpu_runtime SHARED
  ${_JOY_GPU_SOURCES}
  gpu_test_helpers.cpp
)
add_dependencies(joy_gpu_runtime joy_codegen_kernel_cu)
target_include_directories(joy_gpu_runtime PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CUDNN_INCLUDE_DIR}
)
set_target_properties(joy_gpu_runtime PROPERTIES
  CXX_STANDARD 17
  CUDA_STANDARD 17
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF
  # The library is loaded by ctypes; install in build/lib for the tests.
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
target_link_libraries(joy_gpu_runtime PUBLIC
  CUDA::cudart
  CUDA::cublas
  ${CUDNN_LIBRARY}
)
