cmake_minimum_required(VERSION 3.15.0)

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(JOY LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Building JOY - A simplified compiler for Qwen 0.6B")

#########################################################
################ Find Python #############################
#########################################################
find_package(Python3 REQUIRED COMPONENTS Interpreter)

#########################################################
################ Set library targets #####################
#########################################################
set(JOY_DIALECT_LIBRARY "JoyDialect")
set(JOYL_DIALECT_LIBRARY "JoylDialect")
set(JOYH_DIALECT_LIBRARY "JoyhDialect")
set(JOY_RUNTIME_LIBRARY "joy_runtime")

#########################################################
################ Set C++ flags ###########################
#########################################################
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

set(JOY_WARNING_FLAGS "-Wall -Wextra -Wno-unused-parameter -Wwrite-strings")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti -fdiagnostics-color -fPIC ${JOY_WARNING_FLAGS}")

#########################################################
################### Find MLIR ############################
#########################################################
# LLVM_DIR / MLIR_DIR can be overridden by the caller (recommended via
# `cmake -DMLIR_DIR=... -DLLVM_DIR=...` or by exporting both as env vars
# before invoking cmake -- see scripts/init.sh).  When unset we fall
# back to the standard install layouts shipped by upstream LLVM packages
# (apt.llvm.org Debian/Ubuntu, dnf llvm18 RHEL/Rocky/Alma).
if(NOT DEFINED MLIR_DIR)
    foreach(_p
        "/usr/lib/llvm-18/lib/cmake/mlir"
        "/usr/lib64/cmake/mlir"
        "/usr/lib/cmake/mlir"
        "/opt/llvm-18/lib/cmake/mlir"
        "/usr/local/lib/cmake/mlir")
        if(EXISTS "${_p}/MLIRConfig.cmake")
            set(MLIR_DIR "${_p}")
            break()
        endif()
    endforeach()
endif()

if(NOT DEFINED LLVM_DIR)
    foreach(_p
        "/usr/lib/llvm-18/lib/cmake/llvm"
        "/usr/lib64/cmake/llvm"
        "/usr/lib/cmake/llvm"
        "/opt/llvm-18/lib/cmake/llvm"
        "/usr/local/lib/cmake/llvm")
        if(EXISTS "${_p}/LLVMConfig.cmake")
            set(LLVM_DIR "${_p}")
            break()
        endif()
    endforeach()
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#########################################################
################ Add subdirectories ######################
#########################################################
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

message(STATUS "JOY build configuration complete")
