#!/bin/bash
# JOY Project Build Script
# This script configures and builds the JOY compiler

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

echo "========================================"
echo "  JOY Project Build"
echo "========================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Build directory: ${BUILD_DIR}"

# Load environment if available
ENV_FILE="${BUILD_DIR}/env.sh"
if [ -f "${ENV_FILE}" ]; then
    echo "Loading environment from: ${ENV_FILE}"
    source "${ENV_FILE}"
else
    echo "WARNING: Environment file not found. Running init.sh..."
    "${SCRIPT_DIR}/init.sh"
    if [ -f "${ENV_FILE}" ]; then
        source "${ENV_FILE}"
    fi
fi

# Check for required tools
if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found!"
    exit 1
fi

# Parse command line arguments
BUILD_TYPE="Release"
CLEAN_BUILD=0
NINJA_BUILD=1
JOBS=$(nproc)

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --no-ninja)
            NINJA_BUILD=0
            shift
            ;;
        -j)
            JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--debug|--release] [--clean] [--no-ninja] [-j jobs]"
            exit 1
            ;;
    esac
done

echo "Build type: ${BUILD_TYPE}"
echo "Clean build: ${CLEAN_BUILD}"
echo "Jobs: ${JOBS}"

# Clean build if requested
if [ ${CLEAN_BUILD} -eq 1 ]; then
    echo "Performing clean build..."
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
fi

# Create build directory if it doesn't exist
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Determine generator
GENERATOR="Unix Makefiles"
if [ ${NINJA_BUILD} -eq 1 ] && command -v ninja &> /dev/null; then
    GENERATOR="Ninja"
    echo "Using Ninja build system"
else
    echo "Using Make build system"
fi

# Find LLVM and MLIR.  We prefer LLVM/MLIR 18 in the standard upstream
# install layouts; the user can override either by exporting LLVM_DIR /
# MLIR_DIR before this script runs, or by passing -DLLVM_DIR=... /
# -DMLIR_DIR=... to cmake directly.
SYSTEM_LLVM_PATHS=(
    "/usr/lib/llvm-18/lib/cmake/llvm"
    "/usr/lib64/cmake/llvm"
    "/usr/lib/cmake/llvm"
    "/usr/local/lib/cmake/llvm"
    "/opt/llvm-18/lib/cmake/llvm"
    "/opt/llvm/lib/cmake/llvm"
)

SYSTEM_MLIR_PATHS=(
    "/usr/lib/llvm-18/lib/cmake/mlir"
    "/usr/lib64/cmake/mlir"
    "/usr/lib/cmake/mlir"
    "/usr/local/lib/cmake/mlir"
    "/opt/llvm-18/lib/cmake/mlir"
    "/opt/llvm/lib/cmake/mlir"
)

# Find LLVM
if [ -z "${LLVM_DIR}" ]; then
    for path in "${SYSTEM_LLVM_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/LLVMConfig.cmake" ]; then
            LLVM_DIR="$path"
            echo "Using system LLVM at: ${LLVM_DIR}"
            break
        fi
    done
fi

# Find MLIR
if [ -z "${MLIR_DIR}" ]; then
    for path in "${SYSTEM_MLIR_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/MLIRConfig.cmake" ]; then
            MLIR_DIR="$path"
            echo "Using system MLIR at: ${MLIR_DIR}"
            break
        fi
    done
fi

# If MLIR not found separately, try to find it relative to LLVM
if [ -z "${MLIR_DIR}" ]; then
    MLIR_CANDIDATE="${LLVM_DIR}/../mlir"
    if [ -d "${MLIR_CANDIDATE}" ] && [ -f "${MLIR_CANDIDATE}/MLIRConfig.cmake" ]; then
        MLIR_DIR="${MLIR_CANDIDATE}"
        echo "Using MLIR relative to LLVM at: ${MLIR_DIR}"
    fi
fi

if [ -z "${MLIR_DIR}" ]; then
    echo "ERROR: MLIR directory not found.  Install LLVM 18 + MLIR 18"
    echo "       (e.g. apt-get install mlir-18-tools libmlir-18-dev   for"
    echo "       Debian/Ubuntu, or dnf install mlir18 mlir18-devel for"
    echo "       RHEL/Rocky/Alma), then either export MLIR_DIR=..."
    echo "       pointing at the directory containing MLIRConfig.cmake"
    echo "       or pass -DMLIR_DIR=... to cmake."
    exit 1
fi

# Configure with CMake
echo ""
echo "========================================"
echo "  Configuring CMake..."
echo "========================================"

echo "LLVM_DIR: ${LLVM_DIR}"
echo "MLIR_DIR: ${MLIR_DIR}"

cmake -G "${GENERATOR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DLLVM_DIR="${LLVM_DIR}" \
    -DMLIR_DIR="${MLIR_DIR}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    "${PROJECT_ROOT}"

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed!"
    exit 1
fi

# Build
echo ""
echo "========================================"
echo "  Building..."
echo "========================================"

if [ "${GENERATOR}" = "Ninja" ]; then
    ninja -j${JOBS}
else
    make -j${JOBS}
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "  Build Complete!"
echo "========================================"
echo ""
echo "Build artifacts are in: ${BUILD_DIR}"
echo "Libraries: ${BUILD_DIR}/lib"
echo "Binaries: ${BUILD_DIR}/bin"
echo ""
