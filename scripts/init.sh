#!/bin/bash
# JOY Project Initialization Script
# This script sets up the build environment for the JOY compiler

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "========================================"
echo "  JOY Project Initialization"
echo "========================================"
echo "Project root: ${PROJECT_ROOT}"

# Find LLVM/MLIR 18 installation.
#
# 1) If the caller already exported LLVM_DIR / MLIR_DIR / LLVM18_HOME we
#    honour them verbatim (no autodetection, no surprise).
# 2) Otherwise we look in the standard upstream LLVM-18 package layouts:
#      apt.llvm.org (Debian/Ubuntu)                 -> /usr/lib/llvm-18/lib/cmake
#      dnf llvm18 / mlir18 (RHEL/Rocky/Alma/Fedora) -> /usr/lib64/cmake
#      manual source build                          -> /opt/llvm-18/lib/cmake
#                                                   -> /usr/local/lib/cmake
SYSTEM_LLVM_PATHS=(
    "/usr/lib/llvm-18/lib/cmake/llvm"
    "/usr/lib64/cmake/llvm"
    "/usr/lib/cmake/llvm"
    "/usr/local/lib/cmake/llvm"
    "/opt/llvm-18/lib/cmake/llvm"
    "/opt/llvm/lib/cmake/llvm"
)

if [ -n "${LLVM18_HOME:-}" ]; then
    SYSTEM_LLVM_PATHS=("${LLVM18_HOME}/lib/cmake/llvm"
                       "${LLVM18_HOME}/lib64/cmake/llvm"
                       "${SYSTEM_LLVM_PATHS[@]}")
fi

# Honour an already-exported LLVM_DIR verbatim.
if [ -n "${LLVM_DIR:-}" ] && [ -f "${LLVM_DIR}/LLVMConfig.cmake" ]; then
    echo "Using user-provided LLVM_DIR: ${LLVM_DIR}"
else
    LLVM_DIR=""
    for path in "${SYSTEM_LLVM_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/LLVMConfig.cmake" ]; then
            export LLVM_DIR="$path"
            echo "Using system LLVM at: ${LLVM_DIR}"
            break
        fi
    done
fi

if [ -z "${LLVM_DIR}" ]; then
    echo "ERROR: Could not find LLVM 18 installation!"
    echo "       Install LLVM 18 (apt.llvm.org / dnf llvm18 / source) or"
    echo "       export LLVM_DIR=<dir containing LLVMConfig.cmake> before"
    echo "       running scripts/init.sh."
    exit 1
fi

# Find MLIR 18 the same way as LLVM 18.
SYSTEM_MLIR_PATHS=(
    "/usr/lib/llvm-18/lib/cmake/mlir"
    "/usr/lib64/cmake/mlir"
    "/usr/lib/cmake/mlir"
    "/usr/local/lib/cmake/mlir"
    "/opt/llvm-18/lib/cmake/mlir"
    "/opt/llvm/lib/cmake/mlir"
)

if [ -n "${LLVM18_HOME:-}" ]; then
    SYSTEM_MLIR_PATHS=("${LLVM18_HOME}/lib/cmake/mlir"
                       "${LLVM18_HOME}/lib64/cmake/mlir"
                       "${SYSTEM_MLIR_PATHS[@]}")
fi

if [ -n "${MLIR_DIR:-}" ] && [ -f "${MLIR_DIR}/MLIRConfig.cmake" ]; then
    echo "Using user-provided MLIR_DIR: ${MLIR_DIR}"
else
    MLIR_DIR=""
    for path in "${SYSTEM_MLIR_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/MLIRConfig.cmake" ]; then
            export MLIR_DIR="$path"
            echo "Found MLIR at: ${MLIR_DIR}"
            break
        fi
    done
fi

# If system MLIR not found, try relative to LLVM
if [ -z "${MLIR_DIR}" ]; then
    MLIR_DIR="${LLVM_DIR}/../mlir"
    if [ ! -d "${MLIR_DIR}" ]; then
        MLIR_DIR="${LLVM_DIR}/../cmake/mlir"
    fi
    
    if [ -d "${MLIR_DIR}" ]; then
        export MLIR_DIR
        echo "Found MLIR relative to LLVM at: ${MLIR_DIR}"
    else
        echo "WARNING: MLIR directory not found at ${MLIR_DIR}"
        echo "Will rely on LLVM's MLIR integration"
    fi
fi

# Create build directory
BUILD_DIR="${PROJECT_ROOT}/build"
if [ ! -d "${BUILD_DIR}" ]; then
    mkdir -p "${BUILD_DIR}"
    echo "Created build directory: ${BUILD_DIR}"
fi

# Export environment variables
export PATH="${LLVM_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${LLVM_DIR}/lib:${LD_LIBRARY_PATH}"

# Save environment to file for later use
ENV_FILE="${PROJECT_ROOT}/build/env.sh"
cat > "${ENV_FILE}" <<EOF
#!/bin/bash
# Auto-generated environment file for JOY
export LLVM_DIR="${LLVM_DIR}"
export MLIR_DIR="${MLIR_DIR}"
export PATH="${LLVM_DIR}/bin:\${PATH}"
export LD_LIBRARY_PATH="${LLVM_DIR}/lib:\${LD_LIBRARY_PATH}"
EOF

chmod +x "${ENV_FILE}"

echo ""
echo "========================================"
echo "  Initialization Complete!"
echo "========================================"
echo ""
echo "Environment file created at: ${ENV_FILE}"
echo "To build the project, run:"
echo "  cd ${PROJECT_ROOT}"
echo "  ./scripts/build.sh"
echo ""
