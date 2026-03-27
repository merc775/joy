//===- JoyOps.cpp - Joy operations implementation ---------------------------===//
//
// Joy Compiler - Operations implementation
//
//===----------------------------------------------------------------------===//

#include "joy/dialect/joy/JoyDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::joy;

//===----------------------------------------------------------------------===//
// Joy Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "joy/dialect/joy/JoyOps.cpp.inc"
