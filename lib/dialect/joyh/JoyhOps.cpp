//===- JoyhOps.cpp - Joyh operations implementation ------------------------===//
//
// Joy Compiler - HAL operations implementation
//
//===----------------------------------------------------------------------===//

#include "joy/dialect/joyh/JoyhDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::joyh;

//===----------------------------------------------------------------------===//
// Joyh Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "joy/dialect/joyh/JoyhOps.cpp.inc"
