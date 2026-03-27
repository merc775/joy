//===- JoylOps.cpp - Joyl operations implementation ------------------------===//
//
// Joy Compiler - Low-level operations implementation
//
//===----------------------------------------------------------------------===//

#include "joy/dialect/joyl/JoylDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::joyl;

//===----------------------------------------------------------------------===//
// Joyl Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "joy/dialect/joyl/JoylOps.cpp.inc"
