//===- JoylDialect.cpp - Joyl dialect implementation ----------------------===//
//
// Joy Compiler - Low-level dialect implementation
//
//===----------------------------------------------------------------------===//

#include "joy/dialect/joyl/JoylDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::joyl;

//===----------------------------------------------------------------------===//
// Joyl Dialect
//===----------------------------------------------------------------------===//

#include "joy/dialect/joyl/JoylDialect.cpp.inc"

void JoylDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "joy/dialect/joyl/JoylOps.cpp.inc"
      >();
}
