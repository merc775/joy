//===- JoyhDialect.cpp - Joyh dialect implementation ----------------------===//
//
// Joy Compiler - HAL dialect implementation
//
//===----------------------------------------------------------------------===//

#include "joy/dialect/joyh/JoyhDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::joyh;

//===----------------------------------------------------------------------===//
// Joyh Dialect
//===----------------------------------------------------------------------===//

#include "joy/dialect/joyh/JoyhDialect.cpp.inc"

void JoyhDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "joy/dialect/joyh/JoyhOps.cpp.inc"
      >();
}
