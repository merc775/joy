//===- JoyDialect.cpp - Joy dialect implementation -------------------------===//
//
// Joy Compiler - Dialect implementation
//
//===----------------------------------------------------------------------===//

#include "joy/dialect/joy/JoyDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::joy;

//===----------------------------------------------------------------------===//
// Joy Dialect
//===----------------------------------------------------------------------===//

#include "joy/dialect/joy/JoyDialect.cpp.inc"

void JoyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "joy/dialect/joy/JoyOps.cpp.inc"
      >();
}
