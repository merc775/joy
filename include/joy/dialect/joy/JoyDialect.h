//===- JoyDialect.h - Joy dialect -------------------------------*- C++ -*-===//
//
// Joy Compiler - Dialect interface
//
//===----------------------------------------------------------------------===//

#ifndef JOY_DIALECT_JOY_JOYDIALECT_H
#define JOY_DIALECT_JOY_JOYDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "joy/dialect/joy/JoyDialect.h.inc"

#define GET_OP_CLASSES
#include "joy/dialect/joy/JoyOps.h.inc"

#endif // JOY_DIALECT_JOY_JOYDIALECT_H
