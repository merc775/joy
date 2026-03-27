//===- JoyhDialect.h - Joyh dialect -----------------------------*- C++ -*-===//
//
// Joy Compiler - HAL dialect interface
//
//===----------------------------------------------------------------------===//

#ifndef JOY_DIALECT_JOYH_JOYHDIALECT_H
#define JOY_DIALECT_JOYH_JOYHDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "joy/dialect/joyh/JoyhDialect.h.inc"

#define GET_OP_CLASSES
#include "joy/dialect/joyh/JoyhOps.h.inc"

#endif // JOY_DIALECT_JOYH_JOYHDIALECT_H
