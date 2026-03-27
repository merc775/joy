//===- JoylDialect.h - Joyl dialect -----------------------------*- C++ -*-===//
//
// Joy Compiler - Low-level dialect interface
//
//===----------------------------------------------------------------------===//

#ifndef JOY_DIALECT_JOYL_JOYLDIALECT_H
#define JOY_DIALECT_JOYL_JOYLDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "joy/dialect/joyl/JoylDialect.h.inc"

#define GET_OP_CLASSES
#include "joy/dialect/joyl/JoylOps.h.inc"

#endif // JOY_DIALECT_JOYL_JOYLDIALECT_H
