//===- JoyhDialect.td - Joyh dialect definition ------------*- tablegen -*-===//
//
// Joy Compiler - Hardware Abstraction Layer dialect
//
//===----------------------------------------------------------------------===//

#ifndef JOYH_DIALECT
#define JOYH_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Joyh Dialect Definition
//===----------------------------------------------------------------------===//

def Joyh_Dialect : Dialect {
  let name = "joyh";
  let summary = "Joyh hardware abstraction layer dialect";
  let description = [{
    The Joyh dialect represents GPU custom calls that dispatch to
    backend operator libraries (cuDNN, CUTLASS, etc.) at runtime.
  }];
  let cppNamespace = "::mlir::joyh";
  let usePropertiesForAttributes = 0;
}

//===----------------------------------------------------------------------===//
// Joyh Operation Base Class
//===----------------------------------------------------------------------===//

class Joyh_Op<string mnemonic, list<Trait> traits = []> :
    Op<Joyh_Dialect, mnemonic, traits>;

#endif // JOYH_DIALECT
