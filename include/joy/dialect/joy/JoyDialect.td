//===- JoyDialect.td - Joy dialect definition --------------*- tablegen -*-===//
//
// Joy Compiler - Simplified dialect for Qwen 0.6B
//
//===----------------------------------------------------------------------===//

#ifndef JOY_DIALECT
#define JOY_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Joy Dialect Definition
//===----------------------------------------------------------------------===//

def Joy_Dialect : Dialect {
  let name = "joy";
  let summary = "Joy high-level operations dialect for Qwen 0.6B";
  let description = [{
    The Joy dialect contains high-level operations used in Qwen 0.6B model.
  }];
  let cppNamespace = "::mlir::joy";
  let usePropertiesForAttributes = 0;
}

//===----------------------------------------------------------------------===//
// Joy Type Definitions
//===----------------------------------------------------------------------===//

class Joy_Type<string name, string typeMnemonic> : TypeDef<Joy_Dialect, name> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Joy Operation Base Class
//===----------------------------------------------------------------------===//

class Joy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Joy_Dialect, mnemonic, traits>;

#endif // JOY_DIALECT
