//===- SimOps.td - `sim` dialect ops -----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for `sim`.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMOPS_TD
#define CIRCT_DIALECT_SIM_SIMOPS_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "circt/Dialect/Sim/SimDialect.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"

class SimOp<string mnemonic, list<Trait> traits = []> :
    Op<SimDialect, mnemonic, traits>;

def PlusArgsTestOp : SimOp<"plusargs.test", [Pure]> {
  let summary = "SystemVerilog `$test$plusargs` call";

  let arguments = (ins StrAttr:$formatString);
  let results = (outs I1:$found);
  let assemblyFormat = "$formatString attr-dict";
}

def PlusArgsValueOp : SimOp<"plusargs.value", [Pure]> {
  let summary = "SystemVerilog `$value$plusargs` call";

  let arguments = (ins StrAttr:$formatString);
  let results = (outs I1:$found, AnyType:$result);
  let assemblyFormat = "$formatString attr-dict `:` type($result)";
}

def FinishOp : SimOp<"finish"> {
  let summary = "Simulation finish condition";

  let arguments = (ins ClockType:$clk, I1:$cond);
  let results = (outs);

  let assemblyFormat = "$clk `,` $cond attr-dict";
}

def FatalOp : SimOp<"fatal"> {
  let summary = "Simulation failure condition";

  let arguments = (ins ClockType:$clk, I1:$cond);
  let results = (outs);

  let assemblyFormat = "$clk `,` $cond attr-dict";
}

def DPIFuncOp : SimOp<"func.dpi",
     [IsolatedFromAbove, Symbol, OpAsmOpInterface,
      FunctionOpInterface, HasParent<"mlir::ModuleOp">]> {
  let summary = "A System Verilog function";
  let description = [{
    `sim.func.dpi` reresents SystemVerilog DPI function in SV spec 13.4.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<ModuleType>:$module_type,
    OptionalAttr<DictArrayAttr>:$per_argument_attrs,
    OptionalAttr<LocationArrayAttr>:$argument_locs,
    OptionalAttr<StrAttr>:$verilogName
  );
  let regions = (region AnyRegion:$body);


  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    mlir::FunctionType getFunctionType() {
      return getModuleType().getFuncType();
    }


    void setFunctionTypeAttr(mlir::TypeAttr mlirType) {
      setModuleType(cast<hw::ModuleType>(mlirType.getValue()));
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    ::mlir::Region *getCallableRegion() { return nullptr; }
  }];
}

def DPICallOp : SimOp<"func.dpi.call",
     [CallOpInterface, AttrSizedOperandSegments, 
      DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "A call option for DPI function with optional clock and enable";
  let description = [{
    `sim.func.dpi.call` reresents SystemVerilog DPI function call. 

  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                      Optional<ClockType>:$clock,
                      Optional<I1>:$enable,
                      Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $callee `(` $inputs `)` (`clock` $clock^)? (`enable` $enable^)?
    attr-dict `:` functional-type($inputs, results)
  }];

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getInputs();
    }
    MutableOperandRange getArgOperandsMutable() {
      return getInputsMutable();
    }
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getCalleeAttrName(), callee.get<mlir::SymbolRefAttr>());
    }
  }];

}

#endif // CIRCT_DIALECT_SIM_SIMOPS_TD
