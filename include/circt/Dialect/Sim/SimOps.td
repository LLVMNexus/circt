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
include "circt/Dialect/Sim/SimDialect.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "circt/Dialect/HW/HWTypes.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

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

def DPIImportAndCallOp : SimOp<"dpi.import_and_call"> {
  let summary = "Import and call DPI functions";

  let arguments = (ins StrAttr:$function_name,
                       ClockType:$clock,
                       Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = 
    "$function_name $clock `(` $inputs `)` attr-dict `:` functional-type($inputs, results)";
}

def DPIImportOp : SimOp<"dpi.import", [IsolatedFromAbove, Symbol,
      FunctionOpInterface, HasParent<"mlir::ModuleOp">]> {
  let summary = "Import a DPI function";

  let arguments = (ins 
      SymbolNameAttr:$sym_name,
      TypeAttrOf<ModuleType>:$module_type,
      OptionalAttr<StrAttr>:$verilogName);

  let assemblyFormat = "$sym_name (`function_name` $verilogName^)? attr-dict `:` $module_type $body";
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    mlir::FunctionType getFunctionType() {
      return getModuleType().getFuncType();
    }

    void setFunctionTypeAttr(mlir::TypeAttr mlirType) {
      // FIXME: This is really unsafe ...... probably we should implement our own 
      //        FunctionOpInterface for named/ordered types.
      assert(false);
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    ::mlir::Region *getCallableRegion() { return nullptr; }
  }];
}

def DPICallOp : SimOp<"dpi.call", [CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Simulation failure condition";

  let arguments = (ins FlatSymbolRefAttr:$callee, 
      ClockType:$clock, Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = "$callee $clock `(` $inputs `)` attr-dict `:` functional-type($inputs, results)";
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
