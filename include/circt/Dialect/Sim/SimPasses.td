//===-- SimPasses.td - Sim pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the Sim dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMPASSES
#define CIRCT_DIALECT_SIM_SIMPASSES

include "mlir/Pass/PassBase.td"

def SeparateImportAndCall: Pass<"separate-import-and-call-op", "mlir::ModuleOp"> {
  let summary = "Lower sim.dpi.import_and_call op into import and call ops";
  let constructor = "circt::sim::createSeparateImportAndCall()";
}

#endif // CIRCT_DIALECT_SIM_SIMPASSES
