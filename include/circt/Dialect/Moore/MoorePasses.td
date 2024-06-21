//===--- Passes.td - Moore pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the Moore dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREPASSES_TD
#define CIRCT_DIALECT_MOORE_MOOREPASSES_TD

include "mlir/Pass/PassBase.td"

def InfoCollection : Pass<"info-collection", "mlir::ModuleOp"> {
    let summary = "Collect all nets and variables declarations";
    let description = [{
      This pass will perform a pre-pass over the IR to find all concurrently-
      assigned nets and variables. Basically go through all net/variable 
      declarations and collect the connected value in a hash map (either the
      initial value in net declaration or the assigned value).
    }];
    let constructor = "circt::moore::createInfoCollectionPass()";
}

def SimplifyProcedures : Pass<"moore-simplify-procedures", "moore::SVModuleOp"> {
    let summary = "Simplify procedures";
    let description = [{
      Cause we want to introduce mem2reg in the moore dialect to eliminate the
      local temporary variables, if the local variabels exist in the procedure
      body, it can be promoted by mem2reg. But global/module-level variables
      don't be promoted. So this pass is aimed at inserting a local "shadow"
      variable in always blocks for every module-level variable that the
      process modifies. Then perform mem2reg to eliminate the local "shadow"
      variable.
    }];
    let constructor = "circt::moore::createSimplifyProceduresPass()";
}

def SimplifyAssigns : Pass<"moore-simplify-assigns", "moore::SVModuleOp"> {
    let summary = "Simplify assigns";
    let description = [{
      It's used to desassemble the LHS of assignments that have a form like
      "{a, b} = c" onto "a = c[9001:42];" and "b = c[41:0]". Aimed at
      conveniently lowering this kind of assignment.
    }];
    let constructor = "circt::moore::createSimplifyAssignsPass()";
}

#endif // CIRCT_DIALECT_MOORE_MOOREPASSES_TD
