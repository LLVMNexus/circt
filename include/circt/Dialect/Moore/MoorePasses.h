//===- Passes.h - Moore pass entry points -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREPASSES_H
#define CIRCT_DIALECT_MOORE_MOOREPASSES_H

#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace moore {

#define GEN_PASS_DECL
#include "circt/Dialect/Moore/MoorePasses.h.inc"

class InfoCollection {
  // A map used to collect nets or variables and their values.
  // Only record the value produced by the last assignment op,
  // including the declaration assignment.
  DenseMap<Operation *, Value> assignmentChains;

public:
  void addValue(Operation *op);

  auto getValue(Operation *op) { return assignmentChains.lookup(op); }
};

extern InfoCollection decl;
std::unique_ptr<mlir::Pass> createInfoCollectionPass();

std::unique_ptr<mlir::Pass> createSimplifyProceduresPass();
std::unique_ptr<mlir::Pass> createSimplifyAssignsPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Moore/MoorePasses.h.inc"

} // namespace moore
} // namespace circt

#endif // CIRCT_DIALECT_MOORE_MOOREPASSES_H
