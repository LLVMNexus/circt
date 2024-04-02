//===- HWMemSimImpl.cpp - HW Memory Implementation Pass -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass converts generated FIRRTL memory modules to
// simulation models.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

//===----------------------------------------------------------------------===//
// SeparateImportAndCall Pass
//===----------------------------------------------------------------------===//

namespace circt {
namespace sim {
#define GEN_PASS_DEF_SEPARATEIMPORTANDCALL
#include "circt/Dialect/Sim/SimPasses.h.inc"
} // namespace sim
} // namespace circt

using namespace circt;
using namespace sim;
using namespace hw;
namespace {
struct SeparateImportAndCallPass
    : public circt::sim::impl::SeparateImportAndCallBase<
          SeparateImportAndCallPass> {
  using SeparateImportAndCallBase::SeparateImportAndCallBase;

  void runOnOperation() override;
};

} // end anonymous namespace

void SeparateImportAndCallPass::runOnOperation() {
  auto topModule = getOperation();

  llvm::DenseMap<std::pair<StringAttr, hw::ModuleType>, sim::DPIImportOp>
      dpiDefinitions;

  // TOOD: Map-reduce parallely
  SmallVector<sim::DPIImportAndCallOp> ops;
  topModule.walk([&](sim::DPIImportAndCallOp op) { ops.push_back(op); });

  for (auto op : ops) {
    op.getFunctionNameAttr();
    auto input_types = op.getInputs().getTypes();
    auto output_types = op.getResultTypes();
    mlir::ImplicitLocOpBuilder builder(op.getLoc(), op);
    SmallVector<hw::ModulePort> ports;
    ports.reserve(input_types.size() + output_types.size());

    for (auto [idx, in_type] : llvm::enumerate(input_types)) {
      ModulePort port;
      port.dir = ModulePort::Direction::Input;
      // FIXME: Fix this once DPI intrinsic is deprecated.
      port.name = builder.getStringAttr(Twine("in_") + Twine(idx));
      port.type = in_type;
      ports.push_back(port);
    }

    for (auto [idx, out_type] : llvm::enumerate(output_types)) {
      ModulePort port;
      port.dir = ModulePort::Direction::Output;
      // FIXME: Fix this once DPI intrinsic is deprecated.
      port.name = builder.getStringAttr(Twine("out_") + Twine(idx));
      port.type = out_type;
      ports.push_back(port);
    }

    auto modType = hw::ModuleType::get(&getContext(), ports);
    auto &dpiImport = dpiDefinitions[{op.getFunctionNameAttr(), modType}];
    if (!dpiImport) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(topModule.getBody());
      // TODO: Legalize symbols.
      dpiImport = builder.create<sim::DPIImportOp>(op.getFunctionNameAttr(),
                                                   modType, StringAttr());
    }

    auto call = builder.create<sim::DPICallOp>(op.getResultTypes(),
                                               op.getFunctionNameAttr(),
                                               op.getClock(), op.getInputs());
    op.replaceAllUsesWith(call.getResults());
    op.erase();
  }
}

std::unique_ptr<mlir::Pass> circt::sim::createSeparateImportAndCall() {
  return std::make_unique<SeparateImportAndCallPass>();
}
