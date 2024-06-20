//===- DeleteLocalVar.cpp - Delete local temporary variables --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SimplifyAssigns pass.
// It's used to disassemble the moore.concat_ref. Which is tricky to lower
// directly. For example, disassemble "{a, b} = c" onto "a = c[7:3]"
// and "b = c[2:0]".
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_SIMPLIFYASSIGNS
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;

namespace {
struct SimplifyAssignsPass
    : public circt::moore::impl::SimplifyAssignsBase<SimplifyAssignsPass> {
  void runOnOperation() override;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createSimplifyAssignsPass() {
  return std::make_unique<SimplifyAssignsPass>();
}

void SimplifyAssignsPass::runOnOperation() {
  getOperation()->walk([&](ConcatRefOp concatRefOp) {
    mlir::OpBuilder builder(&getContext());

    // moore.concat_ref only as the LHS of assignments.
    if (auto *user = *concatRefOp->getUsers().begin()) {
      auto src = user->getOperand(1);
      auto srcWidth = cast<UnpackedType>(src.getType()).getBitSize().value();

      // Handle all operands of moore.concat_ref.
      for (auto operand : concatRefOp.getValues()) {
        auto type = cast<RefType>(operand.getType()).getNestedType();
        auto width = type.getBitSize().value();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(user);
        auto i32 = moore::IntType::getInt(&getContext(), 32);
        auto lowBit = builder.create<ConstantOp>(concatRefOp.getLoc(), i32,
                                                 srcWidth - width);
        auto extract =
            builder.create<ExtractOp>(src.getLoc(), type, src, lowBit);

        // Update the real bit width of RHS of assignment. Like "c" the above
        // description mentioned.
        srcWidth = srcWidth - width;

        // Estimate which assignment needs to be created.
        if (isa<ContinuousAssignOp>(user))
          builder.create<ContinuousAssignOp>(
              user->getLoc(), operand.getDefiningOp()->getResult(0), extract);
        else if (isa<BlockingAssignOp>(user))
          builder.create<BlockingAssignOp>(
              user->getLoc(), operand.getDefiningOp()->getResult(0), extract);
        else
          builder.create<NonBlockingAssignOp>(
              user->getLoc(), operand.getDefiningOp()->getResult(0), extract);
      }

      // Delete the original moore.concat_ref and its user--moore.*assign.
      user->erase();
      concatRefOp->erase();
    }
    return WalkResult::advance();
  });
}
