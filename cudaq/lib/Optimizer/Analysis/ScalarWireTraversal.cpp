/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/ScalarWireTraversal.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;

// Returns whether `nested` is inside `outer` through only single-block
// `cc.scope` operations. Any other enclosing region prevents traversal.
static bool entersSingleBlockLexicalScopesOnly(Block *nested, Block *outer) {
  while (nested != outer) {
    if (!nested)
      return false;
    auto scope = dyn_cast_or_null<cudaq::cc::ScopeOp>(nested->getParentOp());
    if (!scope || !cudaq::opt::isSupportedScalarWireScope(scope))
      return false;
    nested = scope->getBlock();
  }
  return true;
}

/// Returns the `cc.continue` that forwards values from a single-block lexical
/// scope. Its positional mapping to the scope results is validated here.
static std::optional<cudaq::cc::ContinueOp>
getSingleBlockScopeContinue(cudaq::cc::ScopeOp scope) {
  if (!scope || !scope.getInitRegion().hasOneBlock())
    return std::nullopt;
  auto cont = dyn_cast<cudaq::cc::ContinueOp>(
      scope.getInitRegion().front().getTerminator());
  if (!cont || cont.getNumOperands() != scope->getNumResults())
    return std::nullopt;
  for (auto [operand, result] :
       llvm::zip(cont.getOperands(), scope->getResults())) {
    bool operandIsWire = isa<cudaq::quake::WireType>(operand.getType());
    bool resultIsWire = isa<cudaq::quake::WireType>(result.getType());
    if (operandIsWire != resultIsWire)
      return std::nullopt;
    if ((cudaq::quake::isQuantumType(operand.getType()) && !operandIsWire) ||
        (cudaq::quake::isQuantumType(result.getType()) && !resultIsWire))
      return std::nullopt;
  }
  return cont;
}

bool cudaq::opt::isSupportedScalarWireScope(Operation *operation) {
  auto scope = dyn_cast_or_null<cudaq::cc::ScopeOp>(operation);
  if (!scope || !getSingleBlockScopeContinue(scope))
    return false;

  bool foundUnwind = false;
  scope.getInitRegion().walk([&](Operation *nested) {
    if (isa<cudaq::cc::UnwindBreakOp, cudaq::cc::UnwindContinueOp,
            cudaq::cc::UnwindReturnOp>(nested)) {
      foundUnwind = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return !foundUnwind;
}

Block *cudaq::opt::getScalarWireTraversalRoot(Block *block) {
  while (block) {
    auto scope = dyn_cast_or_null<cudaq::cc::ScopeOp>(block->getParentOp());
    if (!scope || !isSupportedScalarWireScope(scope))
      return block;
    block = scope->getBlock();
  }
  return nullptr;
}

/// Return whether an operation can be followed as a direct scalar-wire step.
/// Calls, region operations, and terminators require control-flow semantics
/// that this helper deliberately does not model.
static bool isDirectScalarWireStep(Operation *operation) {
  return !isa<CallOpInterface>(operation) && operation->getNumRegions() == 0 &&
         !operation->hasTrait<OpTrait::IsTerminator>();
}

/// Follows a `cc.continue` operand to the corresponding enclosing scope result.
static std::optional<cudaq::opt::ScalarWireStep>
traverseScopeForward(OpOperand *use) {
  auto cont = dyn_cast<cudaq::cc::ContinueOp>(use->getOwner());
  if (!cont)
    return std::nullopt;
  auto scope = dyn_cast<cudaq::cc::ScopeOp>(cont->getParentOp());
  if (!scope || !cudaq::opt::isSupportedScalarWireScope(scope))
    return std::nullopt;
  // `cc.continue` forwards each operand to the scope result at the same
  // position, so the operand number identifies the outgoing scalar wire.
  unsigned index = use->getOperandNumber();
  if (index >= scope->getNumResults() ||
      !isa<cudaq::quake::WireType>(scope->getResult(index).getType()))
    return std::nullopt;
  Value result = scope->getResult(index);
  // Do not cross a scope result that forks after the lexical boundary.
  if (!result.hasOneUse())
    return std::nullopt;
  return cudaq::opt::ScalarWireStep{result, scope, use};
}

std::optional<cudaq::opt::ScalarWireStep>
cudaq::opt::traverseScalarWire(Value wire) {
  if (!isa<cudaq::quake::WireType>(wire.getType()) || !wire.hasOneUse())
    return std::nullopt;

  OpOperand *use = &*wire.getUses().begin();
  Operation *user = use->getOwner();
  // A `cc.continue` forwards a value defined inside a lexical scope to its
  // corresponding scope result.
  if (isa<cudaq::cc::ContinueOp>(user))
    return traverseScopeForward(use);
  if (!isDirectScalarWireStep(user))
    return std::nullopt;
  // Values defined outside a lexical scope are captured implicitly. Accept
  // the use only when every intervening region is a supported scope.
  if (!entersSingleBlockLexicalScopesOnly(user->getBlock(),
                                          wire.getParentBlock()))
    return std::nullopt;
  return ScalarWireStep{wire, user};
}
