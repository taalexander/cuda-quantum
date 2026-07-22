/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/CommutationAnalysis.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Matchers.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>

using namespace mlir;

using cudaq::quake::detail::CommutationAnalysis;
using cudaq::quake::detail::CommutationReason;
using cudaq::quake::detail::CommutationResult;
using cudaq::quake::detail::CommutationStatus;

namespace {
using QuantumIdentity = std::uint32_t;
using OperationPair = std::pair<Operation *, Operation *>;
using BorrowKey = std::pair<Attribute, std::int32_t>;

struct ControlUse {
  QuantumIdentity identity;
  bool negated;

  bool operator==(const ControlUse &) const = default;
};

struct OperationView {
  OperationView(Operation *operation,
                cudaq::quake::OperatorInterface operatorInterface)
      : operation(operation), interface(operatorInterface) {}

  Operation *operation;
  cudaq::quake::OperatorInterface interface;
  llvm::SmallVector<ControlUse> controls;
  llvm::SmallVector<QuantumIdentity> targets;
  llvm::DenseSet<QuantumIdentity> support;
  llvm::DenseSet<QuantumIdentity> targetIdentities;
  llvm::DenseMap<QuantumIdentity, bool> controlPolarities;
};

struct PauliAction {
  explicit PauliAction(bool isFixed) : isFixed(isFixed) {}

  llvm::DenseMap<QuantumIdentity, char> terms;
  bool isFixed;
};
} // namespace

static CommutationResult commutes(CommutationReason reason) {
  return {CommutationStatus::Commutes, reason};
}

static CommutationResult doesNotCommute(CommutationReason reason) {
  return {CommutationStatus::DoesNotCommute, reason};
}

static CommutationResult indeterminate(CommutationReason reason) {
  return {CommutationStatus::Indeterminate, reason};
}

static OperationPair getCanonicalPair(Operation *lhs, Operation *rhs) {
  if (std::less<Operation *>{}(rhs, lhs))
    std::swap(lhs, rhs);
  return {lhs, rhs};
}

static bool isCustomUnitary(Operation *operation) {
  return isa<cudaq::quake::CustomUnitaryCallOp,
             cudaq::quake::CustomUnitaryConstantOp>(operation);
}

static bool isSupportedSharedOperation(Operation *operation) {
  return isa<cudaq::quake::HOp, cudaq::quake::XOp, cudaq::quake::YOp,
             cudaq::quake::ZOp, cudaq::quake::SOp, cudaq::quake::TOp,
             cudaq::quake::SwapOp, cudaq::quake::R1Op, cudaq::quake::RxOp,
             cudaq::quake::RyOp, cudaq::quake::RzOp, cudaq::quake::PhasedRxOp,
             cudaq::quake::U2Op, cudaq::quake::U3Op, cudaq::quake::ExpPauliOp>(
      operation);
}

static bool isComputationalDiagonal(Operation *operation) {
  return isa<cudaq::quake::ZOp, cudaq::quake::SOp, cudaq::quake::TOp,
             cudaq::quake::R1Op, cudaq::quake::RzOp>(operation);
}

static bool isXAxis(Operation *operation) {
  return isa<cudaq::quake::XOp, cudaq::quake::RxOp>(operation);
}

static bool isYAxis(Operation *operation) {
  return isa<cudaq::quake::YOp, cudaq::quake::RyOp>(operation);
}

static bool isZAxis(Operation *operation) {
  return isa<cudaq::quake::ZOp, cudaq::quake::SOp, cudaq::quake::TOp,
             cudaq::quake::R1Op, cudaq::quake::RzOp>(operation);
}

static bool areExactParameters(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  if (lhs.getType() != rhs.getType())
    return false;

  Attribute lhsConstant;
  Attribute rhsConstant;
  return matchPattern(lhs, m_Constant(&lhsConstant)) &&
         matchPattern(rhs, m_Constant(&rhsConstant)) &&
         lhsConstant == rhsConstant;
}

static bool areExactParameters(cudaq::quake::OperatorInterface lhs,
                               cudaq::quake::OperatorInterface rhs) {
  auto lhsParameters = lhs.getParameters();
  auto rhsParameters = rhs.getParameters();
  return lhsParameters.size() == rhsParameters.size() &&
         llvm::equal(lhsParameters, rhsParameters, [](Value lhs, Value rhs) {
           return areExactParameters(lhs, rhs);
         });
}

static bool haveSameTargets(const OperationView &lhs,
                            const OperationView &rhs) {
  if (lhs.targets.size() != rhs.targets.size())
    return false;
  if (isa<cudaq::quake::SwapOp>(lhs.operation) &&
      isa<cudaq::quake::SwapOp>(rhs.operation))
    return lhs.targets.size() == 2 && ((lhs.targets[0] == rhs.targets[0] &&
                                        lhs.targets[1] == rhs.targets[1]) ||
                                       (lhs.targets[0] == rhs.targets[1] &&
                                        lhs.targets[1] == rhs.targets[0]));
  return lhs.targets == rhs.targets;
}

static std::optional<llvm::StringRef> getPauliWord(const OperationView &view) {
  auto expPauli = dyn_cast<cudaq::quake::ExpPauliOp>(view.operation);
  if (!expPauli)
    return std::nullopt;
  auto literal = expPauli.getPauliLiteralAttr();
  if (!literal || literal.getValue().size() != view.targets.size())
    return std::nullopt;
  if (!llvm::all_of(literal.getValue(), [](char pauli) {
        return pauli == 'I' || pauli == 'X' || pauli == 'Y' || pauli == 'Z';
      }))
    return std::nullopt;
  return literal.getValue();
}

static bool hasSupportedPauliWord(const OperationView &view) {
  return !isa<cudaq::quake::ExpPauliOp>(view.operation) ||
         getPauliWord(view).has_value();
}

static bool haveSameOperation(const OperationView &lhs,
                              const OperationView &rhs) {
  if (!isSupportedSharedOperation(lhs.operation) ||
      !isSupportedSharedOperation(rhs.operation) ||
      lhs.operation->getName() != rhs.operation->getName() ||
      lhs.controls != rhs.controls || !haveSameTargets(lhs, rhs) ||
      !areExactParameters(lhs.interface, rhs.interface))
    return false;

  if (isa<cudaq::quake::ExpPauliOp>(lhs.operation))
    return getPauliWord(lhs) == getPauliWord(rhs);
  return true;
}

static bool haveSameAxisTargetAction(const OperationView &lhs,
                                     const OperationView &rhs) {
  if (lhs.targets.size() != 1 || rhs.targets.size() != 1 ||
      lhs.targets.front() != rhs.targets.front())
    return false;
  if ((isXAxis(lhs.operation) && isXAxis(rhs.operation)) ||
      (isYAxis(lhs.operation) && isYAxis(rhs.operation)) ||
      (isZAxis(lhs.operation) && isZAxis(rhs.operation)))
    return true;

  auto lhsPhasedRx = dyn_cast<cudaq::quake::PhasedRxOp>(lhs.operation);
  auto rhsPhasedRx = dyn_cast<cudaq::quake::PhasedRxOp>(rhs.operation);
  return lhsPhasedRx && rhsPhasedRx &&
         areExactParameters(lhsPhasedRx.getParameters()[1],
                            rhsPhasedRx.getParameters()[1]);
}

static std::optional<PauliAction> getPauliAction(const OperationView &view) {
  char pauli = 0;
  if (isa<cudaq::quake::XOp>(view.operation))
    pauli = 'X';
  else if (isa<cudaq::quake::YOp>(view.operation))
    pauli = 'Y';
  else if (isa<cudaq::quake::ZOp>(view.operation))
    pauli = 'Z';

  if (pauli) {
    if (view.targets.size() != 1)
      return std::nullopt;
    PauliAction action(/*isFixed=*/true);
    action.terms.try_emplace(view.targets.front(), pauli);
    return action;
  }

  auto word = getPauliWord(view);
  if (!word)
    return std::nullopt;
  PauliAction action(/*isFixed=*/false);
  action.terms.reserve(view.targets.size());
  for (auto [identity, character] : llvm::zip(view.targets, *word))
    action.terms.try_emplace(identity, character);
  return action;
}

static unsigned getPauliAnticommutationParity(const PauliAction &lhs,
                                              const PauliAction &rhs) {
  const auto *smaller = &lhs.terms;
  const auto *larger = &rhs.terms;
  if (larger->size() < smaller->size())
    std::swap(smaller, larger);

  unsigned parity = 0;
  for (auto [identity, pauli] : *smaller) {
    auto other = larger->find(identity);
    if (other != larger->end() && pauli != 'I' && other->second != 'I' &&
        pauli != other->second)
      parity ^= 1;
  }
  return parity;
}

static bool supportsAreDisjoint(const OperationView &lhs,
                                const OperationView &rhs) {
  const auto *smaller = &lhs.support;
  const auto *larger = &rhs.support;
  if (larger->size() < smaller->size())
    std::swap(smaller, larger);
  return llvm::none_of(*smaller, [&](QuantumIdentity identity) {
    return larger->contains(identity);
  });
}

static bool hasTargetControlCrossover(const OperationView &lhs,
                                      const OperationView &rhs) {
  return llvm::any_of(lhs.targetIdentities,
                      [&](QuantumIdentity identity) {
                        return rhs.controlPolarities.contains(identity);
                      }) ||
         llvm::any_of(rhs.targetIdentities, [&](QuantumIdentity identity) {
           return lhs.controlPolarities.contains(identity);
         });
}

static bool diagonalOverlapsOnlyControls(const OperationView &diagonal,
                                         const OperationView &controlled) {
  if (!isComputationalDiagonal(diagonal.operation) ||
      controlled.controls.empty())
    return false;

  bool hasOverlap = false;
  auto checkIdentity = [&](QuantumIdentity identity) {
    bool isSharedControl = controlled.controlPolarities.contains(identity);
    bool isSharedTarget = controlled.targetIdentities.contains(identity);
    hasOverlap |= isSharedControl || isSharedTarget;
    return !isSharedTarget;
  };

  for (ControlUse control : diagonal.controls)
    if (!checkIdentity(control.identity))
      return false;
  for (QuantumIdentity target : diagonal.targets)
    if (!checkIdentity(target))
      return false;
  return hasOverlap;
}

static bool targetSupportsAreDisjoint(const OperationView &lhs,
                                      const OperationView &rhs) {
  const auto *smaller = &lhs.targetIdentities;
  const auto *larger = &rhs.targetIdentities;
  if (larger->size() < smaller->size())
    std::swap(smaller, larger);
  return llvm::none_of(*smaller, [&](QuantumIdentity identity) {
    return larger->contains(identity);
  });
}

static bool targetActionsCommute(const OperationView &lhs,
                                 const OperationView &rhs) {
  if (targetSupportsAreDisjoint(lhs, rhs))
    return true;
  if (lhs.operation->getName() == rhs.operation->getName() &&
      haveSameTargets(lhs, rhs) &&
      areExactParameters(lhs.interface, rhs.interface)) {
    if (!isa<cudaq::quake::ExpPauliOp>(lhs.operation) ||
        getPauliWord(lhs) == getPauliWord(rhs))
      return true;
  }
  if (isComputationalDiagonal(lhs.operation) &&
      isComputationalDiagonal(rhs.operation))
    return true;
  if (haveSameAxisTargetAction(lhs, rhs))
    return true;

  auto lhsPauli = getPauliAction(lhs);
  auto rhsPauli = getPauliAction(rhs);
  return lhsPauli && rhsPauli &&
         getPauliAnticommutationParity(*lhsPauli, *rhsPauli) == 0;
}

static bool haveMutuallyExclusiveControls(const OperationView &lhs,
                                          const OperationView &rhs) {
  const auto *smaller = &lhs.controlPolarities;
  const auto *larger = &rhs.controlPolarities;
  if (larger->size() < smaller->size())
    std::swap(smaller, larger);
  for (auto [identity, negated] : *smaller) {
    auto other = larger->find(identity);
    if (other != larger->end() && negated != other->second)
      return true;
  }
  return false;
}

struct CommutationAnalysis::Impl {
  explicit Impl(Block &block);

  CommutationResult evaluate(Operation *lhs, Operation *rhs);
  std::optional<CommutationReason> getView(OperationView &view) const;
  void buildIdentityMap();
  void propagateOperatorIdentities(cudaq::quake::OperatorInterface op);
  void propagateIdentity(Value input, Value result);

  Block *block;
  QuantumIdentity nextIdentity = 0;
  llvm::DenseMap<Value, QuantumIdentity> identities;
  llvm::DenseMap<BorrowKey, QuantumIdentity> borrowedIdentities;
  llvm::DenseMap<OperationPair, CommutationResult> cache;
};

CommutationAnalysis::Impl::Impl(Block &block) : block(&block) {
  buildIdentityMap();
}

void CommutationAnalysis::Impl::propagateIdentity(Value input, Value result) {
  auto identity = identities.find(input);
  if (identity != identities.end())
    identities.try_emplace(result, identity->second);
}

void CommutationAnalysis::Impl::propagateOperatorIdentities(
    cudaq::quake::OperatorInterface op) {
  llvm::SmallVector<Value> wireInputs;
  for (Value control : op.getControls())
    if (isa<cudaq::quake::WireType>(control.getType()))
      wireInputs.push_back(control);
  for (Value target : op.getTargets())
    if (isa<cudaq::quake::WireType>(target.getType()))
      wireInputs.push_back(target);

  auto wireResults = op.getWires();
  if (wireInputs.size() != wireResults.size() ||
      llvm::any_of(wireResults, [](Value result) {
        return !isa<cudaq::quake::WireType>(result.getType());
      }))
    return;
  for (auto [input, result] : llvm::zip(wireInputs, wireResults))
    propagateIdentity(input, result);
}

void CommutationAnalysis::Impl::buildIdentityMap() {
  for (BlockArgument argument : block->getArguments())
    if (isa<cudaq::quake::WireType>(argument.getType()))
      identities.try_emplace(argument, nextIdentity++);

  for (Operation &operation : *block) {
    if (auto nullWire = dyn_cast<cudaq::quake::NullWireOp>(operation)) {
      identities.try_emplace(nullWire.getResult(), nextIdentity++);
      continue;
    }
    if (auto borrowWire = dyn_cast<cudaq::quake::BorrowWireOp>(operation)) {
      BorrowKey key{borrowWire.getSetNameAttr(), borrowWire.getIdentity()};
      auto [identity, inserted] =
          borrowedIdentities.try_emplace(key, nextIdentity);
      if (inserted)
        ++nextIdentity;
      identities.try_emplace(borrowWire.getResult(), identity->second);
      continue;
    }
    if (auto toControl = dyn_cast<cudaq::quake::ToControlOp>(operation)) {
      propagateIdentity(toControl.getQubit(), toControl.getResult());
      continue;
    }
    if (auto fromControl = dyn_cast<cudaq::quake::FromControlOp>(operation)) {
      propagateIdentity(fromControl.getCtrlbit(), fromControl.getResult());
      continue;
    }
    if (auto operatorInterface =
            dyn_cast<cudaq::quake::OperatorInterface>(operation))
      propagateOperatorIdentities(operatorInterface);
  }
}

std::optional<CommutationReason>
CommutationAnalysis::Impl::getView(OperationView &view) const {
  auto negatedControls = view.interface.getNegatedControls();
  auto controls = view.interface.getControls();
  if (negatedControls && negatedControls->size() != controls.size())
    return CommutationReason::MalformedControlPolarity;

  llvm::DenseSet<QuantumIdentity> seenIdentities;
  view.controls.reserve(controls.size());
  for (auto [index, control] : llvm::enumerate(controls)) {
    if (!isa<cudaq::quake::WireType, cudaq::quake::ControlType>(
            control.getType()))
      return CommutationReason::UnsupportedQuantumOperandType;
    auto identity = identities.find(control);
    if (identity == identities.end() ||
        !seenIdentities.insert(identity->second).second)
      return CommutationReason::AmbiguousQuantumIdentity;
    view.controls.push_back(
        {identity->second, negatedControls && (*negatedControls)[index]});
    view.support.insert(identity->second);
    view.controlPolarities.try_emplace(
        identity->second, negatedControls && (*negatedControls)[index]);
  }

  auto targets = view.interface.getTargets();
  view.targets.reserve(targets.size());
  for (Value target : targets) {
    if (!isa<cudaq::quake::WireType>(target.getType()))
      return CommutationReason::UnsupportedQuantumOperandType;
    auto identity = identities.find(target);
    if (identity == identities.end() ||
        !seenIdentities.insert(identity->second).second)
      return CommutationReason::AmbiguousQuantumIdentity;
    view.targets.push_back(identity->second);
    view.support.insert(identity->second);
    view.targetIdentities.insert(identity->second);
  }
  return std::nullopt;
}

CommutationResult CommutationAnalysis::Impl::evaluate(Operation *lhs,
                                                      Operation *rhs) {
  auto lhsInterface = dyn_cast<cudaq::quake::OperatorInterface>(lhs);
  auto rhsInterface = dyn_cast<cudaq::quake::OperatorInterface>(rhs);
  if (!lhsInterface || !rhsInterface)
    return indeterminate(CommutationReason::UnsupportedOperationKind);

  OperationView lhsView{lhs, lhsInterface};
  OperationView rhsView{rhs, rhsInterface};
  if (auto reason = getView(lhsView))
    return indeterminate(*reason);
  if (auto reason = getView(rhsView))
    return indeterminate(*reason);

  if (supportsAreDisjoint(lhsView, rhsView))
    return commutes(CommutationReason::DisjointSupport);
  if (isCustomUnitary(lhs) || isCustomUnitary(rhs))
    return indeterminate(CommutationReason::NoApplicableRule);
  if (!isSupportedSharedOperation(lhs) || !isSupportedSharedOperation(rhs))
    return indeterminate(CommutationReason::NoApplicableRule);
  if (!hasSupportedPauliWord(lhsView) || !hasSupportedPauliWord(rhsView))
    return indeterminate(CommutationReason::UnsupportedPauliWord);

  // U commutes with itself and its exact adjoint because UU^-1 = U^-1U = I.
  if (haveSameOperation(lhsView, rhsView))
    return commutes(CommutationReason::SameOperation);

  // Diagonal matrices commute because their products are pointwise scalar
  // products in the same computational basis.
  if (isComputationalDiagonal(lhs) && isComputationalDiagonal(rhs))
    return commutes(CommutationReason::ComputationalDiagonal);

  // Rotations generated by the same Pauli axis commute. PhasedRx operations
  // require the same phase so their generators match.
  if (lhsView.controls.empty() && rhsView.controls.empty() &&
      haveSameAxisTargetAction(lhsView, rhsView))
    return commutes(CommutationReason::SameAxis);

  // Pauli products obey PQ = (-1)^m QP, where m is the number of aligned
  // anti-commuting factors. Odd parity proves a negative only for fixed Paulis.
  if (lhsView.controls.empty() && rhsView.controls.empty()) {
    auto lhsPauli = getPauliAction(lhsView);
    auto rhsPauli = getPauliAction(rhsView);
    if (lhsPauli && rhsPauli) {
      if (getPauliAnticommutationParity(*lhsPauli, *rhsPauli) == 0)
        return commutes(CommutationReason::EvenPauliParity);
      if (lhsPauli->isFixed && rhsPauli->isFixed)
        return doesNotCommute(CommutationReason::OddPauliParity);
      return indeterminate(CommutationReason::NoApplicableRule);
    }
  }

  // A diagonal action commutes with a controlled operation when every shared
  // identity belongs only to the latter's computational-basis projector.
  if (diagonalOverlapsOnlyControls(lhsView, rhsView) ||
      diagonalOverlapsOnlyControls(rhsView, lhsView))
    return commutes(CommutationReason::DiagonalOnControls);

  // With no target-control crossover, commuting target actions and commuting
  // control projectors make every term of the controlled products commute.
  if ((!lhsView.controls.empty() || !rhsView.controls.empty()) &&
      !hasTargetControlCrossover(lhsView, rhsView) &&
      targetActionsCommute(lhsView, rhsView))
    return commutes(CommutationReason::CompatibleControlledTargets);

  // Opposite polarity on a shared control gives disjoint projectors (PQ = 0),
  // so the controlled operations commute regardless of their target actions.
  if (!lhsView.controls.empty() && !rhsView.controls.empty() &&
      !hasTargetControlCrossover(lhsView, rhsView) &&
      haveMutuallyExclusiveControls(lhsView, rhsView))
    return commutes(CommutationReason::MutuallyExclusiveControls);

  return indeterminate(CommutationReason::NoApplicableRule);
}

llvm::StringRef
cudaq::quake::detail::getCommutationReasonId(CommutationReason reason) {
  switch (reason) {
  case CommutationReason::DisjointSupport:
    return "disjoint-support";
  case CommutationReason::SameOperation:
    return "same-operation";
  case CommutationReason::ComputationalDiagonal:
    return "computational-diagonal";
  case CommutationReason::SameAxis:
    return "same-axis";
  case CommutationReason::EvenPauliParity:
    return "even-pauli-parity";
  case CommutationReason::OddPauliParity:
    return "odd-pauli-parity";
  case CommutationReason::DiagonalOnControls:
    return "diagonal-on-controls";
  case CommutationReason::CompatibleControlledTargets:
    return "compatible-controlled-targets";
  case CommutationReason::MutuallyExclusiveControls:
    return "mutually-exclusive-controls";
  case CommutationReason::NullOperation:
    return "null-operation";
  case CommutationReason::DifferentBlocks:
    return "different-blocks";
  case CommutationReason::UnsupportedOperationKind:
    return "unsupported-operation-kind";
  case CommutationReason::UnsupportedQuantumOperandType:
    return "unsupported-quantum-operand-type";
  case CommutationReason::MalformedControlPolarity:
    return "malformed-control-polarity";
  case CommutationReason::AmbiguousQuantumIdentity:
    return "ambiguous-quantum-identity";
  case CommutationReason::UnsupportedPauliWord:
    return "unsupported-pauli-word";
  case CommutationReason::NoApplicableRule:
    return "no-applicable-rule";
  }
  llvm_unreachable("unhandled commutation reason");
}

CommutationAnalysis::CommutationAnalysis(Block &block)
    : impl(std::make_unique<Impl>(block)) {}

CommutationAnalysis::~CommutationAnalysis() = default;

CommutationResult CommutationAnalysis::getResult(Operation *lhs,
                                                 Operation *rhs) {
  if (!lhs || !rhs)
    return indeterminate(CommutationReason::NullOperation);
  if (lhs->getBlock() != impl->block || rhs->getBlock() != impl->block)
    return indeterminate(CommutationReason::DifferentBlocks);

  OperationPair key = getCanonicalPair(lhs, rhs);
  auto cached = impl->cache.find(key);
  if (cached != impl->cache.end())
    return cached->second;
  auto result = impl->evaluate(lhs, rhs);
  impl->cache.try_emplace(key, result);
  return result;
}

bool CommutationAnalysis::canCommute(Operation *lhs, Operation *rhs) {
  return static_cast<bool>(getResult(lhs, rhs));
}
