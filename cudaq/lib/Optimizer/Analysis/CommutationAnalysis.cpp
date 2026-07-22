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

using cudaq::quake::Pauli;
using cudaq::quake::PauliWord;
using cudaq::quake::detail::CommutationAnalysis;
using cudaq::quake::detail::CommutationReason;
using cudaq::quake::detail::CommutationResult;
using cudaq::quake::detail::CommutationStatus;

namespace {
// Block-local label shared by SSA values that represent the same qubit.
using QubitId = std::uint32_t;
using OperationPair = std::pair<Operation *, Operation *>;
using BorrowKey = std::pair<Attribute, std::int32_t>;

struct ControlUse {
  QubitId qubitId;
  bool negated;

  bool operator==(const ControlUse &) const = default;
};

// Query-local view of a Quake operator's controls, targets, and support,
// expressed using analysis-local qubit identifiers.
struct OperationView {
  OperationView(Operation *operation,
                cudaq::quake::OperatorInterface operatorInterface)
      : operation(operation), interface(operatorInterface) {}

  // Underlying operation used for kind-specific commutation rules.
  Operation *operation;
  // Quake interface used to access parameters and quantum operands.
  cudaq::quake::OperatorInterface interface;
  // Controls in operand order, including their positive or negative polarity.
  llvm::SmallVector<ControlUse> controls;
  // Targets in operand order, preserving positional gate semantics.
  llvm::SmallVector<QubitId> targets;
  // Unique union of control and target qubits.
  llvm::DenseSet<QubitId> support;
  // Target lookup used by overlap and crossover rules.
  llvm::DenseSet<QubitId> targetQubitIds;
  // Control lookup and polarity used by controlled-operation rules.
  llvm::DenseMap<QubitId, bool> controlPolarities;
};

// Analysis-local Pauli product keyed by qubit rather than IR target order.
// This normalized form makes shared-qubit parity checks order-independent.
struct PauliAction {
  llvm::DenseMap<QubitId, Pauli> terms;
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
  // std::less provides a total order for unrelated pointers. The order has no
  // semantic meaning; it only makes the symmetric cache key canonical.
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

// Exact Pauli operators support a negative commutation proof from odd parity.
// ExpPauli rotations do not because their angles may make them commute.
static bool isPauliOperator(Operation *operation) {
  return isa<cudaq::quake::XOp, cudaq::quake::YOp, cudaq::quake::ZOp>(
      operation);
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

static bool isComputationalDiagonal(Operation *operation) {
  // TODO: Add other computational-basis diagonal operations here without
  // broadening the single-target Z-axis predicate.
  return isZAxis(operation);
}

static bool areExactParameterValues(Value lhs, Value rhs) {
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

static bool haveExactParameters(cudaq::quake::OperatorInterface lhs,
                                cudaq::quake::OperatorInterface rhs) {
  auto lhsParameters = lhs.getParameters();
  auto rhsParameters = rhs.getParameters();
  return lhsParameters.size() == rhsParameters.size() &&
         llvm::equal(lhsParameters, rhsParameters, areExactParameterValues);
}

static bool haveSameTargets(const OperationView &lhs,
                            const OperationView &rhs) {
  if (lhs.targets.size() != rhs.targets.size())
    return false;
  // Swap is symmetric in its two targets, so reversed target order represents
  // the same operation.
  if (isa<cudaq::quake::SwapOp>(lhs.operation) &&
      isa<cudaq::quake::SwapOp>(rhs.operation))
    return lhs.targets.size() == 2 && ((lhs.targets[0] == rhs.targets[0] &&
                                        lhs.targets[1] == rhs.targets[1]) ||
                                       (lhs.targets[0] == rhs.targets[1] &&
                                        lhs.targets[1] == rhs.targets[0]));
  return lhs.targets == rhs.targets;
}

static std::optional<PauliWord> getLiteralPaulis(const OperationView &view) {
  auto expPauli = dyn_cast<cudaq::quake::ExpPauliOp>(view.operation);
  if (!expPauli)
    return std::nullopt;
  auto literal = expPauli.getPauliLiteralAttr();
  if (!literal || literal.getValue().size() != view.targets.size())
    return std::nullopt;
  return cudaq::quake::symbolizePauliWord(literal.getValue());
}

static bool hasSupportedPauliWord(const OperationView &view) {
  return !isa<cudaq::quake::ExpPauliOp>(view.operation) ||
         getLiteralPaulis(view).has_value();
}

static bool haveSameOperation(const OperationView &lhs,
                              const OperationView &rhs) {
  // Match the operation kind and every action-bearing interface value. Adjoint
  // state may differ because an operation commutes with its exact inverse.
  if (!isSupportedSharedOperation(lhs.operation) ||
      !isSupportedSharedOperation(rhs.operation) ||
      lhs.operation->getName() != rhs.operation->getName() ||
      lhs.controls != rhs.controls || !haveSameTargets(lhs, rhs) ||
      !haveExactParameters(lhs.interface, rhs.interface))
    return false;

  // ExpPauli stores part of its action in the Pauli word rather than among the
  // OperatorInterface parameters.
  if (isa<cudaq::quake::ExpPauliOp>(lhs.operation))
    return getLiteralPaulis(lhs) == getLiteralPaulis(rhs);
  return true;
}

static bool haveSameAxisTargetAction(const OperationView &lhs,
                                     const OperationView &rhs) {
  if (lhs.targets.size() != 1 || rhs.targets.size() != 1 ||
      lhs.targets.front() != rhs.targets.front())
    return false;
  // Gates in the same standard axis family commute even when their rotation
  // angles differ.
  if ((isXAxis(lhs.operation) && isXAxis(rhs.operation)) ||
      (isYAxis(lhs.operation) && isYAxis(rhs.operation)) ||
      (isZAxis(lhs.operation) && isZAxis(rhs.operation)))
    return true;

  // This rule proves commutation when the axis-defining PhasedRx phase values
  // match exactly; rotation angles may differ.
  auto lhsPhasedRx = dyn_cast<cudaq::quake::PhasedRxOp>(lhs.operation);
  auto rhsPhasedRx = dyn_cast<cudaq::quake::PhasedRxOp>(rhs.operation);
  return lhsPhasedRx && rhsPhasedRx &&
         areExactParameterValues(lhsPhasedRx.getParameters()[1],
                                 rhsPhasedRx.getParameters()[1]);
}

// Normalize an exact Pauli operator or literal ExpPauli word into Pauli symbols
// keyed by the block-local qubits on which they act.
static std::optional<PauliAction> getPauliAction(const OperationView &view) {
  std::optional<Pauli> pauli;
  if (isa<cudaq::quake::XOp>(view.operation))
    pauli = Pauli::X;
  else if (isa<cudaq::quake::YOp>(view.operation))
    pauli = Pauli::Y;
  else if (isa<cudaq::quake::ZOp>(view.operation))
    pauli = Pauli::Z;

  if (pauli) {
    if (view.targets.size() != 1)
      return std::nullopt;
    PauliAction action;
    action.terms.try_emplace(view.targets.front(), *pauli);
    return action;
  }

  auto paulis = getLiteralPaulis(view);
  if (!paulis)
    return std::nullopt;
  PauliAction action;
  action.terms.reserve(view.targets.size());
  for (auto [qubitId, symbol] : llvm::zip(view.targets, *paulis))
    action.terms.try_emplace(qubitId, symbol);
  return action;
}

static bool hasOddPauliAnticommutationParity(const PauliAction &lhs,
                                             const PauliAction &rhs) {
  const auto *smaller = &lhs.terms;
  const auto *larger = &rhs.terms;
  if (larger->size() < smaller->size())
    std::swap(smaller, larger);

  bool hasOddParity = false;
  for (auto [qubitId, pauli] : *smaller) {
    auto other = larger->find(qubitId);
    if (other != larger->end() && pauli != Pauli::I &&
        other->second != Pauli::I && pauli != other->second)
      hasOddParity = !hasOddParity;
  }
  return hasOddParity;
}

// Operations with no shared control or target qubit commute independently of
// their gate semantics.
static bool haveDisjointQuantumSupport(const OperationView &lhs,
                                       const OperationView &rhs) {
  const auto *smaller = &lhs.support;
  const auto *larger = &rhs.support;
  if (larger->size() < smaller->size())
    std::swap(smaller, larger);
  return llvm::none_of(
      *smaller, [&](QubitId qubitId) { return larger->contains(qubitId); });
}

static bool hasTargetControlCrossover(const OperationView &lhs,
                                      const OperationView &rhs) {
  return llvm::any_of(lhs.targetQubitIds,
                      [&](QubitId qubitId) {
                        return rhs.controlPolarities.contains(qubitId);
                      }) ||
         llvm::any_of(rhs.targetQubitIds, [&](QubitId qubitId) {
           return lhs.controlPolarities.contains(qubitId);
         });
}

static bool diagonalOverlapsOnlyControls(const OperationView &diagonal,
                                         const OperationView &controlled) {
  if (!isComputationalDiagonal(diagonal.operation) ||
      controlled.controls.empty())
    return false;

  bool hasOverlap = false;
  auto checkQubit = [&](QubitId qubitId) {
    bool isSharedControl = controlled.controlPolarities.contains(qubitId);
    bool isSharedTarget = controlled.targetQubitIds.contains(qubitId);
    hasOverlap |= isSharedControl || isSharedTarget;
    return !isSharedTarget;
  };

  for (ControlUse control : diagonal.controls)
    if (!checkQubit(control.qubitId))
      return false;
  for (QubitId target : diagonal.targets)
    if (!checkQubit(target))
      return false;
  return hasOverlap;
}

// Controlled operations may share controls; this predicate checks only whether
// their target actions are disjoint.
static bool haveDisjointTargetSupport(const OperationView &lhs,
                                      const OperationView &rhs) {
  const auto *smaller = &lhs.targetQubitIds;
  const auto *larger = &rhs.targetQubitIds;
  if (larger->size() < smaller->size())
    std::swap(smaller, larger);
  return llvm::none_of(
      *smaller, [&](QubitId qubitId) { return larger->contains(qubitId); });
}

static bool targetActionsCommute(const OperationView &lhs,
                                 const OperationView &rhs) {
  if (haveDisjointTargetSupport(lhs, rhs))
    return true;
  if (lhs.operation->getName() == rhs.operation->getName() &&
      haveSameTargets(lhs, rhs) &&
      haveExactParameters(lhs.interface, rhs.interface)) {
    if (!isa<cudaq::quake::ExpPauliOp>(lhs.operation) ||
        getLiteralPaulis(lhs) == getLiteralPaulis(rhs))
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
         !hasOddPauliAnticommutationParity(*lhsPauli, *rhsPauli);
}

static bool haveMutuallyExclusiveControls(const OperationView &lhs,
                                          const OperationView &rhs) {
  const auto *smaller = &lhs.controlPolarities;
  const auto *larger = &rhs.controlPolarities;
  if (larger->size() < smaller->size())
    std::swap(smaller, larger);
  for (auto [qubitId, negated] : *smaller) {
    auto other = larger->find(qubitId);
    if (other != larger->end() && negated != other->second)
      return true;
  }
  return false;
}

static void propagateQubitId(llvm::DenseMap<Value, QubitId> &qubitIds,
                             Value input, Value result) {
  auto qubitId = qubitIds.find(input);
  if (qubitId != qubitIds.end())
    qubitIds.try_emplace(result, qubitId->second);
}

static void propagateOperatorQubitIds(llvm::DenseMap<Value, QubitId> &qubitIds,
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
    propagateQubitId(qubitIds, input, result);
}

static void buildQubitIdMap(Block &block,
                            llvm::DenseMap<Value, QubitId> &qubitIds) {
  QubitId nextQubitId = 0;
  llvm::DenseMap<BorrowKey, QubitId> borrowedQubitIds;

  for (BlockArgument argument : block.getArguments())
    if (isa<cudaq::quake::WireType>(argument.getType()))
      qubitIds.try_emplace(argument, nextQubitId++);

  for (Operation &operation : block) {
    if (auto nullWire = dyn_cast<cudaq::quake::NullWireOp>(operation)) {
      qubitIds.try_emplace(nullWire.getResult(), nextQubitId++);
      continue;
    }
    if (auto borrowWire = dyn_cast<cudaq::quake::BorrowWireOp>(operation)) {
      BorrowKey key{borrowWire.getSetNameAttr(), borrowWire.getIdentity()};
      auto [qubitId, inserted] = borrowedQubitIds.try_emplace(key, nextQubitId);
      if (inserted)
        ++nextQubitId;
      qubitIds.try_emplace(borrowWire.getResult(), qubitId->second);
      continue;
    }
    if (auto toControl = dyn_cast<cudaq::quake::ToControlOp>(operation)) {
      propagateQubitId(qubitIds, toControl.getQubit(), toControl.getResult());
      continue;
    }
    if (auto fromControl = dyn_cast<cudaq::quake::FromControlOp>(operation)) {
      propagateQubitId(qubitIds, fromControl.getCtrlbit(),
                       fromControl.getResult());
      continue;
    }
    if (auto operatorInterface =
            dyn_cast<cudaq::quake::OperatorInterface>(operation))
      propagateOperatorQubitIds(qubitIds, operatorInterface);
  }
}

static std::optional<CommutationReason>
getView(OperationView &view, const llvm::DenseMap<Value, QubitId> &qubitIds) {
  auto negatedControls = view.interface.getNegatedControls();
  auto controls = view.interface.getControls();
  if (negatedControls && negatedControls->size() != controls.size())
    return CommutationReason::MalformedControlPolarity;

  llvm::DenseSet<QubitId> seenQubitIds;
  view.controls.reserve(controls.size());
  for (auto [index, control] : llvm::enumerate(controls)) {
    if (!isa<cudaq::quake::WireType, cudaq::quake::ControlType>(
            control.getType()))
      return CommutationReason::UnsupportedQuantumOperandType;
    auto qubitId = qubitIds.find(control);
    if (qubitId == qubitIds.end())
      return CommutationReason::UnmappedQubitId;
    if (!seenQubitIds.insert(qubitId->second).second)
      return CommutationReason::DuplicateQubitOperand;
    view.controls.push_back(
        {qubitId->second, negatedControls && (*negatedControls)[index]});
    view.support.insert(qubitId->second);
    view.controlPolarities.try_emplace(
        qubitId->second, negatedControls && (*negatedControls)[index]);
  }

  auto targets = view.interface.getTargets();
  view.targets.reserve(targets.size());
  for (Value target : targets) {
    if (!isa<cudaq::quake::WireType>(target.getType()))
      return CommutationReason::UnsupportedQuantumOperandType;
    auto qubitId = qubitIds.find(target);
    if (qubitId == qubitIds.end())
      return CommutationReason::UnmappedQubitId;
    if (!seenQubitIds.insert(qubitId->second).second)
      return CommutationReason::DuplicateQubitOperand;
    view.targets.push_back(qubitId->second);
    view.support.insert(qubitId->second);
    view.targetQubitIds.insert(qubitId->second);
  }
  return std::nullopt;
}

static CommutationResult
evaluate(Operation *lhs, Operation *rhs,
         const llvm::DenseMap<Value, QubitId> &qubitIds) {
  auto lhsInterface = dyn_cast<cudaq::quake::OperatorInterface>(lhs);
  auto rhsInterface = dyn_cast<cudaq::quake::OperatorInterface>(rhs);
  if (!lhsInterface || !rhsInterface)
    return indeterminate(CommutationReason::UnsupportedOperationKind);

  OperationView lhsView{lhs, lhsInterface};
  OperationView rhsView{rhs, rhsInterface};
  if (auto reason = getView(lhsView, qubitIds))
    return indeterminate(*reason);
  if (auto reason = getView(rhsView, qubitIds))
    return indeterminate(*reason);

  // Operators on disjoint qubits commute because
  // (A tensor I)(I tensor B) = A tensor B = (I tensor B)(A tensor I).
  if (haveDisjointQuantumSupport(lhsView, rhsView))
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

  // Computational-basis diagonal matrices satisfy D1 D2 = D2 D1 because
  // their products are pointwise scalar products in the same basis.
  if (isComputationalDiagonal(lhs) && isComputationalDiagonal(rhs))
    return commutes(CommutationReason::ComputationalDiagonal);

  // Operators that are functions of the same Pauli axis P commute because
  // f(P) g(P) = g(P) f(P). This rule recognizes PhasedRx axes only when their
  // phase values match exactly.
  if (lhsView.controls.empty() && rhsView.controls.empty() &&
      haveSameAxisTargetAction(lhsView, rhsView))
    return commutes(CommutationReason::SameAxis);

  // Pauli products obey PQ = (-1)^m QP, where m is the number of aligned
  // anti-commuting factors. Odd parity proves a negative only for exact Pauli
  // operators, not parameterized ExpPauli rotations.
  if (lhsView.controls.empty() && rhsView.controls.empty()) {
    auto lhsPauli = getPauliAction(lhsView);
    auto rhsPauli = getPauliAction(rhsView);
    if (lhsPauli && rhsPauli) {
      if (!hasOddPauliAnticommutationParity(*lhsPauli, *rhsPauli))
        return commutes(CommutationReason::EvenPauliParity);
      if (isPauliOperator(lhs) && isPauliOperator(rhs))
        return doesNotCommute(CommutationReason::OddPauliParity);
      return indeterminate(CommutationReason::NoApplicableRule);
    }
  }

  // A diagonal action D commutes with a computational-basis control projector
  // P because DP = PD. This applies when every shared qubit is only a control
  // of the other operation, never one of its targets.
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
  case CommutationReason::UnmappedQubitId:
    return "unmapped-qubit-id";
  case CommutationReason::DuplicateQubitOperand:
    return "duplicate-qubit-operand";
  case CommutationReason::UnsupportedPauliWord:
    return "unsupported-pauli-word";
  case CommutationReason::NoApplicableRule:
    return "no-applicable-rule";
  }
  llvm_unreachable("unhandled commutation reason");
}

CommutationAnalysis::CommutationAnalysis(Block &block) : block(&block) {
  buildQubitIdMap(block, qubitIds);
}

CommutationResult CommutationAnalysis::getResult(Operation *lhs,
                                                 Operation *rhs) {
  if (!lhs || !rhs)
    return indeterminate(CommutationReason::NullOperation);
  if (lhs->getBlock() != block || rhs->getBlock() != block)
    return indeterminate(CommutationReason::DifferentBlocks);

  OperationPair key = getCanonicalPair(lhs, rhs);
  auto cached = cache.find(key);
  if (cached != cache.end())
    return cached->second;
  auto result = evaluate(lhs, rhs, qubitIds);
  cache.try_emplace(key, result);
  return result;
}

bool CommutationAnalysis::canCommute(Operation *lhs, Operation *rhs) {
  return static_cast<bool>(getResult(lhs, rhs));
}
