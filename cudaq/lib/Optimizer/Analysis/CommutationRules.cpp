/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CommutationRules.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <utility>

using namespace mlir;

using cudaq::quake::detail::CanonicalOperationPair;
using cudaq::quake::detail::CommutationFailureReason;
using cudaq::quake::detail::DescriptorConstructionResult;
using cudaq::quake::detail::NormalizedOperatorDescriptor;
using cudaq::quake::detail::OperatorKind;
using cudaq::quake::detail::QuantumOperand;
using cudaq::quake::detail::QuantumOperandRole;

static std::optional<OperatorKind> getOperatorKind(Operation *operation) {
  return llvm::TypeSwitch<Operation *, std::optional<OperatorKind>>(operation)
      .Case<cudaq::quake::HOp>([](auto) { return OperatorKind::H; })
      .Case<cudaq::quake::XOp>([](auto) { return OperatorKind::X; })
      .Case<cudaq::quake::YOp>([](auto) { return OperatorKind::Y; })
      .Case<cudaq::quake::ZOp>([](auto) { return OperatorKind::Z; })
      .Case<cudaq::quake::SOp>([](auto) { return OperatorKind::S; })
      .Case<cudaq::quake::TOp>([](auto) { return OperatorKind::T; })
      .Case<cudaq::quake::SwapOp>([](auto) { return OperatorKind::Swap; })
      .Case<cudaq::quake::R1Op>([](auto) { return OperatorKind::R1; })
      .Case<cudaq::quake::RxOp>([](auto) { return OperatorKind::Rx; })
      .Case<cudaq::quake::RyOp>([](auto) { return OperatorKind::Ry; })
      .Case<cudaq::quake::RzOp>([](auto) { return OperatorKind::Rz; })
      .Case<cudaq::quake::PhasedRxOp>(
          [](auto) { return OperatorKind::PhasedRx; })
      .Case<cudaq::quake::U2Op>([](auto) { return OperatorKind::U2; })
      .Case<cudaq::quake::U3Op>([](auto) { return OperatorKind::U3; })
      .Case<cudaq::quake::ExpPauliOp>(
          [](auto) { return OperatorKind::ExpPauli; })
      .Default([](Operation *) { return std::nullopt; });
}

static bool isSupportedControl(Value value) {
  return isa<cudaq::quake::WireType, cudaq::quake::ControlType>(
      value.getType());
}

static bool isSupportedTarget(Value value) {
  return isa<cudaq::quake::WireType>(value.getType());
}

static bool isSupportedPauliWord(llvm::StringRef pauliWord,
                                 std::size_t targetCount) {
  return pauliWord.size() == targetCount &&
         llvm::all_of(pauliWord, [](char pauli) {
           return pauli == 'I' || pauli == 'X' || pauli == 'Y' || pauli == 'Z';
         });
}

bool QuantumOperand::operator==(const QuantumOperand &other) const {
  return value == other.value && role == other.role && negated == other.negated;
}

bool NormalizedOperatorDescriptor::operator==(
    const NormalizedOperatorDescriptor &other) const {
  return kind == other.kind && parameters == other.parameters &&
         quantumOperands == other.quantumOperands &&
         pauliWord == other.pauliWord && isAdjoint == other.isAdjoint;
}

DescriptorConstructionResult::DescriptorConstructionResult(
    std::optional<NormalizedOperatorDescriptor> descriptor,
    std::optional<CommutationFailureReason> failureReason)
    : descriptor(std::move(descriptor)), failureReason(failureReason) {}

DescriptorConstructionResult DescriptorConstructionResult::getSuccess(
    NormalizedOperatorDescriptor descriptor) {
  return {std::move(descriptor), std::nullopt};
}

DescriptorConstructionResult
DescriptorConstructionResult::getFailure(CommutationFailureReason reason) {
  return {std::nullopt, reason};
}

DescriptorConstructionResult::operator bool() const {
  return descriptor.has_value();
}

const NormalizedOperatorDescriptor &
DescriptorConstructionResult::operator*() const {
  assert(descriptor && "cannot access a failed descriptor result");
  return *descriptor;
}

const NormalizedOperatorDescriptor *
DescriptorConstructionResult::operator->() const {
  return &**this;
}

CommutationFailureReason
DescriptorConstructionResult::getFailureReason() const {
  assert(failureReason && "successful descriptor result has no failure");
  return *failureReason;
}

DescriptorConstructionResult
cudaq::quake::detail::getNormalizedOperatorDescriptor(Operation *operation) {
  if (!operation)
    return DescriptorConstructionResult::getFailure(
        CommutationFailureReason::NullOperation);

  auto kind = getOperatorKind(operation);
  if (!kind)
    return DescriptorConstructionResult::getFailure(
        CommutationFailureReason::UnsupportedOperation);

  auto operatorInterface = dyn_cast<cudaq::quake::OperatorInterface>(operation);
  assert(operatorInterface && "recognized Quake operator lacks its interface");
  if (*kind == OperatorKind::ExpPauli) {
    auto expPauli = cast<cudaq::quake::ExpPauliOp>(operation);
    if (!expPauli.getPauliLiteralAttr() || expPauli.getParameters().size() > 1)
      return DescriptorConstructionResult::getFailure(
          CommutationFailureReason::UnsupportedOperation);
  }

  NormalizedOperatorDescriptor descriptor{
      *kind,
      llvm::SmallVector<Value>(operatorInterface.getParameters()),
      {},
      {},
      operatorInterface.isAdj()};

  auto controls = operatorInterface.getControls();
  auto negatedControls = operatorInterface.getNegatedControls();
  if (negatedControls && negatedControls->size() != controls.size())
    return DescriptorConstructionResult::getFailure(
        CommutationFailureReason::InvalidControlPolarity);

  descriptor.quantumOperands.reserve(controls.size() +
                                     operatorInterface.getTargets().size());
  for (auto [index, control] : llvm::enumerate(controls)) {
    if (!isSupportedControl(control))
      return DescriptorConstructionResult::getFailure(
          CommutationFailureReason::UnsupportedQubitOperand);
    bool isNegated = negatedControls && (*negatedControls)[index];
    descriptor.quantumOperands.push_back(
        {control, QuantumOperandRole::Control, isNegated});
  }

  auto targetStart = descriptor.quantumOperands.size();
  for (Value target : operatorInterface.getTargets()) {
    if (!isSupportedTarget(target))
      return DescriptorConstructionResult::getFailure(
          CommutationFailureReason::UnsupportedQubitOperand);
    descriptor.quantumOperands.push_back(
        {target, QuantumOperandRole::Target, false});
  }

  if (*kind == OperatorKind::Swap) {
    auto targets = llvm::MutableArrayRef(descriptor.quantumOperands)
                       .drop_front(targetStart);
    llvm::sort(
        targets, [](const QuantumOperand &lhs, const QuantumOperand &rhs) {
          return std::less<const void *>{}(lhs.value.getAsOpaquePointer(),
                                           rhs.value.getAsOpaquePointer());
        });
  } else if (*kind == OperatorKind::ExpPauli) {
    descriptor.pauliWord =
        cast<cudaq::quake::ExpPauliOp>(operation).getPauliLiteralAttr();
    if (!isSupportedPauliWord(descriptor.pauliWord.getValue(),
                              descriptor.quantumOperands.size() - targetStart))
      return DescriptorConstructionResult::getFailure(
          CommutationFailureReason::UnsupportedOperation);
  }

  return DescriptorConstructionResult::getSuccess(std::move(descriptor));
}

CanonicalOperationPair::CanonicalOperationPair(Operation *lhs, Operation *rhs) {
  if (std::less<Operation *>{}(rhs, lhs))
    std::swap(lhs, rhs);
  first = lhs;
  second = rhs;
}

bool CanonicalOperationPair::operator==(
    const CanonicalOperationPair &other) const {
  return first == other.first && second == other.second;
}

llvm::hash_code
cudaq::quake::detail::hash_value(const CanonicalOperationPair &pair) {
  return llvm::hash_combine(pair.getFirst(), pair.getSecond());
}
