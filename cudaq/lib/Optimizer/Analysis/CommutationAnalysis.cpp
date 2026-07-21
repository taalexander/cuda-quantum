/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/CommutationAnalysis.h"
#include "llvm/Support/ErrorHandling.h"

using namespace cudaq::quake::detail;

llvm::StringRef cudaq::quake::detail::getCommutationProofReasonId(
    CommutationProofReason reason) {
  switch (reason) {
  case CommutationProofReason::DisjointSupport:
    return "disjoint-support";
  case CommutationProofReason::SameOperation:
    return "same-operation";
  case CommutationProofReason::ComputationalDiagonal:
    return "computational-diagonal";
  case CommutationProofReason::SameAxis:
    return "same-axis";
  case CommutationProofReason::PauliParity:
    return "pauli-parity";
  case CommutationProofReason::DiagonalOnControls:
    return "diagonal-on-controls";
  case CommutationProofReason::CompatibleControlledTargets:
    return "compatible-controlled-targets";
  case CommutationProofReason::MutuallyExclusiveControls:
    return "mutually-exclusive-controls";
  }
  llvm_unreachable("unhandled commutation proof reason");
}

llvm::StringRef cudaq::quake::detail::getCommutationFailureReasonId(
    CommutationFailureReason reason) {
  switch (reason) {
  case CommutationFailureReason::NullOperation:
    return "null-operation";
  case CommutationFailureReason::DifferentBlocks:
    return "different-blocks";
  case CommutationFailureReason::UnsupportedOperation:
    return "unsupported-operation";
  case CommutationFailureReason::UnsupportedQubitOperand:
    return "unsupported-qubit-operand";
  case CommutationFailureReason::InvalidControlPolarity:
    return "invalid-control-polarity";
  case CommutationFailureReason::AmbiguousQuantumValue:
    return "ambiguous-quantum-value";
  case CommutationFailureReason::NotProven:
    return "not-proven";
  }
  llvm_unreachable("unhandled commutation failure reason");
}

CommutationResult::CommutationResult(
    CommutationRelation relation,
    std::optional<CommutationProofReason> proofReason,
    std::optional<CommutationFailureReason> failureReason)
    : relation(relation), proofReason(proofReason),
      failureReason(failureReason) {}

CommutationResult
CommutationResult::getCommutes(CommutationProofReason reason) {
  return {CommutationRelation::Commutes, reason, std::nullopt};
}

CommutationResult
CommutationResult::getDoesNotCommute(CommutationProofReason reason) {
  return {CommutationRelation::DoesNotCommute, reason, std::nullopt};
}

CommutationResult
CommutationResult::getUnknown(CommutationFailureReason reason) {
  return {CommutationRelation::Unknown, std::nullopt, reason};
}
