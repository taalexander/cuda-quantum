/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/StringRef.h"
#include <optional>

namespace cudaq::quake::detail {

enum class CommutationRelation { Commutes, DoesNotCommute, Unknown };

enum class CommutationProofReason {
  DisjointSupport,
  SameOperation,
  ComputationalDiagonal,
  SameAxis,
  PauliParity,
  DiagonalOnControls,
  CompatibleControlledTargets,
  MutuallyExclusiveControls
};

enum class CommutationFailureReason {
  NullOperation,
  DifferentBlocks,
  UnsupportedOperation,
  UnsupportedQubitOperand,
  InvalidControlPolarity,
  AmbiguousQuantumValue,
  NotProven
};

llvm::StringRef getCommutationProofReasonId(CommutationProofReason reason);
llvm::StringRef getCommutationFailureReasonId(CommutationFailureReason reason);

/// A structural commutation decision and the reason supporting it.
class CommutationResult {
public:
  /// Return a proved exact commutation result.
  static CommutationResult getCommutes(CommutationProofReason reason);
  /// Return a result proving that the operations do not commute.
  static CommutationResult getDoesNotCommute(CommutationProofReason reason);
  /// Return an unproved result with its conservative failure classification.
  static CommutationResult getUnknown(CommutationFailureReason reason);

  CommutationRelation getRelation() const { return relation; }
  std::optional<CommutationProofReason> getProofReason() const {
    return proofReason;
  }
  std::optional<CommutationFailureReason> getFailureReason() const {
    return failureReason;
  }

private:
  CommutationResult(CommutationRelation relation,
                    std::optional<CommutationProofReason> proofReason,
                    std::optional<CommutationFailureReason> failureReason);

  CommutationRelation relation;
  std::optional<CommutationProofReason> proofReason;
  std::optional<CommutationFailureReason> failureReason;
};

} // namespace cudaq::quake::detail
