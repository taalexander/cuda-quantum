/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Value.h"
#include <cstdint>
#include <utility>

namespace mlir {
class Block;
class Operation;
} // namespace mlir

namespace cudaq::quake::detail {

/// The outcome of a commutation query.
enum class CommutationStatus { Commutes, DoesNotCommute, Indeterminate };

/// The rule or limitation that produced a commutation status.
enum class CommutationReason {
  // Reasons paired with CommutationStatus::Commutes.
  /// The operations have disjoint block-local quantum support.
  DisjointSupport,
  /// The supported operations have the same action and placement, optionally
  /// with opposite adjoint states.
  SameOperation,
  /// Both operations are diagonal in the computational basis.
  ComputationalDiagonal,
  /// Both operations rotate about the same axis. Rotation angles may differ;
  /// `PhasedRx` rotation angles may differ, but their axis-defining phase
  /// parameters must be the same SSA value or equal constants.
  SameAxis,
  /// Pauli products have even anti-commutation parity on shared targets.
  EvenPauliParity,
  /// A diagonal operation overlaps the other operation only on controls.
  DiagonalOnControls,
  /// Controlled operations have commuting target actions and no target-control
  /// crossover.
  CompatibleControlledTargets,
  /// Opposite polarity on a shared control makes the control predicates
  /// mutually exclusive.
  MutuallyExclusiveControls,

  // Reasons paired with CommutationStatus::DoesNotCommute.
  /// Fixed Pauli products have odd anti-commutation parity on shared targets.
  OddPauliParity,

  // Reasons paired with CommutationStatus::Indeterminate.
  /// At least one query operation is null.
  NullOperation,
  /// At least one operation is outside the analyzed block.
  DifferentBlocks,
  /// At least one operation does not implement Quake `OperatorInterface`.
  UnsupportedOperationKind,
  /// A quantum operand is not a supported scalar wire or control value.
  UnsupportedQuantumOperandType,
  /// Control polarity metadata does not match the control operands.
  MalformedControlPolarity,
  /// The analysis cannot establish a unique block-local quantum identity.
  AmbiguousQuantumIdentity,
  /// An `ExpPauli` word is dynamic or is not aligned literal `I/X/Y/Z` data.
  UnsupportedPauliWord,
  /// Supported operations did not satisfy an available structural rule.
  NoApplicableRule
};

/// Return the stable textual identifier for a commutation reason.
llvm::StringRef getCommutationReasonId(CommutationReason reason);

/// A structural commutation outcome and its classification.
struct CommutationResult {
  CommutationStatus status;
  CommutationReason reason;

  /// True only when structural analysis proved exact commutation.
  explicit operator bool() const {
    return status == CommutationStatus::Commutes;
  }
};

/// Read-only commutation analysis for operations in one block.
///
/// The analysis is primarily designed to prove when reordering is safe.
/// `DoesNotCommute` is returned only for the limited cases where an available
/// rule proves noncommutation. `Indeterminate` means that the available rules
/// established neither result. It does not imply either commutation or
/// noncommutation.
///
/// Compiler transformations must treat both `DoesNotCommute` and
/// `Indeterminate` as not safe to reorder. The separate statuses preserve the
/// distinction between proven noncommutation and the absence of a proof.
///
/// Any mutation of the block invalidates the analysis instance. The caller
/// must discard it before querying the changed block.
class CommutationAnalysis {
public:
  explicit CommutationAnalysis(mlir::Block &block);

  CommutationAnalysis(const CommutationAnalysis &) = delete;
  CommutationAnalysis &operator=(const CommutationAnalysis &) = delete;

  /// Return the detailed symmetric relation between two operations.
  CommutationResult getResult(mlir::Operation *lhs, mlir::Operation *rhs);

  /// Return true only when exact commutation has been proven.
  bool canCommute(mlir::Operation *lhs, mlir::Operation *rhs);

private:
  mlir::Block *block;
  llvm::DenseMap<mlir::Value, std::uint32_t> identities;
  llvm::DenseMap<std::pair<mlir::Operation *, mlir::Operation *>,
                 CommutationResult>
      cache;
};

} // namespace cudaq::quake::detail
