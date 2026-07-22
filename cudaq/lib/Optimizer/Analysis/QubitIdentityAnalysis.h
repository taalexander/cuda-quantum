/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"
#include <cstdint>
#include <optional>

namespace mlir {
class Block;
}

namespace cudaq::quake::detail {

/// Tracks physical-qubit identity for scalar `!quake.wire` and
/// `!quake.control` SSA values within one block of valid Quake value-form IR.
/// `CommutationAnalysis` uses these identities to determine whether operations
/// act on the same or disjoint physical qubits.
///
/// Block arguments, `quake.null_wire`, and `quake.borrow_wire` establish local
/// identities. The analysis propagates them through supported operators,
/// measurements, resets, and wire/control conversions. Identity does not imply
/// quantum-state equivalence. For example, measurement and reset preserve the
/// physical qubit while changing its state.
///
/// The analysis does not follow identity through calls, references, aggregates,
/// or block edges. Values with unsupported or ambiguous lineage remain
/// unidentified. Any mutation of the block invalidates the analysis.
class QubitIdentityAnalysis {
public:
  using QubitId = std::uint32_t;

  explicit QubitIdentityAnalysis(mlir::Block &block);

  /// Return the analysis-local qubit identifier, or no value when lineage is
  /// unsupported or ambiguous.
  std::optional<QubitId> getQubitId(mlir::Value value) const;

private:
  llvm::DenseMap<mlir::Value, QubitId> qubitIds;
};

} // namespace cudaq::quake::detail
