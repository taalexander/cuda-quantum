/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
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

/// Block-local identity of the underlying qubit represented by scalar Quake
/// wire and control SSA values. Identity does not imply quantum-state
/// equivalence: measurement and reset preserve identity while changing state.
///
/// Operators, measurement, reset, and wire/control conversions propagate known
/// identities. Scalar block arguments establish local identities without
/// correlation to predecessor values. Calls, references, and aggregates remain
/// unmapped. Any block mutation invalidates the analysis.
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
