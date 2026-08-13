/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/IR/Value.h"
#include <optional>

namespace cudaq::opt {

/// One exact use-def step along a scalar wire. `wire` is the value reached in
/// the forward direction. A direct step retains the input wire, while a scope
/// step crosses to the matching scope result.
/// `continueOperand` is non-null only for the latter.
struct ScalarWireStep {
  mlir::Value wire;
  mlir::Operation *operation;
  mlir::OpOperand *continueOperand = nullptr;
};

/// Follows one exact scalar-wire step forward. It follows direct def-use links
/// and `cc.continue` forwarding through single-block lexical
/// scopes, with continuation operands mapped to scope results by index.
/// All other boundaries return `std::nullopt`. Callers decide whether the
/// reached operation is suitable for their analysis or rewrite.
std::optional<ScalarWireStep> traverseScalarWire(mlir::Value wire);

/// Returns true when `operation` is a single-block `cc.scope` whose ordinary
/// exit has a valid positional scalar-wire interface and no unwind exit.
bool isSupportedScalarWireScope(mlir::Operation *operation);

/// Returns the outermost block connected to `block` only through supported
/// scalar-wire scopes.
mlir::Block *getScalarWireTraversalRoot(mlir::Block *block);

} // namespace cudaq::opt
