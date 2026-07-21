/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Analysis/CommutationAnalysis.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include <optional>

namespace mlir {
class Operation;
} // namespace mlir

namespace cudaq::quake::detail {

enum class OperatorKind {
  H,
  X,
  Y,
  Z,
  S,
  T,
  Swap,
  R1,
  Rx,
  Ry,
  Rz,
  PhasedRx,
  U2,
  U3,
  ExpPauli
};

enum class QuantumOperandRole { Control, Target };

struct QuantumOperand {
  mlir::Value value;
  QuantumOperandRole role;
  bool negated;

  bool operator==(const QuantumOperand &other) const;
};

struct NormalizedOperatorDescriptor {
  OperatorKind kind;
  llvm::SmallVector<mlir::Value> parameters;
  llvm::SmallVector<QuantumOperand> quantumOperands;
  mlir::StringAttr pauliWord;
  bool isAdjoint;

  bool operator==(const NormalizedOperatorDescriptor &other) const;
};

class DescriptorConstructionResult {
public:
  static DescriptorConstructionResult
  getSuccess(NormalizedOperatorDescriptor descriptor);
  static DescriptorConstructionResult
  getFailure(CommutationFailureReason reason);

  explicit operator bool() const;
  const NormalizedOperatorDescriptor &operator*() const;
  const NormalizedOperatorDescriptor *operator->() const;
  CommutationFailureReason getFailureReason() const;

private:
  DescriptorConstructionResult(
      std::optional<NormalizedOperatorDescriptor> descriptor,
      std::optional<CommutationFailureReason> failureReason);

  std::optional<NormalizedOperatorDescriptor> descriptor;
  std::optional<CommutationFailureReason> failureReason;
};

DescriptorConstructionResult
getNormalizedOperatorDescriptor(mlir::Operation *operation);

class CanonicalOperationPair {
public:
  CanonicalOperationPair(mlir::Operation *lhs, mlir::Operation *rhs);

  mlir::Operation *getFirst() const { return first; }
  mlir::Operation *getSecond() const { return second; }
  bool operator==(const CanonicalOperationPair &other) const;

private:
  mlir::Operation *first;
  mlir::Operation *second;
};

llvm::hash_code hash_value(const CanonicalOperationPair &pair);

} // namespace cudaq::quake::detail
