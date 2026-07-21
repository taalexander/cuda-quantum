/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
#include "cudaq/Optimizer/Analysis/CommutationAnalysis.h"
#include "CommutationRules.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "gtest/gtest.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
// clang-format on

using namespace mlir;

using cudaq::quake::detail::CanonicalOperationPair;
using cudaq::quake::detail::CommutationFailureReason;
using cudaq::quake::detail::CommutationProofReason;
using cudaq::quake::detail::CommutationRelation;
using cudaq::quake::detail::CommutationResult;
using cudaq::quake::detail::getCommutationFailureReasonId;
using cudaq::quake::detail::getCommutationProofReasonId;
using cudaq::quake::detail::getNormalizedOperatorDescriptor;
using cudaq::quake::detail::hash_value;
using cudaq::quake::detail::OperatorKind;
using cudaq::quake::detail::QuantumOperand;
using cudaq::quake::detail::QuantumOperandRole;

namespace {
class CommutationDescriptorTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<cudaq::cc::CCDialect>();
    context.loadDialect<cudaq::quake::QuakeDialect>();
    module = OwningOpRef<ModuleOp>(ModuleOp::create(UnknownLoc::get(&context)));
  }

  func::FuncOp createKernel(llvm::StringRef name, ArrayRef<Type> inputTypes) {
    OpBuilder builder(&context);
    builder.setInsertionPointToEnd(module->getBody());
    auto function =
        func::FuncOp::create(builder, builder.getUnknownLoc(), name,
                             builder.getFunctionType(inputTypes, TypeRange{}));
    function.addEntryBlock();
    return function;
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};
} // namespace

TEST(CommutationResultTest, ExposesStableReasonIdentifiers) {
  EXPECT_EQ(getCommutationProofReasonId(CommutationProofReason::SameOperation),
            "same-operation");
  EXPECT_EQ(
      getCommutationProofReasonId(CommutationProofReason::DisjointSupport),
      "disjoint-support");
  EXPECT_EQ(getCommutationProofReasonId(CommutationProofReason::PauliParity),
            "pauli-parity");
  EXPECT_EQ(getCommutationProofReasonId(
                CommutationProofReason::MutuallyExclusiveControls),
            "mutually-exclusive-controls");

  EXPECT_EQ(
      getCommutationFailureReasonId(CommutationFailureReason::NullOperation),
      "null-operation");
  EXPECT_EQ(
      getCommutationFailureReasonId(CommutationFailureReason::DifferentBlocks),
      "different-blocks");
  EXPECT_EQ(getCommutationFailureReasonId(
                CommutationFailureReason::UnsupportedOperation),
            "unsupported-operation");
  EXPECT_EQ(getCommutationFailureReasonId(
                CommutationFailureReason::UnsupportedQubitOperand),
            "unsupported-qubit-operand");
  EXPECT_EQ(getCommutationFailureReasonId(
                CommutationFailureReason::InvalidControlPolarity),
            "invalid-control-polarity");
  EXPECT_EQ(getCommutationFailureReasonId(CommutationFailureReason::NotProven),
            "not-proven");
}

TEST(CommutationResultTest, KeepsProofsAndFailuresDisjoint) {
  auto commutes =
      CommutationResult::getCommutes(CommutationProofReason::DisjointSupport);
  EXPECT_EQ(commutes.getRelation(), CommutationRelation::Commutes);
  EXPECT_EQ(commutes.getProofReason(), CommutationProofReason::DisjointSupport);
  EXPECT_FALSE(commutes.getFailureReason().has_value());

  auto doesNotCommute =
      CommutationResult::getDoesNotCommute(CommutationProofReason::PauliParity);
  EXPECT_EQ(doesNotCommute.getRelation(), CommutationRelation::DoesNotCommute);
  EXPECT_EQ(doesNotCommute.getProofReason(),
            CommutationProofReason::PauliParity);
  EXPECT_FALSE(doesNotCommute.getFailureReason().has_value());

  auto unknown =
      CommutationResult::getUnknown(CommutationFailureReason::NotProven);
  EXPECT_EQ(unknown.getRelation(), CommutationRelation::Unknown);
  EXPECT_FALSE(unknown.getProofReason().has_value());
  EXPECT_EQ(unknown.getFailureReason(), CommutationFailureReason::NotProven);
}

TEST_F(CommutationDescriptorTest, CanonicalOperationPairIsSymmetric) {
  auto wireType = cudaq::quake::WireType::get(&context);
  auto function = createKernel("pair", {wireType, wireType});
  OpBuilder builder(&function.front(), function.front().begin());
  auto location = builder.getUnknownLoc();
  auto *x =
      cudaq::quake::XOp::create(builder, location, function.getArgument(0))
          .getOperation();
  auto *z =
      cudaq::quake::ZOp::create(builder, location, function.getArgument(1))
          .getOperation();

  CanonicalOperationPair xz(x, z);
  CanonicalOperationPair zx(z, x);
  EXPECT_EQ(xz, zx);
  EXPECT_EQ(hash_value(xz), hash_value(zx));
  EXPECT_EQ(xz.getFirst(), zx.getFirst());
  EXPECT_EQ(xz.getSecond(), zx.getSecond());
}

TEST_F(CommutationDescriptorTest, NormalizesSwapTargetOrder) {
  auto wireType = cudaq::quake::WireType::get(&context);
  auto function = createKernel("swap", {wireType, wireType});
  OpBuilder builder(&function.front(), function.front().begin());
  auto location = builder.getUnknownLoc();
  auto q0 = function.getArgument(0);
  auto q1 = function.getArgument(1);
  auto *forward = cudaq::quake::SwapOp::create(builder, location, ValueRange{},
                                               ValueRange{q0, q1})
                      .getOperation();
  auto *reverse = cudaq::quake::SwapOp::create(builder, location, ValueRange{},
                                               ValueRange{q1, q0})
                      .getOperation();

  auto forwardDescriptor = getNormalizedOperatorDescriptor(forward);
  auto reverseDescriptor = getNormalizedOperatorDescriptor(reverse);
  ASSERT_TRUE(forwardDescriptor);
  ASSERT_TRUE(reverseDescriptor);
  EXPECT_EQ(*forwardDescriptor, *reverseDescriptor);
  ASSERT_EQ(forwardDescriptor->quantumOperands.size(), 2u);
  EXPECT_EQ(forwardDescriptor->quantumOperands[0].role,
            QuantumOperandRole::Target);
  EXPECT_EQ(forwardDescriptor->quantumOperands,
            reverseDescriptor->quantumOperands);
}

TEST_F(CommutationDescriptorTest, PreservesControlOrderAndPolarity) {
  auto wireType = cudaq::quake::WireType::get(&context);
  auto controlType = cudaq::quake::ControlType::get(&context);
  auto function = createKernel("controls", {controlType, wireType, wireType});
  OpBuilder builder(&function.front(), function.front().begin());
  auto location = builder.getUnknownLoc();
  auto q0 = function.getArgument(0);
  auto q1 = function.getArgument(1);
  auto q2 = function.getArgument(2);
  auto polarities = builder.getDenseBoolArrayAttr({true, false});
  auto *operation =
      cudaq::quake::XOp::create(builder, location, false, ValueRange{},
                                ValueRange{q0, q1}, ValueRange{q2}, polarities)
          .getOperation();

  auto descriptor = getNormalizedOperatorDescriptor(operation);
  ASSERT_TRUE(descriptor);
  ASSERT_EQ(descriptor->quantumOperands.size(), 3u);
  EXPECT_EQ(descriptor->quantumOperands[0],
            (QuantumOperand{q0, QuantumOperandRole::Control, true}));
  EXPECT_EQ(descriptor->quantumOperands[1],
            (QuantumOperand{q1, QuantumOperandRole::Control, false}));
  EXPECT_EQ(descriptor->quantumOperands[2],
            (QuantumOperand{q2, QuantumOperandRole::Target, false}));
}

TEST_F(CommutationDescriptorTest, PreservesDynamicParametersStructurally) {
  auto wireType = cudaq::quake::WireType::get(&context);
  auto parameterType = Float64Type::get(&context);
  auto function = createKernel("parameter", {parameterType, wireType});
  OpBuilder builder(&function.front(), function.front().begin());
  auto *operation = cudaq::quake::RxOp::create(
                        builder, builder.getUnknownLoc(), true,
                        ValueRange{function.getArgument(0)}, ValueRange{},
                        ValueRange{function.getArgument(1)})
                        .getOperation();

  auto descriptor = getNormalizedOperatorDescriptor(operation);
  ASSERT_TRUE(descriptor);
  EXPECT_EQ(descriptor->kind, OperatorKind::Rx);
  EXPECT_EQ(descriptor->parameters,
            (llvm::SmallVector<Value>{function.getArgument(0)}));
  EXPECT_TRUE(descriptor->isAdjoint);
}

TEST_F(CommutationDescriptorTest, DescribesSupportedOperatorFamilies) {
  auto wireType = cudaq::quake::WireType::get(&context);
  auto parameterType = Float64Type::get(&context);
  auto function = createKernel("families", {parameterType, parameterType,
                                            parameterType, wireType, wireType});
  OpBuilder builder(&function.front(), function.front().begin());
  auto location = builder.getUnknownLoc();
  auto p0 = function.getArgument(0);
  auto p1 = function.getArgument(1);
  auto p2 = function.getArgument(2);
  auto q0 = function.getArgument(3);
  auto q1 = function.getArgument(4);

  llvm::SmallVector<std::pair<Operation *, OperatorKind>> cases{
      {cudaq::quake::HOp::create(builder, location, q0).getOperation(),
       OperatorKind::H},
      {cudaq::quake::XOp::create(builder, location, q0).getOperation(),
       OperatorKind::X},
      {cudaq::quake::YOp::create(builder, location, q0).getOperation(),
       OperatorKind::Y},
      {cudaq::quake::ZOp::create(builder, location, q0).getOperation(),
       OperatorKind::Z},
      {cudaq::quake::SOp::create(builder, location, q0).getOperation(),
       OperatorKind::S},
      {cudaq::quake::TOp::create(builder, location, q0).getOperation(),
       OperatorKind::T},
      {cudaq::quake::SwapOp::create(builder, location, ValueRange{},
                                    ValueRange{q0, q1})
           .getOperation(),
       OperatorKind::Swap},
      {cudaq::quake::R1Op::create(builder, location, ValueRange{p0},
                                  ValueRange{}, ValueRange{q0})
           .getOperation(),
       OperatorKind::R1},
      {cudaq::quake::RxOp::create(builder, location, ValueRange{p0},
                                  ValueRange{}, ValueRange{q0})
           .getOperation(),
       OperatorKind::Rx},
      {cudaq::quake::RyOp::create(builder, location, ValueRange{p0},
                                  ValueRange{}, ValueRange{q0})
           .getOperation(),
       OperatorKind::Ry},
      {cudaq::quake::RzOp::create(builder, location, ValueRange{p0},
                                  ValueRange{}, ValueRange{q0})
           .getOperation(),
       OperatorKind::Rz},
      {cudaq::quake::PhasedRxOp::create(builder, location, ValueRange{p0, p1},
                                        ValueRange{}, ValueRange{q0})
           .getOperation(),
       OperatorKind::PhasedRx},
      {cudaq::quake::U2Op::create(builder, location, ValueRange{p0, p1},
                                  ValueRange{}, ValueRange{q0})
           .getOperation(),
       OperatorKind::U2},
      {cudaq::quake::U3Op::create(builder, location, ValueRange{p0, p1, p2},
                                  ValueRange{}, ValueRange{q0})
           .getOperation(),
       OperatorKind::U3},
      {cudaq::quake::ExpPauliOp::create(builder, location, ValueRange{p0},
                                        ValueRange{}, ValueRange{q0}, "X")
           .getOperation(),
       OperatorKind::ExpPauli}};

  for (auto [operation, expectedKind] : cases) {
    auto descriptor = getNormalizedOperatorDescriptor(operation);
    ASSERT_TRUE(descriptor) << operation->getName().getStringRef().str();
    EXPECT_EQ(descriptor->kind, expectedKind);
    if (expectedKind == OperatorKind::ExpPauli)
      EXPECT_EQ(descriptor->pauliWord.getValue(), "X");
  }
}

TEST_F(CommutationDescriptorTest, RejectsMalformedLiteralExpPauliDescriptors) {
  auto wireType = cudaq::quake::WireType::get(&context);
  auto parameterType = Float64Type::get(&context);
  auto function =
      createKernel("malformed_pauli", {parameterType, parameterType, wireType});
  OpBuilder builder(&function.front(), function.front().begin());
  auto location = builder.getUnknownLoc();
  auto p0 = function.getArgument(0);
  auto p1 = function.getArgument(1);
  auto q0 = function.getArgument(2);

  llvm::SmallVector<Operation *> cases{
      cudaq::quake::ExpPauliOp::create(builder, location, ValueRange{p0},
                                       ValueRange{}, ValueRange{q0}, "XX")
          .getOperation(),
      cudaq::quake::ExpPauliOp::create(builder, location, ValueRange{p0},
                                       ValueRange{}, ValueRange{q0}, "A")
          .getOperation(),
      cudaq::quake::ExpPauliOp::create(builder, location, ValueRange{p0, p1},
                                       ValueRange{}, ValueRange{q0}, "X")
          .getOperation()};

  for (Operation *operation : cases) {
    auto descriptor = getNormalizedOperatorDescriptor(operation);
    ASSERT_FALSE(descriptor);
    EXPECT_EQ(descriptor.getFailureReason(),
              CommutationFailureReason::UnsupportedOperation);
  }
}

TEST_F(CommutationDescriptorTest, ClassifiesUnsupportedInputsConservatively) {
  auto nullDescriptor = getNormalizedOperatorDescriptor(nullptr);
  EXPECT_FALSE(nullDescriptor);
  EXPECT_EQ(nullDescriptor.getFailureReason(),
            CommutationFailureReason::NullOperation);

  auto refType = cudaq::quake::RefType::get(&context);
  auto veqType = cudaq::quake::VeqType::get(&context, 2);
  auto function = createKernel("unsupported", {refType, veqType});
  OpBuilder builder(&function.front(), function.front().begin());
  auto location = builder.getUnknownLoc();

  auto *constant =
      arith::ConstantIntOp::create(builder, location, 0, 64).getOperation();
  auto nonOperator = getNormalizedOperatorDescriptor(constant);
  EXPECT_FALSE(nonOperator);
  EXPECT_EQ(nonOperator.getFailureReason(),
            CommutationFailureReason::UnsupportedOperation);

  auto *dynamicPauli = cudaq::quake::ExpPauliOp::create(
                           builder, location, ValueRange{}, ValueRange{},
                           ValueRange{function.getArgument(0)}, Value{})
                           .getOperation();
  auto unsupportedOperator = getNormalizedOperatorDescriptor(dynamicPauli);
  EXPECT_FALSE(unsupportedOperator);
  EXPECT_EQ(unsupportedOperator.getFailureReason(),
            CommutationFailureReason::UnsupportedOperation);

  for (auto operand : function.getArguments()) {
    auto *operatorWithUnsupportedOperand =
        cudaq::quake::XOp::create(builder, location, operand).getOperation();
    auto unsupportedOperand =
        getNormalizedOperatorDescriptor(operatorWithUnsupportedOperand);
    EXPECT_FALSE(unsupportedOperand);
    EXPECT_EQ(unsupportedOperand.getFailureReason(),
              CommutationFailureReason::UnsupportedQubitOperand);
  }
}
