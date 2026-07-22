/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
#include "cudaq/Optimizer/Analysis/CommutationAnalysis.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <utility>
#include <vector>
// clang-format on

using namespace mlir;

using cudaq::quake::detail::CommutationAnalysis;
using cudaq::quake::detail::CommutationReason;
using cudaq::quake::detail::CommutationResult;
using cudaq::quake::detail::CommutationStatus;
using cudaq::quake::detail::getCommutationReasonId;

namespace {
class CommutationAnalysisTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<cudaq::cc::CCDialect>();
    context.loadDialect<cudaq::quake::QuakeDialect>();
    module = OwningOpRef<ModuleOp>(ModuleOp::create(UnknownLoc::get(&context)));
  }

  func::FuncOp createKernel(llvm::StringRef name,
                            ArrayRef<Type> inputTypes = {}) {
    builder.setInsertionPointToEnd(module->getBody());
    auto function = func::FuncOp::create(
        builder, loc, name, builder.getFunctionType(inputTypes, {}));
    function->setAttr("cudaq-kernel", builder.getUnitAttr());
    function.addEntryBlock();
    builder.setInsertionPointToStart(&function.front());
    return function;
  }

  Value createWire() {
    return cudaq::quake::NullWireOp::create(builder, loc, wireType());
  }

  Value createConstant(double value) {
    return arith::ConstantFloatOp::create(builder, loc, builder.getF64Type(),
                                          APFloat(value));
  }

  template <typename Op>
  Op createGate(ValueRange parameters, ValueRange controls,
                ValueRange targets) {
    llvm::SmallVector<Type> wireResults;
    for (Value control : controls)
      if (isa<cudaq::quake::WireType>(control.getType()))
        wireResults.push_back(control.getType());
    for (Value target : targets)
      if (isa<cudaq::quake::WireType>(target.getType()))
        wireResults.push_back(target.getType());
    return Op::create(builder, loc, TypeRange{wireResults}, UnitAttr{},
                      parameters, controls, targets, DenseBoolArrayAttr{});
  }

  template <typename Op>
  Op createGate(ValueRange controls, ValueRange targets) {
    return createGate<Op>(ValueRange{}, controls, targets);
  }

  template <typename Op>
  Op createGate(Value target) {
    return createGate<Op>(ValueRange{}, ValueRange{target});
  }

  cudaq::quake::ExpPauliOp createExpPauli(ValueRange parameters,
                                          ValueRange targets,
                                          llvm::StringRef word) {
    llvm::SmallVector<Type> wireResults(targets.size(), wireType());
    return cudaq::quake::ExpPauliOp::create(
        builder, loc, TypeRange{wireResults}, UnitAttr{}, parameters,
        ValueRange{}, targets, DenseBoolArrayAttr{}, Value{},
        builder.getStringAttr(word));
  }

  cudaq::quake::ExpPauliOp createExpPauli(ValueRange parameters,
                                          ValueRange targets, Value word) {
    llvm::SmallVector<Type> wireResults(targets.size(), wireType());
    return cudaq::quake::ExpPauliOp::create(
        builder, loc, TypeRange{wireResults}, UnitAttr{}, parameters,
        ValueRange{}, targets, DenseBoolArrayAttr{}, word, StringAttr{});
  }

  void finishFunction() {
    llvm::SmallVector<Value> unusedWires;
    for (Operation &operation : *builder.getInsertionBlock())
      for (Value result : operation.getResults())
        if (isa<cudaq::quake::WireType>(result.getType()) && result.use_empty())
          unusedWires.push_back(result);
    for (Value wire : unusedWires)
      cudaq::quake::SinkOp::create(builder, loc, TypeRange{}, wire);
    func::ReturnOp::create(builder, loc);
  }

  template <typename Op>
  static Value getWire(Op operation, unsigned index = 0) {
    return operation.getWires()[index];
  }

  static void expectPair(CommutationAnalysis &analysis, Operation *lhs,
                         Operation *rhs, CommutationStatus status,
                         CommutationReason reason) {
    auto forward = analysis.getResult(lhs, rhs);
    auto reverse = analysis.getResult(rhs, lhs);
    EXPECT_EQ(forward.status, status);
    EXPECT_EQ(forward.reason, reason);
    EXPECT_EQ(reverse.status, status);
    EXPECT_EQ(reverse.reason, reason);
    EXPECT_EQ(analysis.canCommute(lhs, rhs),
              status == CommutationStatus::Commutes);
    EXPECT_EQ(analysis.canCommute(rhs, lhs),
              status == CommutationStatus::Commutes);
  }

  cudaq::quake::WireType wireType() {
    return cudaq::quake::WireType::get(&context);
  }

  cudaq::quake::ControlType controlType() {
    return cudaq::quake::ControlType::get(&context);
  }

  MLIRContext context;
  OpBuilder builder{&context};
  Location loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module;
};
} // namespace

TEST(PauliWordTest, Symbolize) {
  auto word = cudaq::quake::symbolizePauliWord("IXYZ");
  ASSERT_TRUE(word);
  ASSERT_EQ(word->size(), 4u);
  EXPECT_EQ((*word)[0], cudaq::quake::Pauli::I);
  EXPECT_EQ((*word)[1], cudaq::quake::Pauli::X);
  EXPECT_EQ((*word)[2], cudaq::quake::Pauli::Y);
  EXPECT_EQ((*word)[3], cudaq::quake::Pauli::Z);
  EXPECT_FALSE(cudaq::quake::symbolizePauliWord("XA"));
}

TEST(CommutationResultTest, ResultContract) {
  const std::vector<std::pair<CommutationReason, llvm::StringRef>> cases{
      {CommutationReason::DisjointSupport, "disjoint-support"},
      {CommutationReason::SameOperation, "same-operation"},
      {CommutationReason::ComputationalDiagonal, "computational-diagonal"},
      {CommutationReason::SameAxis, "same-axis"},
      {CommutationReason::EvenPauliParity, "even-pauli-parity"},
      {CommutationReason::OddPauliParity, "odd-pauli-parity"},
      {CommutationReason::DiagonalOnControls, "diagonal-on-controls"},
      {CommutationReason::CompatibleControlledTargets,
       "compatible-controlled-targets"},
      {CommutationReason::MutuallyExclusiveControls,
       "mutually-exclusive-controls"},
      {CommutationReason::NullOperation, "null-operation"},
      {CommutationReason::DifferentBlocks, "different-blocks"},
      {CommutationReason::UnsupportedOperationKind,
       "unsupported-operation-kind"},
      {CommutationReason::UnsupportedQuantumOperandType,
       "unsupported-quantum-operand-type"},
      {CommutationReason::MalformedControlPolarity,
       "malformed-control-polarity"},
      {CommutationReason::UnmappedQubitId, "unmapped-qubit-id"},
      {CommutationReason::DuplicateQubitOperand, "duplicate-qubit-operand"},
      {CommutationReason::UnsupportedPauliWord, "unsupported-pauli-word"},
      {CommutationReason::NoApplicableRule, "no-applicable-rule"}};
  for (auto [reason, identifier] : cases)
    EXPECT_EQ(getCommutationReasonId(reason), identifier);

  EXPECT_TRUE(static_cast<bool>(CommutationResult{
      CommutationStatus::Commutes, CommutationReason::DisjointSupport}));
  EXPECT_FALSE(static_cast<bool>(CommutationResult{
      CommutationStatus::DoesNotCommute, CommutationReason::OddPauliParity}));
  EXPECT_FALSE(static_cast<bool>(CommutationResult{
      CommutationStatus::Indeterminate, CommutationReason::NoApplicableRule}));
}

TEST_F(CommutationAnalysisTest, DisjointSupport) {
  auto function = createKernel("disjoint");
  Value q0 = createWire();
  Value q1 = createWire();
  auto x = createGate<cudaq::quake::XOp>(q0);
  auto h = createGate<cudaq::quake::HOp>(q1);
  finishFunction();

  CommutationAnalysis analysis(function.front());
  expectPair(analysis, x, h, CommutationStatus::Commutes,
             CommutationReason::DisjointSupport);
}

TEST_F(CommutationAnalysisTest, SameOperation) {
  auto function = createKernel("same_operation", {builder.getF64Type()});
  Value theta = function.getArgument(0);
  Value q0 = createWire();
  Value q1 = createWire();
  Value q2 = createWire();
  Value q3 = createWire();
  auto rx0 = createGate<cudaq::quake::RxOp>(ValueRange{theta}, ValueRange{},
                                            ValueRange{q0});
  auto rx1 = createGate<cudaq::quake::RxOp>(ValueRange{theta}, ValueRange{},
                                            ValueRange{getWire(rx0)});
  rx1.setIsAdj(true);
  auto swap0 = createGate<cudaq::quake::SwapOp>(ValueRange{},
                                                ValueRange{getWire(rx1), q1});
  auto swap1 = createGate<cudaq::quake::SwapOp>(
      ValueRange{}, ValueRange{getWire(swap0, 1), getWire(swap0, 0)});
  Value zero0 = createConstant(0.0);
  Value zero1 = createConstant(0.0);
  Value one0 = createConstant(1.0);
  Value one1 = createConstant(1.0);
  auto u20 = createGate<cudaq::quake::U2Op>(ValueRange{zero0, one0},
                                            ValueRange{}, ValueRange{q2});
  auto u21 = createGate<cudaq::quake::U2Op>(
      ValueRange{zero1, one1}, ValueRange{}, ValueRange{getWire(u20)});
  auto u30 = createGate<cudaq::quake::U3Op>(ValueRange{theta, zero0, one0},
                                            ValueRange{}, ValueRange{q3});
  auto u31 = createGate<cudaq::quake::U3Op>(
      ValueRange{theta, zero0, one0}, ValueRange{}, ValueRange{getWire(u30)});
  finishFunction();

  CommutationAnalysis analysis(function.front());
  expectPair(analysis, rx0, rx1, CommutationStatus::Commutes,
             CommutationReason::SameOperation);
  expectPair(analysis, swap0, swap1, CommutationStatus::Commutes,
             CommutationReason::SameOperation);
  expectPair(analysis, u20, u21, CommutationStatus::Commutes,
             CommutationReason::SameOperation);
  expectPair(analysis, u30, u31, CommutationStatus::Commutes,
             CommutationReason::SameOperation);
}

TEST_F(CommutationAnalysisTest, ComputationalDiagonal) {
  auto function = createKernel("diagonal");
  Value q = createWire();
  Value angle = createConstant(0.5);
  auto z = createGate<cudaq::quake::ZOp>(q);
  auto s = createGate<cudaq::quake::SOp>(getWire(z));
  auto t = createGate<cudaq::quake::TOp>(getWire(s));
  auto r1 = createGate<cudaq::quake::R1Op>(ValueRange{angle}, ValueRange{},
                                           ValueRange{getWire(t)});
  auto rz = createGate<cudaq::quake::RzOp>(ValueRange{angle}, ValueRange{},
                                           ValueRange{getWire(r1)});
  finishFunction();

  CommutationAnalysis analysis(function.front());
  expectPair(analysis, z, s, CommutationStatus::Commutes,
             CommutationReason::ComputationalDiagonal);
  expectPair(analysis, s, t, CommutationStatus::Commutes,
             CommutationReason::ComputationalDiagonal);
  expectPair(analysis, r1, rz, CommutationStatus::Commutes,
             CommutationReason::ComputationalDiagonal);
}

TEST_F(CommutationAnalysisTest, SameAxis) {
  auto function = createKernel("same_axis");
  Value q0 = createWire();
  Value q1 = createWire();
  Value q2 = createWire();
  Value angle0 = createConstant(0.5);
  Value angle1 = createConstant(1.0);
  Value phase = createConstant(0.25);
  Value otherPhase = createConstant(0.75);
  auto x = createGate<cudaq::quake::XOp>(q0);
  auto rx = createGate<cudaq::quake::RxOp>(ValueRange{angle0}, ValueRange{},
                                           ValueRange{getWire(x)});
  auto phased0 = createGate<cudaq::quake::PhasedRxOp>(
      ValueRange{angle0, phase}, ValueRange{}, ValueRange{q1});
  auto phased1 = createGate<cudaq::quake::PhasedRxOp>(
      ValueRange{angle1, phase}, ValueRange{}, ValueRange{getWire(phased0)});
  auto phased2 = createGate<cudaq::quake::PhasedRxOp>(
      ValueRange{angle0, otherPhase}, ValueRange{},
      ValueRange{getWire(phased1)});
  auto y = createGate<cudaq::quake::YOp>(q2);
  auto ry = createGate<cudaq::quake::RyOp>(ValueRange{angle0}, ValueRange{},
                                           ValueRange{getWire(y)});
  finishFunction();

  CommutationAnalysis analysis(function.front());
  expectPair(analysis, x, rx, CommutationStatus::Commutes,
             CommutationReason::SameAxis);
  expectPair(analysis, phased0, phased1, CommutationStatus::Commutes,
             CommutationReason::SameAxis);
  expectPair(analysis, phased1, phased2, CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);
  expectPair(analysis, y, ry, CommutationStatus::Commutes,
             CommutationReason::SameAxis);
}

TEST_F(CommutationAnalysisTest, PauliParity) {
  auto function = createKernel("pauli_parity");
  Value q0 = createWire();
  Value q1 = createWire();
  Value q2 = createWire();
  Value q3 = createWire();
  Value angle = createConstant(0.5);
  auto xx = createExpPauli(ValueRange{angle}, ValueRange{q0, q1}, "XX");
  auto zz = createExpPauli(ValueRange{angle},
                           ValueRange{getWire(xx, 0), getWire(xx, 1)}, "ZZ");
  auto x = createGate<cudaq::quake::XOp>(q2);
  auto z = createGate<cudaq::quake::ZOp>(getWire(x));
  auto expX = createExpPauli(ValueRange{angle}, ValueRange{q3}, "X");
  auto expZ = createGate<cudaq::quake::ZOp>(getWire(expX));
  finishFunction();

  CommutationAnalysis analysis(function.front());
  expectPair(analysis, xx, zz, CommutationStatus::Commutes,
             CommutationReason::EvenPauliParity);
  expectPair(analysis, x, z, CommutationStatus::DoesNotCommute,
             CommutationReason::OddPauliParity);
  expectPair(analysis, expX, expZ, CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);
}

TEST_F(CommutationAnalysisTest, DiagonalOnControls) {
  auto function = createKernel("diagonal_on_controls");
  Value control = createWire();
  Value target = createWire();
  auto z = createGate<cudaq::quake::ZOp>(control);
  auto controlledX =
      createGate<cudaq::quake::XOp>(ValueRange{getWire(z)}, ValueRange{target});
  finishFunction();

  CommutationAnalysis analysis(function.front());
  expectPair(analysis, z, controlledX, CommutationStatus::Commutes,
             CommutationReason::DiagonalOnControls);
}

TEST_F(CommutationAnalysisTest, CompatibleControlledTargets) {
  auto function = createKernel("compatible_targets");
  Value controlWire = createWire();
  Value target = createWire();
  Value angle = createConstant(0.5);
  Value control = cudaq::quake::ToControlOp::create(builder, loc, controlType(),
                                                    controlWire);
  auto controlledX =
      createGate<cudaq::quake::XOp>(ValueRange{control}, ValueRange{target});
  auto controlledRx = createGate<cudaq::quake::RxOp>(
      ValueRange{angle}, ValueRange{control}, ValueRange{getWire(controlledX)});
  Value crossoverControl = createWire();
  Value crossoverTarget = createWire();
  auto crossoverX = createGate<cudaq::quake::XOp>(ValueRange{crossoverControl},
                                                  ValueRange{crossoverTarget});
  auto crossoverZ = createGate<cudaq::quake::ZOp>(
      ValueRange{getWire(crossoverX, 1)}, ValueRange{getWire(crossoverX, 0)});
  finishFunction();

  CommutationAnalysis analysis(function.front());
  expectPair(analysis, controlledX, controlledRx, CommutationStatus::Commutes,
             CommutationReason::CompatibleControlledTargets);
  expectPair(analysis, crossoverX, crossoverZ, CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);
}

TEST_F(CommutationAnalysisTest, MutuallyExclusiveControls) {
  auto function = createKernel("exclusive_controls");
  Value controlWire = createWire();
  Value target = createWire();
  Value control = cudaq::quake::ToControlOp::create(builder, loc, controlType(),
                                                    controlWire);
  auto controlledX =
      createGate<cudaq::quake::XOp>(ValueRange{control}, ValueRange{target});
  auto controlledY = createGate<cudaq::quake::YOp>(
      ValueRange{control}, ValueRange{getWire(controlledX)});
  controlledY.setNegatedQubitControlsAttr(
      builder.getDenseBoolArrayAttr({true}));
  finishFunction();

  CommutationAnalysis analysis(function.front());
  expectPair(analysis, controlledX, controlledY, CommutationStatus::Commutes,
             CommutationReason::MutuallyExclusiveControls);
}

TEST_F(CommutationAnalysisTest, ConservativeOutcomes) {
  auto function = createKernel("conservative");
  Value q0 = createWire();
  Value q1 = createWire();
  Value angle = createConstant(0.5);
  auto x = createGate<cudaq::quake::XOp>(q0);
  auto h = createGate<cudaq::quake::HOp>(getWire(x));

  Value control =
      cudaq::quake::ToControlOp::create(builder, loc, controlType(), q1);
  auto malformed = createGate<cudaq::quake::XOp>(ValueRange{control},
                                                 ValueRange{getWire(h)});
  malformed.setNegatedQubitControlsAttr(
      builder.getDenseBoolArrayAttr({true, false}));

  auto badPauli =
      createExpPauli(ValueRange{angle}, ValueRange{getWire(malformed)}, "XX");
  auto z = createGate<cudaq::quake::ZOp>(getWire(badPauli));
  cudaq::quake::SinkOp::create(builder, loc, TypeRange{}, getWire(z));
  auto returnOp = func::ReturnOp::create(builder, loc);

  CommutationAnalysis analysis(function.front());
  expectPair(analysis, nullptr, x, CommutationStatus::Indeterminate,
             CommutationReason::NullOperation);
  expectPair(analysis, x, returnOp, CommutationStatus::Indeterminate,
             CommutationReason::UnsupportedOperationKind);
  expectPair(analysis, x, h, CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);
  expectPair(analysis, malformed, z, CommutationStatus::Indeterminate,
             CommutationReason::MalformedControlPolarity);
  expectPair(analysis, badPauli, z, CommutationStatus::Indeterminate,
             CommutationReason::UnsupportedPauliWord);

  auto aggregate =
      createKernel("aggregate", {cudaq::quake::VeqType::get(&context, 2)});
  auto aggregateX =
      cudaq::quake::XOp::create(builder, loc, aggregate.getArgument(0));
  finishFunction();
  CommutationAnalysis aggregateAnalysis(aggregate.front());
  expectPair(aggregateAnalysis, aggregateX, aggregateX,
             CommutationStatus::Indeterminate,
             CommutationReason::UnsupportedQuantumOperandType);

  auto duplicate = createKernel("duplicate_role");
  Value duplicateWire = createWire();
  auto duplicateX = createGate<cudaq::quake::XOp>(ValueRange{duplicateWire},
                                                  ValueRange{duplicateWire});
  finishFunction();
  CommutationAnalysis duplicateAnalysis(duplicate.front());
  expectPair(duplicateAnalysis, duplicateX, duplicateX,
             CommutationStatus::Indeterminate,
             CommutationReason::DuplicateQubitOperand);

  auto dynamicPauli =
      createKernel("dynamic_pauli", {cudaq::cc::CharspanType::get(&context)});
  Value dynamicTarget = createWire();
  Value dynamicAngle = createConstant(0.5);
  auto dynamicExp =
      createExpPauli(ValueRange{dynamicAngle}, ValueRange{dynamicTarget},
                     dynamicPauli.getArgument(0));
  auto dynamicZ = createGate<cudaq::quake::ZOp>(getWire(dynamicExp));
  finishFunction();
  CommutationAnalysis dynamicAnalysis(dynamicPauli.front());
  expectPair(dynamicAnalysis, dynamicExp, dynamicZ,
             CommutationStatus::Indeterminate,
             CommutationReason::UnsupportedPauliWord);
}

TEST_F(CommutationAnalysisTest, BlockLocalQubitIds) {
  builder.setInsertionPointToEnd(module->getBody());
  cudaq::quake::WireSetOp::create(builder, loc, "wires", 2, ElementsAttr{});
  auto borrowed = createKernel("borrowed");
  auto q0a =
      cudaq::quake::BorrowWireOp::create(builder, loc, wireType(), "wires", 0);
  auto x0a = createGate<cudaq::quake::XOp>(q0a.getResult());
  cudaq::quake::ReturnWireOp::create(builder, loc, getWire(x0a));
  auto q0b =
      cudaq::quake::BorrowWireOp::create(builder, loc, wireType(), "wires", 0);
  auto x0b = createGate<cudaq::quake::XOp>(q0b.getResult());
  cudaq::quake::ReturnWireOp::create(builder, loc, getWire(x0b));
  auto q1 =
      cudaq::quake::BorrowWireOp::create(builder, loc, wireType(), "wires", 1);
  auto h1 = createGate<cudaq::quake::HOp>(q1.getResult());
  cudaq::quake::ReturnWireOp::create(builder, loc, getWire(h1));
  finishFunction();
  CommutationAnalysis borrowedAnalysis(borrowed.front());
  expectPair(borrowedAnalysis, x0a, x0b, CommutationStatus::Commutes,
             CommutationReason::SameOperation);
  expectPair(borrowedAnalysis, x0a, h1, CommutationStatus::Commutes,
             CommutationReason::DisjointSupport);

  auto converted = createKernel("converted");
  Value q = createWire();
  Value angle = createConstant(0.5);
  auto x = createGate<cudaq::quake::XOp>(q);
  Value control = cudaq::quake::ToControlOp::create(builder, loc, controlType(),
                                                    getWire(x));
  Value returned =
      cudaq::quake::FromControlOp::create(builder, loc, wireType(), control);
  auto rx = createGate<cudaq::quake::RxOp>(ValueRange{angle}, ValueRange{},
                                           ValueRange{returned});
  finishFunction();
  CommutationAnalysis convertedAnalysis(converted.front());
  expectPair(convertedAnalysis, x, rx, CommutationStatus::Commutes,
             CommutationReason::SameAxis);

  auto other = createKernel("other");
  Value otherWire = createWire();
  auto otherZ = createGate<cudaq::quake::ZOp>(otherWire);
  finishFunction();
  expectPair(convertedAnalysis, x, otherZ, CommutationStatus::Indeterminate,
             CommutationReason::DifferentBlocks);

  auto arguments = createKernel("arguments", {wireType(), wireType()});
  auto argumentX = createGate<cudaq::quake::XOp>(arguments.getArgument(0));
  auto argumentH = createGate<cudaq::quake::HOp>(arguments.getArgument(1));
  finishFunction();
  CommutationAnalysis argumentAnalysis(arguments.front());
  expectPair(argumentAnalysis, argumentX, argumentH,
             CommutationStatus::Commutes, CommutationReason::DisjointSupport);

  auto mixed = createKernel("mixed_results");
  Value wireControl = createWire();
  Value reusableWire = createWire();
  Value mixedTarget = createWire();
  Value reusableControl = cudaq::quake::ToControlOp::create(
      builder, loc, controlType(), reusableWire);
  auto mixedX = createGate<cudaq::quake::XOp>(
      ValueRange{wireControl, reusableControl}, ValueRange{mixedTarget});
  auto controlZ = createGate<cudaq::quake::ZOp>(getWire(mixedX, 0));
  auto targetX = createGate<cudaq::quake::XOp>(getWire(mixedX, 1));
  finishFunction();
  CommutationAnalysis mixedAnalysis(mixed.front());
  expectPair(mixedAnalysis, mixedX, controlZ, CommutationStatus::Commutes,
             CommutationReason::DiagonalOnControls);
  expectPair(mixedAnalysis, mixedX, targetX, CommutationStatus::Commutes,
             CommutationReason::CompatibleControlledTargets);

  builder.setInsertionPointToEnd(module->getBody());
  func::FuncOp::create(builder, loc, "wire_source",
                       builder.getFunctionType({}, {wireType()}));
  auto callResult = createKernel("call_result");
  auto call = func::CallOp::create(builder, loc, "wire_source",
                                   TypeRange{wireType()}, ValueRange{});
  auto callX = createGate<cudaq::quake::XOp>(call.getResult(0));
  auto callZ = createGate<cudaq::quake::ZOp>(getWire(callX));
  finishFunction();
  CommutationAnalysis callAnalysis(callResult.front());
  expectPair(callAnalysis, callX, callZ, CommutationStatus::Indeterminate,
             CommutationReason::UnmappedQubitId);
}

TEST_F(CommutationAnalysisTest, RebuildAfterMutation) {
  auto function = createKernel("mutation");
  Value q0 = createWire();
  Value q1 = createWire();
  auto x = createGate<cudaq::quake::XOp>(q0);
  auto h = createGate<cudaq::quake::HOp>(q1);
  finishFunction();
  {
    CommutationAnalysis analysis(function.front());
    EXPECT_TRUE(analysis.canCommute(x, h));
  }

  h->setOperand(0, x.getTarget());
  CommutationAnalysis rebuilt(function.front());
  expectPair(rebuilt, x, h, CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);
}
