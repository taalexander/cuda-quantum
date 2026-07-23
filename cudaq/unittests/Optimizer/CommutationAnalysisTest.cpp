/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/CommutationAnalysis.h"
#include "gtest/gtest.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include <utility>
#include <vector>

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
  }

  OwningOpRef<ModuleOp> parseModule(llvm::StringRef source) {
    auto parsed = parseSourceString<ModuleOp>(source, &context);
    if (parsed && succeeded(verify(*parsed)))
      return parsed;
    return {};
  }

  static func::FuncOp getFunction(ModuleOp module, llvm::StringRef name) {
    auto function = module.lookupSymbol<func::FuncOp>(name);
    EXPECT_TRUE(function);
    return function;
  }

  static llvm::SmallVector<Operation *>
  getOperators(ModuleOp module, llvm::StringRef functionName) {
    llvm::SmallVector<Operation *> operators;
    auto function = getFunction(module, functionName);
    if (!function)
      return operators;
    for (Operation &operation : function.front())
      if (isa<cudaq::quake::OperatorInterface>(operation))
        operators.push_back(&operation);
    return operators;
  }

  func::FuncOp createMutationKernel() {
    module = OwningOpRef<ModuleOp>(ModuleOp::create(UnknownLoc::get(&context)));
    builder.setInsertionPointToEnd(module->getBody());
    auto function = func::FuncOp::create(builder, loc, "mutation",
                                         builder.getFunctionType({}, {}));
    function->setAttr("cudaq-kernel", builder.getUnitAttr());
    function.addEntryBlock();
    builder.setInsertionPointToStart(&function.front());
    return function;
  }

  Value createWire() {
    return cudaq::quake::NullWireOp::create(builder, loc, wireType());
  }

  template <typename Op>
  Op createGate(Value target) {
    return Op::create(builder, loc, TypeRange{wireType()}, UnitAttr{},
                      ValueRange{}, ValueRange{}, ValueRange{target},
                      DenseBoolArrayAttr{});
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
  auto module = parseModule(R"mlir(
    module {
      func.func @disjoint() {
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %x = quake.x %q0 : (!quake.wire) -> !quake.wire
        %h = quake.h %q1 : (!quake.wire) -> !quake.wire
        quake.sink %x : !quake.wire
        quake.sink %h : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto operators = getOperators(*module, "disjoint");
  ASSERT_EQ(operators.size(), 2u);
  auto function = getFunction(*module, "disjoint");
  CommutationAnalysis analysis(function.front());
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::DisjointSupport);
}

TEST_F(CommutationAnalysisTest, SameOperation) {
  auto module = parseModule(R"mlir(
    module {
      func.func @same_operation(%theta: f64) {
        %zero0 = arith.constant 0.0 : f64
        %zero1 = arith.constant 0.0 : f64
        %one0 = arith.constant 1.0 : f64
        %one1 = arith.constant 1.0 : f64
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %q2 = quake.null_wire
        %q3 = quake.null_wire
        %rx0 = quake.rx (%theta) %q0 : (f64, !quake.wire) -> !quake.wire
        %rx1 = quake.rx<adj> (%theta) %rx0 : (f64, !quake.wire) -> !quake.wire
        %swap0:2 = quake.swap %rx1, %q1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %swap1:2 = quake.swap %swap0#1, %swap0#0 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %u20 = quake.u2 (%zero0, %one0) %q2 : (f64, f64, !quake.wire) -> !quake.wire
        %u21 = quake.u2 (%zero1, %one1) %u20 : (f64, f64, !quake.wire) -> !quake.wire
        %u30 = quake.u3 (%theta, %zero0, %one0) %q3 : (f64, f64, f64, !quake.wire) -> !quake.wire
        %u31 = quake.u3 (%theta, %zero0, %one0) %u30 : (f64, f64, f64, !quake.wire) -> !quake.wire
        quake.sink %swap1#0 : !quake.wire
        quake.sink %swap1#1 : !quake.wire
        quake.sink %u21 : !quake.wire
        quake.sink %u31 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto operators = getOperators(*module, "same_operation");
  ASSERT_EQ(operators.size(), 8u);
  auto function = getFunction(*module, "same_operation");
  CommutationAnalysis analysis(function.front());
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::SameOperation);
  expectPair(analysis, operators[2], operators[3], CommutationStatus::Commutes,
             CommutationReason::SameOperation);
  expectPair(analysis, operators[4], operators[5], CommutationStatus::Commutes,
             CommutationReason::SameOperation);
  expectPair(analysis, operators[6], operators[7], CommutationStatus::Commutes,
             CommutationReason::SameOperation);
}

TEST_F(CommutationAnalysisTest, ComputationalDiagonal) {
  auto module = parseModule(R"mlir(
    module {
      func.func @diagonal() {
        %angle = arith.constant 5.0e-1 : f64
        %q = quake.null_wire
        %z = quake.z %q : (!quake.wire) -> !quake.wire
        %s = quake.s %z : (!quake.wire) -> !quake.wire
        %t = quake.t %s : (!quake.wire) -> !quake.wire
        %r1 = quake.r1 (%angle) %t : (f64, !quake.wire) -> !quake.wire
        %rz = quake.rz (%angle) %r1 : (f64, !quake.wire) -> !quake.wire
        quake.sink %rz : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto operators = getOperators(*module, "diagonal");
  ASSERT_EQ(operators.size(), 5u);
  auto function = getFunction(*module, "diagonal");
  CommutationAnalysis analysis(function.front());
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::ComputationalDiagonal);
  expectPair(analysis, operators[1], operators[2], CommutationStatus::Commutes,
             CommutationReason::ComputationalDiagonal);
  expectPair(analysis, operators[3], operators[4], CommutationStatus::Commutes,
             CommutationReason::ComputationalDiagonal);
}

TEST_F(CommutationAnalysisTest, SameAxis) {
  auto module = parseModule(R"mlir(
    module {
      func.func @same_axis() {
        %angle0 = arith.constant 5.0e-1 : f64
        %angle1 = arith.constant 1.0 : f64
        %phase = arith.constant 2.5e-1 : f64
        %other_phase = arith.constant 7.5e-1 : f64
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %q2 = quake.null_wire
        %x = quake.x %q0 : (!quake.wire) -> !quake.wire
        %rx = quake.rx (%angle0) %x : (f64, !quake.wire) -> !quake.wire
        %p0 = quake.phased_rx (%angle0, %phase) %q1 : (f64, f64, !quake.wire) -> !quake.wire
        %p1 = quake.phased_rx (%angle1, %phase) %p0 : (f64, f64, !quake.wire) -> !quake.wire
        %p2 = quake.phased_rx (%angle0, %other_phase) %p1 : (f64, f64, !quake.wire) -> !quake.wire
        %y = quake.y %q2 : (!quake.wire) -> !quake.wire
        %ry = quake.ry (%angle0) %y : (f64, !quake.wire) -> !quake.wire
        quake.sink %rx : !quake.wire
        quake.sink %p2 : !quake.wire
        quake.sink %ry : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto operators = getOperators(*module, "same_axis");
  ASSERT_EQ(operators.size(), 7u);
  auto function = getFunction(*module, "same_axis");
  CommutationAnalysis analysis(function.front());
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::SameAxis);
  expectPair(analysis, operators[2], operators[3], CommutationStatus::Commutes,
             CommutationReason::SameAxis);
  expectPair(analysis, operators[3], operators[4],
             CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);
  expectPair(analysis, operators[5], operators[6], CommutationStatus::Commutes,
             CommutationReason::SameAxis);
}

TEST_F(CommutationAnalysisTest, PauliParity) {
  auto module = parseModule(R"mlir(
    module {
      func.func @pauli_parity() {
        %angle = arith.constant 5.0e-1 : f64
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %q2 = quake.null_wire
        %q3 = quake.null_wire
        %xx:2 = quake.exp_pauli (%angle) %q0, %q1 to "XX" : (f64, !quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %zz:2 = quake.exp_pauli (%angle) %xx#0, %xx#1 to "ZZ" : (f64, !quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %x = quake.x %q2 : (!quake.wire) -> !quake.wire
        %z = quake.z %x : (!quake.wire) -> !quake.wire
        %exp_x = quake.exp_pauli (%angle) %q3 to "X" : (f64, !quake.wire) -> !quake.wire
        %exp_z = quake.z %exp_x : (!quake.wire) -> !quake.wire
        quake.sink %zz#0 : !quake.wire
        quake.sink %zz#1 : !quake.wire
        quake.sink %z : !quake.wire
        quake.sink %exp_z : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto operators = getOperators(*module, "pauli_parity");
  ASSERT_EQ(operators.size(), 6u);
  auto function = getFunction(*module, "pauli_parity");
  CommutationAnalysis analysis(function.front());
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::EvenPauliParity);
  expectPair(analysis, operators[2], operators[3],
             CommutationStatus::DoesNotCommute,
             CommutationReason::OddPauliParity);
  expectPair(analysis, operators[4], operators[5],
             CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);
}

TEST_F(CommutationAnalysisTest, DiagonalOnControls) {
  auto module = parseModule(R"mlir(
    module {
      func.func @diagonal_on_controls() {
        %control = quake.null_wire
        %target = quake.null_wire
        %z = quake.z %control : (!quake.wire) -> !quake.wire
        %cx:2 = quake.x [%z] %target : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %cx#0 : !quake.wire
        quake.sink %cx#1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto operators = getOperators(*module, "diagonal_on_controls");
  ASSERT_EQ(operators.size(), 2u);
  auto function = getFunction(*module, "diagonal_on_controls");
  CommutationAnalysis analysis(function.front());
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::DiagonalOnControls);
}

TEST_F(CommutationAnalysisTest, CompatibleControlledTargets) {
  auto module = parseModule(R"mlir(
    module {
      func.func @compatible_targets() {
        %angle = arith.constant 5.0e-1 : f64
        %control_wire = quake.null_wire
        %target = quake.null_wire
        %control = quake.to_ctrl %control_wire : (!quake.wire) -> !quake.control
        %cx = quake.x [%control] %target : (!quake.control, !quake.wire) -> !quake.wire
        %crx = quake.rx (%angle) [%control] %cx : (f64, !quake.control, !quake.wire) -> !quake.wire
        %cross_control = quake.null_wire
        %cross_target = quake.null_wire
        %cross_x:2 = quake.x [%cross_control] %cross_target : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %cross_z:2 = quake.z [%cross_x#1] %cross_x#0 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %crx : !quake.wire
        quake.sink %cross_z#0 : !quake.wire
        quake.sink %cross_z#1 : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto operators = getOperators(*module, "compatible_targets");
  ASSERT_EQ(operators.size(), 4u);
  auto function = getFunction(*module, "compatible_targets");
  CommutationAnalysis analysis(function.front());
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::CompatibleControlledTargets);
  expectPair(analysis, operators[2], operators[3],
             CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);
}

TEST_F(CommutationAnalysisTest, MutuallyExclusiveControls) {
  auto module = parseModule(R"mlir(
    module {
      func.func @exclusive_controls() {
        %control_wire = quake.null_wire
        %target = quake.null_wire
        %control = quake.to_ctrl %control_wire : (!quake.wire) -> !quake.control
        %x = quake.x [%control] %target : (!quake.control, !quake.wire) -> !quake.wire
        %y = quake.y [%control neg [true]] %x : (!quake.control, !quake.wire) -> !quake.wire
        quake.sink %y : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);
  auto operators = getOperators(*module, "exclusive_controls");
  ASSERT_EQ(operators.size(), 2u);
  auto function = getFunction(*module, "exclusive_controls");
  CommutationAnalysis analysis(function.front());
  expectPair(analysis, operators[0], operators[1], CommutationStatus::Commutes,
             CommutationReason::MutuallyExclusiveControls);
}

TEST_F(CommutationAnalysisTest, ConservativeOutcomes) {
  auto module = parseModule(R"mlir(
    module {
      func.func private @wire_source() -> !quake.wire
      func.func private @unitary_generator()
      func.func private @other_unitary_generator()
      func.func @conservative() {
        %q0 = quake.null_wire
        %x = quake.x %q0 : (!quake.wire) -> !quake.wire
        %h = quake.h %x : (!quake.wire) -> !quake.wire
        quake.sink %h : !quake.wire
        return
      }
      func.func @aggregate(%q: !quake.veq<2>) {
        quake.x %q : (!quake.veq<2>) -> ()
        return
      }
      func.func @duplicate_role() {
        %control_wire = quake.null_wire
        %target = quake.null_wire
        %control = quake.to_ctrl %control_wire : (!quake.wire) -> !quake.control
        %x = quake.x [%control, %control] %target : (!quake.control, !quake.control, !quake.wire) -> !quake.wire
        quake.sink %x : !quake.wire
        return
      }
      func.func @dynamic_pauli(%word: !cc.charspan) {
        %angle = arith.constant 5.0e-1 : f64
        %q = quake.null_wire
        %exp = quake.exp_pauli (%angle) %q to %word : (f64, !quake.wire, !cc.charspan) -> !quake.wire
        %z = quake.z %exp : (!quake.wire) -> !quake.wire
        quake.sink %z : !quake.wire
        return
      }
      func.func @call_result() {
        %q = call @wire_source() : () -> !quake.wire
        %x = quake.x %q : (!quake.wire) -> !quake.wire
        %z = quake.z %x : (!quake.wire) -> !quake.wire
        quake.sink %z : !quake.wire
        return
      }
      func.func @opaque_unitaries() {
        %angle0 = arith.constant 2.5e-1 : f64
        %angle1 = arith.constant 5.0e-1 : f64
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %q2 = quake.null_wire
        %u0 = quake.custom_unitary_call @unitary_generator %q0 : (!quake.wire) -> !quake.wire
        %u1 = quake.custom_unitary_call @unitary_generator %q1 : (!quake.wire) -> !quake.wire
        %u2 = quake.custom_unitary_call @unitary_generator<adj> %u0 : (!quake.wire) -> !quake.wire
        %u3 = quake.custom_unitary_call @other_unitary_generator %u2 : (!quake.wire) -> !quake.wire
        %u4 = quake.custom_unitary_call @unitary_generator(%angle0) %q2 : (f64, !quake.wire) -> !quake.wire
        %u5 = quake.custom_unitary_call @unitary_generator(%angle1) %u4 : (f64, !quake.wire) -> !quake.wire
        quake.sink %u1 : !quake.wire
        quake.sink %u3 : !quake.wire
        quake.sink %u5 : !quake.wire
        return
      }
      func.func @constant_unitaries() {
        %q = quake.null_wire
        %u0 = quake.custom_unitary_constant @unitary_matrix %q : (!quake.wire) -> !quake.wire
        %u1 = quake.custom_unitary_constant @unitary_matrix<adj> %u0 : (!quake.wire) -> !quake.wire
        quake.sink %u1 : !quake.wire
        return
      }
      cc.global constant private @unitary_matrix (dense<[(1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00)]> : tensor<4xcomplex<f64>>) : !cc.array<complex<f64> x 4>
    })mlir");
  ASSERT_TRUE(module);
  auto operators = getOperators(*module, "conservative");
  ASSERT_EQ(operators.size(), 2u);
  auto function = getFunction(*module, "conservative");
  auto *returnOp = function.front().getTerminator();
  CommutationAnalysis analysis(function.front());
  expectPair(analysis, nullptr, operators[0], CommutationStatus::Indeterminate,
             CommutationReason::NullOperation);
  expectPair(analysis, operators[0], returnOp, CommutationStatus::Indeterminate,
             CommutationReason::UnsupportedOperationKind);
  expectPair(analysis, operators[0], operators[1],
             CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);

  auto aggregate = getFunction(*module, "aggregate");
  auto aggregateOperators = getOperators(*module, "aggregate");
  ASSERT_EQ(aggregateOperators.size(), 1u);
  CommutationAnalysis aggregateAnalysis(aggregate.front());
  expectPair(aggregateAnalysis, aggregateOperators[0], aggregateOperators[0],
             CommutationStatus::Indeterminate,
             CommutationReason::UnsupportedQuantumOperandType);

  auto duplicate = getFunction(*module, "duplicate_role");
  auto duplicateOperators = getOperators(*module, "duplicate_role");
  ASSERT_EQ(duplicateOperators.size(), 1u);
  CommutationAnalysis duplicateAnalysis(duplicate.front());
  expectPair(duplicateAnalysis, duplicateOperators[0], duplicateOperators[0],
             CommutationStatus::Indeterminate,
             CommutationReason::DuplicateQubitOperand);

  auto dynamicPauli = getFunction(*module, "dynamic_pauli");
  auto dynamicOperators = getOperators(*module, "dynamic_pauli");
  ASSERT_EQ(dynamicOperators.size(), 2u);
  CommutationAnalysis dynamicAnalysis(dynamicPauli.front());
  expectPair(dynamicAnalysis, dynamicOperators[0], dynamicOperators[1],
             CommutationStatus::Indeterminate,
             CommutationReason::UnsupportedPauliWord);

  auto callResult = getFunction(*module, "call_result");
  auto callOperators = getOperators(*module, "call_result");
  ASSERT_EQ(callOperators.size(), 2u);
  CommutationAnalysis callAnalysis(callResult.front());
  expectPair(callAnalysis, callOperators[0], callOperators[1],
             CommutationStatus::Indeterminate,
             CommutationReason::UnmappedQubitId);

  auto opaque = getFunction(*module, "opaque_unitaries");
  auto opaqueOperators = getOperators(*module, "opaque_unitaries");
  ASSERT_EQ(opaqueOperators.size(), 6u);
  CommutationAnalysis opaqueAnalysis(opaque.front());
  expectPair(opaqueAnalysis, opaqueOperators[0], opaqueOperators[1],
             CommutationStatus::Commutes, CommutationReason::DisjointSupport);
  expectPair(opaqueAnalysis, opaqueOperators[0], opaqueOperators[2],
             CommutationStatus::Commutes, CommutationReason::SameOperation);
  expectPair(opaqueAnalysis, opaqueOperators[0], opaqueOperators[3],
             CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);
  expectPair(opaqueAnalysis, opaqueOperators[4], opaqueOperators[5],
             CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);

  auto constant = getFunction(*module, "constant_unitaries");
  auto constantOperators = getOperators(*module, "constant_unitaries");
  ASSERT_EQ(constantOperators.size(), 2u);
  CommutationAnalysis constantAnalysis(constant.front());
  expectPair(constantAnalysis, constantOperators[0], constantOperators[1],
             CommutationStatus::Commutes, CommutationReason::SameOperation);
}

TEST_F(CommutationAnalysisTest, BlockLocalQubitIds) {
  auto module = parseModule(R"mlir(
    module {
      func.func @other() {
        %q = quake.null_wire
        %z = quake.z %q : (!quake.wire) -> !quake.wire
        quake.sink %z : !quake.wire
        return
      }
      func.func @mixed_results() {
        %wire_control = quake.null_wire
        %reusable_wire = quake.null_wire
        %target = quake.null_wire
        %control = quake.to_ctrl %reusable_wire : (!quake.wire) -> !quake.control
        %mixed_x:2 = quake.x [%wire_control, %control] %target : (!quake.wire, !quake.control, !quake.wire) -> (!quake.wire, !quake.wire)
        %control_z = quake.z %mixed_x#0 : (!quake.wire) -> !quake.wire
        %target_x = quake.x %mixed_x#1 : (!quake.wire) -> !quake.wire
        quake.sink %control_z : !quake.wire
        quake.sink %target_x : !quake.wire
        return
      }
    })mlir");
  ASSERT_TRUE(module);

  auto mixed = getFunction(*module, "mixed_results");
  auto mixedOperators = getOperators(*module, "mixed_results");
  ASSERT_EQ(mixedOperators.size(), 3u);
  CommutationAnalysis mixedAnalysis(mixed.front());
  expectPair(mixedAnalysis, mixedOperators[0], mixedOperators[1],
             CommutationStatus::Commutes,
             CommutationReason::DiagonalOnControls);
  expectPair(mixedAnalysis, mixedOperators[0], mixedOperators[2],
             CommutationStatus::Commutes,
             CommutationReason::CompatibleControlledTargets);

  auto otherOperators = getOperators(*module, "other");
  ASSERT_EQ(otherOperators.size(), 1u);
  expectPair(mixedAnalysis, mixedOperators[0], otherOperators[0],
             CommutationStatus::Indeterminate,
             CommutationReason::DifferentBlocks);
}

TEST_F(CommutationAnalysisTest, RebuildAfterMutation) {
  auto function = createMutationKernel();
  Value q0 = createWire();
  Value q1 = createWire();
  auto x = createGate<cudaq::quake::XOp>(q0);
  auto h = createGate<cudaq::quake::HOp>(q1);
  finishFunction();
  {
    CommutationAnalysis analysis(function.front());
    EXPECT_TRUE(analysis.canCommute(x, h));
  }

  x.getWires().front().getUses().begin()->getOwner()->erase();
  h->setOperand(0, x.getWires().front());
  ASSERT_TRUE(succeeded(verify(*module)));
  CommutationAnalysis rebuilt(function.front());
  expectPair(rebuilt, x, h, CommutationStatus::Indeterminate,
             CommutationReason::NoApplicableRule);
}
