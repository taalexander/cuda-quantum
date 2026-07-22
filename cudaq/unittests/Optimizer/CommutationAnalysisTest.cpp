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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include <cmath>
#include <complex>
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
using Complex = std::complex<double>;

struct Matrix {
  std::size_t dimension;
  std::vector<Complex> elements;
};

static Matrix multiply(const Matrix &lhs, const Matrix &rhs) {
  EXPECT_EQ(lhs.dimension, rhs.dimension);
  Matrix product{lhs.dimension,
                 std::vector<Complex>(lhs.dimension * lhs.dimension)};
  for (std::size_t row = 0; row < lhs.dimension; ++row)
    for (std::size_t column = 0; column < lhs.dimension; ++column)
      for (std::size_t inner = 0; inner < lhs.dimension; ++inner)
        product.elements[row * lhs.dimension + column] +=
            lhs.elements[row * lhs.dimension + inner] *
            rhs.elements[inner * lhs.dimension + column];
  return product;
}

static Matrix tensorProduct(const Matrix &lhs, const Matrix &rhs) {
  Matrix product{lhs.dimension * rhs.dimension,
                 std::vector<Complex>(lhs.dimension * rhs.dimension *
                                      lhs.dimension * rhs.dimension)};
  for (std::size_t lhsRow = 0; lhsRow < lhs.dimension; ++lhsRow)
    for (std::size_t lhsColumn = 0; lhsColumn < lhs.dimension; ++lhsColumn)
      for (std::size_t rhsRow = 0; rhsRow < rhs.dimension; ++rhsRow)
        for (std::size_t rhsColumn = 0; rhsColumn < rhs.dimension;
             ++rhsColumn) {
          auto row = lhsRow * rhs.dimension + rhsRow;
          auto column = lhsColumn * rhs.dimension + rhsColumn;
          product.elements[row * product.dimension + column] =
              lhs.elements[lhsRow * lhs.dimension + lhsColumn] *
              rhs.elements[rhsRow * rhs.dimension + rhsColumn];
        }
  return product;
}

static Matrix controlled(const Matrix &target, bool negated) {
  Matrix result{4, std::vector<Complex>(16)};
  for (std::size_t control = 0; control < 2; ++control)
    for (std::size_t targetRow = 0; targetRow < 2; ++targetRow)
      for (std::size_t targetColumn = 0; targetColumn < 2; ++targetColumn) {
        auto row = 2 * control + targetRow;
        auto column = 2 * control + targetColumn;
        bool enabled = control != static_cast<std::size_t>(negated);
        result.elements[4 * row + column] =
            enabled ? target.elements[2 * targetRow + targetColumn]
                    : Complex(targetRow == targetColumn);
      }
  return result;
}

static bool commute(const Matrix &lhs, const Matrix &rhs) {
  auto lhsRhs = multiply(lhs, rhs);
  auto rhsLhs = multiply(rhs, lhs);
  for (std::size_t index = 0; index < lhsRhs.elements.size(); ++index)
    if (std::abs(lhsRhs.elements[index] - rhsLhs.elements[index]) > 1.0e-12)
      return false;
  return true;
}

class CommutationAnalysisTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<cf::ControlFlowDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<cudaq::cc::CCDialect>();
    context.loadDialect<cudaq::quake::QuakeDialect>();
  }

  OwningOpRef<ModuleOp> parse(llvm::StringRef source) {
    auto module = parseSourceString<ModuleOp>(source, &context);
    EXPECT_TRUE(module);
    return module;
  }

  static func::FuncOp getFunction(ModuleOp module, llvm::StringRef name) {
    return module.lookupSymbol<func::FuncOp>(name);
  }

  static llvm::SmallVector<Operation *> getOperators(Block &block) {
    llvm::SmallVector<Operation *> operators;
    for (Operation &operation : block)
      if (isa<cudaq::quake::OperatorInterface>(operation))
        operators.push_back(&operation);
    return operators;
  }

  static llvm::SmallVector<Operation *> getOperators(func::FuncOp function) {
    return getOperators(function.front());
  }

  static void expectSymmetric(CommutationAnalysis &analysis, Operation *lhs,
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

  MLIRContext context;
};
} // namespace

TEST(CommutationResultTest, ExposesStableReasonIdentifiers) {
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
      {CommutationReason::AmbiguousQuantumIdentity,
       "ambiguous-quantum-identity"},
      {CommutationReason::UnsupportedPauliWord, "unsupported-pauli-word"},
      {CommutationReason::NoApplicableRule, "no-applicable-rule"}};
  for (auto [reason, identifier] : cases)
    EXPECT_EQ(getCommutationReasonId(reason), identifier);
}

TEST(CommutationResultTest, ConvertsOnlyProvedCommutationToTrue) {
  EXPECT_TRUE(static_cast<bool>(CommutationResult{
      CommutationStatus::Commutes, CommutationReason::DisjointSupport}));
  EXPECT_FALSE(static_cast<bool>(CommutationResult{
      CommutationStatus::DoesNotCommute, CommutationReason::OddPauliParity}));
  EXPECT_FALSE(static_cast<bool>(CommutationResult{
      CommutationStatus::Indeterminate, CommutationReason::NoApplicableRule}));
}

TEST(CommutationResultTest, RuleProofsAgreeWithSmallMatrixOracle) {
  const Complex i(0.0, 1.0);
  const Matrix identity{2, {1.0, 0.0, 0.0, 1.0}};
  const Matrix x{2, {0.0, 1.0, 1.0, 0.0}};
  const Matrix y{2, {0.0, -i, i, 0.0}};
  const Matrix z{2, {1.0, 0.0, 0.0, -1.0}};
  const Matrix s{2, {1.0, 0.0, 0.0, i}};
  const double inverseSqrtTwo = 1.0 / std::sqrt(2.0);
  const Matrix h{
      2, {inverseSqrtTwo, inverseSqrtTwo, inverseSqrtTwo, -inverseSqrtTwo}};
  const double angle = 0.37;
  const Matrix rx{2,
                  {std::cos(angle), -i * std::sin(angle), -i * std::sin(angle),
                   std::cos(angle)}};

  struct OracleCase {
    const char *rule;
    Matrix lhs;
    Matrix rhs;
    bool expected;
  };
  std::vector<OracleCase> cases{
      {"disjoint support", tensorProduct(x, identity),
       tensorProduct(identity, h), true},
      {"same operation", h, h, true},
      {"computational diagonal", z, s, true},
      {"same axis", x, rx, true},
      {"even Pauli parity", tensorProduct(x, x), tensorProduct(z, z), true},
      {"odd Pauli parity", x, z, false},
      {"diagonal on controls", tensorProduct(z, identity), controlled(x, false),
       true},
      {"compatible controlled targets", controlled(x, false),
       controlled(rx, false), true},
      {"mutually exclusive controls", controlled(x, false), controlled(y, true),
       true}};

  for (const auto &testCase : cases)
    EXPECT_EQ(commute(testCase.lhs, testCase.rhs), testCase.expected)
        << testCase.rule;
}

TEST_F(CommutationAnalysisTest, QueriesDirectQuakeOperationsSymmetrically) {
  auto module = parse(R"mlir(
    module {
      func.func @kernel() {
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %c1 = arith.constant 1.0 : f64
        %x0 = quake.x %q0 : (!quake.wire) -> !quake.wire
        %h1 = quake.h %q1 : (!quake.wire) -> !quake.wire
        %rx0 = quake.rx (%c1) %x0 : (f64, !quake.wire) -> !quake.wire
        quake.sink %rx0 : !quake.wire
        quake.sink %h1 : !quake.wire
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "kernel");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 3u);

  CommutationAnalysis analysis(function.front());
  expectSymmetric(analysis, operators[0], operators[1],
                  CommutationStatus::Commutes,
                  CommutationReason::DisjointSupport);
  expectSymmetric(analysis, operators[0], operators[2],
                  CommutationStatus::Commutes, CommutationReason::SameAxis);
}

TEST_F(CommutationAnalysisTest, PropagatesMixedWireResultsInQuakeOrder) {
  auto module = parse(R"mlir(
    module {
      func.func @kernel() {
        %wire_control = quake.null_wire
        %reusable_wire = quake.null_wire
        %target = quake.null_wire
        %reusable = quake.to_ctrl %reusable_wire : (!quake.wire) -> !quake.control
        %next:2 = quake.x [%wire_control, %reusable] %target :
            (!quake.wire, !quake.control, !quake.wire) ->
            (!quake.wire, !quake.wire)
        %z_control = quake.z %next#0 : (!quake.wire) -> !quake.wire
        %x_target = quake.x %next#1 : (!quake.wire) -> !quake.wire
        %returned = quake.from_ctrl %reusable : (!quake.control) -> !quake.wire
        quake.sink %z_control : !quake.wire
        quake.sink %x_target : !quake.wire
        quake.sink %returned : !quake.wire
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "kernel");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 3u);

  CommutationAnalysis analysis(function.front());
  expectSymmetric(analysis, operators[0], operators[1],
                  CommutationStatus::Commutes,
                  CommutationReason::DiagonalOnControls);
  expectSymmetric(analysis, operators[0], operators[2],
                  CommutationStatus::Commutes,
                  CommutationReason::CompatibleControlledTargets);
}

TEST_F(CommutationAnalysisTest, ClassifiesInvalidQueryInputs) {
  auto module = parse(R"mlir(
    module {
      func.func @first() {
        %q = quake.null_wire
        %x = quake.x %q : (!quake.wire) -> !quake.wire
        quake.sink %x : !quake.wire
        return
      }
      func.func @second() {
        %q = quake.null_wire
        %z = quake.z %q : (!quake.wire) -> !quake.wire
        quake.sink %z : !quake.wire
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto first = getFunction(*module, "first");
  auto second = getFunction(*module, "second");
  Operation *x = getOperators(first).front();
  Operation *z = getOperators(second).front();
  CommutationAnalysis analysis(first.front());

  expectSymmetric(analysis, nullptr, x, CommutationStatus::Indeterminate,
                  CommutationReason::NullOperation);
  expectSymmetric(analysis, x, z, CommutationStatus::Indeterminate,
                  CommutationReason::DifferentBlocks);
  expectSymmetric(analysis, x, first.front().getTerminator(),
                  CommutationStatus::Indeterminate,
                  CommutationReason::UnsupportedOperationKind);
}

TEST_F(CommutationAnalysisTest, MatchesExactOperationsAndUnorderedSwapTargets) {
  auto module = parse(R"mlir(
    module {
      func.func @kernel(%theta : f64) {
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %q2 = quake.null_wire
        %one0 = arith.constant 1.0 : f64
        %one1 = arith.constant 1.0 : f64
        %rx0 = quake.rx (%theta) %q0 : (f64, !quake.wire) -> !quake.wire
        %rx1 = quake.rx (%theta) %rx0 : (f64, !quake.wire) -> !quake.wire
        %rx2 = quake.rx (%one0) %rx1 : (f64, !quake.wire) -> !quake.wire
        %rx3 = quake.rx (%one1) %rx2 : (f64, !quake.wire) -> !quake.wire
        %swap0:2 = quake.swap %rx3, %q1 :
            (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %swap1:2 = quake.swap %swap0#1, %swap0#0 :
            (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        %swap2:2 = quake.swap %swap1#0, %q2 :
            (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %swap2#0 : !quake.wire
        quake.sink %swap2#1 : !quake.wire
        quake.sink %swap1#1 : !quake.wire
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "kernel");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 7u);
  CommutationAnalysis analysis(function.front());

  expectSymmetric(analysis, operators[0], operators[1],
                  CommutationStatus::Commutes,
                  CommutationReason::SameOperation);
  expectSymmetric(analysis, operators[2], operators[3],
                  CommutationStatus::Commutes,
                  CommutationReason::SameOperation);
  expectSymmetric(analysis, operators[4], operators[5],
                  CommutationStatus::Commutes,
                  CommutationReason::SameOperation);
  expectSymmetric(analysis, operators[5], operators[6],
                  CommutationStatus::Indeterminate,
                  CommutationReason::NoApplicableRule);
}

TEST_F(CommutationAnalysisTest, MatchesExactGeneralOperationsAndPhasedRxAxis) {
  auto module = parse(R"mlir(
    module {
      func.func @kernel(%dynamic : f64) {
        %q = quake.null_wire
        %zero = arith.constant 0.0 : f64
        %negative_zero = arith.constant -0.0 : f64
        %one0 = arith.constant 1.0 : f64
        %one1 = arith.constant 1.0 : f64
        %two = arith.constant 2.0 : f64
        %h0 = quake.h %q : (!quake.wire) -> !quake.wire
        %h1 = quake.h %h0 : (!quake.wire) -> !quake.wire
        %u20 = quake.u2 (%zero, %one0) %h1 :
            (f64, f64, !quake.wire) -> !quake.wire
        %u21 = quake.u2<adj> (%zero, %one1) %u20 :
            (f64, f64, !quake.wire) -> !quake.wire
        %u22 = quake.u2 (%zero, %one0) %u21 :
            (f64, f64, !quake.wire) -> !quake.wire
        %u23 = quake.u2 (%negative_zero, %one1) %u22 :
            (f64, f64, !quake.wire) -> !quake.wire
        %u30 = quake.u3 (%zero, %one0, %two) %u23 :
            (f64, f64, f64, !quake.wire) -> !quake.wire
        %u31 = quake.u3 (%zero, %one1, %two) %u30 :
            (f64, f64, f64, !quake.wire) -> !quake.wire
        %u32 = quake.u3 (%zero, %dynamic, %two) %u31 :
            (f64, f64, f64, !quake.wire) -> !quake.wire
        %p0 = quake.phased_rx (%zero, %one0) %u32 :
            (f64, f64, !quake.wire) -> !quake.wire
        %p1 = quake.phased_rx (%two, %one1) %p0 :
            (f64, f64, !quake.wire) -> !quake.wire
        %p2 = quake.phased_rx (%zero, %two) %p1 :
            (f64, f64, !quake.wire) -> !quake.wire
        quake.sink %p2 : !quake.wire
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "kernel");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 12u);
  CommutationAnalysis analysis(function.front());

  expectSymmetric(analysis, operators[0], operators[1],
                  CommutationStatus::Commutes,
                  CommutationReason::SameOperation);
  expectSymmetric(analysis, operators[2], operators[3],
                  CommutationStatus::Commutes,
                  CommutationReason::SameOperation);
  expectSymmetric(analysis, operators[4], operators[5],
                  CommutationStatus::Indeterminate,
                  CommutationReason::NoApplicableRule);
  expectSymmetric(analysis, operators[6], operators[7],
                  CommutationStatus::Commutes,
                  CommutationReason::SameOperation);
  expectSymmetric(analysis, operators[7], operators[8],
                  CommutationStatus::Indeterminate,
                  CommutationReason::NoApplicableRule);
  expectSymmetric(analysis, operators[9], operators[10],
                  CommutationStatus::Commutes, CommutationReason::SameAxis);
  expectSymmetric(analysis, operators[10], operators[11],
                  CommutationStatus::Indeterminate,
                  CommutationReason::NoApplicableRule);
}

TEST_F(CommutationAnalysisTest, AppliesDiagonalAxisAndFixedPauliRules) {
  auto module = parse(R"mlir(
    module {
      func.func @kernel() {
        %q = quake.null_wire
        %one = arith.constant 1.0 : f64
        %x = quake.x %q : (!quake.wire) -> !quake.wire
        %rx = quake.rx (%one) %x : (f64, !quake.wire) -> !quake.wire
        %y = quake.y %rx : (!quake.wire) -> !quake.wire
        %ry = quake.ry (%one) %y : (f64, !quake.wire) -> !quake.wire
        %z = quake.z %ry : (!quake.wire) -> !quake.wire
        %s = quake.s %z : (!quake.wire) -> !quake.wire
        %t = quake.t %s : (!quake.wire) -> !quake.wire
        %r1 = quake.r1 (%one) %t : (f64, !quake.wire) -> !quake.wire
        %rz = quake.rz (%one) %r1 : (f64, !quake.wire) -> !quake.wire
        %h = quake.h %rz : (!quake.wire) -> !quake.wire
        quake.sink %h : !quake.wire
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "kernel");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 10u);
  CommutationAnalysis analysis(function.front());

  expectSymmetric(analysis, operators[0], operators[1],
                  CommutationStatus::Commutes, CommutationReason::SameAxis);
  expectSymmetric(analysis, operators[0], operators[2],
                  CommutationStatus::DoesNotCommute,
                  CommutationReason::OddPauliParity);
  expectSymmetric(analysis, operators[2], operators[3],
                  CommutationStatus::Commutes, CommutationReason::SameAxis);
  expectSymmetric(analysis, operators[4], operators[5],
                  CommutationStatus::Commutes,
                  CommutationReason::ComputationalDiagonal);
  expectSymmetric(analysis, operators[6], operators[7],
                  CommutationStatus::Commutes,
                  CommutationReason::ComputationalDiagonal);
  expectSymmetric(analysis, operators[7], operators[8],
                  CommutationStatus::Commutes,
                  CommutationReason::ComputationalDiagonal);
  expectSymmetric(analysis, operators[0], operators[9],
                  CommutationStatus::Indeterminate,
                  CommutationReason::NoApplicableRule);
}

TEST_F(CommutationAnalysisTest, AppliesLiteralPauliParityConservatively) {
  auto module = parse(R"mlir(
    module {
      func.func @even() {
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %theta = arith.constant 0.5 : f64
        %xx:2 = quake.exp_pauli (%theta) %q0, %q1 to "XX" :
            (f64, !quake.wire, !quake.wire) ->
            (!quake.wire, !quake.wire)
        %zz:2 = quake.exp_pauli (%theta) %xx#0, %xx#1 to "ZZ" :
            (f64, !quake.wire, !quake.wire) ->
            (!quake.wire, !quake.wire)
        quake.sink %zz#0 : !quake.wire
        quake.sink %zz#1 : !quake.wire
        return
      }
      func.func @odd() {
        %q = quake.null_wire
        %theta = arith.constant 0.5 : f64
        %exp = quake.exp_pauli (%theta) %q to "X" :
            (f64, !quake.wire) -> !quake.wire
        %z = quake.z %exp : (!quake.wire) -> !quake.wire
        quake.sink %z : !quake.wire
        return
      }
      func.func @placement() {
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %q2 = quake.null_wire
        %theta = arith.constant 0.5 : f64
        %xiz:3 = quake.exp_pauli (%theta) %q0, %q1, %q2 to "XIZ" :
            (f64, !quake.wire, !quake.wire, !quake.wire) ->
            (!quake.wire, !quake.wire, !quake.wire)
        %ziy:3 = quake.exp_pauli (%theta) %xiz#0, %xiz#1, %xiz#2 to "ZIY" :
            (f64, !quake.wire, !quake.wire, !quake.wire) ->
            (!quake.wire, !quake.wire, !quake.wire)
        quake.sink %ziy#0 : !quake.wire
        quake.sink %ziy#1 : !quake.wire
        quake.sink %ziy#2 : !quake.wire
        return
      }
      func.func @four_qubit() {
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %q2 = quake.null_wire
        %q3 = quake.null_wire
        %theta = arith.constant 0.5 : f64
        %xizy:4 = quake.exp_pauli (%theta) %q0, %q1, %q2, %q3 to "XIZY" :
            (f64, !quake.wire, !quake.wire, !quake.wire, !quake.wire) ->
            (!quake.wire, !quake.wire, !quake.wire, !quake.wire)
        %ziyx:4 = quake.exp_pauli (%theta) %xizy#0, %xizy#1, %xizy#2,
            %xizy#3 to "ZIYX" :
            (f64, !quake.wire, !quake.wire, !quake.wire, !quake.wire) ->
            (!quake.wire, !quake.wire, !quake.wire, !quake.wire)
        quake.sink %ziyx#0 : !quake.wire
        quake.sink %ziyx#1 : !quake.wire
        quake.sink %ziyx#2 : !quake.wire
        quake.sink %ziyx#3 : !quake.wire
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);

  auto even = getFunction(*module, "even");
  auto evenOperators = getOperators(even);
  CommutationAnalysis evenAnalysis(even.front());
  expectSymmetric(evenAnalysis, evenOperators[0], evenOperators[1],
                  CommutationStatus::Commutes,
                  CommutationReason::EvenPauliParity);

  auto odd = getFunction(*module, "odd");
  auto oddOperators = getOperators(odd);
  CommutationAnalysis oddAnalysis(odd.front());
  expectSymmetric(oddAnalysis, oddOperators[0], oddOperators[1],
                  CommutationStatus::Indeterminate,
                  CommutationReason::NoApplicableRule);

  auto placement = getFunction(*module, "placement");
  auto placementOperators = getOperators(placement);
  CommutationAnalysis placementAnalysis(placement.front());
  expectSymmetric(placementAnalysis, placementOperators[0],
                  placementOperators[1], CommutationStatus::Commutes,
                  CommutationReason::EvenPauliParity);

  auto fourQubit = getFunction(*module, "four_qubit");
  auto fourQubitOperators = getOperators(fourQubit);
  CommutationAnalysis fourQubitAnalysis(fourQubit.front());
  expectSymmetric(fourQubitAnalysis, fourQubitOperators[0],
                  fourQubitOperators[1], CommutationStatus::Indeterminate,
                  CommutationReason::NoApplicableRule);
}

TEST_F(CommutationAnalysisTest, AppliesControlledOperationRules) {
  auto module = parse(R"mlir(
    module {
      func.func @kernel() {
        %control_wire = quake.null_wire
        %target = quake.null_wire
        %theta = arith.constant 0.5 : f64
        %z_control = quake.z %control_wire : (!quake.wire) -> !quake.wire
        %x_control = quake.x %z_control : (!quake.wire) -> !quake.wire
        %control = quake.to_ctrl %x_control :
            (!quake.wire) -> !quake.control
        %rx_target = quake.rx (%theta) %target :
            (f64, !quake.wire) -> !quake.wire
        %cx0 = quake.x [%control] %rx_target :
            (!quake.control, !quake.wire) -> !quake.wire
        %cy_same = quake.y [%control] %cx0 :
            (!quake.control, !quake.wire) -> !quake.wire
        %cy_neg = quake.y [%control neg [true]] %cy_same :
            (!quake.control, !quake.wire) -> !quake.wire
        %returned = quake.from_ctrl %control :
            (!quake.control) -> !quake.wire
        quake.sink %returned : !quake.wire
        quake.sink %cy_neg : !quake.wire
        return
      }
      func.func @multi_control() {
        %c0_wire = quake.null_wire
        %c1_wire = quake.null_wire
        %target = quake.null_wire
        %theta = arith.constant 0.5 : f64
        %z_c1 = quake.z %c1_wire : (!quake.wire) -> !quake.wire
        %c0 = quake.to_ctrl %c0_wire : (!quake.wire) -> !quake.control
        %c1 = quake.to_ctrl %z_c1 : (!quake.wire) -> !quake.control
        %x = quake.x [%c0, %c1 neg [false, false]] %target :
            (!quake.control, !quake.control, !quake.wire) -> !quake.wire
        %rx = quake.rx (%theta) [%c0, %c1 neg [false, false]] %x :
            (f64, !quake.control, !quake.control, !quake.wire) -> !quake.wire
        %y = quake.y [%c0, %c1 neg [false, true]] %rx :
            (!quake.control, !quake.control, !quake.wire) -> !quake.wire
        %z = quake.z [%c0] %y :
            (!quake.control, !quake.wire) -> !quake.wire
        %rz = quake.rz (%theta) [%c0, %c1 neg [true, false]] %z :
            (f64, !quake.control, !quake.control, !quake.wire) -> !quake.wire
        %c0_result = quake.from_ctrl %c0 : (!quake.control) -> !quake.wire
        %c1_result = quake.from_ctrl %c1 : (!quake.control) -> !quake.wire
        quake.sink %c0_result : !quake.wire
        quake.sink %c1_result : !quake.wire
        quake.sink %rz : !quake.wire
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "kernel");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 6u);
  CommutationAnalysis analysis(function.front());

  expectSymmetric(analysis, operators[0], operators[3],
                  CommutationStatus::Commutes,
                  CommutationReason::DiagonalOnControls);
  expectSymmetric(analysis, operators[1], operators[3],
                  CommutationStatus::Indeterminate,
                  CommutationReason::NoApplicableRule);
  expectSymmetric(analysis, operators[2], operators[3],
                  CommutationStatus::Commutes,
                  CommutationReason::CompatibleControlledTargets);
  expectSymmetric(analysis, operators[3], operators[4],
                  CommutationStatus::Indeterminate,
                  CommutationReason::NoApplicableRule);
  expectSymmetric(analysis, operators[3], operators[5],
                  CommutationStatus::Commutes,
                  CommutationReason::MutuallyExclusiveControls);
  expectSymmetric(analysis, operators[0], operators[5],
                  CommutationStatus::Commutes,
                  CommutationReason::DiagonalOnControls);

  auto multiControl = getFunction(*module, "multi_control");
  auto multiOperators = getOperators(multiControl);
  CommutationAnalysis multiAnalysis(multiControl.front());
  expectSymmetric(multiAnalysis, multiOperators[0], multiOperators[1],
                  CommutationStatus::Commutes,
                  CommutationReason::DiagonalOnControls);
  expectSymmetric(multiAnalysis, multiOperators[1], multiOperators[2],
                  CommutationStatus::Commutes,
                  CommutationReason::CompatibleControlledTargets);
  expectSymmetric(multiAnalysis, multiOperators[1], multiOperators[3],
                  CommutationStatus::Commutes,
                  CommutationReason::MutuallyExclusiveControls);
  expectSymmetric(multiAnalysis, multiOperators[4], multiOperators[5],
                  CommutationStatus::Commutes,
                  CommutationReason::ComputationalDiagonal);
}

TEST_F(CommutationAnalysisTest, LimitsOpaqueAndUnsupportedSharedSupport) {
  auto module = parse(R"mlir(
    module {
      func.func private @generator()
      func.func @kernel(%ref : !quake.ref) {
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %custom = quake.custom_unitary_call @generator %q0 :
            (!quake.wire) -> !quake.wire
        %x = quake.x %custom : (!quake.wire) -> !quake.wire
        %h = quake.h %q1 : (!quake.wire) -> !quake.wire
        quake.x %ref : (!quake.ref) -> ()
        quake.sink %x : !quake.wire
        quake.sink %h : !quake.wire
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "kernel");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 4u);
  CommutationAnalysis analysis(function.front());

  expectSymmetric(analysis, operators[0], operators[1],
                  CommutationStatus::Indeterminate,
                  CommutationReason::NoApplicableRule);
  expectSymmetric(analysis, operators[0], operators[0],
                  CommutationStatus::Indeterminate,
                  CommutationReason::NoApplicableRule);
  expectSymmetric(analysis, operators[0], operators[2],
                  CommutationStatus::Commutes,
                  CommutationReason::DisjointSupport);
  expectSymmetric(analysis, operators[1], operators[3],
                  CommutationStatus::Indeterminate,
                  CommutationReason::UnsupportedQuantumOperandType);
}

TEST_F(CommutationAnalysisTest, TracksIdentitySourcesConversionsAndBoundaries) {
  auto module = parse(R"mlir(
    module {
      func.func private @wire_source() -> !quake.wire
      quake.wire_set @wires[4]
      func.func @borrowed() {
        %q0a = quake.borrow_wire @wires[0] : !quake.wire
        %q0b = quake.borrow_wire @wires[0] : !quake.wire
        %q1 = quake.borrow_wire @wires[1] : !quake.wire
        %x0a = quake.x %q0a : (!quake.wire) -> !quake.wire
        %x0b = quake.x %q0b : (!quake.wire) -> !quake.wire
        %h1 = quake.h %q1 : (!quake.wire) -> !quake.wire
        quake.return_wire %x0a : !quake.wire
        quake.return_wire %x0b : !quake.wire
        quake.return_wire %h1 : !quake.wire
        return
      }
      func.func @opaque(%control : !quake.control) {
        %target = quake.null_wire
        %cx = quake.x [%control] %target :
            (!quake.control, !quake.wire) -> !quake.wire
        %z = quake.z %cx : (!quake.wire) -> !quake.wire
        quake.sink %z : !quake.wire
        return
      }
      func.func @arguments(%q0 : !quake.wire, %q1 : !quake.wire) {
        %x = quake.x %q0 : (!quake.wire) -> !quake.wire
        %h = quake.h %q1 : (!quake.wire) -> !quake.wire
        quake.sink %x : !quake.wire
        quake.sink %h : !quake.wire
        return
      }
      func.func @round_trip() {
        %q = quake.null_wire
        %theta = arith.constant 0.5 : f64
        %x = quake.x %q : (!quake.wire) -> !quake.wire
        %control = quake.to_ctrl %x : (!quake.wire) -> !quake.control
        %returned = quake.from_ctrl %control : (!quake.control) -> !quake.wire
        %rx = quake.rx (%theta) %returned :
            (f64, !quake.wire) -> !quake.wire
        quake.sink %rx : !quake.wire
        return
      }
      func.func @call_result() {
        %q = func.call @wire_source() : () -> !quake.wire
        %x = quake.x %q : (!quake.wire) -> !quake.wire
        %z = quake.z %x : (!quake.wire) -> !quake.wire
        quake.sink %z : !quake.wire
        return
      }
      func.func @blocks() {
        %q = quake.null_wire
        %x = quake.x %q : (!quake.wire) -> !quake.wire
        cf.br ^next(%x : !quake.wire)
      ^next(%arg : !quake.wire):
        %z = quake.z %arg : (!quake.wire) -> !quake.wire
        quake.sink %z : !quake.wire
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);

  auto borrowed = getFunction(*module, "borrowed");
  auto borrowedOperators = getOperators(borrowed);
  CommutationAnalysis borrowedAnalysis(borrowed.front());
  expectSymmetric(borrowedAnalysis, borrowedOperators[0], borrowedOperators[1],
                  CommutationStatus::Commutes,
                  CommutationReason::SameOperation);
  expectSymmetric(borrowedAnalysis, borrowedOperators[0], borrowedOperators[2],
                  CommutationStatus::Commutes,
                  CommutationReason::DisjointSupport);

  auto opaque = getFunction(*module, "opaque");
  auto opaqueOperators = getOperators(opaque);
  CommutationAnalysis opaqueAnalysis(opaque.front());
  expectSymmetric(opaqueAnalysis, opaqueOperators[0], opaqueOperators[1],
                  CommutationStatus::Indeterminate,
                  CommutationReason::AmbiguousQuantumIdentity);

  auto arguments = getFunction(*module, "arguments");
  auto argumentOperators = getOperators(arguments);
  CommutationAnalysis argumentAnalysis(arguments.front());
  expectSymmetric(argumentAnalysis, argumentOperators[0], argumentOperators[1],
                  CommutationStatus::Commutes,
                  CommutationReason::DisjointSupport);

  auto roundTrip = getFunction(*module, "round_trip");
  auto roundTripOperators = getOperators(roundTrip);
  CommutationAnalysis roundTripAnalysis(roundTrip.front());
  expectSymmetric(roundTripAnalysis, roundTripOperators[0],
                  roundTripOperators[1], CommutationStatus::Commutes,
                  CommutationReason::SameAxis);

  auto callResult = getFunction(*module, "call_result");
  auto callOperators = getOperators(callResult);
  CommutationAnalysis callAnalysis(callResult.front());
  expectSymmetric(callAnalysis, callOperators[0], callOperators[1],
                  CommutationStatus::Indeterminate,
                  CommutationReason::AmbiguousQuantumIdentity);

  auto blocks = getFunction(*module, "blocks");
  auto entryOperators = getOperators(blocks.front());
  auto exitOperators = getOperators(blocks.getBody().back());
  CommutationAnalysis blockAnalysis(blocks.front());
  expectSymmetric(blockAnalysis, entryOperators[0], exitOperators[0],
                  CommutationStatus::Indeterminate,
                  CommutationReason::DifferentBlocks);
}

TEST_F(CommutationAnalysisTest, RejectsMalformedOperatorMetadata) {
  auto module = parse(R"mlir(
    module {
      func.func @kernel() {
        %control_wire = quake.null_wire
        %target = quake.null_wire
        %control = quake.to_ctrl %control_wire :
            (!quake.wire) -> !quake.control
        %bad_control = quake.x [%control neg [true, false]] %target :
            (!quake.control, !quake.wire) -> !quake.wire
        %theta = arith.constant 0.5 : f64
        %bad_pauli = quake.exp_pauli (%theta) %bad_control to "XX" :
            (f64, !quake.wire) -> !quake.wire
        %z = quake.z %bad_pauli : (!quake.wire) -> !quake.wire
        %returned = quake.from_ctrl %control :
            (!quake.control) -> !quake.wire
        quake.sink %returned : !quake.wire
        quake.sink %z : !quake.wire
        return
      }
      func.func @dynamic_pauli(%word : !cc.charspan) {
        %q = quake.null_wire
        %theta = arith.constant 0.5 : f64
        %exp = quake.exp_pauli (%theta) %q to %word :
            (f64, !quake.wire, !cc.charspan) -> !quake.wire
        %z = quake.z %exp : (!quake.wire) -> !quake.wire
        quake.sink %z : !quake.wire
        return
      }
      func.func @aggregate(%targets : !quake.veq<2>) {
        quake.x %targets : (!quake.veq<2>) -> ()
        return
      }
      func.func @duplicate_role() {
        %q = quake.null_wire
        %bad:2 = quake.x [%q] %q :
            (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
        quake.sink %bad#0 : !quake.wire
        quake.sink %bad#1 : !quake.wire
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "kernel");
  auto operators = getOperators(function);
  ASSERT_EQ(operators.size(), 3u);
  CommutationAnalysis analysis(function.front());

  expectSymmetric(analysis, operators[0], operators[2],
                  CommutationStatus::Indeterminate,
                  CommutationReason::MalformedControlPolarity);
  expectSymmetric(analysis, operators[1], operators[2],
                  CommutationStatus::Indeterminate,
                  CommutationReason::UnsupportedPauliWord);

  auto dynamicPauli = getFunction(*module, "dynamic_pauli");
  auto dynamicOperators = getOperators(dynamicPauli);
  CommutationAnalysis dynamicAnalysis(dynamicPauli.front());
  expectSymmetric(dynamicAnalysis, dynamicOperators[0], dynamicOperators[1],
                  CommutationStatus::Indeterminate,
                  CommutationReason::UnsupportedPauliWord);

  auto aggregate = getFunction(*module, "aggregate");
  auto aggregateOperator = getOperators(aggregate).front();
  CommutationAnalysis aggregateAnalysis(aggregate.front());
  expectSymmetric(aggregateAnalysis, aggregateOperator, aggregateOperator,
                  CommutationStatus::Indeterminate,
                  CommutationReason::UnsupportedQuantumOperandType);

  auto duplicateRole = getFunction(*module, "duplicate_role");
  auto duplicateOperator = getOperators(duplicateRole).front();
  CommutationAnalysis duplicateAnalysis(duplicateRole.front());
  expectSymmetric(duplicateAnalysis, duplicateOperator, duplicateOperator,
                  CommutationStatus::Indeterminate,
                  CommutationReason::AmbiguousQuantumIdentity);
}

TEST_F(CommutationAnalysisTest, CallerRebuildsAnalysisAfterBlockMutation) {
  auto module = parse(R"mlir(
    module {
      func.func @kernel() {
        %q0 = quake.null_wire
        %q1 = quake.null_wire
        %x = quake.x %q0 : (!quake.wire) -> !quake.wire
        %h = quake.h %q1 : (!quake.wire) -> !quake.wire
        quake.sink %x : !quake.wire
        quake.sink %h : !quake.wire
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  auto function = getFunction(*module, "kernel");
  auto operators = getOperators(function);
  {
    CommutationAnalysis analysis(function.front());
    EXPECT_TRUE(analysis.canCommute(operators[0], operators[1]));
  }

  auto x = cast<cudaq::quake::XOp>(operators[0]);
  operators[1]->setOperand(0, x.getTarget());
  CommutationAnalysis rebuilt(function.front());
  auto result = rebuilt.getResult(operators[0], operators[1]);
  EXPECT_EQ(result.status, CommutationStatus::Indeterminate);
  EXPECT_EQ(result.reason, CommutationReason::NoApplicableRule);
}
