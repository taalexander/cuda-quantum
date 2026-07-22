/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

// clang-format off
#include "QubitIdentityAnalysis.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
// clang-format on

using namespace mlir;

using cudaq::quake::detail::QubitIdentityAnalysis;

TEST(QubitIdentityAnalysisTest, ScalarValueFormsAndLimitations) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<cudaq::cc::CCDialect>();
  context.loadDialect<cudaq::quake::QuakeDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto module = OwningOpRef<ModuleOp>(ModuleOp::create(loc));
  auto wireType = cudaq::quake::WireType::get(&context);
  auto controlType = cudaq::quake::ControlType::get(&context);

  builder.setInsertionPointToEnd(module->getBody());
  cudaq::quake::WireSetOp::create(builder, loc, "wires", 2, ElementsAttr{});
  auto wireSource = func::FuncOp::create(
      builder, loc, "wire_source",
      builder.getFunctionType(TypeRange{}, TypeRange{wireType}));
  wireSource.setPrivate();
  auto function = func::FuncOp::create(
      builder, loc, "identity",
      builder.getFunctionType(
          TypeRange{wireType, controlType,
                    cudaq::quake::VeqType::get(&context, 2)},
          TypeRange{}));
  function.addEntryBlock();
  builder.setInsertionPointToStart(&function.front());

  Value initial = cudaq::quake::NullWireOp::create(builder, loc, wireType);
  auto x = cudaq::quake::XOp::create(builder, loc, TypeRange{wireType},
                                     UnitAttr{}, ValueRange{}, ValueRange{},
                                     ValueRange{initial}, DenseBoolArrayAttr{});
  auto reset = cudaq::quake::ResetOp::create(builder, loc, TypeRange{wireType},
                                             x.getWires().front());
  Value distinct = cudaq::quake::NullWireOp::create(builder, loc, wireType);
  auto measurement = cudaq::quake::MzOp::create(
      builder, loc,
      TypeRange{
          cudaq::cc::StdvecType::get(cudaq::quake::MeasureType::get(&context)),
          wireType, wireType},
      ValueRange{reset.getWires().front(), distinct}, StringAttr{});
  Value control = cudaq::quake::ToControlOp::create(
      builder, loc, controlType, measurement.getWires().front());
  Value returned =
      cudaq::quake::FromControlOp::create(builder, loc, wireType, control);

  auto borrow0a =
      cudaq::quake::BorrowWireOp::create(builder, loc, wireType, "wires", 0);
  cudaq::quake::ReturnWireOp::create(builder, loc, borrow0a.getResult());
  auto borrow0b =
      cudaq::quake::BorrowWireOp::create(builder, loc, wireType, "wires", 0);
  cudaq::quake::ReturnWireOp::create(builder, loc, borrow0b.getResult());
  auto borrow1 =
      cudaq::quake::BorrowWireOp::create(builder, loc, wireType, "wires", 1);
  cudaq::quake::ReturnWireOp::create(builder, loc, borrow1.getResult());
  auto call = func::CallOp::create(builder, loc, "wire_source",
                                   TypeRange{wireType}, ValueRange{});
  Value reference = cudaq::quake::AllocaOp::create(builder, loc);
  Value unwrapped =
      cudaq::quake::UnwrapOp::create(builder, loc, wireType, reference);
  cudaq::quake::SinkOp::create(builder, loc, TypeRange{}, returned);
  cudaq::quake::SinkOp::create(builder, loc, TypeRange{},
                               measurement.getWires()[1]);
  cudaq::quake::SinkOp::create(builder, loc, TypeRange{}, call.getResult(0));
  cudaq::quake::SinkOp::create(builder, loc, TypeRange{},
                               function.getArgument(0));
  Value controlArgumentWire = cudaq::quake::FromControlOp::create(
      builder, loc, wireType, function.getArgument(1));
  cudaq::quake::SinkOp::create(builder, loc, TypeRange{}, controlArgumentWire);
  cudaq::quake::WrapOp::create(builder, loc, unwrapped, reference);
  func::ReturnOp::create(builder, loc);

  ASSERT_TRUE(succeeded(verify(*module)));
  QubitIdentityAnalysis analysis(function.front());
  auto initialId = analysis.getQubitId(initial);
  ASSERT_TRUE(initialId);
  EXPECT_EQ(initialId, analysis.getQubitId(x.getWires().front()));
  EXPECT_EQ(initialId, analysis.getQubitId(reset.getWires().front()));
  EXPECT_EQ(initialId, analysis.getQubitId(measurement.getWires().front()));
  EXPECT_EQ(initialId, analysis.getQubitId(control));
  EXPECT_EQ(initialId, analysis.getQubitId(returned));

  ASSERT_TRUE(analysis.getQubitId(function.getArgument(0)));
  ASSERT_TRUE(analysis.getQubitId(function.getArgument(1)));
  EXPECT_EQ(analysis.getQubitId(function.getArgument(1)),
            analysis.getQubitId(controlArgumentWire));
  ASSERT_TRUE(analysis.getQubitId(distinct));
  EXPECT_NE(initialId, analysis.getQubitId(distinct));
  EXPECT_EQ(analysis.getQubitId(distinct),
            analysis.getQubitId(measurement.getWires()[1]));
  EXPECT_EQ(analysis.getQubitId(borrow0a), analysis.getQubitId(borrow0b));
  EXPECT_NE(analysis.getQubitId(borrow0a), analysis.getQubitId(borrow1));
  EXPECT_FALSE(analysis.getQubitId(function.getArgument(2)));
  EXPECT_FALSE(analysis.getQubitId(call.getResult(0)));
  EXPECT_FALSE(analysis.getQubitId(unwrapped));
}
