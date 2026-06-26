/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.  *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/KernelExecution.h"
#include "common/ResultReconstruction.h"
#include "nlohmann/json.hpp"
#include "cudaq/algorithms/sample/policy.h"
#include <gtest/gtest.h>
#include <optional>
#include <string>

// The reconstruction reorder/projection behavior reached through a policy's
// resultOutputMap is exercised end-to-end in
// runtime/test/test_kernel_execution_maps.cpp. The cases below cover what that
// integration test does not touch: the value semantics of activeDeviceQubits,
// the policy metadata API that derives the map for mapped executions (and
// leaves it empty for unmapped ones), and validateExecutionMetadata's
// back-compat position fallback.

static cudaq::KernelExecution createKernelExecution(std::string name) {
  return cudaq::KernelExecution(name, "code", std::nullopt, std::nullopt);
}

TEST(KernelExecutionTester, CopiesActiveDeviceQubitsByValue) {
  auto original = createKernelExecution("kernel");
  original.activeDeviceQubits = {3};

  auto copied = original;
  original.activeDeviceQubits[0] = 11;
  EXPECT_EQ(copied.activeDeviceQubits, (cudaq::ActiveDeviceQubits{3}));
}

TEST(KernelExecutionTester, AssignmentCopiesActiveDeviceQubitsIndependently) {
  auto lhs = createKernelExecution("lhs");
  auto rhs = createKernelExecution("rhs");
  rhs.activeDeviceQubits = {3};

  lhs = rhs;
  rhs.activeDeviceQubits[0] = 11;
  EXPECT_EQ(lhs.activeDeviceQubits, (cudaq::ActiveDeviceQubits{3}));
}

// sample_policy and observe_policy share the same setKernelExecutionMetadata
// base, so the sample policy stands in for both. A mapped execution carries
// active device qubits and derives the result map from the enriched
// output_names; an unmapped execution skips local reconstruction and leaves the
// map empty.
TEST(KernelExecutionTester, PoliciesDeriveResultMapFromOutputNames) {
  nlohmann::json outputNames =
      nlohmann::json::parse(R"([[[0,[4,"alpha",0]]]])");

  auto mapped = cudaq::KernelExecution("sample", "code", std::nullopt,
                                       std::nullopt, outputNames);
  mapped.activeDeviceQubits = {4};

  cudaq::sample_policy mappedPolicy;
  mappedPolicy.setKernelExecutionMetadata(mapped);
  ASSERT_EQ(mappedPolicy.resultOutputMap.outputs.size(), 1);
  EXPECT_EQ(mappedPolicy.resultOutputMap.outputs[0].bitIndex, 4);
  EXPECT_EQ(mappedPolicy.resultOutputMap.outputs[0].outputName, "alpha");
  EXPECT_EQ(mappedPolicy.resultOutputMap.outputs[0].outputPosition, 0);

  auto unmapped = cudaq::KernelExecution("unmapped", "code", std::nullopt,
                                         std::nullopt, outputNames);
  cudaq::sample_policy unmappedPolicy;
  unmappedPolicy.setKernelExecutionMetadata(unmapped);
  EXPECT_TRUE(unmappedPolicy.resultOutputMap.outputs.empty());
}

// validateExecutionMetadata accepts legacy two-element output-location tuples:
// with no explicit position, positions fall back to the dense result index, so
// the dense-position invariant still holds.
TEST(KernelExecutionTester, ValidatesLegacyTwoTuplePositionFallback) {
  auto outputNames =
      nlohmann::json::parse(R"([[[0,[2,"alpha"]],[1,[5,"beta"]]]])");
  EXPECT_NO_THROW(cudaq::validateExecutionMetadata({}, outputNames));
}
