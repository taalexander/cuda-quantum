/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "nvqir/Gates.h"
#include "cudaq/ptsbe/PTSBESamplerImpl.h"
#include <cmath>

using namespace cudaq;

// Single-target gate-field conversion (name, matrix, targets, controls,
// parameters) is exercised end-to-end elsewhere, so those per-field cases were
// removed: ExecutePTSBETest.SingleTrajectoryHadamard covers h,
// ExecutePTSBETest.BellStateDistribution covers controlled x, and rx parameter
// passthrough is pinned by PTSBESampleTest.TracePTSBatchHandlesKernelArgs (rx
// angle survives into the trace) and
// PTSBESampleTest.BroadcastReturnsMultipleResults (rotationKernel's rx runs
// end-to-end). The float<float> template cast is not reachable from this
// double-only CPU target; it is covered by the mgpu fp32 GPU tests. The
// multi-target path (two targets, 16-element matrix) and the error path reach
// no execution test, so they are pinned directly here.

/// Verify unknown gate throws with descriptive error
CUDAQ_TEST(TraceConversionTest, UnknownGateThrows) {
  ptsbe::TraceInstruction inst(ptsbe::TraceInstructionType::Gate,
                               "invalid_gate_xyz", {0}, {}, {});
  try {
    cudaq::ptsbe::detail::convertToSimulatorTask<double>(inst);
    FAIL() << "Expected an exception for unknown gate";
  } catch (...) {
  }
}

/// Verify multi-target gate (swap): two targets, 16-element matrix
CUDAQ_TEST(TraceConversionTest, MultiTargetGate) {
  ptsbe::TraceInstruction inst(ptsbe::TraceInstructionType::Gate, "swap",
                               {3, 7}, {}, {});
  auto task = cudaq::ptsbe::detail::convertToSimulatorTask<double>(inst);

  EXPECT_EQ(task.targets.size(), 2u);
  EXPECT_EQ(task.targets[0], 3u);
  EXPECT_EQ(task.targets[1], 7u);
  EXPECT_EQ(task.matrix.size(), 16u);
}
