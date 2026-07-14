/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/KrausTrajectory.h"
#include "cudaq/ptsbe/PTSBESamplerImpl.h"
#include <cmath>
#include <string>
#include <vector>

using namespace cudaq;

namespace {

// The Kraus-selection behavior under test is which Gate ops the merge emits
// and where, so tests assert on the Gate subsequence of the replay-op list.
// The holder keeps the gate cache and replay alive for as long as the
// returned pointers are used; the pointers stay valid across the move
// because they point into heap storage.
struct MergedGates {
  std::vector<ptsbe::detail::GateTask<double>> gateCache;
  ptsbe::detail::TrajectoryReplay<double> replay;
  std::vector<const ptsbe::detail::GateTask<double> *> gates;

  std::size_t size() const { return gates.size(); }
  bool empty() const { return gates.empty(); }
  const ptsbe::detail::GateTask<double> &operator[](std::size_t i) const {
    return *gates[i];
  }
};

MergedGates mergeToGates(const std::vector<ptsbe::TraceInstruction> &ptsbeTrace,
                         const cudaq::KrausTrajectory &trajectory) {
  MergedGates merged;
  merged.gateCache = ptsbe::detail::convertTraceGates<double>(ptsbeTrace);
  merged.replay = ptsbe::detail::mergeSitesWithTrajectory<double>(
      ptsbeTrace, merged.gateCache, trajectory);
  for (const auto &op : merged.replay.ops)
    if (op.kind == ptsbe::detail::ReplayOpKind::Gate)
      merged.gates.push_back(op.task);
  return merged;
}

} // namespace

// No-noise passthrough and single Z-error insertion at a Noise position are
// covered end-to-end by ExecutePTSBETest.SingleTrajectoryHadamard (empty
// trajectory) and ExecutePTSBETest.TrajectoryWithNoiseInsertion; the removed
// NoNoiseInsertions and SingleNoiseInsertion unit cases duplicated them.

/// Verify two consecutive noise entries at the same gate
CUDAQ_TEST(MergeSitesWithTrajectoryTest, MultipleNoiseEntriesAfterGate) {
  // Trace: [0] H on q0, [1] Noise on q0, [2] Noise on q1
  std::vector<ptsbe::TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {0},
       {},
       {},
       depolarization_channel(0.1)},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {1},
       {},
       {},
       depolarization_channel(0.1)},
  };

  // X on qubit 0 at trace pos 1, Z on qubit 1 at trace pos 2
  std::vector<KrausSelection> selections = {
      KrausSelection(1, {0}, "h", 1, true),
      KrausSelection(2, {1}, "h", 3, true)};
  KrausTrajectory trajectory(0, selections, 0.05, 5);

  auto gates = mergeToGates(ptsbeTrace, trajectory);

  ASSERT_EQ(gates.size(), 3u);
  EXPECT_EQ(gates[0].operationName, "h");
  EXPECT_EQ(gates[1].operationName, "x");
  EXPECT_EQ(gates[1].targets[0], 0u);
  EXPECT_EQ(gates[2].operationName, "z");
  EXPECT_EQ(gates[2].targets[0], 1u);
}

/// Verify invalid circuit_location throws error
CUDAQ_TEST(MergeSitesWithTrajectoryTest, InvalidCircuitLocationThrows) {
  std::vector<ptsbe::TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
  };

  // circuit_location = 5 is beyond the trace
  std::vector<KrausSelection> selections = {
      KrausSelection(5, {0}, "h", 2, true)};
  KrausTrajectory trajectory(0, selections, 0.1, 10);

  try {
    mergeToGates(ptsbeTrace, trajectory);
    FAIL() << "Expected an exception for invalid circuit_location";
  } catch (...) {
  }
}

/// Verify noise at the last trace position works
CUDAQ_TEST(MergeSitesWithTrajectoryTest, NoiseAtLastPosition) {
  // Trace: [0] H on q0, [1] X on q1, [2] Noise on q1
  std::vector<ptsbe::TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {ptsbe::TraceInstructionType::Gate, "x", {1}, {}, {}},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {1},
       {},
       {},
       depolarization_channel(0.1)},
  };

  // Z error at trace position 2 (the Noise entry)
  std::vector<KrausSelection> selections = {
      KrausSelection(2, {1}, "x", 3, true)};
  KrausTrajectory trajectory(0, selections, 0.1, 10);

  auto gates = mergeToGates(ptsbeTrace, trajectory);

  // Should be: H, X, noise(Z)
  ASSERT_EQ(gates.size(), 3u);
  EXPECT_EQ(gates[0].operationName, "h");
  EXPECT_EQ(gates[1].operationName, "x");
  EXPECT_EQ(gates[2].operationName, "z");
}

// Pure identity skipping (IdentityNoiseSkipped, AllIdentitySkipsAllNoise) is
// pinned by the MixedIdentityAndErrorNoise case below, which exercises the same
// identity-skip branch alongside a real error, and end-to-end by
// ExecutePTSBETest.MultiQubitWithSelectiveNoise; the pure-identity duplicates
// were removed.

/// Verify mixed identity and error noise insertions
CUDAQ_TEST(MergeSitesWithTrajectoryTest, MixedIdentityAndErrorNoise) {
  // Trace: [0] H q0, [1] Noise q0, [2] X q1, [3] Noise q1, [4] Z q0
  std::vector<ptsbe::TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {0},
       {},
       {},
       depolarization_channel(0.1)},
      {ptsbe::TraceInstructionType::Gate, "x", {1}, {}, {}},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {1},
       {},
       {},
       depolarization_channel(0.1)},
      {ptsbe::TraceInstructionType::Gate, "z", {0}, {}, {}},
  };

  // IDENTITY at trace pos 1, Y error (index 2) at trace pos 3
  std::vector<KrausSelection> selections = {
      KrausSelection(1, {0}, "h", 0), KrausSelection(3, {1}, "x", 2, true)};
  KrausTrajectory trajectory(0, selections, 0.2, 20);

  auto gates = mergeToGates(ptsbeTrace, trajectory);

  // H, (identity skipped), X, noise(Y), Z
  ASSERT_EQ(gates.size(), 4u);
  EXPECT_EQ(gates[0].operationName, "h");
  EXPECT_EQ(gates[1].operationName, "x");
  EXPECT_EQ(gates[2].operationName, "y");
  EXPECT_EQ(gates[2].targets[0], 1u);
  EXPECT_EQ(gates[3].operationName, "z");
}

/// Verify empty trace produces an empty replay list
CUDAQ_TEST(MergeSitesWithTrajectoryTest, EmptyTrace) {
  std::vector<ptsbe::TraceInstruction> ptsbeTrace;

  KrausTrajectory trajectory(0, {}, 1.0, 100);
  auto gates = mergeToGates(ptsbeTrace, trajectory);
  EXPECT_TRUE(gates.empty());
}

// Noise inserted after every gate (NoiseOnEveryGate) is covered by
// MultipleNoiseEntriesAfterGate and MixedIdentityAndErrorNoise above (both emit
// multiple error insertions across the trace) and end-to-end by
// ExecutePTSBETest.TrajectoryWithNoiseInsertion; the removed case duplicated
// that multi-insertion behavior.
