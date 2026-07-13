/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/KrausTrajectory.h"
#include "cudaq/ptsbe/PTSBEExecutionData.h"
#include "cudaq/ptsbe/PTSBEOptions.h"
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/ptsbe/PTSBESamplerImpl.h"
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

using namespace cudaq::ptsbe;

namespace {

// Bell pair built THROUGH a mid-circuit measurement: mz(q0) collapses the
// superposition, then cx copies the recorded outcome onto q1. rec0 == rec1
// on every shot iff replay collapses at the measurement site.
auto midCircuitBellKernel = []() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  mz(q[0]);
  x<cudaq::ctrl>(q[0], q[1]);
  mz(q[1]);
};

auto measureResetMeasureKernel = []() __qpu__ {
  cudaq::qubit q;
  h(q);
  mz(q);
  reset(q);
  mz(q);
};

auto deterministicFlipKernel = []() __qpu__ {
  cudaq::qubit q;
  mz(q);
  cudaq::apply_noise<cudaq::bit_flip_channel>(1.0, q);
  mz(q);
};

auto coinFlipNoiseKernel = []() __qpu__ {
  cudaq::qubit q;
  mz(q);
  cudaq::apply_noise<cudaq::bit_flip_channel>(0.5, q);
  mz(q);
};

// Per-shot records are fixed-width strings in record-index order: character
// i of each record is the outcome of the measurement site whose
// record_index is i.
std::vector<std::string> recordsFrom(const cudaq::sample_result &result,
                                     std::size_t expectedShots,
                                     std::size_t expectedWidth) {
  auto records = result.sequential_data();
  EXPECT_EQ(records.size(), expectedShots);
  for (const auto &record : records)
    EXPECT_EQ(record.size(), expectedWidth);
  return records;
}

} // namespace

// ============================================================================
// PER-SHOT REPLAY CORRECTNESS (generic path, qpp simulator)
// ============================================================================

// End-to-end Bell collapse through cudaq::ptsbe::sample: mz(q0) is genuinely
// mid-circuit (cx touches q0 afterwards), so the trace build must flag the
// batch and the replay path must collapse before the cx copies the outcome.
// Any single-qubit Pauli error after h keeps the copy exact, so rec0 == rec1
// holds on every shot of every trajectory.
CUDAQ_TEST(McmReplayTest, BellCollapseEndToEndMidCircuitMeasurement) {
  cudaq::set_random_seed(42);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.05));

  sample_options options;
  options.shots = 100;
  options.noise = noise;
  options.ptsbe.include_sequential_data = true;

  auto result = sample(options, midCircuitBellKernel);
  EXPECT_EQ(result.get_total_shots(), options.shots);

  auto records = recordsFrom(result, options.shots, 2);
  for (const auto &record : records)
    EXPECT_EQ(record[0], record[1]) << "decorrelated Bell record: " << record;

  EXPECT_EQ(result.count("01"), 0u);
  EXPECT_EQ(result.count("10"), 0u);
  EXPECT_EQ(result.count("00") + result.count("11"), options.shots);
}

// h; mz; reset; mz collapses, resets to |0>, and measures again. The second
// record bit must be 0 on every shot regardless of trajectory; the first
// bit must show both outcomes over 200 shots.
CUDAQ_TEST(McmReplayTest, ResetForcesSecondRecordBitToZero) {
  cudaq::set_random_seed(42);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.05));

  sample_options options;
  options.shots = 200;
  options.noise = noise;
  options.ptsbe.include_sequential_data = true;

  auto result = sample(options, measureResetMeasureKernel);
  EXPECT_EQ(result.get_total_shots(), options.shots);

  auto records = recordsFrom(result, options.shots, 2);
  bool sawZeroFirstBit = false;
  bool sawOneFirstBit = false;
  for (const auto &record : records) {
    EXPECT_EQ(record[1], '0') << "reset did not zero the qubit: " << record;
    if (record[0] == '0')
      sawZeroFirstBit = true;
    else
      sawOneFirstBit = true;
  }
  EXPECT_TRUE(sawZeroFirstBit);
  EXPECT_TRUE(sawOneFirstBit);

  EXPECT_EQ(result.count("00") + result.count("10"), options.shots);
}

// ============================================================================
// NOISE INTERLEAVED WITH MEASUREMENT SITES
// ============================================================================

// A bit-flip channel with probability 1 between two mz on the same qubit:
// the only trajectory carrying shots selects the X operator, so every record
// must read 0 before the flip and 1 after it.
CUDAQ_TEST(McmReplayTest, InterleavedBitFlipFlipsRecordDeterministically) {
  sample_options options;
  options.shots = 50;
  options.ptsbe.include_sequential_data = true;
  options.ptsbe.return_execution_data = true;

  auto result = sample(options, deterministicFlipKernel);
  EXPECT_EQ(result.get_total_shots(), options.shots);

  auto records = recordsFrom(result, options.shots, 2);
  for (const auto &record : records)
    EXPECT_EQ(record, "01");
  EXPECT_EQ(result.count("01"), options.shots);

  ASSERT_TRUE(result.has_execution_data());
  const auto &data = result.execution_data();
  ASSERT_FALSE(data.trajectories.empty());
  for (const auto &traj : data.trajectories)
    EXPECT_EQ(traj.countErrors(), 1u)
        << "trajectory " << traj.trajectory_id
        << " carries shots but did not select the X operator";
}

// With p = 0.5 both trajectories carry shots. Records must follow the
// selected trajectory exactly: the identity trajectory yields 00 on every
// one of its shots, the X trajectory yields 01. Execution data identifies
// which trajectory produced which records.
CUDAQ_TEST(McmReplayTest, InterleavedBitFlipRecordsFollowSelectedTrajectory) {
  cudaq::set_random_seed(42);

  sample_options options;
  options.shots = 100;
  options.ptsbe.include_sequential_data = true;
  options.ptsbe.return_execution_data = true;

  auto result = sample(options, coinFlipNoiseKernel);
  EXPECT_EQ(result.get_total_shots(), options.shots);

  ASSERT_TRUE(result.has_execution_data());
  const auto &data = result.execution_data();

  bool sawIdentityTrajectory = false;
  bool sawErrorTrajectory = false;
  for (const auto &traj : data.trajectories) {
    if (traj.num_shots == 0)
      continue;
    const std::string expected = traj.countErrors() == 0 ? "00" : "01";
    if (traj.countErrors() == 0)
      sawIdentityTrajectory = true;
    else
      sawErrorTrajectory = true;

    std::size_t total = 0;
    for (const auto &[bits, count] : traj.measurement_counts) {
      EXPECT_EQ(bits, expected)
          << "trajectory " << traj.trajectory_id << " produced record " << bits;
      total += count;
    }
    EXPECT_EQ(total, traj.num_shots);
  }
  EXPECT_TRUE(sawIdentityTrajectory);
  EXPECT_TRUE(sawErrorTrajectory);

  EXPECT_EQ(result.count("00") + result.count("01"), options.shots);
}

// ============================================================================
// RECORD DECORRELATION ACROSS SHOTS OF ONE TRAJECTORY
// ============================================================================

// One noise-free trajectory, 100 shots, an MCM site with a 50/50 outcome:
// each shot must replay its measurement independently (maxShotsPerPath = 1),
// so both record values appear. If all shots shared one drawn outcome, every
// record would be identical. The trailing x pins collapse per shot: the
// second bit is always the negation of the first.
CUDAQ_TEST(McmReplayTest, RecordsDecorrelatedAcrossShotsOfOneTrajectory) {
  cudaq::set_random_seed(42);

  PTSBatch batch;
  batch.trace = {
      {TraceInstructionType::Gate, "h", {0}, {}, {}},
      {TraceInstructionType::Measurement, "mz", {0}, {}, {}, std::nullopt, 0},
      {TraceInstructionType::Gate, "x", {0}, {}, {}},
      {TraceInstructionType::Measurement, "mz", {0}, {}, {}, std::nullopt, 1},
  };
  batch.includeSequentialData = true;
  batch.hasMidCircuitMeasurement = true;
  batch.maxShotsPerPath = 1;

  const std::size_t shots = 100;
  batch.trajectories.push_back(cudaq::KrausTrajectory(0, {}, 1.0, shots));

  auto results = detail::samplePTSBEWithLifecycle(batch);
  auto result = detail::aggregateResults(results);

  auto records = recordsFrom(result, shots, 2);
  for (const auto &record : records)
    EXPECT_NE(record[0], record[1])
        << "x after collapse must negate the record: " << record;

  EXPECT_GT(result.count("01"), 0u) << "all shots shared one measurement draw";
  EXPECT_GT(result.count("10"), 0u) << "all shots shared one measurement draw";
  EXPECT_EQ(result.count("01") + result.count("10"), shots);
}

// ============================================================================
// PTSBatch UNIT TESTS
// ============================================================================

// PTSBatch is a public struct, so hand-built traces may carry record
// indices in any order. The record width is the maximum over all measuring
// instructions of record_index + target count; deriving it from the last
// measuring instruction would under-size the per-shot record and the replay
// loop would then write past the end of the record string.
CUDAQ_TEST(McmReplayTest, NumRecordBitsIsMaxOverAllMeasuringInstructions) {
  PTSBatch batch;
  batch.trace = {
      {TraceInstructionType::Measurement, "mz", {0}, {}, {}, std::nullopt, 2},
      {TraceInstructionType::Gate, "x", {1}, {}, {}},
      {TraceInstructionType::Measurement, "mz", {1}, {}, {}, std::nullopt, 0},
  };
  EXPECT_EQ(batch.numRecordBits(), 3u);
}

CUDAQ_TEST(McmReplayTest, NumRecordBitsIsZeroWithoutMeasuringInstructions) {
  PTSBatch batch;
  batch.trace = {
      {TraceInstructionType::Gate, "h", {0}, {}, {}},
  };
  EXPECT_EQ(batch.numRecordBits(), 0u);
}

// ============================================================================
// mergeSitesWithTrajectory UNIT TESTS
// ============================================================================

namespace {

PTSBETrace makeSiteTrace() {
  PTSBETrace trace = {
      {TraceInstructionType::Gate, "h", {0}, {}, {}},
      {TraceInstructionType::Noise,
       "depolarization",
       {0},
       {},
       {},
       cudaq::depolarization_channel(0.1)},
      {TraceInstructionType::MeasureReset, "mz", {0}, {}, {}, std::nullopt, 0},
      {TraceInstructionType::Gate, "x", {0}, {}, {}},
      {TraceInstructionType::Measurement, "mz", {0}, {}, {}, std::nullopt, 1},
      {TraceInstructionType::Reset, "reset", {1}, {}, {}},
      {TraceInstructionType::Measurement, "mz", {1}, {}, {}, std::nullopt, 2},
  };
  return trace;
}

} // namespace

// Replay ops come out in trace order with one op per site: Gate for circuit
// gates and selected noise operators, Measure / Reset / MeasureReset for
// their sites, carrying the site's qubits.
CUDAQ_TEST(McmReplayTest, MergeSitesEmitsOpKindsInTraceOrder) {
  auto trace = makeSiteTrace();

  // Z error (Kraus index 3) at the Noise entry, trace position 1.
  std::vector<cudaq::KrausSelection> selections = {
      cudaq::KrausSelection(1, {0}, "h", 3, true)};
  cudaq::KrausTrajectory trajectory(0, selections, 0.1, 10);

  auto gateCache = detail::convertTraceGates<double>(trace);
  auto replay = mergeSitesWithTrajectory<double>(trace, gateCache, trajectory);
  const auto &ops = replay.ops;

  ASSERT_EQ(ops.size(), 7u);
  EXPECT_EQ(ops[0].kind, ReplayOpKind::Gate);
  EXPECT_EQ(ops[0].task->operationName, "h");
  EXPECT_EQ(ops[1].kind, ReplayOpKind::Gate);
  EXPECT_EQ(ops[1].task->operationName, "z");
  EXPECT_EQ(ops[2].kind, ReplayOpKind::MeasureReset);
  EXPECT_EQ(ops[2].qubits, (std::vector<std::size_t>{0}));
  EXPECT_EQ(ops[3].kind, ReplayOpKind::Gate);
  EXPECT_EQ(ops[3].task->operationName, "x");
  EXPECT_EQ(ops[4].kind, ReplayOpKind::Measure);
  EXPECT_EQ(ops[4].qubits, (std::vector<std::size_t>{0}));
  EXPECT_EQ(ops[5].kind, ReplayOpKind::Reset);
  EXPECT_EQ(ops[5].qubits, (std::vector<std::size_t>{1}));
  EXPECT_EQ(ops[6].kind, ReplayOpKind::Measure);
  EXPECT_EQ(ops[6].qubits, (std::vector<std::size_t>{1}));
}

// recordOffset on each replay op equals the trace instruction's record_index.
// Gate and Reset ops record nothing.
CUDAQ_TEST(McmReplayTest, MergeSitesRecordOffsetsMatchTraceRecordIndices) {
  auto trace = makeSiteTrace();
  cudaq::KrausTrajectory trajectory(0, {}, 1.0, 10);

  auto gateCache = detail::convertTraceGates<double>(trace);
  auto replay = mergeSitesWithTrajectory<double>(trace, gateCache, trajectory);
  const auto &ops = replay.ops;

  ASSERT_EQ(ops.size(), 6u);
  EXPECT_FALSE(ops[0].recordOffset.has_value());
  ASSERT_TRUE(ops[1].recordOffset.has_value());
  EXPECT_EQ(*ops[1].recordOffset, 0u);
  EXPECT_FALSE(ops[2].recordOffset.has_value());
  ASSERT_TRUE(ops[3].recordOffset.has_value());
  EXPECT_EQ(*ops[3].recordOffset, 1u);
  EXPECT_FALSE(ops[4].recordOffset.has_value());
  ASSERT_TRUE(ops[5].recordOffset.has_value());
  EXPECT_EQ(*ops[5].recordOffset, 2u);
}

// includeIdentity = true must give every trajectory of the same trace an
// identical op-kind sequence: identity selections occupy a Gate slot instead
// of vanishing, keeping gate-slot alignment across trajectories.
CUDAQ_TEST(McmReplayTest, MergeSitesIncludeIdentityKeepsSlotAlignment) {
  PTSBETrace trace = {
      {TraceInstructionType::Gate, "h", {0}, {}, {}},
      {TraceInstructionType::Noise,
       "depolarization",
       {0},
       {},
       {},
       cudaq::depolarization_channel(0.1)},
      {TraceInstructionType::Measurement, "mz", {0}, {}, {}, std::nullopt, 0},
      {TraceInstructionType::Gate, "x", {1}, {}, {}},
      {TraceInstructionType::Noise,
       "depolarization",
       {1},
       {},
       {},
       cudaq::depolarization_channel(0.1)},
      {TraceInstructionType::Measurement, "mz", {1}, {}, {}, std::nullopt, 1},
  };

  std::vector<cudaq::KrausSelection> identitySelections = {
      cudaq::KrausSelection(1, {0}, "h", 0),
      cudaq::KrausSelection(4, {1}, "x", 0)};
  cudaq::KrausTrajectory identityTrajectory(0, identitySelections, 0.81, 81);

  std::vector<cudaq::KrausSelection> errorSelections = {
      cudaq::KrausSelection(1, {0}, "h", 1, true),
      cudaq::KrausSelection(4, {1}, "x", 3, true)};
  cudaq::KrausTrajectory errorTrajectory(1, errorSelections, 0.01, 1);

  auto gateCache = detail::convertTraceGates<double>(trace);
  auto identityReplay = mergeSitesWithTrajectory<double>(
      trace, gateCache, identityTrajectory, /*includeIdentity=*/true);
  auto errorReplay = mergeSitesWithTrajectory<double>(
      trace, gateCache, errorTrajectory, /*includeIdentity=*/true);
  const auto &identityOps = identityReplay.ops;
  const auto &errorOps = errorReplay.ops;

  ASSERT_EQ(identityOps.size(), 6u);
  ASSERT_EQ(errorOps.size(), identityOps.size());
  for (std::size_t i = 0; i < identityOps.size(); ++i) {
    EXPECT_EQ(identityOps[i].kind, errorOps[i].kind) << "slot " << i;
    EXPECT_EQ(identityOps[i].recordOffset, errorOps[i].recordOffset)
        << "slot " << i;
  }
  EXPECT_EQ(identityOps[1].kind, ReplayOpKind::Gate);
  EXPECT_EQ(errorOps[1].task->operationName, "x");
  EXPECT_EQ(errorOps[4].task->operationName, "z");

  // Without includeIdentity the identity trajectory drops its noise slots.
  auto compactReplay =
      mergeSitesWithTrajectory<double>(trace, gateCache, identityTrajectory);
  const auto &compactOps = compactReplay.ops;
  ASSERT_EQ(compactOps.size(), 4u);
  EXPECT_EQ(compactOps[0].kind, ReplayOpKind::Gate);
  EXPECT_EQ(compactOps[1].kind, ReplayOpKind::Measure);
  EXPECT_EQ(compactOps[2].kind, ReplayOpKind::Gate);
  EXPECT_EQ(compactOps[3].kind, ReplayOpKind::Measure);
}

// ============================================================================
// MAX-SHOTS-PER-SLOT ENVIRONMENT OVERRIDE
// ============================================================================

namespace {

PTSBETrace makeMcmTrace() {
  return {
      {TraceInstructionType::Gate, "h", {0}, {}, {}},
      {TraceInstructionType::Measurement, "mz", {0}, {}, {}, std::nullopt, 0},
      {TraceInstructionType::Gate, "x", {0}, {}, {}},
      {TraceInstructionType::Measurement, "mz", {0}, {}, {}, std::nullopt, 1},
  };
}

struct ScopedEnvVar {
  ScopedEnvVar(const char *name, const char *value) : name(name) {
    setenv(name, value, 1);
  }
  ~ScopedEnvVar() { unsetenv(name); }
  const char *name;
};

} // namespace

// CUDAQ_PTSBE_MAX_SHOTS_PER_PATH overrides both the automatic selection and
// an explicitly set PTSBEOptions::max_shots_per_path.
CUDAQ_TEST(McmReplayTest, EnvVarOverridesMaxShotsPerPath) {
  PTSBEOptions options;
  options.max_shots_per_path = 7;

  ScopedEnvVar env("CUDAQ_PTSBE_MAX_SHOTS_PER_PATH", "4");
  auto batch = detail::buildPTSBatchFromTrace(makeMcmTrace(), options, 16);

  EXPECT_TRUE(batch.hasMidCircuitMeasurement);
  EXPECT_EQ(batch.maxShotsPerPath, 4u);
}

CUDAQ_TEST(McmReplayTest, MaxShotsPerPathDefaultsWithoutEnvOverride) {
  PTSBEOptions options;

  auto autoBatch = detail::buildPTSBatchFromTrace(makeMcmTrace(), options, 16);
  EXPECT_EQ(autoBatch.maxShotsPerPath, 1u);

  options.max_shots_per_path = 7;
  auto explicitBatch =
      detail::buildPTSBatchFromTrace(makeMcmTrace(), options, 16);
  EXPECT_EQ(explicitBatch.maxShotsPerPath, 7u);
}

CUDAQ_TEST(McmReplayTest, EnvVarMaxShotsPerPathRejectsNonNumeric) {
  PTSBEOptions options;

  ScopedEnvVar env("CUDAQ_PTSBE_MAX_SHOTS_PER_PATH", "many");
  EXPECT_ANY_THROW(detail::buildPTSBatchFromTrace(makeMcmTrace(), options, 16));
}

// ============================================================================
// RECORDS OPT-IN CONTRACT
// ============================================================================
//
// Mid-circuit results default to counts over unique full records: a lossless
// sufficient statistic for decoding, since shots are exchangeable and
// duplicate records decode identically. The per-shot list is opt-in via
// include_sequential_data. These tests replace the earlier contract where
// mid-circuit measurement forced the sequential-data channel on.

// buildPTSBatchFromTrace must not force sequential data on for mid-circuit
// traces; the option alone controls it.
CUDAQ_TEST(McmReplayTest, BuildBatchMcmDefaultsToCountsOnly) {
  PTSBEOptions options;
  auto batch = detail::buildPTSBatchFromTrace(makeMcmTrace(), options, 16);
  EXPECT_TRUE(batch.hasMidCircuitMeasurement);
  EXPECT_FALSE(batch.includeSequentialData);

  options.include_sequential_data = true;
  auto optInBatch = detail::buildPTSBatchFromTrace(makeMcmTrace(), options, 16);
  EXPECT_TRUE(optInBatch.includeSequentialData);
}

// A hand-built mid-circuit batch without sequential data executes and
// aggregates full records into counts.
CUDAQ_TEST(McmReplayTest, McmBatchCountsOnlyExecutesAndAggregates) {
  cudaq::set_random_seed(42);

  PTSBatch batch;
  batch.trace = {
      {TraceInstructionType::Gate, "h", {0}, {}, {}},
      {TraceInstructionType::Gate, "x", {1}, {0}, {}},
      {TraceInstructionType::Measurement, "mz", {0}, {}, {}, std::nullopt, 0},
      {TraceInstructionType::Measurement, "mz", {1}, {}, {}, std::nullopt, 1},
  };
  batch.includeSequentialData = false;
  batch.hasMidCircuitMeasurement = true;
  batch.maxShotsPerPath = 1;

  const std::size_t shots = 100;
  batch.trajectories.push_back(cudaq::KrausTrajectory(0, {}, 1.0, shots));

  auto results = detail::samplePTSBEWithLifecycle(batch);
  auto result = detail::aggregateResults(results);

  EXPECT_TRUE(result.sequential_data().empty());
  EXPECT_EQ(result.count("01"), 0u);
  EXPECT_EQ(result.count("10"), 0u);
  EXPECT_GT(result.count("00"), 0u);
  EXPECT_GT(result.count("11"), 0u);
  EXPECT_EQ(result.count("00") + result.count("11"), shots);
}

// End-to-end default through cudaq::ptsbe::sample: counts over full
// records, empty sequential data; opting in returns the per-shot list.
CUDAQ_TEST(McmReplayTest, McmSampleDefaultsToCountsOnlyOptInReturnsShotList) {
  cudaq::set_random_seed(42);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.05));

  sample_options options;
  options.shots = 100;
  options.noise = noise;

  auto result = sample(options, midCircuitBellKernel);
  EXPECT_EQ(result.get_total_shots(), options.shots);
  EXPECT_TRUE(result.sequential_data().empty());
  EXPECT_EQ(result.count("00") + result.count("11"), options.shots);
  EXPECT_EQ(result.count("01"), 0u);
  EXPECT_EQ(result.count("10"), 0u);

  options.ptsbe.include_sequential_data = true;
  auto optInResult = sample(options, midCircuitBellKernel);
  EXPECT_EQ(optInResult.sequential_data().size(), options.shots);
}
