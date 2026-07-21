/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/PTSBESampler.h"
#include "cudaq/ptsbe/PTSBESamplerImpl.h"
#include <cmath>
#include <map>
#include <type_traits>

using namespace cudaq;

namespace {

struct MockPTSBESimulator {
  mutable bool sampleWithPTSBE_called = false;

  std::vector<cudaq::sample_result>
  sampleWithPTSBE(const ptsbe::PTSBatch &batch) {
    sampleWithPTSBE_called = true;
    return {};
  }
};

struct MockBatchSimulator : ptsbe::BatchSimulator {
  mutable bool sampleWithPTSBE_called = false;

  std::vector<cudaq::sample_result>
  sampleWithPTSBE(const ptsbe::PTSBatch &batch) {
    sampleWithPTSBE_called = true;
    return {};
  }
};

struct MockImportanceBatchSimulator : ptsbe::BatchSimulator,
                                      ptsbe::ImportanceBatchSimulator {
  bool importanceCalled = false;
  bool returnUnitWeightHistogram = false;
  bool returnCompressedUnitWeightHistogram = false;
  bool returnSparseWeightedBins = false;
  std::optional<ptsbe::ImportanceExecutionRequest> lastRequest;

  std::vector<cudaq::sample_result>
  sampleWithPTSBE(const ptsbe::PTSBatch &) override {
    return {};
  }

  ptsbe::ImportanceExecutionResult sampleWithPTSBEImportance(
      const ptsbe::PTSBatch &,
      ptsbe::ImportanceExecutionRequest request) override {
    importanceCalled = true;
    lastRequest = std::move(request);
    ptsbe::ImportanceExecutionResult result;
    result.bins =
        returnSparseWeightedBins
            ? std::vector<ptsbe::detail::LogMassBin>{{"0", std::log(0.999)},
                                                     {"1", std::log(0.001)}}
            : std::vector<ptsbe::detail::LogMassBin>{{"0", std::log(0.25)},
                                                     {"1", std::log(0.75)}};
    if (returnUnitWeightHistogram)
      result.unitWeightHistogram =
          returnCompressedUnitWeightHistogram
              ? std::vector<ptsbe::detail::CountBin>{{"0", 1}, {"1", 3}}
              : std::vector<ptsbe::detail::CountBin>{{"0", 3}, {"1", 7}};
    result.diagnostics.representedParticles = lastRequest->proposalParticles;
    result.diagnostics.logSumWeights =
        std::log(static_cast<double>(lastRequest->proposalParticles));
    result.diagnostics.logSumSquaredWeights =
        std::log(static_cast<double>(lastRequest->proposalParticles));
    result.diagnostics.effectiveSampleSize = lastRequest->proposalParticles;
    return result;
  }
};

struct NonPTSBESimulator {
  void execute(const ptsbe::PTSBatch &) {}
};

static_assert(std::is_base_of_v<ptsbe::BatchSimulator, MockBatchSimulator>);
static_assert(!std::is_base_of_v<ptsbe::BatchSimulator, MockPTSBESimulator>);
static_assert(std::is_base_of_v<ptsbe::ImportanceBatchSimulator,
                                MockImportanceBatchSimulator>);

} // namespace

/// Test: ptsbe::PTSBatch compiles and can hold trajectory data
CUDAQ_TEST(PTSBEInterfaceTest, PTSBatchWithTrajectories) {
  ptsbe::PTSBatch batch;

  for (size_t i = 0; i < 5; ++i) {
    KrausTrajectory traj;
    traj.trajectory_id = i;
    traj.num_shots = (i + 1) * 200;
    batch.trajectories.push_back(traj);
  }

  EXPECT_EQ(batch.trajectories.size(), 5);
  EXPECT_EQ(batch.trajectories[2].num_shots, 600);
}

CUDAQ_TEST(PTSBEInterfaceTest,
           BudgetedCountedWaveAllocatesExactlyNForEveryResampler) {
  for (const auto resampler :
       {ptsbe::detail::FinalResampler::Multinomial,
        ptsbe::detail::FinalResampler::Residual,
        ptsbe::detail::FinalResampler::ResidualStratified}) {
    ptsbe::PTSBatch batch;
    batch.trace = {
        {ptsbe::TraceInstructionType::Measurement, "mz", {0}, {}, {}}};
    cudaq::KrausTrajectory root;
    root.num_shots = 10;
    batch.trajectories.push_back(root);
    batch.includeSequentialData = true;
    batch.maxLiveStates = 4;
    batch.importanceExperiment =
        std::make_shared<const ptsbe::detail::ImportanceExperimentState>(
            ptsbe::detail::ImportanceExperimentState{
                {.mode = ptsbe::detail::NonUnitaryMode::CountedWave,
                 .resampler = resampler,
                 .proposalParticles = 4},
                20260720});

    MockImportanceBatchSimulator simulator;
    simulator.returnUnitWeightHistogram = true;
    simulator.returnCompressedUnitWeightHistogram = true;
    const auto result =
        ptsbe::detail::finalizeImportancePTSBE(simulator, batch);

    ASSERT_TRUE(simulator.lastRequest);
    EXPECT_EQ(simulator.lastRequest->proposalParticles, 4);
    EXPECT_EQ(simulator.lastRequest->mode,
              ptsbe::detail::NonUnitaryMode::CountedWave);
    EXPECT_TRUE(simulator.lastRequest->krausProposals.empty());
    EXPECT_EQ(result.get_total_shots(), 10);
    const auto sequential = result.sequential_data();
    EXPECT_EQ(sequential.size(), 10);
    std::map<std::string, std::size_t> sequentialCounts;
    for (const auto &record : sequential)
      ++sequentialCounts[record];
    const auto aggregateCounts = result.to_map();
    ASSERT_EQ(sequentialCounts.size(), aggregateCounts.size());
    for (const auto &[record, count] : sequentialCounts)
      EXPECT_EQ(count, aggregateCounts.at(record));
  }
}

CUDAQ_TEST(PTSBEInterfaceTest, ImportanceFinalizerOmitsZeroCountRecords) {
  ptsbe::PTSBatch batch;
  batch.trace = {{ptsbe::TraceInstructionType::Measurement, "mz", {0}, {}, {}}};
  cudaq::KrausTrajectory root;
  root.num_shots = 1;
  batch.trajectories.push_back(root);
  batch.maxLiveStates = 4;
  batch.importanceExperiment =
      std::make_shared<const ptsbe::detail::ImportanceExperimentState>(
          ptsbe::detail::ImportanceExperimentState{
              {.mode = ptsbe::detail::NonUnitaryMode::Importance,
               .resampler = ptsbe::detail::FinalResampler::ResidualStratified},
              20260720});

  MockImportanceBatchSimulator simulator;
  simulator.returnSparseWeightedBins = true;
  const auto result = ptsbe::detail::finalizeImportancePTSBE(simulator, batch);

  const auto counts = result.to_map();
  ASSERT_EQ(counts.size(), 1);
  EXPECT_EQ(counts.begin()->second, 1u);
}

CUDAQ_TEST(PTSBEInterfaceTest,
           ImportanceFinalizerMaterializesOrdinaryExactCountResult) {
  ptsbe::PTSBatch batch;
  batch.trace = {{ptsbe::TraceInstructionType::Measurement, "mz", {0}, {}, {}}};
  cudaq::KrausTrajectory root;
  root.num_shots = 17;
  batch.trajectories.push_back(root);
  batch.includeSequentialData = true;
  batch.maxLiveStates = 4;
  batch.importanceExperiment =
      std::make_shared<const ptsbe::detail::ImportanceExperimentState>(
          ptsbe::detail::ImportanceExperimentState{
              {.mode = ptsbe::detail::NonUnitaryMode::Importance,
               .resampler = ptsbe::detail::FinalResampler::ResidualStratified},
              20260703});

  MockImportanceBatchSimulator simulator;
  auto result = ptsbe::detail::finalizeImportancePTSBE(simulator, batch);

  EXPECT_TRUE(simulator.importanceCalled);
  EXPECT_EQ(result.get_total_shots(), 17u);
  const auto sequential = result.sequential_data();
  EXPECT_EQ(sequential.size(), 17u);
  std::map<std::string, std::size_t> sequentialCounts;
  for (const auto &record : sequential)
    ++sequentialCounts[record];
  const auto aggregateCounts = result.to_map();
  ASSERT_EQ(sequentialCounts.size(), aggregateCounts.size());
  for (const auto &[record, count] : sequentialCounts)
    EXPECT_EQ(count, aggregateCounts.at(record));
}

CUDAQ_TEST(PTSBEInterfaceTest,
           ImportanceFinalizerPreservesExactUnitWeightHistogram) {
  ptsbe::PTSBatch batch;
  batch.trace = {{ptsbe::TraceInstructionType::Measurement, "mz", {0}, {}, {}}};
  cudaq::KrausTrajectory root;
  root.num_shots = 10;
  batch.trajectories.push_back(root);
  batch.maxLiveStates = 4;
  batch.importanceExperiment =
      std::make_shared<const ptsbe::detail::ImportanceExperimentState>(
          ptsbe::detail::ImportanceExperimentState{
              {.mode = ptsbe::detail::NonUnitaryMode::Importance,
               .resampler = ptsbe::detail::FinalResampler::Multinomial},
              20260704});

  MockImportanceBatchSimulator simulator;
  simulator.returnUnitWeightHistogram = true;
  const auto result = ptsbe::detail::finalizeImportancePTSBE(simulator, batch);

  const auto counts = result.to_map();
  EXPECT_EQ(counts.at("0"), 3u);
  EXPECT_EQ(counts.at("1"), 7u);
}

/// Test: Trajectory with KrausSelection noise insertions
CUDAQ_TEST(PTSBEInterfaceTest, TrajectoryWithNoise) {
  KrausTrajectory traj;
  traj.trajectory_id = 0;
  traj.num_shots = 1000;

  // Add noise selections
  traj.kraus_selections.push_back(KrausSelection(0, {0}, "h", 0));
  traj.kraus_selections.push_back(KrausSelection(1, {0, 1}, "cx", 2, true));
  traj.kraus_selections.push_back(KrausSelection(2, {1}, "x", 1, true));

  EXPECT_EQ(traj.kraus_selections.size(), 3);
  EXPECT_EQ(traj.kraus_selections[1].qubits.size(), 2);
  EXPECT_EQ(traj.kraus_selections[2].op_name, "x");
}

/// Test: Shot allocation across multiple trajectories
CUDAQ_TEST(PTSBEInterfaceTest, ShotAllocation) {
  ptsbe::PTSBatch batch;

  // Different shot counts per trajectory
  std::vector<size_t> shot_counts = {500, 300, 150, 50};

  for (size_t i = 0; i < shot_counts.size(); ++i) {
    KrausTrajectory traj;
    traj.trajectory_id = i;
    traj.num_shots = shot_counts[i];
    batch.trajectories.push_back(traj);
  }

  size_t total = 0;
  for (const auto &t : batch.trajectories)
    total += t.num_shots;

  EXPECT_EQ(total, 1000);
}

/// Test: Zero-shot trajectory (probability thresholding edge case)
CUDAQ_TEST(PTSBEInterfaceTest, ZeroShotTrajectory) {
  ptsbe::PTSBatch batch;

  KrausTrajectory zero_traj;
  zero_traj.trajectory_id = 0;
  zero_traj.num_shots = 0;
  batch.trajectories.push_back(zero_traj);

  KrausTrajectory normal_traj;
  normal_traj.trajectory_id = 1;
  normal_traj.num_shots = 1000;
  batch.trajectories.push_back(normal_traj);

  EXPECT_EQ(batch.trajectories[0].num_shots, 0);
  EXPECT_EQ(batch.trajectories[1].num_shots, 1000);
}

/// Test: Empty batch (validation edge case)
CUDAQ_TEST(PTSBEInterfaceTest, EmptyBatch) {
  ptsbe::PTSBatch batch;

  EXPECT_TRUE(batch.trajectories.empty());
}

/// Test: Clean trajectory without noise
CUDAQ_TEST(PTSBEInterfaceTest, CleanTrajectory) {
  KrausTrajectory traj;
  traj.trajectory_id = 0;
  traj.num_shots = 500;

  EXPECT_TRUE(traj.kraus_selections.empty());
  EXPECT_EQ(traj.num_shots, 500);
}

/// Test: Runtime dispatch calls sampleWithPTSBE for ptsbe::BatchSimulator
/// implementers
CUDAQ_TEST(PTSBEInterfaceTest, RuntimeDispatchCallsMock) {
  MockBatchSimulator ptsbe_sim;
  ptsbe::PTSBatch batch;

  ptsbe_sim.sampleWithPTSBE(batch);
  EXPECT_TRUE(ptsbe_sim.sampleWithPTSBE_called);

  constexpr bool nonPtsbeIsBatchSimulator =
      std::is_base_of_v<ptsbe::BatchSimulator, NonPTSBESimulator>;
  EXPECT_FALSE(nonPtsbeIsBatchSimulator);
}

/// Test: ptsbe::BatchSimulator inheritance is the dispatch contract
CUDAQ_TEST(PTSBEInterfaceTest, BatchSimulatorDispatchContract) {
  constexpr bool mockBatchIsBatchSimulator =
      std::is_base_of_v<ptsbe::BatchSimulator, MockBatchSimulator>;
  constexpr bool mockPtsbeIsBatchSimulator =
      std::is_base_of_v<ptsbe::BatchSimulator, MockPTSBESimulator>;
  EXPECT_TRUE(mockBatchIsBatchSimulator);
  EXPECT_FALSE(mockPtsbeIsBatchSimulator);
}

CUDAQ_TEST(PTSBEInterfaceTest, OptionalImportanceInterfaceContract) {
  MockBatchSimulator oldOnly;
  EXPECT_THROW(ptsbe::requireImportanceBatchSimulator(oldOnly),
               std::runtime_error);

  MockImportanceBatchSimulator dual;
  auto &selected = ptsbe::requireImportanceBatchSimulator(dual);
  EXPECT_EQ(&selected, static_cast<ptsbe::ImportanceBatchSimulator *>(&dual));
}
