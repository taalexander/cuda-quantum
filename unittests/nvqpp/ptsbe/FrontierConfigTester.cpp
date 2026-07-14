/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/NoiseModel.h"
#include "common/Trace.h"
#include "cudaq/ptsbe/NoiseExtractor.h"
#include "cudaq/ptsbe/PTSBEOptions.h"
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/ptsbe/PTSBESamplerImpl.h"
#include "cudaq/ptsbe/strategies/ProbabilisticSamplingStrategy.h"
#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace cudaq::ptsbe;
using cudaq::ptsbe::detail::mergeSitesWithTrajectory;
using cudaq::ptsbe::detail::ReplayOpKind;

namespace {

/// Run `fn` and require it to throw `Exception` whose message contains
/// `needle`.
template <typename Exception, typename Fn>
void expectThrowContains(Fn &&fn, const std::string &needle) {
  try {
    fn();
    FAIL() << "Expected exception containing '" << needle << "'";
  } catch (const Exception &e) {
    EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
        << "message was: " << e.what();
  }
}

cudaq::Trace hMeasureTrace() {
  cudaq::Trace trace;
  trace.appendInstruction("h", {}, {}, {cudaq::QuditInfo(2, 0)});
  trace.appendMeasurement("mz", {cudaq::QuditInfo(2, 0)});
  return trace;
}

PTSBatch buildBatch(const cudaq::noise_model &noise,
                    const PTSBEOptions &options, std::size_t shots) {
  auto trace = detail::buildPTSBETrace(hMeasureTrace(), noise);
  return detail::buildPTSBatchFromTrace(std::move(trace), options, shots);
}

cudaq::noise_model bitFlipOnH(double p = 0.5) {
  cudaq::noise_model noise;
  noise.add_channel("h", {0}, cudaq::bit_flip_channel(p));
  return noise;
}

cudaq::noise_model amplitudeDampingOnH(double gamma = 0.2) {
  cudaq::noise_model noise;
  noise.add_channel("h", {0}, cudaq::amplitude_damping_channel(gamma));
  return noise;
}

} // namespace

TEST(FrontierConfigTest, OptionsDefaultsUnset) {
  PTSBEOptions options;
  EXPECT_FALSE(options.max_live_states.has_value());
  EXPECT_FALSE(options.allow_non_unitary);
}

TEST(FrontierConfigTest, KnobsUnsetKeepTerminalBehavior) {
  auto batch = buildBatch(cudaq::noise_model{}, PTSBEOptions{}, 100);

  EXPECT_FALSE(batch.maxLiveStates.has_value());
  EXPECT_EQ(batch.maxShotsPerPath, 0);
}

TEST(FrontierConfigTest, MaxLiveStatesDefaultsToOneShotPerPath) {
  PTSBEOptions options;
  options.max_live_states = 4;

  auto batch = buildBatch(cudaq::noise_model{}, options, 4);

  ASSERT_TRUE(batch.maxLiveStates.has_value());
  EXPECT_EQ(*batch.maxLiveStates, 4);
  // A set frontier width engages frontier configuration, whose default is one
  // terminal sample per path (T=1), so C_u = N_u.
  EXPECT_EQ(batch.maxShotsPerPath, 1);
  ASSERT_EQ(batch.trajectories.size(), 1);
  EXPECT_EQ(batch.trajectories[0].num_shots, 4);
}

TEST(FrontierConfigTest, NonIntegerTerminalSamplesPerPathError) {
  PTSBEOptions options;
  options.max_live_states = 2;
  options.max_shots_per_path = 2;

  // N_u = 5, C_u = ceil(5/2) = 3: no integer T_u with N_u = C_u * T_u.
  expectThrowContains<std::invalid_argument>(
      [&]() { buildBatch(cudaq::noise_model{}, options, 5); }, "integer");
}

TEST(FrontierConfigTest, ValidConfigurationIsNotClamped) {
  PTSBEOptions options;
  options.max_live_states = 3;
  options.max_shots_per_path = 2;

  // N_u = 6, C_u = 3, T_u = 2: valid, values pass through unchanged.
  auto batch = buildBatch(cudaq::noise_model{}, options, 6);
  EXPECT_EQ(batch.maxShotsPerPath, 2);
  ASSERT_TRUE(batch.maxLiveStates.has_value());
  EXPECT_EQ(*batch.maxLiveStates, 3);
  ASSERT_EQ(batch.trajectories.size(), 1);
  EXPECT_EQ(batch.trajectories[0].num_shots, 6);
}

TEST(FrontierConfigTest, ZeroFrontierWidthRejected) {
  PTSBEOptions options;
  options.max_live_states = 0;
  expectThrowContains<std::invalid_argument>(
      [&]() { buildBatch(cudaq::noise_model{}, options, 4); }, "positive");
}

TEST(FrontierConfigTest, FixedDrawsPreserveDrawTotal) {
  auto trace = detail::buildPTSBETrace(hMeasureTrace(), bitFlipOnH());
  auto sites = detail::extractNoiseSites(trace);

  ProbabilisticSamplingStrategy strategy(
      /*seed=*/7,
      /*max_trajectory_samples=*/std::nullopt,
      /*num_root_draws=*/50);
  auto trajectories = strategy.generateTrajectories(sites.noise_sites, 50);

  std::size_t drawTotal = 0;
  for (const auto &traj : trajectories)
    drawTotal += traj.multiplicity;
  EXPECT_EQ(drawTotal, 50);
}

TEST(FrontierConfigTest, FixedDrawsBeyondMaxTrajectoriesError) {
  cudaq::noise_model noise;
  noise.add_channel("h", {0}, cudaq::depolarization_channel(0.75));
  auto trace = detail::buildPTSBETrace(hMeasureTrace(), noise);
  auto sites = detail::extractNoiseSites(trace);

  // Four near-equiprobable outcomes: 200 draws discover more than 2 unique
  // roots. Reaching max_trajectories before the draws complete is an error,
  // not a stopping rule.
  ProbabilisticSamplingStrategy strategy(
      /*seed=*/7,
      /*max_trajectory_samples=*/std::nullopt,
      /*num_root_draws=*/200);
  expectThrowContains<std::runtime_error>(
      [&]() { strategy.generateTrajectories(sites.noise_sites, 2); },
      "max_trajectories");
}

TEST(FrontierConfigTest, FixedDrawsConflictWithSampleBudget) {
  expectThrowContains<std::invalid_argument>(
      [&]() {
        ProbabilisticSamplingStrategy strategy(
            /*seed=*/7, /*max_trajectory_samples=*/5, /*num_root_draws=*/5);
      },
      "num_root_draws");
}

TEST(FrontierConfigTest, NoiseFreeTraceRunsOneIdentityRoot) {
  auto batch = buildBatch(cudaq::noise_model{}, PTSBEOptions{}, 8);

  ASSERT_EQ(batch.trajectories.size(), 1);
  EXPECT_EQ(batch.trajectories[0].multiplicity, 1);
  EXPECT_DOUBLE_EQ(batch.trajectories[0].probability, 1.0);
  EXPECT_EQ(batch.trajectories[0].num_shots, 8);
}

TEST(FrontierConfigTest, NonUnitaryOnlyWorkloadRunsOneIdentityRoot) {
  PTSBEOptions options;
  options.allow_non_unitary = true;

  // The pure non-unitary workload has zero pre-sampled sites: every branch
  // is selected during replay, so a single identity root carries the shots.
  auto batch = buildBatch(amplitudeDampingOnH(), options, 8);
  ASSERT_EQ(batch.trajectories.size(), 1);
  EXPECT_TRUE(batch.trajectories[0].kraus_selections.empty());
  EXPECT_EQ(batch.trajectories[0].num_shots, 8);
}

TEST(FrontierConfigTest, AdmissionKeepsRawChannelOps) {
  auto trace = detail::buildPTSBETrace(hMeasureTrace(), amplitudeDampingOnH());

  auto result = detail::extractNoiseSites(trace, /*validate_unitary_mixture=*/
                                          true, /*allow_non_unitary=*/true);

  ASSERT_EQ(result.noise_sites.size(), 1);
  const auto &site = result.noise_sites[0];
  EXPECT_TRUE(site.is_non_unitary);
  EXPECT_FALSE(site.channel.is_unitary_mixture());
  // Raw amplitude-damping Kraus operators, not a unitary conversion.
  EXPECT_EQ(site.channel.get_ops().size(), 2);
  EXPECT_FALSE(result.all_unitary_mixtures);
}

TEST(FrontierConfigTest, AdmissionDisabledStillThrows) {
  auto trace = detail::buildPTSBETrace(hMeasureTrace(), amplitudeDampingOnH());

  expectThrowContains<std::invalid_argument>(
      [&]() {
        auto sites =
            detail::extractNoiseSites(trace, /*validate_unitary_mixture=*/true,
                                      /*allow_non_unitary=*/false);
        (void)sites;
      },
      "unitary");
}

TEST(FrontierConfigTest, AdmittedSitesAreNotPreSampled) {
  PTSBEOptions options;
  options.allow_non_unitary = true;

  auto batch = buildBatch(amplitudeDampingOnH(), options, 10);

  // The only noise site is non-unitary, so no root pre-sampling happens:
  // execution runs one identity root and branches at the site during replay.
  ASSERT_EQ(batch.trajectories.size(), 1);
  EXPECT_TRUE(batch.trajectories[0].kraus_selections.empty());
}

TEST(FrontierConfigTest, MergeEmitsKrausBranchForNonUnitarySites) {
  auto trace = detail::buildPTSBETrace(hMeasureTrace(), amplitudeDampingOnH());
  ASSERT_EQ(trace.size(), 3);

  auto gateCache = detail::convertTraceGates<double>(trace);
  cudaq::KrausTrajectory trajectory(0, {}, 1.0, 1);

  auto replay = mergeSitesWithTrajectory<double>(trace, gateCache, trajectory);

  ASSERT_EQ(replay.ops.size(), 3);
  EXPECT_EQ(replay.ops[0].kind, ReplayOpKind::Gate);
  EXPECT_EQ(replay.ops[1].kind, ReplayOpKind::KrausBranch);
  EXPECT_EQ(replay.ops[2].kind, ReplayOpKind::Measure);

  const auto &branch = replay.ops[1];
  ASSERT_NE(branch.channel, nullptr);
  EXPECT_EQ(branch.channel, &trace[1].channel.value());
  EXPECT_EQ(branch.channel->get_ops().size(), 2);
  ASSERT_EQ(branch.qubits.size(), 1);
  EXPECT_EQ(branch.qubits[0], 0);
}

TEST(FrontierConfigTest, GenericDispatchRejectsNonUnitarySites) {
  PTSBEOptions options;
  options.allow_non_unitary = true;

  auto batch = buildBatch(amplitudeDampingOnH(), options, 10);

  expectThrowContains<std::runtime_error>(
      [&]() { detail::samplePTSBEWithLifecycle(batch); }, "non-unitary");
}
