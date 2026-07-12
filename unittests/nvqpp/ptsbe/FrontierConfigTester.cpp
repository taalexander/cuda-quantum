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

/// Deterministic strategy for allocation tests: one trajectory per requested
/// multiplicity, each selecting a distinct Kraus index at every noise site.
class FixedRootStrategy : public PTSSamplingStrategy {
public:
  explicit FixedRootStrategy(std::vector<std::size_t> multiplicities)
      : multiplicities_(std::move(multiplicities)) {}

  [[nodiscard]] std::vector<cudaq::KrausTrajectory>
  generateTrajectories(std::span<const detail::NoisePoint> noise_points,
                       std::size_t /*max_trajectories*/) const override {
    std::vector<cudaq::KrausTrajectory> result;
    for (std::size_t i = 0; i < multiplicities_.size(); ++i) {
      std::vector<cudaq::KrausSelection> selections;
      for (const auto &np : noise_points) {
        auto idx = i % np.channel.size();
        selections.push_back(
            {np.circuit_location, np.qubits, np.op_name, idx, idx != 0});
      }
      cudaq::KrausTrajectory traj(i, std::move(selections),
                                  1.0 / multiplicities_.size());
      traj.multiplicity = multiplicities_[i];
      traj.weight = static_cast<double>(multiplicities_[i]);
      result.push_back(std::move(traj));
    }
    return result;
  }

  [[nodiscard]] const char *name() const override { return "FixedRoot"; }

  [[nodiscard]] std::unique_ptr<PTSSamplingStrategy> clone() const override {
    return std::make_unique<FixedRootStrategy>(*this);
  }

private:
  std::vector<std::size_t> multiplicities_;
};

} // namespace

TEST(FrontierConfigTest, OptionsDefaultsUnset) {
  PTSBEOptions options;
  EXPECT_FALSE(options.num_root_draws.has_value());
  EXPECT_FALSE(options.max_paths_per_root.has_value());
  EXPECT_FALSE(options.max_live_states.has_value());
  EXPECT_FALSE(options.allow_non_unitary);
}

TEST(FrontierConfigTest, BatchCarriesKnobsAndDefaultsToOneShotPerPath) {
  PTSBEOptions options;
  options.num_root_draws = 1;
  options.max_paths_per_root = 8;
  options.max_live_states = 4;

  auto batch = buildBatch(cudaq::noise_model{}, options, 4);

  ASSERT_TRUE(batch.numRootDraws.has_value());
  EXPECT_EQ(*batch.numRootDraws, 1);
  ASSERT_TRUE(batch.maxPathsPerRoot.has_value());
  EXPECT_EQ(*batch.maxPathsPerRoot, 8);
  ASSERT_TRUE(batch.maxLiveStates.has_value());
  EXPECT_EQ(*batch.maxLiveStates, 4);
  // Default configuration is one terminal sample per path (T=1), so
  // C_u = N_u.
  EXPECT_EQ(batch.maxShotsPerPath, 1);
  ASSERT_EQ(batch.trajectories.size(), 1);
  EXPECT_EQ(batch.trajectories[0].num_shots, 4);
}

TEST(FrontierConfigTest, KnobsUnsetKeepTerminalBehavior) {
  auto batch = buildBatch(cudaq::noise_model{}, PTSBEOptions{}, 100);

  EXPECT_FALSE(batch.numRootDraws.has_value());
  EXPECT_FALSE(batch.maxPathsPerRoot.has_value());
  EXPECT_FALSE(batch.maxLiveStates.has_value());
  EXPECT_EQ(batch.maxShotsPerPath, 0);
}

TEST(FrontierConfigTest, RequiredPathsExceedingMaxPathsPerRootErrors) {
  PTSBEOptions options;
  options.max_paths_per_root = 10;

  // Default T=1 makes C_u = N_u = 100 > 10. Errors, no silent clamping.
  expectThrowContains<std::invalid_argument>(
      [&]() { buildBatch(cudaq::noise_model{}, options, 100); },
      "max_paths_per_root");
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
  options.max_paths_per_root = 3;
  options.max_shots_per_path = 2;

  // N_u = 6, C_u = 3, T_u = 2: valid, values pass through unchanged.
  auto batch = buildBatch(cudaq::noise_model{}, options, 6);
  EXPECT_EQ(batch.maxShotsPerPath, 2);
  ASSERT_TRUE(batch.maxPathsPerRoot.has_value());
  EXPECT_EQ(*batch.maxPathsPerRoot, 3);
  ASSERT_EQ(batch.trajectories.size(), 1);
  EXPECT_EQ(batch.trajectories[0].num_shots, 6);
}

TEST(FrontierConfigTest, ZeroFrontierKnobsRejected) {
  {
    PTSBEOptions options;
    options.max_live_states = 0;
    expectThrowContains<std::invalid_argument>(
        [&]() { buildBatch(cudaq::noise_model{}, options, 4); }, "positive");
  }
  {
    PTSBEOptions options;
    options.num_root_draws = 0;
    expectThrowContains<std::invalid_argument>(
        [&]() { buildBatch(cudaq::noise_model{}, options, 4); }, "positive");
  }
  {
    PTSBEOptions options;
    options.max_paths_per_root = 0;
    expectThrowContains<std::invalid_argument>(
        [&]() { buildBatch(cudaq::noise_model{}, options, 4); }, "positive");
  }
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

TEST(FrontierConfigTest, RootMultiplicityAllocationIsExact) {
  PTSBEOptions options;
  options.strategy =
      std::make_shared<FixedRootStrategy>(std::vector<std::size_t>{3, 1});
  options.num_root_draws = 4;

  // N_u = shots * d_u / D exactly: 8 * 3/4 = 6 and 8 * 1/4 = 2.
  auto batch = buildBatch(bitFlipOnH(), options, 8);
  ASSERT_EQ(batch.trajectories.size(), 2);
  EXPECT_EQ(batch.trajectories[0].num_shots, 6);
  EXPECT_EQ(batch.trajectories[1].num_shots, 2);
}

TEST(FrontierConfigTest, RootMultiplicityAllocationNonIntegerErrors) {
  PTSBEOptions options;
  options.strategy =
      std::make_shared<FixedRootStrategy>(std::vector<std::size_t>{3, 1});
  options.num_root_draws = 4;

  // 5 * 3/4 is not an integer, so flat counts cannot preserve outer-root
  // weight.
  expectThrowContains<std::invalid_argument>(
      [&]() { buildBatch(bitFlipOnH(), options, 5); }, "outer-root");
}

TEST(FrontierConfigTest, MismatchedDrawTotalErrors) {
  PTSBEOptions options;
  options.strategy =
      std::make_shared<FixedRootStrategy>(std::vector<std::size_t>{3, 1});
  options.num_root_draws = 5;

  // Strategy produced sum(d_u) = 4 != D = 5.
  expectThrowContains<std::invalid_argument>(
      [&]() { buildBatch(bitFlipOnH(), options, 10); }, "num_root_draws");
}

TEST(FrontierConfigTest, UniformAllocationViolatingRootWeightErrors) {
  PTSBEOptions options;
  options.strategy =
      std::make_shared<FixedRootStrategy>(std::vector<std::size_t>{3, 1});
  options.num_root_draws = 4;
  options.shot_allocation =
      ShotAllocationStrategy(ShotAllocationStrategy::Type::UNIFORM);

  // UNIFORM gives (4, 4); flat counts require N_u/total = d_u/D = (3/4, 1/4).
  expectThrowContains<std::invalid_argument>(
      [&]() { buildBatch(bitFlipOnH(), options, 8); }, "outer-root");
}

TEST(FrontierConfigTest, ZeroShotRootWithRootDrawsErrors) {
  PTSBEOptions options;
  options.strategy =
      std::make_shared<FixedRootStrategy>(std::vector<std::size_t>{1, 3});
  options.num_root_draws = 4;
  options.shot_allocation =
      ShotAllocationStrategy(ShotAllocationStrategy::Type::UNIFORM);

  // UNIFORM with one shot gives (1, 0): the second root has d_u = 3 but
  // N_u = 0, so flat counts cannot preserve N_u/total = d_u/D. Zero-shot
  // roots must not slip past the root-weight validation.
  expectThrowContains<std::invalid_argument>(
      [&]() { buildBatch(bitFlipOnH(), options, 1); }, "outer-root");
}

TEST(FrontierConfigTest, IdentityRootAbsorbsAllRootDraws) {
  PTSBEOptions options;
  options.num_root_draws = 4;

  // D IID draws of a circuit with no pre-sampled sites are one identity root
  // with outer multiplicity D, probability 1, and the full shot allocation.
  auto batch = buildBatch(cudaq::noise_model{}, options, 8);
  ASSERT_EQ(batch.trajectories.size(), 1);
  EXPECT_EQ(batch.trajectories[0].multiplicity, 4);
  EXPECT_DOUBLE_EQ(batch.trajectories[0].probability, 1.0);
  EXPECT_EQ(batch.trajectories[0].num_shots, 8);
}

TEST(FrontierConfigTest, NonUnitaryOnlyWorkloadAcceptsMultipleRootDraws) {
  PTSBEOptions options;
  options.allow_non_unitary = true;
  options.num_root_draws = 4;

  // The pure non-unitary workload has zero pre-sampled sites: every branch
  // is selected during replay, so the identity root carries all D draws.
  auto batch = buildBatch(amplitudeDampingOnH(), options, 8);
  ASSERT_EQ(batch.trajectories.size(), 1);
  EXPECT_EQ(batch.trajectories[0].multiplicity, 4);
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
