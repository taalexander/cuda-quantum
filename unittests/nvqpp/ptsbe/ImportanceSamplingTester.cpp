/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/ImportanceSampling.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>

using cudaq::kraus_op;
using cudaq::ptsbe::detail::aggregateLogMass;
using cudaq::ptsbe::detail::allocateCounts;
using cudaq::ptsbe::detail::applyScaledKraus;
using cudaq::ptsbe::detail::buildKrausProposal;
using cudaq::ptsbe::detail::computeWeightDiagnostics;
using cudaq::ptsbe::detail::CountBin;
using cudaq::ptsbe::detail::counterUniform;
using cudaq::ptsbe::detail::deriveHistoryId;
using cudaq::ptsbe::detail::drawCategorical;
using cudaq::ptsbe::detail::FinalResampler;
using cudaq::ptsbe::detail::ImportanceDrawPurpose;
using cudaq::ptsbe::detail::ImportanceNormalization;
using cudaq::ptsbe::detail::ImportanceRngKey;
using cudaq::ptsbe::detail::LogMassBin;
using cudaq::ptsbe::detail::makeSequentialData;
using cudaq::ptsbe::detail::materializeWeightedRecords;
using cudaq::ptsbe::detail::NonUnitaryMode;
using cudaq::ptsbe::detail::readImportanceExperimentConfig;
using cudaq::ptsbe::detail::splitMultiplicity;
using cudaq::ptsbe::detail::WeightedRecord;

static std::vector<kraus_op> amplitudeDamping(double gamma) {
  return {{{1.0, 0.0, 0.0, std::sqrt(1.0 - gamma)}},
          {{0.0, std::sqrt(gamma), 0.0, 0.0}}};
}

namespace {
struct ScopedEnvVar {
  std::string name;
  std::optional<std::string> previous;

  ScopedEnvVar(const char *variable, const char *value) : name(variable) {
    if (const auto *oldValue = std::getenv(variable))
      previous = oldValue;
    if (value)
      setenv(variable, value, 1);
    else
      unsetenv(variable);
  }

  ~ScopedEnvVar() {
    if (previous)
      setenv(name.c_str(), previous->c_str(), 1);
    else
      unsetenv(name.c_str());
  }
};
} // namespace

CUDAQ_TEST(ImportanceConfigTest, DefaultsSelectExactFrontier) {
  ScopedEnvVar mode("CUDAQ_PTSBE_NONUNITARY_MODE", nullptr);
  ScopedEnvVar normalization("CUDAQ_PTSBE_IMPORTANCE_NORMALIZATION", nullptr);
  ScopedEnvVar resampler("CUDAQ_PTSBE_IMPORTANCE_RESAMPLER", nullptr);
  ScopedEnvVar checkpoint("CUDAQ_PTSBE_IMPORTANCE_CHECKPOINT_SITES", nullptr);
  const auto config = readImportanceExperimentConfig();
  EXPECT_EQ(config.mode, NonUnitaryMode::Frontier);
  EXPECT_EQ(config.normalization, ImportanceNormalization::Site);
  EXPECT_EQ(config.resampler, FinalResampler::ResidualStratified);
  EXPECT_EQ(config.checkpointSites, 16);
}

CUDAQ_TEST(ImportanceConfigTest, ParsesEveryBranchLocalControl) {
  for (const auto &[value, expected] :
       std::vector<std::pair<const char *, NonUnitaryMode>>{
           {"frontier", NonUnitaryMode::Frontier},
           {"counted_wave", NonUnitaryMode::CountedWave},
           {"importance", NonUnitaryMode::Importance}}) {
    ScopedEnvVar variable("CUDAQ_PTSBE_NONUNITARY_MODE", value);
    EXPECT_EQ(readImportanceExperimentConfig().mode, expected);
  }
  {
    ScopedEnvVar variable("CUDAQ_PTSBE_IMPORTANCE_NORMALIZATION", "segment");
    EXPECT_EQ(readImportanceExperimentConfig().normalization,
              ImportanceNormalization::Segment);
  }
  for (const auto &[value, expected] :
       std::vector<std::pair<const char *, FinalResampler>>{
           {"multinomial", FinalResampler::Multinomial},
           {"residual", FinalResampler::Residual},
           {"residual_stratified", FinalResampler::ResidualStratified}}) {
    ScopedEnvVar variable("CUDAQ_PTSBE_IMPORTANCE_RESAMPLER", value);
    EXPECT_EQ(readImportanceExperimentConfig().resampler, expected);
  }
  {
    ScopedEnvVar variable("CUDAQ_PTSBE_IMPORTANCE_CHECKPOINT_SITES", "31");
    EXPECT_EQ(readImportanceExperimentConfig().checkpointSites, 31);
  }
}

CUDAQ_TEST(ImportanceConfigTest, RejectsInvalidControlValues) {
  {
    ScopedEnvVar variable("CUDAQ_PTSBE_NONUNITARY_MODE", "tree_magic");
    EXPECT_THROW(readImportanceExperimentConfig(), std::invalid_argument);
  }
  {
    ScopedEnvVar variable("CUDAQ_PTSBE_IMPORTANCE_NORMALIZATION", "late");
    EXPECT_THROW(readImportanceExperimentConfig(), std::invalid_argument);
  }
  {
    ScopedEnvVar variable("CUDAQ_PTSBE_IMPORTANCE_RESAMPLER", "largest");
    EXPECT_THROW(readImportanceExperimentConfig(), std::invalid_argument);
  }
  {
    ScopedEnvVar variable("CUDAQ_PTSBE_IMPORTANCE_CHECKPOINT_SITES", "0");
    EXPECT_THROW(readImportanceExperimentConfig(), std::invalid_argument);
  }
}

CUDAQ_TEST(ImportanceProposalTest, ConstructsTraceOverDimensionProposal) {
  const double gamma = 0.2;
  const double excitedWeight = 0.3;
  const auto rootExcited = std::sqrt(excitedWeight);
  const auto rootGround = std::sqrt(1.0 - excitedWeight);
  std::vector<kraus_op> operators = {
      {{rootGround, 0.0, 0.0, rootGround * std::sqrt(1.0 - gamma)}},
      {{0.0, rootGround * std::sqrt(gamma), 0.0, 0.0}},
      {{rootExcited * std::sqrt(1.0 - gamma), 0.0, 0.0, rootExcited}},
      {{0.0, 0.0, rootExcited * std::sqrt(gamma), 0.0}}};

  const auto proposal = buildKrausProposal(operators);
  ASSERT_EQ(proposal.dimension, 2);
  ASSERT_EQ(proposal.probabilities.size(), 4);
  EXPECT_NEAR(proposal.probabilities[0],
              excitedWeight * 0.0 + (1.0 - excitedWeight) * (1.0 - gamma / 2.0),
              1e-14);
  EXPECT_NEAR(proposal.probabilities[1], (1.0 - excitedWeight) * gamma / 2.0,
              1e-14);
  EXPECT_NEAR(proposal.probabilities[2], excitedWeight * (1.0 - gamma / 2.0),
              1e-14);
  EXPECT_NEAR(proposal.probabilities[3], excitedWeight * gamma / 2.0, 1e-14);
}

CUDAQ_TEST(ImportanceProposalTest, AmplitudeDampingJumpIsGammaOverTwo) {
  const auto proposal = buildKrausProposal(amplitudeDamping(0.125));
  ASSERT_EQ(proposal.probabilities.size(), 2);
  EXPECT_NEAR(proposal.probabilities[1], 0.0625, 1e-15);
  EXPECT_NEAR(proposal.probabilities[0], 0.9375, 1e-15);
}

CUDAQ_TEST(ImportanceProposalTest, NumericalProposalUsesOneNormalizedVector) {
  const double gamma = 3.999880003599892e-05;
  const double excitedWeight = 0.31675049748507544;
  const auto rootExcited = std::sqrt(excitedWeight);
  const auto rootGround = std::sqrt(1.0 - excitedWeight);
  const std::vector<kraus_op> operators = {
      {{rootGround, 0.0, 0.0, rootGround * std::sqrt(1.0 - gamma)}},
      {{0.0, rootGround * std::sqrt(gamma), 0.0, 0.0}},
      {{rootExcited * std::sqrt(1.0 - gamma), 0.0, 0.0, rootExcited}},
      {{0.0, 0.0, rootExcited * std::sqrt(gamma), 0.0}}};
  const auto proposal = buildKrausProposal(operators);
  EXPECT_DOUBLE_EQ(std::accumulate(proposal.probabilities.begin(),
                                   proposal.probabilities.end(), 0.0),
                   1.0);

  const std::vector<std::complex<double>> state{{0.0, 0.0}, {1.0, 0.0}};
  double reconstructedMass = 0.0;
  for (std::size_t index = 0; index < operators.size(); ++index) {
    const auto selected = applyScaledKraus(state, operators[index],
                                           proposal.probabilities[index]);
    reconstructedMass += proposal.probabilities[index] * selected.normSquared;
  }
  EXPECT_NEAR(reconstructedMass, 1.0, 1e-14);

  const auto withinTolerance = std::sqrt(1.0 - 5e-11);
  const std::vector<kraus_op> numericallyComplete = {
      {{withinTolerance, 0.0, 0.0, withinTolerance}}};
  const auto toleranceProposal = buildKrausProposal(numericallyComplete);
  ASSERT_EQ(toleranceProposal.probabilities.size(), 1);
  EXPECT_DOUBLE_EQ(toleranceProposal.probabilities[0], 1.0);
}

CUDAQ_TEST(ImportanceProposalTest, PreservesFullSupportAndOriginalIndices) {
  const double tiny = 1e-150;
  std::vector<kraus_op> operators = {
      {{1.0, 0.0, 0.0, 1.0}}, {{0.0, 0.0, 0.0, 0.0}}, {{0.0, tiny, tiny, 0.0}}};

  const auto proposal = buildKrausProposal(operators);
  ASSERT_EQ(proposal.probabilities.size(), 2);
  EXPECT_EQ(proposal.originalBranchIndices, (std::vector<std::size_t>{0, 2}));
  EXPECT_GT(proposal.probabilities[1], 0.0);
  EXPECT_EQ(proposal.probabilities[1], tiny * tiny);
}

CUDAQ_TEST(ImportanceProposalTest, RejectsNonzeroSupportBelowFp64Range) {
  const double belowSquaredFp64 = 1e-200;
  const std::vector<kraus_op> operators = {
      {{1.0, 0.0, 0.0, 1.0}}, {{0.0, belowSquaredFp64, belowSquaredFp64, 0.0}}};
  EXPECT_THROW(buildKrausProposal(operators), std::invalid_argument);
}

CUDAQ_TEST(ImportanceProposalTest, RejectsInvalidAndIncompleteChannels) {
  std::vector<kraus_op> incomplete = {
      {{std::sqrt(0.9), 0.0, 0.0, std::sqrt(0.9)}}};
  EXPECT_THROW(buildKrausProposal(incomplete), std::invalid_argument);

  auto nonFinite = amplitudeDamping(0.1);
  nonFinite[0].data[0] = std::numeric_limits<cudaq::real>::quiet_NaN();
  EXPECT_THROW(buildKrausProposal(nonFinite), std::invalid_argument);

  std::vector<kraus_op> empty;
  EXPECT_THROW(buildKrausProposal(empty), std::invalid_argument);
}

CUDAQ_TEST(ImportanceProposalTest, FixedSeedPathUsesStableHistoryKeys) {
  const auto proposal = buildKrausProposal(amplitudeDamping(0.4));
  std::vector<std::size_t> firstPath;
  std::vector<std::size_t> secondPath;
  std::uint64_t firstHistory = 17;
  std::uint64_t secondHistory = 17;
  for (std::uint64_t site = 0; site < 16; ++site) {
    const ImportanceRngKey firstKey{
        20260701, 3, firstHistory, site, ImportanceDrawPurpose::KrausLabel, 0};
    const ImportanceRngKey secondKey{
        20260701, 3, secondHistory, site, ImportanceDrawPurpose::KrausLabel, 0};
    const auto first = drawCategorical(proposal.probabilities, firstKey);
    const auto second = drawCategorical(proposal.probabilities, secondKey);
    firstPath.push_back(first);
    secondPath.push_back(second);
    firstHistory = deriveHistoryId(3, firstHistory, site, 2, first);
    secondHistory = deriveHistoryId(3, secondHistory, site, 2, second);
  }

  EXPECT_EQ(firstPath, secondPath);
  EXPECT_EQ(firstHistory, secondHistory);
  EXPECT_EQ(firstPath, (std::vector<std::size_t>{1, 0, 0, 1, 0, 1, 0, 1, 0, 0,
                                                 0, 0, 0, 0, 0, 1}));
  EXPECT_EQ(firstHistory, 8871991410005924073ULL);
  EXPECT_NE(counterUniform(
                {20260701, 3, 17, 0, ImportanceDrawPurpose::KrausLabel, 0}),
            counterUniform(
                {20260701, 3, 17, 0, ImportanceDrawPurpose::KrausLabel, 1}));
}

CUDAQ_TEST(ImportancePopulationTest, FixedProposalSplitsConserveMultiplicity) {
  const std::vector<double> probabilities{0.125, 0.25, 0.625};
  const std::vector<std::size_t> originalIndices{0, 2, 5};
  const ImportanceRngKey key{
      20260703, 4, 981, 7, ImportanceDrawPurpose::KrausLabel, 0};
  const auto first =
      splitMultiplicity(1000, probabilities, originalIndices, key, 2);
  const auto second =
      splitMultiplicity(1000, probabilities, originalIndices, key, 2);

  ASSERT_EQ(first.size(), 3);
  EXPECT_EQ(first[0].originalBranchIndex, 0);
  EXPECT_EQ(first[1].originalBranchIndex, 2);
  EXPECT_EQ(first[2].originalBranchIndex, 5);
  EXPECT_EQ(first[0].multiplicity + first[1].multiplicity +
                first[2].multiplicity,
            1000);
  EXPECT_EQ(first[0].multiplicity, second[0].multiplicity);
  EXPECT_EQ(first[1].multiplicity, second[1].multiplicity);
  EXPECT_EQ(first[2].multiplicity, second[2].multiplicity);
  EXPECT_EQ(first[0].childHistoryId, second[0].childHistoryId);
  EXPECT_EQ(
      (std::vector<std::uint64_t>{first[0].multiplicity, first[1].multiplicity,
                                  first[2].multiplicity}),
      (std::vector<std::uint64_t>{132, 247, 621}));
}

CUDAQ_TEST(ImportancePopulationTest, SplitIsIndependentOfParentVisitOrder) {
  const std::vector<double> probabilities{0.4, 0.6};
  const std::vector<std::size_t> indices{1, 3};
  const std::vector<std::uint64_t> parentOrder{91, 7, 2001, 44};
  std::map<std::uint64_t, std::vector<std::uint64_t>> forward;
  std::map<std::uint64_t, std::vector<std::uint64_t>> reverse;

  for (const auto parent : parentOrder) {
    const ImportanceRngKey key{
        20260704, 2, parent, 9, ImportanceDrawPurpose::KrausLabel, 0};
    for (const auto &branch :
         splitMultiplicity(37, probabilities, indices, key, 2))
      forward[parent].push_back(branch.multiplicity);
  }
  for (auto iter = parentOrder.rbegin(); iter != parentOrder.rend(); ++iter) {
    const ImportanceRngKey key{
        20260704, 2, *iter, 9, ImportanceDrawPurpose::KrausLabel, 0};
    for (const auto &branch :
         splitMultiplicity(37, probabilities, indices, key, 2))
      reverse[*iter].push_back(branch.multiplicity);
  }
  EXPECT_EQ(forward, reverse);
}

CUDAQ_TEST(ImportancePopulationTest, AggregatesLogMassInRecordOrder) {
  const std::vector<WeightedRecord> records = {{"11", 3, std::log(2.0)},
                                               {"00", 2, std::log(0.5)},
                                               {"11", 1, std::log(4.0)},
                                               {"01", 5, -1000.0}};
  const auto bins = aggregateLogMass(records);
  ASSERT_EQ(bins.size(), 3);
  EXPECT_EQ(bins[0].record, "00");
  EXPECT_EQ(bins[1].record, "01");
  EXPECT_EQ(bins[2].record, "11");
  EXPECT_NEAR(std::exp(bins[0].logMass), 1.0, 1e-14);
  EXPECT_NEAR(std::exp(bins[2].logMass), 10.0, 1e-13);
  EXPECT_DOUBLE_EQ(bins[1].logMass, std::log(5.0) - 1000.0);
}

CUDAQ_TEST(ImportancePopulationTest,
           DuplicateLogMassIsInvariantToInsertionOrder) {
  const std::vector<WeightedRecord> forward = {
      {"0", 1, 82.97779400064942},  {"0", 1, -24.472015375703165},
      {"0", 1, 75.25015060976153},  {"0", 1, -68.47165565883344},
      {"0", 1, 75.268956088263},    {"0", 1, 0.0031019879942},
      {"0", 1, -104.6671289184501}, {"1", 1, -1.0}};
  const std::vector<WeightedRecord> reverse(forward.rbegin(), forward.rend());
  const auto first = aggregateLogMass(forward);
  const auto second = aggregateLogMass(reverse);
  ASSERT_EQ(first.size(), second.size());
  for (std::size_t index = 0; index < first.size(); ++index) {
    EXPECT_EQ(first[index].record, second[index].record);
    EXPECT_DOUBLE_EQ(first[index].logMass, second[index].logMass);
  }
}

CUDAQ_TEST(ImportancePopulationTest, EssUsesPerParticleMultiplicity) {
  const std::vector<WeightedRecord> records = {{"0", 2, std::log(1.0)},
                                               {"1", 3, std::log(2.0)}};
  const auto diagnostics = computeWeightDiagnostics(records);
  EXPECT_EQ(diagnostics.representedParticles, 5);
  EXPECT_NEAR(std::exp(diagnostics.logSumWeights), 8.0, 1e-14);
  EXPECT_NEAR(std::exp(diagnostics.logSumSquaredWeights), 14.0, 1e-14);
  EXPECT_NEAR(diagnostics.effectiveSampleSize, 64.0 / 14.0, 1e-14);
}

CUDAQ_TEST(ImportancePopulationTest, EssIsInvariantToLargeCommonLogShift) {
  const std::vector<WeightedRecord> baseline = {{"0", 1, 0.0}, {"1", 1, 0.0}};
  const std::vector<WeightedRecord> shifted = {{"0", 1, 1e16}, {"1", 1, 1e16}};
  EXPECT_DOUBLE_EQ(computeWeightDiagnostics(baseline).effectiveSampleSize, 2.0);
  EXPECT_DOUBLE_EQ(computeWeightDiagnostics(shifted).effectiveSampleSize, 2.0);
}

CUDAQ_TEST(ImportancePopulationTest, RejectsOutOfRangeLogDiagnostics) {
  const std::vector<WeightedRecord> records = {{"0", 1, -1e308}};
  EXPECT_THROW(computeWeightDiagnostics(records), std::invalid_argument);
}

CUDAQ_TEST(ImportancePopulationTest, RejectsInvalidPopulationInputs) {
  const std::vector<double> invalidProbabilities{
      0.5, std::numeric_limits<double>::infinity()};
  const std::vector<std::size_t> indices{0, 1};
  EXPECT_THROW(
      splitMultiplicity(4, invalidProbabilities, indices,
                        {1, 0, 0, 0, ImportanceDrawPurpose::KrausLabel, 0}, 2),
      std::invalid_argument);
  EXPECT_THROW(
      splitMultiplicity(0, invalidProbabilities, indices,
                        {1, 0, 0, 0, ImportanceDrawPurpose::KrausLabel, 0}, 2),
      std::invalid_argument);

  const std::vector<WeightedRecord> zeroMultiplicity{{"0", 0, 0.0}};
  EXPECT_THROW(aggregateLogMass(zeroMultiplicity), std::invalid_argument);
  const std::vector<WeightedRecord> nonFiniteWeight{
      {"0", 1, std::numeric_limits<double>::infinity()}};
  EXPECT_THROW(computeWeightDiagnostics(nonFiniteWeight),
               std::invalid_argument);
}

static std::uint64_t countTotal(const std::vector<CountBin> &counts) {
  std::uint64_t total = 0;
  for (const auto &bin : counts)
    total += bin.count;
  return total;
}

static std::vector<CountBin>
directResidualMultinomial(const std::vector<LogMassBin> &bins,
                          std::uint64_t shots, std::uint64_t seed) {
  std::vector<double> probabilities;
  double total = 0.0;
  for (const auto &bin : bins) {
    probabilities.push_back(std::exp(bin.logMass));
    total += probabilities.back();
  }
  for (auto &probability : probabilities)
    probability /= total;
  const auto largest =
      std::max_element(probabilities.begin(), probabilities.end());
  *largest +=
      1.0 - std::accumulate(probabilities.begin(), probabilities.end(), 0.0);

  std::vector<CountBin> counts;
  std::vector<double> residuals;
  std::uint64_t deterministicTotal = 0;
  for (std::size_t index = 0; index < bins.size(); ++index) {
    const auto expected = static_cast<double>(shots) * probabilities[index];
    const auto deterministic = static_cast<std::uint64_t>(std::floor(expected));
    counts.push_back({bins[index].record, deterministic});
    residuals.push_back(expected - deterministic);
    deterministicTotal += deterministic;
  }
  for (std::uint64_t ordinal = 0; ordinal < shots - deterministicTotal;
       ++ordinal) {
    const ImportanceRngKey key{
        seed, 0, 0, 0, ImportanceDrawPurpose::FinalAllocation, ordinal};
    ++counts[drawCategorical(residuals, key)].count;
  }
  return counts;
}

CUDAQ_TEST(ImportanceAllocatorTest, EveryAllocatorConservesRequestedCount) {
  const std::vector<LogMassBin> bins = {{"11", std::log(0.1)},
                                        {"00", std::log(0.2)},
                                        {"10", std::log(0.3)},
                                        {"01", std::log(0.4)}};
  const std::vector<std::vector<std::uint64_t>> golden = {
      {202, 394, 302, 105}, {200, 401, 302, 100}, {200, 401, 301, 101}};
  std::size_t goldenIndex = 0;
  for (const auto resampler :
       {FinalResampler::Multinomial, FinalResampler::Residual,
        FinalResampler::ResidualStratified}) {
    const auto counts = allocateCounts(bins, 1003, resampler, 20260705);
    EXPECT_EQ(countTotal(counts), 1003);
    ASSERT_EQ(counts.size(), 4);
    EXPECT_EQ(counts[0].record, "00");
    EXPECT_EQ(counts[1].record, "01");
    EXPECT_EQ(counts[2].record, "10");
    EXPECT_EQ(counts[3].record, "11");
    EXPECT_EQ((std::vector<std::uint64_t>{counts[0].count, counts[1].count,
                                          counts[2].count, counts[3].count}),
              golden[goldenIndex]);
    ++goldenIndex;
  }
}

CUDAQ_TEST(ImportanceAllocatorTest, UnitWeightsPreserveInputIntegerHistogram) {
  const std::vector<WeightedRecord> records = {
      {"11", 3, 0.0}, {"00", 7, 0.0}, {"11", 2, 0.0}};
  for (const auto resampler :
       {FinalResampler::Multinomial, FinalResampler::Residual,
        FinalResampler::ResidualStratified}) {
    const auto counts =
        materializeWeightedRecords(records, 12, resampler, 20260706);
    ASSERT_EQ(counts.size(), 2);
    EXPECT_EQ(counts[0].record, "00");
    EXPECT_EQ(counts[0].count, 7);
    EXPECT_EQ(counts[1].record, "11");
    EXPECT_EQ(counts[1].count, 5);
  }
}

CUDAQ_TEST(ImportanceAllocatorTest, FixedSeedAndInsertionOrderAreStable) {
  const std::vector<LogMassBin> forward = {{"00", std::log(1.25)},
                                           {"01", std::log(2.5)},
                                           {"10", std::log(3.75)},
                                           {"11", std::log(0.125)}};
  const std::vector<LogMassBin> reverse(forward.rbegin(), forward.rend());
  for (const auto resampler :
       {FinalResampler::Multinomial, FinalResampler::Residual,
        FinalResampler::ResidualStratified})
    EXPECT_EQ(allocateCounts(forward, 137, resampler, 20260707),
              allocateCounts(reverse, 137, resampler, 20260707));
}

CUDAQ_TEST(ImportanceAllocatorTest, CoversOneBinZeroRemainderAndTinyResiduals) {
  const std::vector<LogMassBin> oneBin{{"101", 0.0}};
  for (const auto resampler :
       {FinalResampler::Multinomial, FinalResampler::Residual,
        FinalResampler::ResidualStratified}) {
    const auto counts = allocateCounts(oneBin, 19, resampler, 9);
    ASSERT_EQ(counts.size(), 1);
    EXPECT_EQ(counts[0].count, 19);
  }

  const std::vector<LogMassBin> exact = {{"0", std::log(0.25)},
                                         {"1", std::log(0.75)}};
  const auto noRemainder =
      allocateCounts(exact, 100, FinalResampler::ResidualStratified, 11);
  EXPECT_EQ(noRemainder[0].count, 25);
  EXPECT_EQ(noRemainder[1].count, 75);

  const std::vector<LogMassBin> tiny = {
      {"00", 0.0}, {"01", std::log(1e-200)}, {"10", std::log(2e-200)}};
  EXPECT_EQ(countTotal(allocateCounts(tiny, 17,
                                      FinalResampler::ResidualStratified, 13)),
            17);
}

CUDAQ_TEST(ImportanceAllocatorTest,
           HandlesMaximumShotCountWithoutConversionOverflow) {
  const std::vector<LogMassBin> oneBin{{"0", 0.0}};
  const auto counts =
      allocateCounts(oneBin, std::numeric_limits<std::uint64_t>::max(),
                     FinalResampler::Residual, 17);
  ASSERT_EQ(counts.size(), 1);
  EXPECT_EQ(counts[0].count, std::numeric_limits<std::uint64_t>::max());
}

CUDAQ_TEST(ImportanceAllocatorTest, ResidualUsesDirectFloorWithoutSnapping) {
  constexpr std::uint64_t shots = 100000000;
  const std::vector<LogMassBin> bins = {
      {"00", std::log((10000000.0 - 1e-8) / shots)},
      {"01", std::log((20000000.0 + 0.25) / shots)},
      {"10", std::log((70000000.0 - 0.24999999) / shots)}};
  for (std::uint64_t seed = 1; seed <= 64; ++seed)
    EXPECT_EQ(allocateCounts(bins, shots, FinalResampler::Residual, seed),
              directResidualMultinomial(bins, shots, seed))
        << "seed=" << seed;
}

CUDAQ_TEST(ImportanceAllocatorTest,
           LargeResidualHistogramConservesRequestedCount) {
  std::vector<LogMassBin> bins;
  bins.reserve(10000);
  for (std::size_t index = 0; index < 10000; ++index)
    bins.push_back({std::to_string(index), 0.0});

  for (const auto resampler :
       {FinalResampler::Residual, FinalResampler::ResidualStratified})
    EXPECT_EQ(countTotal(allocateCounts(bins, 10000, resampler, 20260702)),
              10000);
}

CUDAQ_TEST(ImportanceAllocatorTest, SequentialShuffleMatchesAggregateCounts) {
  const std::vector<CountBin> counts = {{"11", 3}, {"00", 5}, {"01", 2}};
  const auto first = makeSequentialData(counts, 20260708);
  const auto second = makeSequentialData(counts, 20260708);
  EXPECT_EQ(first, second);
  EXPECT_EQ(first, (std::vector<std::string>{"00", "11", "00", "00", "01", "00",
                                             "11", "11", "01", "00"}));
  const std::vector<CountBin> reverseCounts(counts.rbegin(), counts.rend());
  EXPECT_EQ(first, makeSequentialData(reverseCounts, 20260708));
  EXPECT_NE(first, (std::vector<std::string>{"00", "00", "00", "00", "00", "01",
                                             "01", "11", "11", "11"}));
  ASSERT_EQ(first.size(), 10);
  std::map<std::string, std::uint64_t> histogram;
  for (const auto &record : first)
    ++histogram[record];
  EXPECT_EQ(histogram["00"], 5);
  EXPECT_EQ(histogram["01"], 2);
  EXPECT_EQ(histogram["11"], 3);
}

CUDAQ_TEST(ImportanceAllocatorTest, RejectsZeroMassAndNonFiniteInputs) {
  const std::vector<LogMassBin> empty;
  EXPECT_THROW(allocateCounts(empty, 1, FinalResampler::Multinomial, 1),
               std::invalid_argument);
  const std::vector<LogMassBin> zeroMass{
      {"0", -std::numeric_limits<double>::infinity()}};
  EXPECT_THROW(allocateCounts(zeroMass, 1, FinalResampler::Residual, 1),
               std::invalid_argument);
  const std::vector<LogMassBin> nanMass{
      {"0", std::numeric_limits<double>::quiet_NaN()}};
  EXPECT_THROW(
      allocateCounts(nanMass, 1, FinalResampler::ResidualStratified, 1),
      std::invalid_argument);
}

static double probabilityOne(const std::vector<std::complex<double>> &state) {
  return std::norm(state[1]);
}

static std::vector<std::complex<double>>
collapseOne(const std::vector<std::complex<double>> &state, std::size_t outcome,
            bool reset) {
  const auto probability =
      outcome == 0 ? std::norm(state[0]) : std::norm(state[1]);
  std::vector<std::complex<double>> collapsed(2);
  collapsed[outcome] = state[outcome] / std::sqrt(probability);
  if (reset && outcome == 1)
    std::swap(collapsed[0], collapsed[1]);
  return collapsed;
}

static std::vector<std::complex<double>>
applyRaw(const kraus_op &op, const std::vector<std::complex<double>> &state) {
  std::vector<std::complex<double>> result(op.nRows);
  for (std::size_t row = 0; row < op.nRows; ++row)
    for (std::size_t column = 0; column < op.nCols; ++column)
      result[row] +=
          std::complex<double>(op.data[row * op.nCols + column].real(),
                               op.data[row * op.nCols + column].imag()) *
          state[column];
  return result;
}

CUDAQ_TEST(ImportanceOracleTest, ScaledOperatorCarriesLikelihoodRatio) {
  const double gamma = 0.3;
  const auto operators = amplitudeDamping(gamma);
  const auto proposal = buildKrausProposal(operators);
  const std::vector<std::complex<double>> excited{{0.0, 0.0}, {1.0, 0.0}};

  const auto noJump =
      applyScaledKraus(excited, operators[0], proposal.probabilities[0]);
  EXPECT_NEAR(noJump.normSquared, (1.0 - gamma) / (1.0 - gamma / 2.0), 1e-14);
  EXPECT_NEAR(noJump.logWeightIncrement, std::log(noJump.normSquared), 1e-14);
  EXPECT_NEAR(std::norm(noJump.normalizedState[1]), 1.0, 1e-14);

  const auto jump =
      applyScaledKraus(excited, operators[1], proposal.probabilities[1]);
  EXPECT_NEAR(jump.normSquared, 2.0, 1e-14);
  EXPECT_NEAR(std::norm(jump.normalizedState[0]), 1.0, 1e-14);
}

CUDAQ_TEST(ImportanceOracleTest,
           ForcedSixteenSiteSegmentMatchesSiteNormalization) {
  constexpr double gamma = 0.05;
  constexpr std::size_t sites = 16;
  const auto operators = amplitudeDamping(gamma);
  const auto proposal = buildKrausProposal(operators);
  const double inverseRootTwo = 1.0 / std::sqrt(2.0);
  const std::vector<std::complex<double>> initial{{inverseRootTwo, 0.0},
                                                  {0.0, inverseRootTwo}};

  auto siteState = initial;
  double siteLogWeight = 0.0;
  for (std::size_t site = 0; site < sites; ++site) {
    const auto applied =
        applyScaledKraus(siteState, operators[0], proposal.probabilities[0]);
    siteState = applied.normalizedState;
    siteLogWeight += applied.logWeightIncrement;
  }

  auto segmentState = initial;
  const double scale = 1.0 / std::sqrt(proposal.probabilities[0]);
  for (std::size_t site = 0; site < sites; ++site) {
    segmentState = applyRaw(operators[0], segmentState);
    for (auto &amplitude : segmentState)
      amplitude *= scale;
  }
  const double segmentNormSquared =
      std::norm(segmentState[0]) + std::norm(segmentState[1]);
  const double segmentLogWeight = std::log(segmentNormSquared);
  for (auto &amplitude : segmentState)
    amplitude /= std::sqrt(segmentNormSquared);

  const auto overlap = std::conj(siteState[0]) * segmentState[0] +
                       std::conj(siteState[1]) * segmentState[1];
  const double infidelity = 1.0 - std::norm(overlap);
  EXPECT_NEAR(siteLogWeight, segmentLogWeight, 1e-10);
  EXPECT_LE(std::abs(infidelity), 1e-12);
  EXPECT_NEAR(probabilityOne(siteState), probabilityOne(segmentState), 1e-10);
}

CUDAQ_TEST(ImportanceOracleTest, InterleavedKrausMcmResetKrausMatchesOracle) {
  const double gamma = 0.2;
  const auto operators = amplitudeDamping(gamma);
  const auto proposal = buildKrausProposal(operators);
  const double inverseRootTwo = 1.0 / std::sqrt(2.0);
  const std::vector<std::complex<double>> initial{{inverseRootTwo, 0.0},
                                                  {inverseRootTwo, 0.0}};

  for (std::size_t first = 0; first < operators.size(); ++first) {
    const auto firstState = applyScaledKraus(initial, operators[first],
                                             proposal.probabilities[first]);
    for (std::size_t outcome = 0; outcome < 2; ++outcome) {
      const auto measurementProbability =
          outcome == 0 ? 1.0 - probabilityOne(firstState.normalizedState)
                       : probabilityOne(firstState.normalizedState);
      if (measurementProbability == 0.0)
        continue;
      const auto collapsed =
          collapseOne(firstState.normalizedState, outcome, true);
      for (std::size_t second = 0; second < operators.size(); ++second) {
        const auto secondState = applyScaledKraus(
            collapsed, operators[second], proposal.probabilities[second]);
        const auto proposedMass = proposal.probabilities[first] *
                                  measurementProbability *
                                  proposal.probabilities[second] *
                                  std::exp(firstState.logWeightIncrement +
                                           secondState.logWeightIncrement);

        auto rawFirst = applyRaw(operators[first], initial);
        rawFirst[outcome == 0 ? 1 : 0] = 0.0;
        if (outcome == 1)
          std::swap(rawFirst[0], rawFirst[1]);
        const auto rawSecond = applyRaw(operators[second], rawFirst);
        const auto exactMass =
            std::norm(rawSecond[0]) + std::norm(rawSecond[1]);
        EXPECT_NEAR(proposedMass, exactMass, 1e-13)
            << "first=" << first << " outcome=" << outcome
            << " second=" << second;
      }
    }
  }
}
