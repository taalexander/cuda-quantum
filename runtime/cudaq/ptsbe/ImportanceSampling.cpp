/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ImportanceSampling.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string_view>

static std::uint64_t mix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

static std::uint64_t combineKey(std::uint64_t state, std::uint64_t value) {
  return mix64(state ^ mix64(value));
}

static double logAdd(double left, double right) {
  if (left == -std::numeric_limits<double>::infinity())
    return right;
  if (right == -std::numeric_limits<double>::infinity())
    return left;
  const auto maximum = std::max(left, right);
  return maximum +
         std::log(std::exp(left - maximum) + std::exp(right - maximum));
}

static double stableLogSum(std::vector<double> contributions) {
  if (contributions.empty())
    return -std::numeric_limits<double>::infinity();
  std::sort(contributions.begin(), contributions.end());
  double total = -std::numeric_limits<double>::infinity();
  for (const auto contribution : contributions)
    total = logAdd(total, contribution);
  return total;
}

static void
validateWeightedRecord(const cudaq::ptsbe::detail::WeightedRecord &record) {
  if (record.multiplicity == 0)
    throw std::invalid_argument(
        "weighted-record multiplicity must be positive");
  if (!std::isfinite(record.logWeight))
    throw std::invalid_argument("weighted-record log weight must be finite");
}

static double
validateCategoricalProbabilities(std::span<const double> probabilities) {
  if (probabilities.empty())
    throw std::invalid_argument("categorical probabilities must not be empty");
  double total = 0.0;
  for (const auto probability : probabilities) {
    if (!std::isfinite(probability) || probability < 0.0)
      throw std::invalid_argument(
          "categorical probabilities must be finite and nonnegative");
    total += probability;
  }
  if (!std::isfinite(total) || total <= 0.0)
    throw std::invalid_argument(
        "categorical probability mass must be finite and positive");
  return total;
}

static void normalizeProbabilities(std::vector<double> &probabilities) {
  const auto total = validateCategoricalProbabilities(probabilities);
  for (auto &probability : probabilities)
    probability /= total;
  const auto largest =
      std::max_element(probabilities.begin(), probabilities.end());
  *largest +=
      1.0 - std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
  if (!std::isfinite(*largest) || *largest <= 0.0)
    throw std::invalid_argument(
        "normalizing probabilities lost positive support");
}

cudaq::ptsbe::detail::KrausProposal cudaq::ptsbe::detail::buildKrausProposal(
    std::span<const cudaq::kraus_op> operators, double completenessTolerance) {
  if (operators.empty())
    throw std::invalid_argument("a Kraus channel must contain an operator");
  if (!std::isfinite(completenessTolerance) || completenessTolerance < 0.0)
    throw std::invalid_argument(
        "completeness tolerance must be finite and nonnegative");

  const auto dimension = operators.front().nRows;
  if (dimension == 0 || operators.front().nCols != dimension)
    throw std::invalid_argument("Kraus operators must be nonempty and square");

  std::vector<std::complex<long double>> completeness(dimension * dimension);
  KrausProposal proposal;
  proposal.dimension = dimension;
  for (std::size_t operatorIndex = 0; operatorIndex < operators.size();
       ++operatorIndex) {
    const auto &op = operators[operatorIndex];
    if (op.nRows != dimension || op.nCols != dimension ||
        op.data.size() != dimension * dimension)
      throw std::invalid_argument(
          "Kraus operators must have one common square dimension");

    long double trace = 0.0L;
    bool hasNonzeroEntry = false;
    for (const auto &value : op.data) {
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument(
            "Kraus operators must contain finite values");
      hasNonzeroEntry =
          hasNonzeroEntry || value.real() != 0.0 || value.imag() != 0.0;
      const auto real = static_cast<long double>(value.real());
      const auto imaginary = static_cast<long double>(value.imag());
      trace += real * real + imaginary * imaginary;
    }

    for (std::size_t row = 0; row < dimension; ++row)
      for (std::size_t column = 0; column < dimension; ++column)
        for (std::size_t inner = 0; inner < dimension; ++inner) {
          const auto left = op.data[inner * dimension + row];
          const auto right = op.data[inner * dimension + column];
          completeness[row * dimension + column] +=
              std::conj(std::complex<long double>(left.real(), left.imag())) *
              std::complex<long double>(right.real(), right.imag());
        }

    if (!hasNonzeroEntry)
      continue;
    const auto probability = static_cast<double>(trace / dimension);
    if (!std::isfinite(probability) || probability <= 0.0)
      throw std::invalid_argument(
          "nonzero Kraus operators require finite positive proposal mass");
    proposal.probabilities.push_back(probability);
    proposal.originalBranchIndices.push_back(operatorIndex);
  }

  for (std::size_t row = 0; row < dimension; ++row)
    for (std::size_t column = 0; column < dimension; ++column) {
      const std::complex<long double> expected = row == column ? 1.0L : 0.0L;
      if (std::abs(completeness[row * dimension + column] - expected) >
          completenessTolerance)
        throw std::invalid_argument(
            "Kraus operators do not satisfy the completeness relation");
    }
  if (proposal.probabilities.empty())
    throw std::invalid_argument("a Kraus channel must have nonzero support");
  normalizeProbabilities(proposal.probabilities);
  return proposal;
}

std::uint64_t cudaq::ptsbe::detail::deriveHistoryId(
    std::uint64_t wave, std::uint64_t parentHistoryId, std::uint64_t traceSite,
    std::uint64_t branchKind, std::uint64_t outcome) {
  auto state = mix64(0x5054534245484953ULL);
  state = combineKey(state, wave);
  state = combineKey(state, parentHistoryId);
  state = combineKey(state, traceSite);
  state = combineKey(state, branchKind);
  return combineKey(state, outcome);
}

double cudaq::ptsbe::detail::counterUniform(const ImportanceRngKey &key) {
  auto state = mix64(0x494d5052544e4753ULL);
  state = combineKey(state, key.seed);
  state = combineKey(state, key.wave);
  state = combineKey(state, key.historyId);
  state = combineKey(state, key.traceSite);
  state = combineKey(state, static_cast<std::uint64_t>(key.purpose));
  state = combineKey(state, key.drawOrdinal);
  return static_cast<double>(state >> 11) * 0x1.0p-53;
}

std::size_t
cudaq::ptsbe::detail::drawCategorical(std::span<const double> probabilities,
                                      const ImportanceRngKey &key) {
  const auto total = validateCategoricalProbabilities(probabilities);
  const auto target = counterUniform(key) * total;
  double cumulative = 0.0;
  for (std::size_t index = 0; index < probabilities.size(); ++index) {
    cumulative += probabilities[index];
    if (target < cumulative)
      return index;
  }
  return probabilities.size() - 1;
}

std::vector<cudaq::ptsbe::detail::CountedBranch>
cudaq::ptsbe::detail::splitMultiplicity(
    std::uint64_t multiplicity, std::span<const double> probabilities,
    std::span<const std::size_t> originalBranchIndices,
    const ImportanceRngKey &baseKey, std::uint64_t branchKind) {
  if (probabilities.size() != originalBranchIndices.size())
    throw std::invalid_argument(
        "proposal probabilities and branch indices must have equal size");
  validateCategoricalProbabilities(probabilities);
  if (multiplicity >
      std::numeric_limits<std::uint64_t>::max() - baseKey.drawOrdinal)
    throw std::invalid_argument("proposal draw ordinal would overflow");

  std::vector<std::uint64_t> counts(probabilities.size());
  for (std::uint64_t ordinal = 0; ordinal < multiplicity; ++ordinal) {
    auto key = baseKey;
    key.drawOrdinal += ordinal;
    ++counts[drawCategorical(probabilities, key)];
  }

  std::vector<CountedBranch> branches;
  branches.reserve(probabilities.size());
  for (std::size_t index = 0; index < probabilities.size(); ++index) {
    if (counts[index] == 0)
      continue;
    const auto originalIndex = originalBranchIndices[index];
    branches.push_back(
        {originalIndex, counts[index],
         deriveHistoryId(baseKey.wave, baseKey.historyId, baseKey.traceSite,
                         branchKind, originalIndex)});
  }
  return branches;
}

std::vector<cudaq::ptsbe::detail::LogMassBin>
cudaq::ptsbe::detail::aggregateLogMass(
    std::span<const WeightedRecord> records) {
  std::map<std::string, std::vector<double>> contributionsByRecord;
  for (const auto &record : records) {
    validateWeightedRecord(record);
    const auto contribution =
        std::log(static_cast<double>(record.multiplicity)) + record.logWeight;
    contributionsByRecord[record.record].push_back(contribution);
  }

  std::vector<LogMassBin> bins;
  bins.reserve(contributionsByRecord.size());
  for (auto &[record, contributions] : contributionsByRecord)
    bins.push_back({std::move(record), stableLogSum(std::move(contributions))});
  return bins;
}

cudaq::ptsbe::detail::WeightDiagnostics
cudaq::ptsbe::detail::computeWeightDiagnostics(
    std::span<const WeightedRecord> records) {
  if (records.empty())
    throw std::invalid_argument(
        "weight diagnostics require at least one record");
  WeightDiagnostics diagnostics;
  double maximumLogWeight = -std::numeric_limits<double>::infinity();
  for (const auto &record : records) {
    validateWeightedRecord(record);
    if (record.multiplicity > std::numeric_limits<std::uint64_t>::max() -
                                  diagnostics.representedParticles)
      throw std::invalid_argument("represented-particle count would overflow");
    diagnostics.representedParticles += record.multiplicity;
    maximumLogWeight = std::max(maximumLogWeight, record.logWeight);
  }
  long double scaledSum = 0.0L;
  long double scaledSquaredSum = 0.0L;
  for (const auto &record : records) {
    const auto shifted = static_cast<long double>(record.logWeight) -
                         static_cast<long double>(maximumLogWeight);
    const auto scaledWeight = std::exp(shifted);
    scaledSum += static_cast<long double>(record.multiplicity) * scaledWeight;
    scaledSquaredSum += static_cast<long double>(record.multiplicity) *
                        scaledWeight * scaledWeight;
  }
  if (!(scaledSum > 0.0L) || !(scaledSquaredSum > 0.0L))
    throw std::invalid_argument("scaled weight sums must be positive");
  if (maximumLogWeight > std::numeric_limits<double>::max() / 2.0)
    throw std::invalid_argument(
        "squared-weight log diagnostic exceeds FP64 range");
  const auto logSumWeights =
      static_cast<long double>(maximumLogWeight) + std::log(scaledSum);
  const auto logSumSquaredWeights =
      2.0L * static_cast<long double>(maximumLogWeight) +
      std::log(scaledSquaredSum);
  diagnostics.logSumWeights = static_cast<double>(logSumWeights);
  diagnostics.logSumSquaredWeights = static_cast<double>(logSumSquaredWeights);
  if (!std::isfinite(diagnostics.logSumWeights) ||
      !std::isfinite(diagnostics.logSumSquaredWeights))
    throw std::invalid_argument("weight log diagnostics exceed FP64 range");
  diagnostics.effectiveSampleSize =
      static_cast<double>(scaledSum * scaledSum / scaledSquaredSum);
  diagnostics.maximumNormalizedWeight = static_cast<double>(1.0L / scaledSum);
  if (!std::isfinite(diagnostics.effectiveSampleSize))
    throw std::invalid_argument("effective sample size is not finite");
  if (!std::isfinite(diagnostics.maximumNormalizedWeight) ||
      diagnostics.maximumNormalizedWeight <= 0.0 ||
      diagnostics.maximumNormalizedWeight > 1.0)
    throw std::invalid_argument("maximum normalized weight is outside (0, 1]");
  return diagnostics;
}

std::vector<cudaq::ptsbe::detail::CountBin>
cudaq::ptsbe::detail::allocateCounts(std::span<const LogMassBin> inputBins,
                                     std::uint64_t shots,
                                     FinalResampler resampler,
                                     std::uint64_t seed) {
  if (inputBins.empty())
    throw std::invalid_argument(
        "final allocation requires positive record mass");
  std::map<std::string, std::vector<double>> contributionsByRecord;
  for (const auto &bin : inputBins) {
    if (!std::isfinite(bin.logMass))
      throw std::invalid_argument("record log mass must be finite");
    contributionsByRecord[bin.record].push_back(bin.logMass);
  }
  std::map<std::string, double> canonicalMass;
  for (auto &[record, contributions] : contributionsByRecord)
    canonicalMass.emplace(std::move(record),
                          stableLogSum(std::move(contributions)));

  const auto maximum =
      std::max_element(canonicalMass.begin(), canonicalMass.end(),
                       [](const auto &left, const auto &right) {
                         return left.second < right.second;
                       })
          ->second;
  std::vector<double> probabilities;
  probabilities.reserve(canonicalMass.size());
  double scaledTotal = 0.0;
  for (const auto &[record, logMass] : canonicalMass) {
    const auto mass = std::exp(logMass - maximum);
    probabilities.push_back(mass);
    scaledTotal += mass;
  }
  if (!std::isfinite(scaledTotal) || scaledTotal <= 0.0)
    throw std::invalid_argument(
        "final allocation requires positive finite mass");
  for (auto &probability : probabilities)
    probability /= scaledTotal;
  normalizeProbabilities(probabilities);

  std::vector<CountBin> counts;
  counts.reserve(canonicalMass.size());
  for (const auto &[record, logMass] : canonicalMass)
    counts.push_back({record, 0});
  const auto removeZeroCounts = [&]() {
    std::erase_if(
        counts, [](const auto &bin) { return bin.count == std::uint64_t{0}; });
  };
  if (shots == 0)
    return {};

  const auto drawMultinomial = [&](std::span<const double> masses,
                                   std::uint64_t draws,
                                   std::uint64_t ordinalOffset) {
    for (std::uint64_t ordinal = 0; ordinal < draws; ++ordinal) {
      const ImportanceRngKey key{seed,
                                 0,
                                 0,
                                 0,
                                 ImportanceDrawPurpose::FinalAllocation,
                                 ordinalOffset + ordinal};
      ++counts[drawCategorical(masses, key)].count;
    }
  };

  if (resampler == FinalResampler::Multinomial) {
    drawMultinomial(probabilities, shots, 0);
    removeZeroCounts();
    return counts;
  }

  std::vector<double> residuals(probabilities.size());
  std::uint64_t deterministicTotal = 0;
  for (std::size_t index = 0; index < probabilities.size(); ++index) {
    const auto expected = static_cast<long double>(shots) *
                          static_cast<long double>(probabilities[index]);
    const auto floored = std::floor(expected);
    if (floored < 0.0L ||
        floored >
            static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
      throw std::invalid_argument(
          "deterministic allocation is outside uint64 range");
    const auto deterministic = static_cast<std::uint64_t>(floored);
    if (deterministic >
        std::numeric_limits<std::uint64_t>::max() - deterministicTotal)
      throw std::invalid_argument(
          "deterministic allocation count would overflow");
    counts[index].count = deterministic;
    deterministicTotal += deterministic;
    residuals[index] =
        static_cast<double>(expected - static_cast<long double>(deterministic));
  }
  if (deterministicTotal > shots)
    throw std::invalid_argument(
        "deterministic allocation exceeds requested count");
  const auto remainder = shots - deterministicTotal;
  if (remainder == 0) {
    removeZeroCounts();
    return counts;
  }

  const auto residualTotal =
      std::accumulate(residuals.begin(), residuals.end(), 0.0);
  if (!std::isfinite(residualTotal) || residualTotal <= 0.0)
    throw std::invalid_argument(
        "fractional residual mass must be finite and positive");

  if (resampler == FinalResampler::Residual) {
    drawMultinomial(residuals, remainder, 0);
    removeZeroCounts();
    return counts;
  }

  for (std::uint64_t stratum = 0; stratum < remainder; ++stratum) {
    const ImportanceRngKey key{
        seed, 0, 0, 0, ImportanceDrawPurpose::FinalAllocation, stratum};
    const auto point = (static_cast<double>(stratum) + counterUniform(key)) /
                       static_cast<double>(remainder) * residualTotal;
    double cumulative = 0.0;
    std::size_t selected = residuals.size() - 1;
    for (std::size_t index = 0; index < residuals.size(); ++index) {
      cumulative += residuals[index];
      if (point < cumulative) {
        selected = index;
        break;
      }
    }
    ++counts[selected].count;
  }
  removeZeroCounts();
  return counts;
}

std::vector<cudaq::ptsbe::detail::CountBin>
cudaq::ptsbe::detail::materializeWeightedRecords(
    std::span<const WeightedRecord> records, std::uint64_t shots,
    FinalResampler resampler, std::uint64_t seed) {
  if (records.empty())
    throw std::invalid_argument(
        "weighted-record materialization requires input");
  bool allUnitWeight = true;
  std::uint64_t representedParticles = 0;
  std::map<std::string, std::uint64_t> nativeHistogram;
  for (const auto &record : records) {
    validateWeightedRecord(record);
    allUnitWeight = allUnitWeight && record.logWeight == 0.0;
    if (record.multiplicity >
            std::numeric_limits<std::uint64_t>::max() - representedParticles ||
        record.multiplicity > std::numeric_limits<std::uint64_t>::max() -
                                  nativeHistogram[record.record])
      throw std::invalid_argument(
          "weighted-record multiplicity would overflow");
    representedParticles += record.multiplicity;
    nativeHistogram[record.record] += record.multiplicity;
  }
  if (allUnitWeight && representedParticles == shots) {
    std::vector<CountBin> counts;
    counts.reserve(nativeHistogram.size());
    for (auto &[record, count] : nativeHistogram)
      counts.push_back({std::move(record), count});
    return counts;
  }
  return allocateCounts(aggregateLogMass(records), shots, resampler, seed);
}

std::vector<std::string>
cudaq::ptsbe::detail::makeSequentialData(std::span<const CountBin> inputCounts,
                                         std::uint64_t seed) {
  std::map<std::string, std::uint64_t> canonicalCounts;
  std::uint64_t total = 0;
  for (const auto &bin : inputCounts) {
    if (bin.count > std::numeric_limits<std::uint64_t>::max() - total ||
        bin.count > std::numeric_limits<std::uint64_t>::max() -
                        canonicalCounts[bin.record])
      throw std::invalid_argument("sequential-data count would overflow");
    total += bin.count;
    canonicalCounts[bin.record] += bin.count;
  }
  if (total > std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument(
        "sequential-data size exceeds addressable memory");
  std::vector<std::string> records;
  records.reserve(static_cast<std::size_t>(total));
  for (const auto &[record, count] : canonicalCounts)
    for (std::uint64_t index = 0; index < count; ++index)
      records.push_back(record);

  for (std::size_t index = records.size(); index > 1; --index) {
    const ImportanceRngKey key{
        seed,
        0,
        0,
        0,
        ImportanceDrawPurpose::SequentialShuffle,
        static_cast<std::uint64_t>(records.size() - index)};
    const auto selected = static_cast<std::size_t>(counterUniform(key) *
                                                   static_cast<double>(index));
    std::swap(records[index - 1], records[selected]);
  }
  return records;
}

cudaq::ptsbe::detail::ScaledKrausState cudaq::ptsbe::detail::applyScaledKraus(
    std::span<const std::complex<double>> normalizedState,
    const cudaq::kraus_op &op, double proposalProbability) {
  if (!std::isfinite(proposalProbability) || proposalProbability <= 0.0)
    throw std::invalid_argument(
        "Kraus proposal probability must be finite and positive");
  if (op.nRows == 0 || op.nRows != op.nCols ||
      op.data.size() != op.nRows * op.nCols ||
      normalizedState.size() != op.nCols)
    throw std::invalid_argument(
        "Kraus operator and state dimensions do not match");

  double inputNormSquared = 0.0;
  for (const auto &amplitude : normalizedState) {
    if (!std::isfinite(amplitude.real()) || !std::isfinite(amplitude.imag()))
      throw std::invalid_argument("Kraus input state must be finite");
    inputNormSquared += std::norm(amplitude);
  }
  if (std::abs(inputNormSquared - 1.0) > 1e-12)
    throw std::invalid_argument("Kraus input state must be normalized");

  ScaledKrausState result;
  result.normalizedState.resize(op.nRows);
  const auto inverseRootProposal = 1.0 / std::sqrt(proposalProbability);
  for (std::size_t row = 0; row < op.nRows; ++row)
    for (std::size_t column = 0; column < op.nCols; ++column) {
      const auto value = op.data[row * op.nCols + column];
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument(
            "Kraus operator must contain finite values");
      result.normalizedState[row] +=
          std::complex<double>(value.real(), value.imag()) *
          normalizedState[column] * inverseRootProposal;
    }
  for (const auto &amplitude : result.normalizedState)
    result.normSquared += std::norm(amplitude);
  if (!std::isfinite(result.normSquared))
    throw std::invalid_argument("scaled Kraus state norm is not finite");
  if (result.normSquared == 0.0) {
    result.logWeightIncrement = -std::numeric_limits<double>::infinity();
    return result;
  }
  result.logWeightIncrement = std::log(result.normSquared);
  const auto inverseNorm = 1.0 / std::sqrt(result.normSquared);
  for (auto &amplitude : result.normalizedState)
    amplitude *= inverseNorm;
  return result;
}

cudaq::ptsbe::detail::ImportanceExperimentConfig
cudaq::ptsbe::detail::readImportanceExperimentConfig() {
  ImportanceExperimentConfig config;
  if (const auto *value = std::getenv("CUDAQ_PTSBE_NONUNITARY_MODE")) {
    const std::string_view mode(value);
    if (mode == "frontier")
      config.mode = NonUnitaryMode::Frontier;
    else if (mode == "counted_wave")
      config.mode = NonUnitaryMode::CountedWave;
    else if (mode == "importance")
      config.mode = NonUnitaryMode::Importance;
    else
      throw std::invalid_argument("CUDAQ_PTSBE_NONUNITARY_MODE must be "
                                  "frontier, counted_wave, or importance");
  }
  if (const auto *value = std::getenv("CUDAQ_PTSBE_IMPORTANCE_NORMALIZATION")) {
    const std::string_view normalization(value);
    if (normalization == "site")
      config.normalization = ImportanceNormalization::Site;
    else if (normalization == "segment")
      config.normalization = ImportanceNormalization::Segment;
    else
      throw std::invalid_argument(
          "CUDAQ_PTSBE_IMPORTANCE_NORMALIZATION must be site or segment");
  }
  if (const auto *value = std::getenv("CUDAQ_PTSBE_IMPORTANCE_RESAMPLER")) {
    const std::string_view resampler(value);
    if (resampler == "multinomial")
      config.resampler = FinalResampler::Multinomial;
    else if (resampler == "residual")
      config.resampler = FinalResampler::Residual;
    else if (resampler == "residual_stratified")
      config.resampler = FinalResampler::ResidualStratified;
    else
      throw std::invalid_argument(
          "CUDAQ_PTSBE_IMPORTANCE_RESAMPLER must be multinomial, residual, or "
          "residual_stratified");
  }
  if (const auto *value =
          std::getenv("CUDAQ_PTSBE_IMPORTANCE_CHECKPOINT_SITES")) {
    std::size_t parsed = 0;
    const std::string_view text(value);
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error != std::errc{} || end != text.data() + text.size() || parsed == 0)
      throw std::invalid_argument(
          "CUDAQ_PTSBE_IMPORTANCE_CHECKPOINT_SITES must be a positive integer");
    config.checkpointSites = parsed;
  }
  if (const auto *value = std::getenv("CUDAQ_PTSBE_IMPORTANCE_PROPOSALS")) {
    std::size_t parsed = 0;
    const std::string_view text(value);
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error != std::errc{} || end != text.data() + text.size() || parsed == 0)
      throw std::invalid_argument(
          "CUDAQ_PTSBE_IMPORTANCE_PROPOSALS must be a positive integer");
    config.proposalParticles = parsed;
  }
  return config;
}
