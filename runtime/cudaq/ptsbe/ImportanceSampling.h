/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/NoiseModel.h"
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace cudaq::ptsbe::detail {

struct KrausProposal {
  std::vector<double> probabilities;
  std::vector<std::size_t> originalBranchIndices;
  std::size_t dimension = 0;
};

enum class ImportanceDrawPurpose : std::uint64_t {
  KrausLabel = 1,
  UnitaryLabel = 2,
  MeasurementOutcome = 3,
  TerminalRecord = 4,
  FinalAllocation = 5,
  SequentialShuffle = 6
};

struct ImportanceRngKey {
  std::uint64_t seed = 0;
  std::uint64_t wave = 0;
  std::uint64_t historyId = 0;
  std::uint64_t traceSite = 0;
  ImportanceDrawPurpose purpose = ImportanceDrawPurpose::KrausLabel;
  std::uint64_t drawOrdinal = 0;
};

struct CountedBranch {
  std::size_t originalBranchIndex = 0;
  std::uint64_t multiplicity = 0;
  std::uint64_t childHistoryId = 0;
};

struct WeightedRecord {
  std::string record;
  std::uint64_t multiplicity = 0;
  double logWeight = 0.0;
};

struct LogMassBin {
  std::string record;
  double logMass = 0.0;
};

struct WeightDiagnostics {
  std::uint64_t representedParticles = 0;
  double logSumWeights = 0.0;
  double logSumSquaredWeights = 0.0;
  double effectiveSampleSize = 0.0;
};

enum class FinalResampler { Multinomial, Residual, ResidualStratified };

enum class NonUnitaryMode { Frontier, CountedWave, Importance };

enum class ImportanceNormalization { Site, Segment };

struct ImportanceExperimentConfig {
  NonUnitaryMode mode = NonUnitaryMode::Frontier;
  ImportanceNormalization normalization = ImportanceNormalization::Site;
  FinalResampler resampler = FinalResampler::ResidualStratified;
  std::size_t checkpointSites = 16;
};

struct ImportanceExperimentState {
  ImportanceExperimentConfig config;
  std::uint64_t seed = 0;
};

struct CountBin {
  std::string record;
  std::uint64_t count = 0;

  bool operator==(const CountBin &) const = default;
};

struct ScaledKrausState {
  std::vector<std::complex<double>> normalizedState;
  double logWeightIncrement = 0.0;
  double normSquared = 0.0;
};

KrausProposal buildKrausProposal(std::span<const cudaq::kraus_op> operators,
                                 double completenessTolerance = 1e-10);

std::uint64_t deriveHistoryId(std::uint64_t wave, std::uint64_t parentHistoryId,
                              std::uint64_t traceSite, std::uint64_t branchKind,
                              std::uint64_t outcome);

double counterUniform(const ImportanceRngKey &key);

std::size_t drawCategorical(std::span<const double> probabilities,
                            const ImportanceRngKey &key);

std::vector<CountedBranch>
splitMultiplicity(std::uint64_t multiplicity,
                  std::span<const double> probabilities,
                  std::span<const std::size_t> originalBranchIndices,
                  const ImportanceRngKey &baseKey, std::uint64_t branchKind);

std::vector<LogMassBin>
aggregateLogMass(std::span<const WeightedRecord> records);

WeightDiagnostics
computeWeightDiagnostics(std::span<const WeightedRecord> records);

std::vector<CountBin> allocateCounts(std::span<const LogMassBin> bins,
                                     std::uint64_t shots,
                                     FinalResampler resampler,
                                     std::uint64_t seed);

std::vector<CountBin>
materializeWeightedRecords(std::span<const WeightedRecord> records,
                           std::uint64_t shots, FinalResampler resampler,
                           std::uint64_t seed);

std::vector<std::string> makeSequentialData(std::span<const CountBin> counts,
                                            std::uint64_t seed);

ScaledKrausState
applyScaledKraus(std::span<const std::complex<double>> normalizedState,
                 const cudaq::kraus_op &op, double proposalProbability);

ImportanceExperimentConfig readImportanceExperimentConfig();

} // namespace cudaq::ptsbe::detail
