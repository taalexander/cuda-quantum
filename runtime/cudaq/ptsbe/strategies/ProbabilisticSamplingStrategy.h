/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "../PTSSamplingStrategy.h"
#include "cudaq/algorithms/broadcast.h"
#include <optional>
#include <random>
#include <stdexcept>

namespace cudaq::ptsbe {

/// @brief Probabilistic trajectory sampling strategy
/// Samples trajectories randomly based on their occurrence probabilities.
class ProbabilisticSamplingStrategy : public PTSSamplingStrategy {
public:
  /// @brief Construct with optional random seed and trajectory sample count
  /// @param seed Random seed for reproducibility. When `nullopt` (default),
  /// uses CUDAQ's global random seed if set, otherwise std::random_device.
  /// @param max_trajectory_samples Maximum number of Monte Carlo draws before
  /// giving up on discovering new unique trajectories. The loop stops early
  /// once max_trajectories unique patterns are found, so the actual draw
  /// count may be less. Every draw contributes to exactly one trajectory's
  /// multiplicity, preserving unbiased MC estimation.
  /// When `nullopt` (default), a budget is auto-calculated as a small
  /// multiplier of max_trajectories.
  /// @param num_root_draws Fixed number D of independent root draws.
  /// When set, generateTrajectories performs exactly D draws and then
  /// deduplicates, so multiplicities sum to D; discovering more than
  /// max_trajectories unique roots before the draws complete is an error,
  /// not a stopping rule. Mutually exclusive with max_trajectory_samples.
  /// @throws std::invalid_argument if both max_trajectory_samples and
  /// num_root_draws are set
  explicit ProbabilisticSamplingStrategy(
      std::optional<std::uint64_t> seed = std::nullopt,
      std::optional<std::size_t> max_trajectory_samples = std::nullopt,
      std::optional<std::size_t> num_root_draws = std::nullopt)
      : rng_(seed.value_or(cudaq::get_random_seed() != 0
                               ? cudaq::get_random_seed()
                               : std::random_device{}())),
        max_trajectory_samples_(max_trajectory_samples),
        num_root_draws_(num_root_draws) {
    if (max_trajectory_samples_ && num_root_draws_)
      throw std::invalid_argument(
          "ProbabilisticSamplingStrategy: max_trajectory_samples and "
          "num_root_draws are mutually exclusive; fixed root draws define "
          "the exact draw count.");
  }

  /// @brief Destructor
  ~ProbabilisticSamplingStrategy() override;

  /// @brief Generate trajectories via probability-weighted Monte Carlo sampling
  /// @param noise_points Noise information from circuit analysis
  /// @param max_trajectories Maximum number of unique trajectories to return.
  /// The loop draws MC samples until this many unique patterns are found or
  /// the budget is exhausted. Every draw contributes to a trajectory's
  /// multiplicity, so the resulting weights are unbiased MC frequency
  /// estimates suitable for PROPORTIONAL shot allocation.
  /// @return Vector of unique trajectories with accumulated multiplicities
  [[nodiscard]] std::vector<cudaq::KrausTrajectory>
  generateTrajectories(std::span<const detail::NoisePoint> noise_points,
                       std::size_t max_trajectories) const override;

  /// @brief Get strategy name
  /// @return "Probabilistic"
  [[nodiscard]] const char *name() const override { return "Probabilistic"; }

  /// @brief Clone this strategy
  /// @return Unique pointer to a copy
  [[nodiscard]] std::unique_ptr<PTSSamplingStrategy> clone() const override {
    return std::make_unique<ProbabilisticSamplingStrategy>(*this);
  }

private:
  mutable std::mt19937_64 rng_;
  std::optional<std::size_t> max_trajectory_samples_;
  std::optional<std::size_t> num_root_draws_;
};

} // namespace cudaq::ptsbe
