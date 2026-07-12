/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "ShotAllocationStrategy.h"
#include <cstddef>
#include <memory>
#include <optional>

namespace cudaq::ptsbe {

// Forward declaration
class PTSSamplingStrategy;

/// @brief Configuration options for PTSBE execution.
///
/// Controls whether PTSBEExecutionData output is produced and which sampling
/// strategy to use.
///
/// The return_execution_data flag controls production of the full
/// PTSBEExecutionData, which bundles circuit instructions, trajectory
/// specifications, and per-trajectory measurement outcomes as a single unit.
///
struct PTSBEOptions {
  /// Produce PTSBEExecutionData (instructions + trajectories + measurement
  /// counts) in the sample result. Default false to avoid overhead when not
  /// needed.
  bool return_execution_data = false;

  /// Populate per-shot sequential bitstring data on the result. When false
  /// (default), only aggregated counts are produced.
  bool include_sequential_data = false;

  /// Maximum number of unique trajectories to generate. When `nullopt`,
  /// defaults to the number of shots.
  std::optional<std::size_t> max_trajectories = std::nullopt;

  /// Maximum shots executed per batch slot. Capping the slot size splits a
  /// trajectory's shots across slots so their measurement records stay
  /// decorrelated. When `nullopt` (default), selected automatically: 1 when
  /// the trace contains mid-circuit measurement or reset or when any
  /// frontier knob (num_root_draws, max_paths_per_root, max_live_states) is
  /// set, unlimited otherwise. An explicit 0 forces unlimited. The
  /// environment variable
  /// `CUDAQ_PTSBE_MAX_SHOTS_PER_PATH` takes precedence over this option.
  std::optional<std::size_t> max_shots_per_path = std::nullopt;

  /// Fixed number D of independent root draws performed before
  /// deduplication. When set, the default probabilistic strategy performs
  /// exactly D draws (discovering more than max_trajectories unique roots is
  /// an error, not a stopping rule), the sum of root multiplicities must
  /// equal D, and PROPORTIONAL shot allocation becomes the exact root-weight
  /// split N_u = shots * d_u / D. Flat results additionally require every
  /// N_u * D to be divisible by d_u * shots; violations are errors, never
  /// silently adjusted. When `nullopt`, root draws follow the strategy's own
  /// budgeting and no root-weight conditions are enforced.
  std::optional<std::size_t> num_root_draws = std::nullopt;

  /// Maximum replay paths sampled for one root. The required path count is
  /// C_u = ceil(N_u / max_shots_per_path); a configuration whose required
  /// C_u exceeds this limit is an error (no silent clamping). When `nullopt`,
  /// the path count per root is unbounded.
  std::optional<std::size_t> max_paths_per_root = std::nullopt;

  /// Maximum statevectors resident in one path group of the branching
  /// frontier executor. Deterministic capacity management, not statistical
  /// tuning: replay paths are processed in groups no larger than this. When
  /// `nullopt`, the executor chooses its capacity.
  std::optional<std::size_t> max_live_states = std::nullopt;

  /// Admit general (non-unitary) Kraus channels. Admitted channels keep
  /// their raw operators and are not pre-sampled into root trajectories;
  /// branch outcomes are drawn from their true state-dependent probabilities
  /// during replay. Requires a BatchSimulator backend; the generic
  /// per-trajectory sampler rejects non-unitary sites.
  bool allow_non_unitary = false;

  /// Custom sampling strategy. If `nullptr`, uses default strategy.
  std::shared_ptr<PTSSamplingStrategy> strategy = nullptr;

  /// Strategy for allocating shots across trajectories.
  /// Defaults to PROPORTIONAL.
  ShotAllocationStrategy shot_allocation{};
};

} // namespace cudaq::ptsbe
