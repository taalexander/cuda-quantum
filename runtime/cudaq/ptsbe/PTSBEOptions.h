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
  /// (default), only aggregated counts are produced; for kernels with
  /// mid-circuit measurement or reset those counts are over unique full
  /// records, a lossless sufficient statistic for the per-shot table.
  bool include_sequential_data = false;

  /// Maximum number of unique trajectories to generate. When `nullopt`,
  /// defaults to the number of shots.
  std::optional<std::size_t> max_trajectories = std::nullopt;

  /// Maximum shots sharing one replay of a trajectory (one replay path).
  /// Capping the path size splits a trajectory's shots across independent
  /// replay paths so their measurement records stay decorrelated. When
  /// `nullopt` (default), selected automatically: 1 when
  /// the trace contains mid-circuit measurement or reset or when
  /// max_live_states is set, unlimited otherwise. An explicit 0 forces
  /// unlimited. On the batched frontier executor (max_live_states set),
  /// 0 selects multiplicity carrying: one replay path carries the whole
  /// root shot count as slot-0 multiplicity, branch and measure sites split
  /// that multiplicity, and identical (trajectory x syndrome) histories merge
  /// into one counted leaf. This is the tree-mode saturation behavior; a
  /// positive cap instead ties terminalSamplesPerPath shots together per path
  /// and draws them independently at the leaf. The environment variable
  /// `CUDAQ_PTSBE_MAX_SHOTS_PER_PATH` takes precedence over this option.
  std::optional<std::size_t> max_shots_per_path = std::nullopt;

  /// Requested live frontier width: the maximum number of statevectors that
  /// stay live in one path group of the branching frontier executor.
  /// Deterministic capacity management, not statistical tuning: replay paths
  /// are processed in groups no larger than this. The resident allocation
  /// rounds this width up to the next power of two; the executor reports both
  /// the requested width and the rounded capacity in its frontier metrics.
  /// When `nullopt`, the executor picks a width automatically: for kernels
  /// with mid-circuit measurement, reset, or admitted non-unitary Kraus
  /// channels it uses one live state per replay path, bounded by the
  /// memory-derived batch size (floor 1). Non-unitary channels therefore need
  /// no explicit width.
  std::optional<std::size_t> max_live_states = std::nullopt;

  /// Admit general (non-unitary) Kraus channels. Admitted channels keep
  /// their raw operators and are not pre-sampled into root trajectories;
  /// branch outcomes are drawn from their true state-dependent probabilities
  /// during replay on the branching frontier executor. When max_live_states
  /// is unset the executor selects the frontier width automatically, so
  /// non-unitary sampling needs no capacity knob. Requires a BatchSimulator
  /// backend; the generic per-trajectory sampler rejects non-unitary sites.
  bool allow_non_unitary = false;

  /// Fold unitary-mixture (Pauli) noise into the live branching frontier as
  /// UnitaryBranch sites instead of pre-sampling it into flat independent
  /// roots. Trajectories that agree early then share their evolved prefix: a
  /// shared prefix node evolves once and clones at branch nodes, with the
  /// split drawn from the channel's fixed state-independent weights (no GPU
  /// probability readback). The ahead-of-time global dedup is preserved; this
  /// only retains the shared-prefix structure the flat list discards. Requires
  /// the single-process batched frontier executor (max_live_states set or
  /// auto-selected). The intended large-state fallback is that at B = 1 the
  /// frontier reduces to the flat independent population; that reduction is not
  /// yet enforced in the multiplicity-carrying tree path, which instead stops
  /// with an error when a trajectory's cumulative branch fan-out would exceed
  /// max_live_states (the design's multinomial resample-down-to-B is not yet
  /// implemented). Choose max_live_states at least as large as the trajectory's
  /// distinct-outcome width, or leave unitary noise pre-sampled for the
  /// large-state case. When false (default) unitary noise stays pre-sampled and
  /// every existing result is byte-identical.
  bool unitary_noise_as_branch = false;

  /// Custom sampling strategy. If `nullptr`, uses default strategy.
  std::shared_ptr<PTSSamplingStrategy> strategy = nullptr;

  /// Strategy for allocating shots across trajectories.
  /// Defaults to PROPORTIONAL.
  ShotAllocationStrategy shot_allocation{};
};

} // namespace cudaq::ptsbe
