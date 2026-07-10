/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file PTSBESamplerImpl.h
/// @brief Internal header for PTSBE simulator integration.
///
/// This header exposes template functions and types that depend on
/// `nvqir::CircuitSimulator` internals. It is intended for simulator
/// implementations and tests, not for the public API. The public header
/// PTSBESampler.h provides the stable API (PTSBatch and dispatch entry points)
/// without leaking `nvqir` internals.

#pragma once

#include "KrausTrajectory.h"
#include "PTSBEExecutionData.h"
#include "PTSBESampler.h"
#include "common/Trace.h"
#include "nvqir/CircuitSimulator.h"
#include "nvqir/Gates.h"
#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace cudaq::ptsbe {

// Abstract interface for batch simulator
// Simulators can optionally implement this interface to provide a custom
// implementation of sampleWithPTSBE.
struct BatchSimulator {
  virtual ~BatchSimulator() = default;
  virtual std::vector<cudaq::sample_result>
  sampleWithPTSBE(const PTSBatch &batch) = 0;
};

/// @brief Alias for CircuitSimulator gate task type
template <typename ScalarType>
using GateTask =
    typename nvqir::CircuitSimulatorBase<ScalarType>::GateApplicationTask;

/// @brief Kind of operation replayed at one site of a merged trajectory.
enum class ReplayOpKind { Gate, Measure, Reset, MeasureReset };

/// @brief One site-ordered operation in a merged trajectory replay list.
///
/// Gate ops carry a simulator gate task. Measure, Reset, and MeasureReset
/// ops carry the site's target qubits; measuring ops additionally carry the
/// position of their bit in the per-shot measurement record.
template <typename ScalarType>
struct ReplayOp {
  ReplayOpKind kind;
  GateTask<ScalarType> task;
  std::vector<std::size_t> qubits;
  std::optional<std::size_t> recordOffset;

  explicit ReplayOp(GateTask<ScalarType> gateTask)
      : kind(ReplayOpKind::Gate), task(std::move(gateTask)) {}

  ReplayOp(ReplayOpKind kind, std::vector<std::size_t> qubits,
           std::optional<std::size_t> recordOffset)
      : kind(kind), task({}, {}, {}, {}, {}), qubits(std::move(qubits)),
        recordOffset(recordOffset) {}
};

/// @brief Walk the PTSBE trace and build the site-ordered replay-op list for
/// one trajectory. Gate entries become Gate ops, Noise entries are resolved
/// to Gate ops via the trajectory selections (channel looked up from the
/// trace), and Measurement, Reset, and MeasureReset entries become their op
/// kinds carrying the site's qubits and record offset.
///
/// @param includeIdentity When true, identity Kraus operators are
///   included as Gate ops. Useful if you require all trajectories to have
///   identical op structure.
template <typename ScalarType>
std::vector<ReplayOp<ScalarType>>
mergeSitesWithTrajectory(std::span<const TraceInstruction> ptsbeTrace,
                         const cudaq::KrausTrajectory &trajectory,
                         bool includeIdentity = false);

/// @brief Legacy gate-only view of mergeSitesWithTrajectory: Gate entries
/// become gate tasks and Noise entries are resolved via the trajectory
/// selections, while Measurement, Reset, and MeasureReset entries are
/// dropped. Only valid for terminal-measurement replay, where the simulator
/// samples measurement qubits after applying all gates; traces with
/// mid-circuit measurement or reset need mergeSitesWithTrajectory.
///
/// @param includeIdentity When true, identity Kraus operators are
///   included as gate tasks. Useful if you require all trajectories to have
///   identical gate structure.
template <typename ScalarType>
std::vector<GateTask<ScalarType>>
mergeTasksWithTrajectory(std::span<const TraceInstruction> ptsbeTrace,
                         const cudaq::KrausTrajectory &trajectory,
                         bool includeIdentity = false);

} // namespace cudaq::ptsbe

namespace cudaq::ptsbe::detail {

/// @brief Convert a PTSBE TraceInstruction (Gate type) to a simulator task.
/// Looks up the gate matrix from the registry and maps plain qubit IDs.
template <typename ScalarType>
GateTask<ScalarType> convertToSimulatorTask(const TraceInstruction &inst);

/// @brief Convert a PTSBE trace to a simulator task list, keeping only Gate
/// entries (Noise and Measurement entries are skipped).
template <typename ScalarType>
std::vector<GateTask<ScalarType>>
convertTrace(std::span<const TraceInstruction> ptsbeTrace);

/// @brief Convert a KrausSelection to a GateApplicationTask using the
/// noise channel's unitary operators from the trace instruction.
template <typename ScalarType>
GateTask<ScalarType> krausSelectionToTask(const cudaq::KrausSelection &sel,
                                          const TraceInstruction &noiseInst);

/// @brief Generic PTSBE execution implementation
///
/// For each trajectory of a terminal-measurement-only batch:
/// - Resets simulator to computational zero state
/// - Merges PTSBE trace with trajectory noise selections
/// - Applies merged gate tasks
/// - Samples measurement qubits
///
/// When the batch has mid-circuit measurement or reset
/// (PTSBatch::hasMidCircuitMeasurement), each shot instead replays the
/// site-ordered op list from mergeSitesWithTrajectory: measurement sites
/// collapse the state and write their bit into a fixed-width record string
/// at the site's record offset, and reset sites return their qubits to |0>.
/// Each shot's record becomes one count entry (and one sequential-data entry
/// when requested).
///
/// Returns per-trajectory results for flexibility. Use aggregateResults()
/// to combine into a single sample_result if needed.
///
/// This is the fallback implementation used when a simulator does not
/// provide a custom sampleWithPTSBE() method.
///
/// Caller must set up ExecutionContext and allocate qubits before
/// calling this function. Caller is also responsible for de-allocating qubits
/// and resetting the ExecutionContext after this function returns.
///
/// @tparam ScalarType Simulator scalar type
/// @param simulator Circuit simulator instance (must have ExecutionContext set)
/// @param batch PTSBE specification
/// @return Per-trajectory sample results
/// @throws std::runtime_error if ExecutionContext not set or gate conversion
/// fails
template <typename ScalarType>
std::vector<cudaq::sample_result>
samplePTSBEGeneric(nvqir::CircuitSimulatorBase<ScalarType> &simulator,
                   const PTSBatch &batch);

} // namespace cudaq::ptsbe::detail
