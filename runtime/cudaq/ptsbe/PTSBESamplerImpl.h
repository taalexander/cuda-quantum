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
enum class ReplayOpKind { Gate, Measure, Reset, MeasureReset, KrausBranch };

/// @brief One site-ordered operation in a merged trajectory replay list.
///
/// Gate ops reference a simulator gate task they do not own: trace gates
/// live in the caller's per-batch gate cache (see convertTraceGates) and
/// resolved Kraus selections live in the enclosing TrajectoryReplay. Measure,
/// Reset, and MeasureReset ops carry the site's target qubits; measuring ops
/// additionally carry the position of their bit in the per-shot measurement
/// record. KrausBranch ops mark general (non-unitary) Kraus sites whose
/// branch is selected during replay from its true state-dependent
/// probabilities; they carry the site's qubits and a non-owning pointer to
/// the raw channel in the caller's trace, which must outlive the replay.
template <typename ScalarType>
struct ReplayOp {
  ReplayOpKind kind;
  const GateTask<ScalarType> *task = nullptr;
  std::vector<std::size_t> qubits;
  std::optional<std::size_t> recordOffset;
  const cudaq::kraus_channel *channel = nullptr;

  explicit ReplayOp(const GateTask<ScalarType> *gateTask)
      : kind(ReplayOpKind::Gate), task(gateTask) {}

  ReplayOp(ReplayOpKind kind, std::vector<std::size_t> qubits,
           std::optional<std::size_t> recordOffset)
      : kind(kind), qubits(std::move(qubits)), recordOffset(recordOffset) {}

  ReplayOp(const cudaq::kraus_channel *branchChannel,
           std::vector<std::size_t> qubits)
      : kind(ReplayOpKind::KrausBranch), qubits(std::move(qubits)),
        channel(branchChannel) {}
};

/// @brief Site-ordered replay list for one trajectory.
///
/// Gate ops in `ops` point either into the per-batch gate cache passed to
/// mergeSitesWithTrajectory (which the caller must keep alive for as long as
/// any replay built from it is used) or into `krausTasks`, which owns the
/// tasks converted from this trajectory's Kraus selections. Move-only:
/// copying would leave `ops` pointing into the source's `krausTasks`.
template <typename ScalarType>
struct TrajectoryReplay {
  std::vector<GateTask<ScalarType>> krausTasks;
  std::vector<ReplayOp<ScalarType>> ops;

  TrajectoryReplay() = default;
  TrajectoryReplay(TrajectoryReplay &&) = default;
  TrajectoryReplay &operator=(TrajectoryReplay &&) = default;
  TrajectoryReplay(const TrajectoryReplay &) = delete;
  TrajectoryReplay &operator=(const TrajectoryReplay &) = delete;
};

/// @brief Walk the PTSBE trace and build the site-ordered replay list for
/// one trajectory. Gate entries become Gate ops referencing the corresponding
/// task in `gateCache` (one entry per trace Gate instruction, in trace order,
/// built once per batch by convertTraceGates), unitary-mixture Noise entries
/// are resolved to Gate ops owned by the returned replay via the trajectory
/// selections, non-unitary Noise entries become KrausBranch ops referencing
/// the raw channel in the trace (which must outlive the replay), and
/// Measurement, Reset, and MeasureReset entries become their op kinds
/// carrying the site's qubits and record offset.
///
/// @param includeIdentity When true, identity Kraus operators are
///   included as Gate ops. Useful if you require all trajectories to have
///   identical op structure.
template <typename ScalarType>
TrajectoryReplay<ScalarType>
mergeSitesWithTrajectory(std::span<const TraceInstruction> ptsbeTrace,
                         std::span<const GateTask<ScalarType>> gateCache,
                         const cudaq::KrausTrajectory &trajectory,
                         bool includeIdentity = false);

/// @brief Legacy gate-only view of mergeSitesWithTrajectory: Gate entries
/// become gate tasks and Noise entries are resolved via the trajectory
/// selections, while Measurement, Reset, and MeasureReset entries are
/// dropped. Only valid for terminal-measurement replay, where the simulator
/// samples measurement qubits after applying all gates; traces with
/// mid-circuit measurement or reset need mergeSitesWithTrajectory. Converts
/// the trace's gates on every call and returns owned copies; per-batch
/// callers should build a gate cache once and use mergeSitesWithTrajectory.
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

/// @brief Build the per-batch gate cache: one converted simulator task per
/// Gate instruction in the trace, in trace order (all other instruction
/// types are skipped). mergeSitesWithTrajectory references this cache so
/// each trace gate is converted once per batch instead of once per
/// trajectory. The cache must outlive every TrajectoryReplay built from it.
template <typename ScalarType>
std::vector<GateTask<ScalarType>>
convertTraceGates(std::span<const TraceInstruction> ptsbeTrace);

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
/// Each shot's record becomes one count entry and one sequential-data entry
/// (mid-circuit batches always carry sequential data, a PTSBatch invariant
/// enforced by samplePTSBE).
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
