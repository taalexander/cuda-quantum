/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "KrausTrajectory.h"
#include "common/NoiseModel.h"
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace cudaq::ptsbe {

/// @brief Discriminator for instruction types within the PTSBE execution data.
///
/// Supports mid-circuit measurement and reset: Measurement covers both
/// mid-circuit and terminal sites, and buildPTSBETrace fuses a Measurement
/// immediately followed by a Reset on the same qubit(s) (with no intervening
/// instruction touching them) into a single MeasureReset site. Execution does
/// not branch at measurement sites; outcomes are drawn at their true
/// probabilities during replay, so this container always holds a flat
/// instruction list and flat trajectory list.
enum class TraceInstructionType {
  Gate,        /// Quantum gate operation (H, X, CNOT, RX, etc.)
  Noise,       /// Noise channel location (depolarizing, amplitude_damping, ...)
  Measurement, /// Measurement operation (mid-circuit or terminal)
  Reset,       /// Qubit reset to |0>
  MeasureReset /// Fused measurement immediately followed by reset
};

/// @brief Single operation in the PTSBE execution trace.
///
/// Stores gate, noise channel, or measurement info with plain qubit indices.
/// This is the user-facing trace type exposed to Python via nanobind.
///
struct TraceInstruction {
  /// @brief Instruction category (Gate, Noise, or Measurement)
  TraceInstructionType type;

  /// @brief Operation name (e.g., `h`, `depolarizing`, `mz`)
  std::string name;

  /// @brief Target qubit indices
  std::vector<std::size_t> targets;

  /// @brief Control qubit indices (empty for non-controlled operations)
  std::vector<std::size_t> controls;

  /// @brief Parameters (gate angles or noise channel parameters)
  std::vector<double> params;

  /// @brief Noise channel (populated only for Noise instructions)
  std::optional<cudaq::kraus_channel> channel;

  /// @brief Position of this instruction's bit in the per-shot measurement
  /// record. Assigned densely in trace order by buildPTSBETrace for every
  /// Measurement and MeasureReset instruction; nullopt for all other types.
  std::optional<std::size_t> record_index;

  /// @brief Measurement register name from the kernel (populated for
  /// Measurement and MeasureReset instructions when the kernel names the
  /// result; nullopt for unnamed measurements and all other types)
  std::optional<std::string> register_name;

  /// @brief Default constructor
  TraceInstruction() = default;

  /// @brief Constructor with all fields
  TraceInstruction(TraceInstructionType type, std::string name,
                   std::vector<std::size_t> targets,
                   std::vector<std::size_t> controls,
                   std::vector<double> params,
                   std::optional<cudaq::kraus_channel> channel = std::nullopt,
                   std::optional<std::size_t> record_index = std::nullopt,
                   std::optional<std::string> register_name = std::nullopt)
      : type(type), name(std::move(name)), targets(std::move(targets)),
        controls(std::move(controls)), params(std::move(params)),
        channel(std::move(channel)), record_index(record_index),
        register_name(std::move(register_name)) {}
};

/// @brief Alias for the PTSBE instruction sequence.
using PTSBETrace = std::vector<TraceInstruction>;

/// @brief Number of qubits referenced in a trace (max qubit ID + 1).
/// Returns 0 for an empty trace.
std::size_t numQubits(std::span<const TraceInstruction> trace);

/// @brief Count instructions matching the given type and optional name.
std::size_t countInstructions(std::span<const TraceInstruction> trace,
                              TraceInstructionType type,
                              std::optional<std::string> name = std::nullopt);

/// @brief True when the trace requires mid-circuit measurement handling.
///
/// Holds iff some Measurement or MeasureReset instruction is followed by a
/// later instruction (Gate, Noise, Reset, Measurement, or MeasureReset)
/// touching one of its target qubits, or some Reset sits mid-circuit (a later
/// instruction touches one of its targets). Terminal-only traces return
/// false, including traces whose last operation on a qubit is a Reset or
/// MeasureReset: a trailing reset cannot influence any recorded bit, so such
/// traces execute on the terminal-sampling path (which samples MeasureReset
/// qubits alongside plain Measurement qubits).
bool hasMidCircuitMeasurement(std::span<const TraceInstruction> trace);

/// @brief Container for PTSBE execution data including circuit structure,
/// trajectory specifications, and per-trajectory measurement outcomes.
///
/// The instructions represent the circuit structure (what operations were
/// applied and where noise channels exist), while trajectories represent
/// noise realizations (which Kraus operators were selected) along with
/// the measurement outcomes from executing each realization.
///
/// One execution data container may have many trajectories which reference
/// the noise locations within the instructions.
struct PTSBEExecutionData {
  /// @brief Ordered circuit operations (gates, noise channels, measurements)
  PTSBETrace instructions;

  /// @brief The sampled trajectories
  std::vector<cudaq::KrausTrajectory> trajectories;

  /// @brief Count instructions matching the given type and optional name
  std::size_t
  count_instructions(TraceInstructionType type,
                     std::optional<std::string> name = std::nullopt) const;

  /// @brief Look up a trajectory by its ID
  /// @return Reference to the trajectory if found, std::nullopt otherwise
  std::optional<std::reference_wrapper<const cudaq::KrausTrajectory>>
  get_trajectory(std::size_t trajectoryId) const;
};

} // namespace cudaq::ptsbe
