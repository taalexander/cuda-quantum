/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PTSBESample.h"
#include "NoiseExtractor.h"
#include "ShotAllocationStrategy.h"
#include "strategies/ProbabilisticSamplingStrategy.h"
#include "cudaq/algorithms/sample.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/simulators.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <span>
#include <unordered_map>

namespace cudaq::ptsbe::detail {

void validatePTSBEKernel(const std::string &kernelName,
                         const ExecutionContext & /*ctx*/) {
  if (cudaq::kernelHasConditionalFeedback(kernelName)) {
    throw std::runtime_error(
        "PTSBE does not support dynamic circuits. Kernel '" + kernelName +
        "' contains conditional feedback on measurement outcomes. "
        "The gate sequence must be deterministic for pre-trajectory sampling.");
  }
}

bool warnNamedRegisters(const std::string &kernelName, ExecutionContext &ctx) {
  for (const auto &inst : ctx.kernelTrace) {
    if (inst.type == cudaq::TraceInstructionType::Measurement &&
        inst.register_name && *inst.register_name != "__global__") {
      std::cerr << "WARNING: Kernel \"" << kernelName
                << "\" uses named measurement results but is invoked via "
                   "ptsbe::sample (or ptsbe.sample). PTSBE outputs a single "
                   "global register; "
                   "named sub-registers are not preserved. Use `cudaq::run` "
                   "to retrieve individual measurement results."
                << std::endl;
      return true;
    }
  }
  return false;
}

void validatePTSBEPreconditions(quantum_platform &platform,
                                std::optional<std::size_t> qpu_id) {
  if (qpu_id && platform.is_remote(*qpu_id))
    throw std::runtime_error(
        "PTSBE does not support remote execution. Use a local simulator.");

  if (!platform.is_simulator())
    throw std::runtime_error("PTSBE is only supported on simulators.");

  // noise_model is optional: noise can come from the model (gate-based) and/or
  // from cudaq.apply_noise() in the kernel.
}

std::vector<RecordSite>
buildRecordLayout(std::span<const TraceInstruction> trace) {
  // Reuse the shared last-touch scan so the terminal flag is an O(1) lookup
  // per site and stays consistent with hasMidCircuitMeasurement.
  const auto lastTouch = computeTraceLayout(trace).lastTouchIndex;

  std::vector<RecordSite> layout;
  for (std::size_t i = 0; i < trace.size(); ++i) {
    const auto &inst = trace[i];
    if (inst.type != TraceInstructionType::Measurement &&
        inst.type != TraceInstructionType::MeasureReset)
      continue;
    if (!inst.record_index)
      throw std::runtime_error(
          "PTSBE trace measurement site '" + inst.name + "' at position " +
          std::to_string(i) +
          " has no record_index; the trace was not built through "
          "buildPTSBETrace");
    const std::size_t base = *inst.record_index;
    const bool resets = inst.type == TraceInstructionType::MeasureReset;
    for (std::size_t k = 0; k < inst.targets.size(); ++k) {
      const auto qubit = inst.targets[k];
      layout.push_back({base + k, qubit, resets, lastTouch.at(qubit) == i,
                        inst.register_name});
    }
  }
  return layout;
}

void cleanupTracerQubits(const Trace &kernelTrace) {
  auto numQubits = kernelTrace.getNumQudits();
  if (numQubits == 0)
    return;
  std::vector<std::size_t> qubitIds(numQubits);
  std::iota(qubitIds.begin(), qubitIds.end(), 0);
  cudaq::get_simulator()->deallocateQubits(qubitIds);
}

static std::vector<std::size_t>
extractQubitIds(const std::vector<cudaq::QuditInfo> &qudits) {
  std::vector<std::size_t> ids;
  ids.reserve(qudits.size());
  for (const auto &q : qudits)
    ids.push_back(q.id);
  return ids;
}

/// Register name carried onto Measurement instructions. The default
/// "__global__" register means the measurement is unnamed.
static std::optional<std::string>
measurementRegisterName(const cudaq::Trace::Instruction &inst) {
  if (inst.register_name && *inst.register_name != "__global__")
    return inst.register_name;
  return std::nullopt;
}

namespace {
/// Mutable state threaded through the single-pass trace build: the emitted
/// instructions, the index of the last emitted instruction touching each
/// qubit (drives measure-then-reset fusion), and the next dense record index.
struct TraceBuildState {
  PTSBETrace result;
  std::unordered_map<std::size_t, std::size_t> lastTouch;
  std::size_t nextRecordIndex = 0;
};
} // namespace

/// Append one instruction, updating last-touch tracking and, for measuring
/// sites, assigning the next dense record index (one bit per target qubit).
/// Record indices are the single source of the per-shot record layout shared
/// by all execution paths, which write one record bit per target at
/// record_index + k.
static void emitInstruction(TraceBuildState &state, TraceInstruction inst) {
  const std::size_t idx = state.result.size();
  for (auto q : inst.targets)
    state.lastTouch[q] = idx;
  for (auto q : inst.controls)
    state.lastTouch[q] = idx;
  if (inst.type == TraceInstructionType::Measurement) {
    inst.record_index = state.nextRecordIndex;
    state.nextRecordIndex += inst.targets.size();
  }
  state.result.push_back(std::move(inst));
}

/// Resolve each channel to its unitary mixture and emit a Noise instruction
/// on the given qubits. Empty channels are dropped.
static void appendNoiseChannels(TraceBuildState &state,
                                std::vector<cudaq::kraus_channel> &&channels,
                                const std::vector<std::size_t> &qubits) {
  for (auto &channel : channels) {
    if (channel.empty())
      continue;
    if (!channel.is_unitary_mixture())
      channel.generateUnitaryParameters();
    auto parameters = channel.parameters;
    emitInstruction(state, {TraceInstructionType::Noise,
                            channel.get_type_name(),
                            qubits,
                            {},
                            std::move(parameters),
                            std::move(channel)});
  }
}

/// Fuse a Reset into the preceding Measurement when that measurement is the
/// last emitted instruction touching every reset target and covers exactly
/// the same targets. The fused MeasureReset keeps the measurement's
/// position, record index, and register name; the Reset is dropped.
/// Instructions on other qubits may sit between the pair.
static bool fuseResetIntoMeasurement(TraceBuildState &state,
                                     const std::vector<std::size_t> &targets) {
  if (targets.empty())
    return false;
  auto first = state.lastTouch.find(targets.front());
  if (first == state.lastTouch.end())
    return false;
  const std::size_t measurementIdx = first->second;
  for (auto q : targets) {
    auto found = state.lastTouch.find(q);
    if (found == state.lastTouch.end() || found->second != measurementIdx)
      return false;
  }
  auto &candidate = state.result[measurementIdx];
  if (candidate.type != TraceInstructionType::Measurement ||
      candidate.targets != targets)
    return false;
  candidate.type = TraceInstructionType::MeasureReset;
  return true;
}

static void convertTraceInstruction(const cudaq::Trace::Instruction &inst,
                                    const cudaq::noise_model &noise_model,
                                    TraceBuildState &state) {
  auto targets = extractQubitIds(inst.targets);
  auto controls = extractQubitIds(inst.controls);

  if (inst.type == cudaq::TraceInstructionType::Reset) {
    if (fuseResetIntoMeasurement(state, targets))
      return;
    emitInstruction(state,
                    {TraceInstructionType::Reset, inst.name, std::move(targets),
                     std::move(controls), inst.params});
    return;
  }

  if (inst.type == cudaq::TraceInstructionType::Noise) {
    std::intptr_t key = inst.noise_channel_key.value();
    cudaq::kraus_channel channel = noise_model.get_channel(key, inst.params);
    if (!channel.empty()) {
      if (!channel.is_unitary_mixture())
        channel.generateUnitaryParameters();
      emitInstruction(state, {TraceInstructionType::Noise, inst.name,
                              std::move(targets), std::move(controls),
                              inst.params, std::move(channel)});
    }
    return;
  }

  if (inst.type == cudaq::TraceInstructionType::Gate) {
    auto channels =
        noise_model.get_channels(inst.name, targets, controls, inst.params);
    std::vector<std::size_t> noiseQubits = targets;
    noiseQubits.insert(noiseQubits.end(), controls.begin(), controls.end());
    emitInstruction(state,
                    {TraceInstructionType::Gate, inst.name, std::move(targets),
                     std::move(controls), inst.params});
    appendNoiseChannels(state, std::move(channels), noiseQubits);
    return;
  }

  if (inst.type == cudaq::TraceInstructionType::Measurement) {
    // Measurement noise precedes the Measurement so the recorded bit reflects
    // the noisy state. This keeps terminal-only traces free of instructions
    // after their measurements (hasMidCircuitMeasurement stays false) and
    // gives site-ordered replay the correct semantics.
    appendNoiseChannels(state, noise_model.get_channels("mz", targets, {}, {}),
                        targets);
    emitInstruction(state, {TraceInstructionType::Measurement,
                            inst.name,
                            std::move(targets),
                            {},
                            inst.params,
                            std::nullopt,
                            std::nullopt,
                            measurementRegisterName(inst)});
    return;
  }
}

PTSBETrace buildPTSBETrace(const cudaq::Trace &trace,
                           const cudaq::noise_model &noise_model) {
  TraceBuildState state;
  bool hasMeasurement = false;
  for (const auto &inst : trace) {
    if (inst.type == cudaq::TraceInstructionType::Measurement)
      hasMeasurement = true;
    convertTraceInstruction(inst, noise_model, state);
  }

  // Match standard cudaq::sample() behavior: when the kernel omits explicit
  // mz() calls, measure all allocated qubits. Generate one Noise + Measurement
  // pair per qubit (noise first, matching convertTraceInstruction) so that
  // per-qubit noise channels (registered via add_channel("mz", {q}, ...)) are
  // matched correctly.
  auto n = trace.getNumQudits();
  if (!hasMeasurement && n > 0) {
    for (std::size_t q = 0; q < n; ++q) {
      appendNoiseChannels(state, noise_model.get_channels("mz", {q}, {}, {}),
                          {q});
      emitInstruction(state,
                      {TraceInstructionType::Measurement, "mz", {q}, {}, {}});
    }
  }

  return std::move(state.result);
}

PTSBEExecutionData
buildExecutionDataInstructions(const cudaq::Trace &kernelTrace,
                               const noise_model &noiseModel) {
  PTSBEExecutionData trace;

  trace.instructions = buildPTSBETrace(kernelTrace, noiseModel);
  return trace;
}

void populateExecutionDataTrajectories(
    PTSBEExecutionData &executionData,
    std::vector<cudaq::KrausTrajectory> trajectories,
    std::vector<cudaq::sample_result> perTrajectoryResults) {
  // Populate measurement_counts from parallel-indexed perTrajectoryResults,
  // keeping only trajectories that received at least one shot. Zero-shot
  // trajectories were discovered by MC sampling but never simulated.
  for (std::size_t i = 0;
       i < trajectories.size() && i < perTrajectoryResults.size(); ++i) {
    if (trajectories[i].num_shots == 0)
      continue;
    if (perTrajectoryResults[i].get_total_shots() > 0)
      trajectories[i].measurement_counts = perTrajectoryResults[i].to_map();
    executionData.trajectories.push_back(std::move(trajectories[i]));
  }
}

/// Read CUDAQ_PTSBE_MAX_SHOTS_PER_PATH. When set, it takes precedence over
/// PTSBEOptions::max_shots_per_path; 0 means unlimited.
static std::optional<std::size_t> maxShotsPerPathEnvOverride() {
  const char *value = std::getenv("CUDAQ_PTSBE_MAX_SHOTS_PER_PATH");
  if (!value || !*value)
    return std::nullopt;
  errno = 0;
  char *end = nullptr;
  auto parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0')
    throw std::invalid_argument(
        "Invalid CUDAQ_PTSBE_MAX_SHOTS_PER_PATH value '" + std::string(value) +
        "': expected a non-negative integer (0 = unlimited).");
  return static_cast<std::size_t>(parsed);
}

/// Frontier configuration is engaged when the live-frontier width is set. A
/// set width must be positive; no silent clamping anywhere.
static bool frontierConfigured(std::optional<std::size_t> maxLiveStates) {
  if (!maxLiveStates)
    return false;
  if (*maxLiveStates == 0)
    throw std::invalid_argument("PTSBE frontier configuration: max_live_states "
                                "must be a positive integer.");
  return true;
}

/// Validate the equal-terminal-samples-per-path condition for a configured
/// frontier: per root u with N_u allocated shots,
///   C_u = ceil(N_u / max_shots_per_path) replay paths, and
///   N_u = C_u * T_u must hold with integer T_u (equal terminal samples per
///   path).
/// A remainder is an error; the runtime never adjusts a limit or reallocates.
static void validateFrontierAllocation(const PTSBatch &batch) {
  for (const auto &traj : batch.trajectories) {
    const std::size_t pathShots = traj.num_shots;
    if (pathShots == 0)
      continue;
    const std::size_t requiredPaths =
        batch.maxShotsPerPath == 0
            ? 1
            : (pathShots + batch.maxShotsPerPath - 1) / batch.maxShotsPerPath;
    if (pathShots % requiredPaths != 0)
      throw std::invalid_argument(
          "PTSBE frontier configuration: root " +
          std::to_string(traj.trajectory_id) + " allocates " +
          std::to_string(pathShots) + " shots across " +
          std::to_string(requiredPaths) +
          " replay paths, which admits no equal integer terminal-sample "
          "count T_u with N_u = C_u * T_u.");
  }
}

PTSBatch buildPTSBatchFromTrace(PTSBETrace &&trace, const PTSBEOptions &options,
                                std::size_t shots) {
  PTSBatch batch;

  const bool frontier = frontierConfigured(options.max_live_states);
  batch.maxLiveStates = options.max_live_states;

  batch.trace = std::move(trace);
  batch.hasMidCircuitMeasurement = hasMidCircuitMeasurement(batch.trace);

  auto noiseResult = extractNoiseSites(batch.trace, /*validate=*/true,
                                       options.allow_non_unitary);
  cudaq::info("[ptsbe] Extracted {} noise sites from {} total instructions",
              noiseResult.noise_sites.size(), noiseResult.total_instructions);

  // Mode selection is internal and automatic: there is one frontier executor
  // whose configuration ranges from the fully degenerate flat pre-sample (B=1,
  // no live branch sites) to the prefix-sharing tree. Unitary-mixture (Pauli)
  // noise is folded into the live frontier as UnitaryBranch sites (tree mode)
  // only when the trace already forces per-prefix re-evolution (mid-circuit
  // measurement or admitted non-unitary Kraus) and the frontier is wide enough
  // to co-locate branches (B > 1); the resample bounds live width past B.
  // Otherwise unitary noise stays pre-sampled into flat roots: the degenerate
  // terminal-only and B=1 / large-state configuration, where flat is already
  // optimal and cloning would only add depth-first backtracking cost.
  const bool nonUnitaryPresent =
      options.allow_non_unitary &&
      std::any_of(noiseResult.noise_sites.begin(),
                  noiseResult.noise_sites.end(),
                  [](const NoisePoint &point) { return point.is_non_unitary; });
  const bool reEvolutionForced =
      batch.hasMidCircuitMeasurement || nonUnitaryPresent;
  const bool branchWidthPermits = frontier && *options.max_live_states > 1;
  batch.unitaryNoiseAsBranch = reEvolutionForced && branchWidthPermits;

  auto envMaxShotsPerPath = maxShotsPerPathEnvOverride();
  if (envMaxShotsPerPath)
    cudaq::info("[ptsbe] max shots per path set to {} via "
                "CUDAQ_PTSBE_MAX_SHOTS_PER_PATH",
                *envMaxShotsPerPath);
  // The frontier default is one terminal sample per path (T=1, so
  // C_u = N_u), which always satisfies the integer-allocation conditions.
  // Tree mode instead carries the whole root multiplicity on one path
  // (unlimited, 0) so identical trajectory-and-syndrome histories merge into
  // one counted leaf; its Measure and branch splits divide that multiplicity.
  batch.maxShotsPerPath =
      envMaxShotsPerPath.has_value()
          ? *envMaxShotsPerPath
          : options.max_shots_per_path.value_or(
                (!batch.unitaryNoiseAsBranch &&
                 (batch.hasMidCircuitMeasurement || frontier))
                    ? 1
                    : 0);
  // Mid-circuit replay aggregates per-shot records into counts over unique
  // full records by default; the per-shot list on the sequential-data
  // channel is opt-in.
  batch.includeSequentialData = options.include_sequential_data;

  // Non-unitary sites are never pre-sampled: their branches are selected
  // during replay from true state-dependent probabilities, so only
  // unitary-mixture sites feed root trajectory generation. In tree mode the
  // unitary-mixture sites are folded into the frontier too (UnitaryBranch), so
  // no site is pre-sampled and the noise-free fallback below builds one
  // all-shots identity root carrying every branch site.
  auto sampledSites = std::move(noiseResult.noise_sites);
  if (batch.unitaryNoiseAsBranch)
    sampledSites.clear();
  else if (options.allow_non_unitary)
    std::erase_if(sampledSites,
                  [](const NoisePoint &point) { return point.is_non_unitary; });

  auto strategy = options.strategy
                      ? options.strategy
                      : std::make_shared<ProbabilisticSamplingStrategy>();
  std::size_t maxTrajs = options.max_trajectories.value_or(shots);
  cudaq::info("[ptsbe] Generating trajectories via {} strategy (max {})",
              strategy->name(), maxTrajs);
  batch.trajectories = strategy->generateTrajectories(sampledSites, maxTrajs);

  // A noise-free trace has no pre-sampled noise sites, so strategies produce
  // no trajectories. Execute it as one identity trajectory (probability 1,
  // no Kraus selections, multiplicity 1) so sampling and per-shot records
  // still run.
  if (batch.trajectories.empty() && sampledSites.empty() && shots > 0)
    batch.trajectories.push_back(
        KrausTrajectory::builder().setId(0).setProbability(1.0).build());

  if (!batch.trajectories.empty() && shots > 0)
    allocateShots(batch.trajectories, shots, options.shot_allocation);

  if (frontier)
    validateFrontierAllocation(batch);

  return batch;
}

} // namespace cudaq::ptsbe::detail
