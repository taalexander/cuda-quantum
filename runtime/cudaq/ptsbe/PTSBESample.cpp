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

std::vector<std::size_t>
extractMeasureQubits(std::span<const TraceInstruction> trace) {
  std::vector<std::size_t> qubits;
  for (const auto &inst : trace) {
    if (inst.type != TraceInstructionType::Measurement &&
        inst.type != TraceInstructionType::MeasureReset)
      continue;
    for (auto id : inst.targets) {
      if (std::find(qubits.begin(), qubits.end(), id) == qubits.end())
        qubits.push_back(id);
    }
  }
  return qubits;
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

static void convertTraceInstruction(const cudaq::Trace::Instruction &inst,
                                    const cudaq::noise_model &noise_model,
                                    std::vector<TraceInstruction> &result) {
  auto targets = extractQubitIds(inst.targets);
  auto controls = extractQubitIds(inst.controls);

  if (inst.type == cudaq::TraceInstructionType::Reset) {
    result.push_back({TraceInstructionType::Reset, inst.name, targets, controls,
                      inst.params});
    return;
  }

  if (inst.type == cudaq::TraceInstructionType::Noise) {
    std::intptr_t key = inst.noise_channel_key.value();
    cudaq::kraus_channel channel = noise_model.get_channel(key, inst.params);
    if (!channel.empty()) {
      if (!channel.is_unitary_mixture())
        channel.generateUnitaryParameters();
      result.push_back({TraceInstructionType::Noise, inst.name, targets,
                        controls, inst.params, std::move(channel)});
    }
    return;
  }

  if (inst.type == cudaq::TraceInstructionType::Gate) {
    auto channels =
        noise_model.get_channels(inst.name, targets, controls, inst.params);
    result.push_back({TraceInstructionType::Gate, inst.name, targets, controls,
                      inst.params});

    std::vector<std::size_t> noiseQubits = targets;
    noiseQubits.insert(noiseQubits.end(), controls.begin(), controls.end());
    for (auto &channel : channels) {
      if (channel.empty())
        continue;
      if (!channel.is_unitary_mixture())
        channel.generateUnitaryParameters();
      auto parameters = channel.parameters;
      result.push_back({TraceInstructionType::Noise,
                        channel.get_type_name(),
                        noiseQubits,
                        {},
                        std::move(parameters),
                        std::move(channel)});
    }
    return;
  }

  if (inst.type == cudaq::TraceInstructionType::Measurement) {
    // Measurement noise precedes the Measurement so the recorded bit reflects
    // the noisy state. This keeps terminal-only traces free of instructions
    // after their measurements (hasMidCircuitMeasurement stays false) and
    // gives site-ordered replay the correct semantics.
    auto channels = noise_model.get_channels("mz", targets, {}, {});
    for (auto &channel : channels) {
      if (channel.empty())
        continue;
      if (!channel.is_unitary_mixture())
        channel.generateUnitaryParameters();
      auto parameters = channel.parameters;
      result.push_back({TraceInstructionType::Noise,
                        channel.get_type_name(),
                        targets,
                        {},
                        std::move(parameters),
                        std::move(channel)});
    }

    result.push_back({TraceInstructionType::Measurement,
                      inst.name,
                      targets,
                      {},
                      inst.params,
                      std::nullopt,
                      std::nullopt,
                      measurementRegisterName(inst)});
    return;
  }
}

static bool touchesAnyQubit(const TraceInstruction &inst,
                            const std::vector<std::size_t> &qubits) {
  auto touches = [&qubits](const std::vector<std::size_t> &ids) {
    return std::any_of(ids.begin(), ids.end(), [&qubits](std::size_t id) {
      return std::find(qubits.begin(), qubits.end(), id) != qubits.end();
    });
  };
  return touches(inst.targets) || touches(inst.controls);
}

/// Fuse each Measurement whose next instruction touching its qubits is a
/// Reset on exactly the same targets into a single MeasureReset site. The
/// fused site keeps the measurement's position and fields; the consumed
/// Reset is removed. Instructions on other qubits may sit between the pair.
static void fuseMeasureResetSites(PTSBETrace &trace) {
  std::vector<bool> consumed(trace.size(), false);
  for (std::size_t i = 0; i < trace.size(); ++i) {
    if (trace[i].type != TraceInstructionType::Measurement)
      continue;
    for (std::size_t j = i + 1; j < trace.size(); ++j) {
      if (consumed[j] || !touchesAnyQubit(trace[j], trace[i].targets))
        continue;
      if (trace[j].type == TraceInstructionType::Reset &&
          trace[j].targets == trace[i].targets) {
        trace[i].type = TraceInstructionType::MeasureReset;
        consumed[j] = true;
      }
      break;
    }
  }

  std::size_t out = 0;
  for (std::size_t i = 0; i < trace.size(); ++i) {
    if (consumed[i])
      continue;
    if (out != i)
      trace[out] = std::move(trace[i]);
    ++out;
  }
  trace.resize(out);
}

/// Assign dense record indices in trace order to every Measurement and
/// MeasureReset instruction, advancing by one bit per target qubit. This is
/// the single source of the per-shot record layout shared by all execution
/// paths, which write one record bit per target at record_index + k.
static void assignRecordIndices(PTSBETrace &trace) {
  std::size_t nextRecord = 0;
  for (auto &inst : trace)
    if (inst.type == TraceInstructionType::Measurement ||
        inst.type == TraceInstructionType::MeasureReset) {
      inst.record_index = nextRecord;
      nextRecord += inst.targets.size();
    }
}

PTSBETrace buildPTSBETrace(const cudaq::Trace &trace,
                           const cudaq::noise_model &noise_model) {
  PTSBETrace result;
  bool hasMeasurement = false;
  for (const auto &inst : trace) {
    if (inst.type == cudaq::TraceInstructionType::Measurement)
      hasMeasurement = true;
    convertTraceInstruction(inst, noise_model, result);
  }

  // Match standard cudaq::sample() behavior: when the kernel omits explicit
  // mz() calls, measure all allocated qubits. Generate one Noise + Measurement
  // pair per qubit (noise first, matching convertTraceInstruction) so that
  // per-qubit noise channels (registered via add_channel("mz", {q}, ...)) are
  // matched correctly.
  auto n = trace.getNumQudits();
  if (!hasMeasurement && n > 0) {
    for (std::size_t q = 0; q < n; ++q) {
      auto channels = noise_model.get_channels("mz", {q}, {}, {});
      for (auto &channel : channels) {
        if (channel.empty())
          continue;
        if (!channel.is_unitary_mixture())
          channel.generateUnitaryParameters();
        result.push_back({TraceInstructionType::Noise,
                          channel.get_type_name(),
                          {q},
                          {},
                          {},
                          std::move(channel)});
      }

      result.push_back({TraceInstructionType::Measurement, "mz", {q}, {}, {}});
    }
  }

  fuseMeasureResetSites(result);
  assignRecordIndices(result);
  return result;
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

PTSBatch buildPTSBatchFromTrace(PTSBETrace &&trace, const PTSBEOptions &options,
                                std::size_t shots) {
  PTSBatch batch;

  batch.trace = std::move(trace);
  batch.measureQubits = extractMeasureQubits(batch.trace);
  batch.hasMidCircuitMeasurement = hasMidCircuitMeasurement(batch.trace);
  auto envMaxShotsPerPath = maxShotsPerPathEnvOverride();
  if (envMaxShotsPerPath)
    cudaq::info("[ptsbe] max shots per slot set to {} via "
                "CUDAQ_PTSBE_MAX_SHOTS_PER_PATH",
                *envMaxShotsPerPath);
  batch.maxShotsPerPath = envMaxShotsPerPath.has_value()
                              ? *envMaxShotsPerPath
                              : options.max_shots_per_path.value_or(
                                    batch.hasMidCircuitMeasurement ? 1 : 0);
  // Mid-circuit replay produces per-shot records; sequential data is the
  // channel that carries them, so it is always on for such batches.
  batch.includeSequentialData =
      options.include_sequential_data || batch.hasMidCircuitMeasurement;
  auto noiseResult = extractNoiseSites(batch.trace);
  cudaq::info("[ptsbe] Extracted {} noise sites from {} total instructions",
              noiseResult.noise_sites.size(), noiseResult.total_instructions);

  auto strategy = options.strategy
                      ? options.strategy
                      : std::make_shared<ProbabilisticSamplingStrategy>();
  std::size_t maxTrajs = options.max_trajectories.value_or(shots);
  cudaq::info("[ptsbe] Generating trajectories via {} strategy (max {})",
              strategy->name(), maxTrajs);
  batch.trajectories =
      strategy->generateTrajectories(noiseResult.noise_sites, maxTrajs);

  if (!batch.trajectories.empty() && shots > 0)
    allocateShots(batch.trajectories, shots, options.shot_allocation);

  return batch;
}

} // namespace cudaq::ptsbe::detail
