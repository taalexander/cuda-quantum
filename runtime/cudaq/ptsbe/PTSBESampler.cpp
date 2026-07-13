/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PTSBESamplerImpl.h"
#include "common/Environment.h"
#include "common/ExecutionContext.h"
#include "cudaq/ptsbe/policy.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/simulators.h"
#include <algorithm>
#include <numeric>
#include <span>
#include <stdexcept>

namespace cudaq::ptsbe {

template <typename ScalarType>
TrajectoryReplay<ScalarType>
mergeSitesWithTrajectory(std::span<const TraceInstruction> ptsbeTrace,
                         std::span<const GateTask<ScalarType>> gateCache,
                         const cudaq::KrausTrajectory &trajectory,
                         bool includeIdentity) {
  const auto &selections = trajectory.kraus_selections;

  TrajectoryReplay<ScalarType> replay;
  // Exact reservation keeps krausTasks stable while ops point into it.
  replay.krausTasks.reserve(selections.size());
  replay.ops.reserve(ptsbeTrace.size());

  std::size_t noiseIdx = 0;
  std::size_t gateIdx = 0;
  for (std::size_t i = 0; i < ptsbeTrace.size(); ++i) {
    const auto &inst = ptsbeTrace[i];

    switch (inst.type) {
    case TraceInstructionType::Gate:
      if (gateIdx >= gateCache.size())
        throw std::runtime_error(
            "mergeSitesWithTrajectory: gate cache has " +
            std::to_string(gateCache.size()) +
            " tasks but the trace has more Gate instructions; build the "
            "cache from the same trace with convertTraceGates");
      replay.ops.emplace_back(&gateCache[gateIdx++]);
      break;
    case TraceInstructionType::Noise:
      // Unitary-mixture noise was pre-sampled into the trajectory and is
      // resolved through the selection loop below. Non-unitary channels are
      // never pre-sampled; they branch during replay at their true
      // state-dependent probabilities.
      if (inst.channel && !inst.channel->is_unitary_mixture())
        replay.ops.emplace_back(&inst.channel.value(), inst.targets);
      break;
    case TraceInstructionType::Measurement:
      replay.ops.emplace_back(ReplayOpKind::Measure, inst.targets,
                              inst.record_index);
      break;
    case TraceInstructionType::Reset:
      replay.ops.emplace_back(ReplayOpKind::Reset, inst.targets, std::nullopt);
      break;
    case TraceInstructionType::MeasureReset:
      replay.ops.emplace_back(ReplayOpKind::MeasureReset, inst.targets,
                              inst.record_index);
      break;
    }

    while (noiseIdx < selections.size() &&
           selections[noiseIdx].circuit_location == i) {
      if (includeIdentity || selections[noiseIdx].is_error) {
        replay.krausTasks.push_back(detail::krausSelectionToTask<ScalarType>(
            selections[noiseIdx], inst));
        replay.ops.emplace_back(&replay.krausTasks.back());
      }
      ++noiseIdx;
    }
  }

  if (noiseIdx < selections.size()) {
    throw std::runtime_error(
        "Invalid circuit_location: " +
        std::to_string(selections[noiseIdx].circuit_location) +
        " >= " + std::to_string(ptsbeTrace.size()));
  }

  return replay;
}

template TrajectoryReplay<float>
mergeSitesWithTrajectory<float>(std::span<const TraceInstruction>,
                                std::span<const GateTask<float>>,
                                const cudaq::KrausTrajectory &, bool);
template TrajectoryReplay<double>
mergeSitesWithTrajectory<double>(std::span<const TraceInstruction>,
                                 std::span<const GateTask<double>>,
                                 const cudaq::KrausTrajectory &, bool);

std::size_t PTSBatch::totalShots() const {
  std::size_t total = 0;
  for (const auto &traj : trajectories)
    total += traj.num_shots;
  return total;
}

std::size_t PTSBatch::numRecordBits() const {
  // PTSBatch is public, so a hand-built trace may carry record indices in
  // any order. The max-scan sizes the record correctly regardless; the
  // per-shot replay writes record[record_index + k] and must never run
  // past the end.
  std::size_t bits = 0;
  for (const auto &inst : trace)
    if (inst.record_index)
      bits = std::max(bits, *inst.record_index + inst.targets.size());
  return bits;
}

} // namespace cudaq::ptsbe

namespace cudaq::ptsbe::detail {

template <typename ScalarType>
GateTask<ScalarType> convertToSimulatorTask(const TraceInstruction &inst) {
  std::vector<ScalarType> typedParams;
  typedParams.reserve(inst.params.size());
  for (auto p : inst.params)
    typedParams.push_back(static_cast<ScalarType>(p));

  auto gateName = nvqir::getGateNameFromString(inst.name);
  auto matrix = nvqir::getGateByName<ScalarType>(gateName, typedParams);

  return GateTask<ScalarType>(inst.name, matrix, inst.controls, inst.targets,
                              typedParams);
}

template <typename ScalarType>
std::vector<GateTask<ScalarType>>
convertTraceGates(std::span<const TraceInstruction> ptsbeTrace) {
  std::vector<GateTask<ScalarType>> tasks;
  tasks.reserve(ptsbeTrace.size());
  for (const auto &inst : ptsbeTrace)
    if (inst.type == TraceInstructionType::Gate)
      tasks.push_back(convertToSimulatorTask<ScalarType>(inst));
  return tasks;
}

template <typename ScalarType>
GateTask<ScalarType> krausSelectionToTask(const cudaq::KrausSelection &sel,
                                          const TraceInstruction &noiseInst) {
  if (noiseInst.type != TraceInstructionType::Noise || !noiseInst.channel)
    throw std::runtime_error(
        "krausSelectionToTask: expected Noise instruction with a channel at "
        "circuit_location " +
        std::to_string(sel.circuit_location));
  const auto &channel = noiseInst.channel.value();
  auto k = sel.kraus_operator_index;
  const auto &unitaryDouble = channel.unitary_ops.at(k);
  std::vector<std::complex<ScalarType>> matrix;
  matrix.reserve(unitaryDouble.size());
  for (const auto &elem : unitaryDouble)
    matrix.emplace_back(static_cast<ScalarType>(elem.real()),
                        static_cast<ScalarType>(elem.imag()));
  std::string opName;
  if (k < channel.op_names.size())
    opName = channel.op_names[k];
  else
    opName = channel.get_type_name() + "[" + std::to_string(k) + "]";
  return GateTask<ScalarType>(opName, matrix, {}, sel.qubits, {});
}

template GateTask<float>
convertToSimulatorTask<float>(const TraceInstruction &);
template GateTask<double>
convertToSimulatorTask<double>(const TraceInstruction &);

template std::vector<GateTask<float>>
    convertTraceGates<float>(std::span<const TraceInstruction>);
template std::vector<GateTask<double>>
    convertTraceGates<double>(std::span<const TraceInstruction>);

template GateTask<float>
krausSelectionToTask<float>(const cudaq::KrausSelection &,
                            const TraceInstruction &);
template GateTask<double>
krausSelectionToTask<double>(const cudaq::KrausSelection &,
                             const TraceInstruction &);

cudaq::sample_result
aggregateResults(const std::vector<cudaq::sample_result> &results) {
  if (results.empty())
    return cudaq::sample_result{};

  cudaq::CountsDictionary aggregatedCounts;
  std::vector<std::string> aggregatedSeqData;
  for (const auto &res : results) {
    if (res.get_total_shots() == 0)
      continue;

    for (const auto &[bitstring, count] : res.to_map())
      aggregatedCounts[bitstring] += count;

    auto seq = res.sequential_data();
    if (!seq.empty())
      aggregatedSeqData.insert(aggregatedSeqData.end(),
                               std::make_move_iterator(seq.begin()),
                               std::make_move_iterator(seq.end()));
  }
  cudaq::ExecutionResult er{aggregatedCounts};
  er.sequentialData = std::move(aggregatedSeqData);
  return cudaq::sample_result{std::move(er)};
}

template <typename ScalarType>
std::vector<cudaq::sample_result>
samplePTSBEGeneric(nvqir::CircuitSimulatorBase<ScalarType> &simulator,
                   const PTSBatch &batch) {
  ScopedTraceWithContext("ptsbe::samplePTSBEGeneric",
                         batch.trajectories.size());
  if (!cudaq::getExecutionContext())
    throw std::runtime_error(
        "samplePTSBEGeneric requires ExecutionContext to be set. "
        "Use cudaq::detail::setExecutionContext() before invoking.");

  if (batch.trajectories.empty())
    return {};

  std::size_t totalShots = batch.totalShots();
  if (totalShots == 0)
    return {};

  if (!batch.hasMidCircuitMeasurement && batch.measureQubits.empty())
    return {};

  // Replay-time Kraus branching needs per-branch state probabilities, which
  // nvqir::CircuitSimulator does not expose. Only BatchSimulator backends
  // execute non-unitary sites.
  for (const auto &inst : batch.trace)
    if (inst.type == TraceInstructionType::Noise && inst.channel &&
        !inst.channel->is_unitary_mixture())
      throw std::runtime_error(
          "PTSBE generic replay does not support non-unitary Kraus channels "
          "(channel '" +
          inst.name +
          "'). General Kraus sites branch at state-dependent probabilities "
          "during replay, which requires a BatchSimulator backend.");

  const std::size_t numRecordBits =
      batch.hasMidCircuitMeasurement ? batch.numRecordBits() : 0;

  // Trace gates convert once per batch; each trajectory merge converts only
  // its Kraus selections. The cache must outlive every replay list below.
  const auto gateCache = convertTraceGates<ScalarType>(batch.trace);
  const std::span<const GateTask<ScalarType>> gateCacheView{gateCache};

  std::vector<cudaq::sample_result> results;
  results.reserve(batch.trajectories.size());

  const std::size_t numTrajectories = batch.trajectories.size();
  // Log progress at ~10% intervals (at least every 100 trajectories)
  const std::size_t progressInterval = std::max<std::size_t>(
      1, std::min<std::size_t>(numTrajectories / 10, 100));

  for (std::size_t ti = 0; ti < numTrajectories; ++ti) {
    const auto &traj = batch.trajectories[ti];
    if (traj.num_shots == 0) {
      results.push_back(cudaq::sample_result{
          cudaq::ExecutionResult{cudaq::CountsDictionary{}}});
      continue;
    }

    auto replay = cudaq::ptsbe::mergeSitesWithTrajectory<ScalarType>(
        batch.trace, gateCacheView, traj);

    if (!batch.hasMidCircuitMeasurement) {
      simulator.setToZeroState();

      for (const auto &op : replay.ops)
        if (op.kind == ReplayOpKind::Gate)
          simulator.applyGate(*op.task);
      simulator.flushGateQueue();

      auto execResult = simulator.sample(batch.measureQubits,
                                         static_cast<int>(traj.num_shots),
                                         batch.includeSequentialData);

      cudaq::ExecutionResult er{execResult.counts};
      if (batch.includeSequentialData)
        er.sequentialData = std::move(execResult.sequentialData);
      results.push_back(cudaq::sample_result{std::move(er)});
    } else {
      // Site-ordered per-shot replay: every measuring site (mid-circuit and
      // terminal) collapses via mz and writes its bit into the record, so no
      // separate terminal sampling pass runs. Each shot replays
      // independently to keep records within one trajectory decorrelated.
      // Records aggregate into counts; the per-shot list materializes only
      // when sequential data was requested.
      cudaq::CountsDictionary counts;
      std::vector<std::string> sequentialData;
      if (batch.includeSequentialData)
        sequentialData.reserve(traj.num_shots);

      for (std::size_t shot = 0; shot < traj.num_shots; ++shot) {
        simulator.setToZeroState();
        std::string record(numRecordBits, '0');

        for (const auto &op : replay.ops) {
          switch (op.kind) {
          case ReplayOpKind::Gate:
            simulator.applyGate(*op.task);
            break;
          case ReplayOpKind::Measure:
          case ReplayOpKind::MeasureReset:
            simulator.flushGateQueue();
            for (std::size_t k = 0; k < op.qubits.size(); ++k) {
              const bool bit = simulator.mz(op.qubits[k]);
              if (op.recordOffset)
                record[*op.recordOffset + k] = bit ? '1' : '0';
              if (op.kind == ReplayOpKind::MeasureReset)
                simulator.resetQubit(op.qubits[k]);
            }
            break;
          case ReplayOpKind::Reset:
            simulator.flushGateQueue();
            for (auto qubit : op.qubits)
              simulator.resetQubit(qubit);
            break;
          case ReplayOpKind::KrausBranch:
            // Unreachable: non-unitary sites are rejected before replay.
            throw std::runtime_error(
                "PTSBE generic replay cannot execute a KrausBranch site.");
          }
        }
        simulator.flushGateQueue();

        ++counts[record];
        if (batch.includeSequentialData)
          sequentialData.push_back(std::move(record));
      }

      cudaq::ExecutionResult er{std::move(counts)};
      er.sequentialData = std::move(sequentialData);
      results.push_back(cudaq::sample_result{std::move(er)});
    }

    if ((ti + 1) % progressInterval == 0)
      cudaq::info("[ptsbe] Trajectory progress: {}/{} ({} shots)", ti + 1,
                  numTrajectories, traj.num_shots);
  }

  return results;
}

// Explicit instantiations for samplePTSBEGeneric
template std::vector<cudaq::sample_result>
samplePTSBEGeneric(nvqir::CircuitSimulatorBase<float> &, const PTSBatch &);
template std::vector<cudaq::sample_result>
samplePTSBEGeneric(nvqir::CircuitSimulatorBase<double> &, const PTSBatch &);

namespace {

/// @brief Helper template for simulator dispatch
template <typename SimulatorType>
std::vector<cudaq::sample_result> dispatchPTSBE(SimulatorType &sim,
                                                const PTSBatch &batch) {

  // Check env var to force the generic (per-trajectory sampler) path,
  // bypassing the batched batchMeasure path which has a per-shot GPU loop.
  const bool forceGeneric =
      cudaq::getEnvBool("CUDAQ_PTSBE_FORCE_GENERIC", false);

  if (!forceGeneric) {
    auto *batchSim = dynamic_cast<BatchSimulator *>(&sim);
    if (batchSim) {
      cudaq::info(
          "[ptsbe] Dispatching to BatchSimulator custom implementation");
      return batchSim->sampleWithPTSBE(batch);
    }
  } else {
    cudaq::info("[ptsbe] BatchSimulator dispatch overridden by "
                "CUDAQ_PTSBE_FORCE_GENERIC=1");
  }

  cudaq::info("[ptsbe] Dispatching to generic per-trajectory sampler");
  return samplePTSBEGeneric(sim, batch);
}

} // namespace

std::vector<cudaq::sample_result> samplePTSBE(const PTSBatch &batch) {
  auto *baseSim = nvqir::getCircuitSimulatorInternal();

  if (baseSim->isSinglePrecision()) {
    auto *sim = dynamic_cast<nvqir::CircuitSimulatorBase<float> *>(baseSim);
    if (!sim)
      throw std::runtime_error(
          "Failed to cast simulator to CircuitSimulatorBase<float>");
    return dispatchPTSBE(*sim, batch);
  } else {
    auto *sim = dynamic_cast<nvqir::CircuitSimulatorBase<double> *>(baseSim);
    if (!sim)
      throw std::runtime_error(
          "Failed to cast simulator to CircuitSimulatorBase<double>");
    return dispatchPTSBE(*sim, batch);
  }
}

ptsbe::sample_result finalizePTSBE(const cudaq::ptsbe::sample_policy &policy) {
  if (!policy.batch)
    throw std::runtime_error(
        "ptsbe::sample_policy has no PTSBatch attached. PTSBE cannot be "
        "finalized by name-only dispatch.");

  auto results = samplePTSBE(*policy.batch);
  auto aggregated = aggregateResults(results);
  policy.perTrajectoryResults = std::move(results);
  return ptsbe::sample_result(std::move(aggregated));
}

void allocateBatchQubits(std::size_t nQubits) {
  nvqir::getCircuitSimulatorInternal()->allocateQubits(nQubits);
}

void releaseBatchQubits(std::size_t nQubits) {
  std::vector<std::size_t> qubitIds(nQubits);
  std::iota(qubitIds.begin(), qubitIds.end(), 0);
  auto *ctx = cudaq::getExecutionContext();
  if (ctx)
    cudaq::detail::resetExecutionContext();
  try {
    nvqir::getCircuitSimulatorInternal()->deallocateQubits(qubitIds);
  } catch (...) {
    if (ctx)
      cudaq::detail::setExecutionContext(ctx);
    throw;
  }
  if (ctx)
    cudaq::detail::setExecutionContext(ctx);
}

} // namespace cudaq::ptsbe::detail
