/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PTSBESamplerImpl.h"
#include "common/Environment.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/simulators.h"
#include <algorithm>
#include <numeric>
#include <span>
#include <stdexcept>

namespace cudaq::ptsbe::detail {

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
      replay.ops.emplace_back(inst.targets, inst.record_index,
                              /*resetAfter=*/false);
      break;
    case TraceInstructionType::Reset:
      replay.ops.emplace_back(inst.targets, std::nullopt,
                              /*resetAfter=*/true);
      break;
    case TraceInstructionType::MeasureReset:
      replay.ops.emplace_back(inst.targets, inst.record_index,
                              /*resetAfter=*/true);
      break;
    }

    while (noiseIdx < selections.size() &&
           selections[noiseIdx].circuit_location == i) {
      if (includeIdentity || selections[noiseIdx].is_error) {
        replay.krausTasks.push_back(
            krausSelectionToTask<ScalarType>(selections[noiseIdx], inst));
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

} // namespace cudaq::ptsbe::detail

namespace cudaq::ptsbe {

std::size_t PTSBatch::totalShots() const {
  std::size_t total = 0;
  for (const auto &traj : trajectories)
    total += traj.num_shots;
  return total;
}

std::size_t PTSBatch::numRecordBits() const {
  // Single-sourced through computeTraceLayout: PTSBatch is public, so a
  // hand-built trace may carry record indices in any order, and the layout
  // sizes the record from the maximum record_index + target count. The
  // per-shot replay writes record[record_index + k] and must never run past
  // the end.
  return computeTraceLayout(trace).numRecordBits;
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

  return GateTask<ScalarType>(inst.name, std::move(matrix), inst.controls,
                              inst.targets, std::move(typedParams));
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
  return GateTask<ScalarType>(std::move(opName), std::move(matrix), {},
                              sel.qubits, {});
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

cudaq::sample_result
aggregateResults(std::vector<cudaq::sample_result> &&results) {
  if (results.empty())
    return cudaq::sample_result{};

  // The caller no longer needs the per-trajectory results, so iterate each
  // trajectory's counts in place (to_map() would duplicate the whole map)
  // and move its sequential data out.
  cudaq::CountsDictionary aggregatedCounts;
  std::vector<std::string> aggregatedSeqData;
  for (auto &res : results) {
    if (res.get_total_shots() == 0)
      continue;

    for (auto it = res.cbegin(); it != res.cend(); ++it)
      aggregatedCounts[it->first] += it->second;

    auto seq = res.sequential_data();
    if (!seq.empty())
      aggregatedSeqData.insert(aggregatedSeqData.end(),
                               std::make_move_iterator(seq.begin()),
                               std::make_move_iterator(seq.end()));
  }
  cudaq::ExecutionResult er{std::move(aggregatedCounts)};
  er.sequentialData = std::move(aggregatedSeqData);
  return cudaq::sample_result{std::move(er)};
}

// Per-shot replay primitives for nvqir::CircuitSimulator: measure sites
// collapse each qubit with mz and (for measure-and-reset or bare reset sites)
// return it to |0>. flushGateQueue runs before each measurement and once at
// the end of a shot, because setToZeroState does not clear the pending gate
// queue (see CircuitSimulatorBase::deallocateQubits).
template <typename ScalarType>
struct GenericPerShotPolicy {
  nvqir::CircuitSimulatorBase<ScalarType> &simulator;

  void setToZeroState() { simulator.setToZeroState(); }

  void applyGate(const GateTask<ScalarType> &task) {
    simulator.applyGate(task);
  }

  void measureSite(const ReplayOp<ScalarType> &op, std::string &record) {
    simulator.flushGateQueue();
    if (op.recordOffset) {
      for (std::size_t k = 0; k < op.qubits.size(); ++k) {
        const bool bit = simulator.mz(op.qubits[k]);
        record[*op.recordOffset + k] = bit ? '1' : '0';
        if (op.resetAfter)
          simulator.resetQubit(op.qubits[k]);
      }
    } else {
      for (auto qubit : op.qubits)
        simulator.resetQubit(qubit);
    }
  }

  void finishShot() { simulator.flushGateQueue(); }
};

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

  // Terminal-only batches sample this qubit list after the gate walk; it is
  // read from the trace's measurement sites rather than a precomputed field.
  // Mid-circuit batches record every measuring site during per-shot replay.
  const std::vector<std::size_t> terminalQubits =
      batch.hasMidCircuitMeasurement ? std::vector<std::size_t>{}
                                     : terminalMeasureQubits(batch.trace);
  if (!batch.hasMidCircuitMeasurement && terminalQubits.empty())
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

    auto replay =
        mergeSitesWithTrajectory<ScalarType>(batch.trace, gateCacheView, traj);

    if (!batch.hasMidCircuitMeasurement) {
      simulator.setToZeroState();

      for (const auto &op : replay.ops)
        if (op.kind == ReplayOpKind::Gate)
          simulator.applyGate(*op.task);
      simulator.flushGateQueue();

      auto execResult =
          simulator.sample(terminalQubits, static_cast<int>(traj.num_shots),
                           batch.includeSequentialData);

      cudaq::ExecutionResult er{execResult.counts};
      if (batch.includeSequentialData)
        er.sequentialData = std::move(execResult.sequentialData);
      results.push_back(cudaq::sample_result{std::move(er)});
    } else {
      // Site-ordered per-shot replay: every measuring site (mid-circuit and
      // terminal) collapses via mz and writes its bit into the record, so no
      // separate terminal sampling pass runs. Each shot replays independently
      // to keep records within one trajectory decorrelated. Records aggregate
      // into counts; the per-shot list materializes only when sequential data
      // was requested.
      GenericPerShotPolicy<ScalarType> policy{simulator};
      auto er = replayTrajectoryPerShot<ScalarType>(
          replay.ops, traj.num_shots, numRecordBits,
          batch.includeSequentialData, policy);
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

/// Finalize and tear down the execution context and deallocate qubits.
/// finalizeExecutionContext must precede deallocateQubits because
/// CircuitSimulatorBase::deallocateQubits is a no-op while a context is set.
void teardown(nvqir::CircuitSimulator *sim, cudaq::ExecutionContext &ctx,
              std::size_t nQubits) {
  sim->finalizeExecutionContext(ctx);
  cudaq::detail::resetExecutionContext();
  std::vector<std::size_t> qubitIds(nQubits);
  std::iota(qubitIds.begin(), qubitIds.end(), 0);
  sim->deallocateQubits(qubitIds);
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

std::vector<cudaq::sample_result>
samplePTSBEWithLifecycle(const PTSBatch &batch,
                         const std::string &contextType) {
  ScopedTraceWithContext("ptsbe::samplePTSBEWithLifecycle");
  auto *sim = nvqir::getCircuitSimulatorInternal();
  const auto nQubits = numQubits(batch.trace);

  cudaq::ExecutionContext ctx(contextType, batch.totalShots());
  cudaq::detail::setExecutionContext(&ctx);
  sim->configureExecutionContext(ctx);
  sim->allocateQubits(nQubits);

  // Teardown must run on both normal exit and exception (e.g. std::bad_alloc
  // from a BatchSimulator). Without it, a thrown exception leaves the
  // execution context set, causing all subsequent simulator calls to fail
  // with "Context already set".
  std::vector<cudaq::sample_result> results;
  try {
    results = samplePTSBE(batch);
  } catch (...) {
    teardown(sim, ctx, nQubits);
    throw;
  }

  teardown(sim, ctx, nQubits);
  return results;
}

} // namespace cudaq::ptsbe::detail
