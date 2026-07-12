/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file PTSBESample.h
/// @brief PTSBE sample API and execution internals
///
/// Provides the `cudaq::ptsbe` sample API:
///   ptsbe::sample_options opts;
///   opts.noise = noise_model;
///   auto result = ptsbe::sample(opts, kernel, args...);
///
/// `ptsbe::sample()` returns `ptsbe::sample_result` (subclass of
/// `cudaq::sample_result`) which may optionally carry execution data.
///
/// Mid-circuit measurement and reset are traced and validated: measure/reset
/// sites carry record indices, and measure-then-reset pairs are fused into
/// MeasureReset sites.
///
/// Limitations:
/// - Dynamic circuits (conditional feedback on measurement outcomes, per the
///   kernel's MLIR metadata) are rejected
/// - C++ library-mode kernels with host-side feedback (e.g. `if (mz(q))`)
///   are not detectable by tracing; a single branch is traced silently
/// - Non-unitary noise requires PTSBEOptions::allow_non_unitary and a
///   BatchSimulator backend; admitted channels branch during replay and the
///   generic per-trajectory sampler rejects them
///

#pragma once

#include "NoiseExtractor.h"
#include "PTSBEExecutionData.h"
#include "PTSBEOptions.h"
#include "PTSBESampleResult.h"
#include "PTSBESampler.h"
#include "ShotAllocationStrategy.h"
#include "common/ExecutionContext.h"
#include "common/Future.h"
#include "common/NoiseModel.h"
#include "cudaq/algorithms/broadcast.h"
#include "cudaq/platform.h"
#include "cudaq/platform/QuantumExecutionQueue.h"
#include "cudaq/runtime/logger/logger.h"
#include <future>
#include <optional>
#include <span>
#include <stdexcept>

namespace cudaq::ptsbe::detail {

/// @brief Validate kernel eligibility for PTSBE execution
///
/// Rejects dynamic circuits: kernels whose MLIR metadata flags conditional
/// feedback on measurement outcomes. Mid-circuit measurements (named or
/// unnamed) and resets are accepted. C++ library-mode host feedback is not
/// visible to this check; such kernels trace a single branch silently.
///
/// @param kernelName Name of the kernel being validated
/// @param ctx ExecutionContext populated after kernel tracing (unused;
///        retained for call-site stability)
/// @throws std::runtime_error if kernel is not eligible for PTSBE
void validatePTSBEKernel(const std::string &kernelName,
                         const ExecutionContext &ctx);

/// @brief Warn if kernel uses named measurement registers.
///
/// Only call when per-shot records are unavailable (no sequential data). When
/// records are produced, named registers are preserved as record-site names
/// on the result's record layout and no warning is needed.
///
/// @param kernelName Name of the kernel being validated
/// @param ctx ExecutionContext populated after kernel tracing
/// @return True if a warning was emitted, false otherwise
bool warnNamedRegisters(const std::string &kernelName, ExecutionContext &ctx);

/// @brief Validate platform preconditions for PTSBE execution
void validatePTSBEPreconditions(
    quantum_platform &platform,
    std::optional<std::size_t> qpu_id = std::nullopt);

/// @brief Build the PTSBE instruction sequence from a raw cudaq::Trace.
///
/// Converts Gate, Noise, Measurement, and Reset entries, then fuses each
/// Measurement whose next instruction touching its qubits is a Reset on the
/// same targets into a MeasureReset site, and finally assigns dense record
/// indices in trace order to every Measurement and MeasureReset instruction.
///
/// @param trace Raw circuit trace
/// @param noise_model Noise model used to resolve inline apply_noise channels
/// @return PTSBETrace with resolved channels for Noise entries
[[nodiscard]] PTSBETrace buildPTSBETrace(const cudaq::Trace &trace,
                                         const cudaq::noise_model &noise_model);

/// @brief Extract measured qubit IDs from the trace's measurement entries.
///
/// Scans the trace for Measurement and MeasureReset instructions and collects
/// their target qubit IDs in the order they first appear. Duplicates are
/// suppressed so each qubit appears at most once while preserving the
/// kernel's measurement ordering. MeasureReset sites are included so that a
/// TERMINAL measure-then-reset still records its bit: replay never applies
/// the trailing reset, and sampling the un-reset state yields exactly the
/// measurement outcome. Mid-circuit MeasureReset sites never reach terminal
/// sampling: dispatch routes mid-circuit traces to per-shot replay, which
/// collapses and records every measuring site directly.
///
/// @param trace PTSBE trace
/// @return Ordered, de-duplicated vector of measured qubit indices
std::vector<std::size_t>
extractMeasureQubits(std::span<const TraceInstruction> trace);

/// @brief Build the per-shot record layout from a PTSBE trace.
///
/// Walks Measurement and MeasureReset instructions in trace order and emits
/// one RecordSite per target qubit at record_index + k, mirroring how the
/// replay paths write record bits. `resets` is true for MeasureReset sites;
/// `terminal` is true when no later instruction in the trace touches the
/// site's qubit; `register_name` carries the kernel's measurement name.
///
/// @param trace PTSBE trace with record indices assigned
/// @return Record sites in record-index order
[[nodiscard]] std::vector<RecordSite>
buildRecordLayout(std::span<const TraceInstruction> trace);

/// @brief Deallocate qubit IDs leaked by the tracer context on the simulator
///
/// In the MLIR/JIT path, qubit allocation
/// (__quantum__rt__qubit_allocate_array in NVQIR.cpp) goes directly to the
/// simulator's allocateQubits, which increments the simulator's
/// QuditIdTracker::currentId. However, CircuitSimulator::deallocateQubits
/// is a no-op when an execution context is set (including tracer), so the
/// kernel's qubit deallocation never returns IDs to the simulator's tracker.
///
/// Without cleanup, each PTSBE tracer pass accumulates qubit IDs on the
/// simulator (first call gets [0,1], next gets [2,3], etc.). This causes
/// noise model key mismatches (noise defined for qubit [0] but the trace
/// now has qubit [2]) and eventual memory exhaustion on density-matrix
/// simulators.
///
/// This function collects all qubit IDs from the kernel trace and
/// deallocates them from the simulator. Must be called AFTER
/// with_execution_context returns (when the execution context is null and
/// deallocateQubits will actually execute).
///
/// @param kernelTrace Captured kernel trace containing qubit IDs
void cleanupTracerQubits(const Trace &kernelTrace);

/// @brief Build PTSBEExecutionData with interleaved instructions (no
/// trajectories)
///
/// Converts the internal kernel trace into the user-facing
/// PTSBEExecutionData format. For each gate in the kernel trace, a Gate
/// instruction is added. If the noise model defines noise at that gate, a
/// Noise instruction follows. Measurement instructions are appended for all
/// measured qubits. The trajectories vector is left empty.
///
/// @param kernelTrace Captured kernel trace
/// @param noiseModel Noise model for identifying noise sites
/// @return PTSBEExecutionData with interleaved instructions and empty
///         trajectories
PTSBEExecutionData
buildExecutionDataInstructions(const cudaq::Trace &kernelTrace,
                               const noise_model &noiseModel);

/// @brief Populate trajectories on an existing PTSBEExecutionData
///
/// Takes ownership of trajectories and results. Remaps each KrausSelection's
/// circuit_location from noise-site index to the corresponding Noise
/// instruction index in PTSBEExecutionData.instructions (derived by scanning
/// the instruction list). Populates measurement_counts from per-trajectory
/// execution results.
///
/// @param executionData PTSBEExecutionData to populate (must have instructions
///        already set)
/// @param trajectories Executed trajectories
/// @param perTrajectoryResults Per-trajectory sample results
void populateExecutionDataTrajectories(
    PTSBEExecutionData &executionData,
    std::vector<cudaq::KrausTrajectory> trajectories,
    std::vector<cudaq::sample_result> perTrajectoryResults);

/// @brief Build PTSBatch from a pre-built PTSBE trace
///
/// Extracts noise sites from the trace, generates trajectories via the
/// configured strategy (or default probabilistic), and allocates shots.
///
/// @param trace Pre-built PTSBE trace (moved into the batch)
/// @param options PTSBE configuration options
/// @param shots Total number of shots to allocate
/// @return PTSBatch ready for execution
PTSBatch buildPTSBatchFromTrace(PTSBETrace &&trace, const PTSBEOptions &options,
                                std::size_t shots);

/// @brief Run PTSBE sampling (internal API matching runSampling pattern)
///
/// Captures the kernel trace, builds PTSBEExecutionData, generates
/// trajectories, executes them, and aggregates results. Optionally attaches
/// the execution data (with trajectories and measurement counts) to the
/// result when return_execution_data is enabled.
///
/// The noise model must be set on the platform before calling this function.
///
/// @tparam KernelFunctor Wrapped kernel functor type
/// @param wrappedKernel Functor that invokes the quantum kernel
/// @param platform Reference to the quantum platform
/// @param kernelName Name of the kernel (for diagnostics and conditional
///        feedback detection)
/// @param shots Number of shots for trajectory allocation
/// @param options PTSBE configuration options
/// @return ptsbe::sample_result with optional execution data
/// @throws std::runtime_error if platform is not a simulator, noise model is
///         missing, or dynamic circuit detected
template <typename KernelFunctor>
sample_result runSamplingPTSBE(KernelFunctor &&wrappedKernel,
                               quantum_platform &platform,
                               const std::string &kernelName, std::size_t shots,
                               const PTSBEOptions &options = PTSBEOptions{}) {
  validatePTSBEPreconditions(platform);

  // Use platform noise if set; otherwise empty model
  static const cudaq::noise_model kEmptyNoiseModel;
  const auto *noisePtr = platform.get_noise();
  const auto &noiseModel = noisePtr ? *noisePtr : kEmptyNoiseModel;

  // Stage 0: Capture trace via ExecutionContext("tracer")
  ExecutionContext traceCtx("tracer");
  platform.with_execution_context(traceCtx, [&]() { wrappedKernel(); });
  cleanupTracerQubits(traceCtx.kernelTrace);
  cudaq::info("[ptsbe] Trace captured: {} qubits, {} instructions",
              traceCtx.kernelTrace.getNumQudits(),
              traceCtx.kernelTrace.getNumInstructions());

  // Stage 1: Validate kernel eligibility (no dynamic circuits)
  validatePTSBEKernel(kernelName, traceCtx);

  // Stage 2: Build PTSBE trace once, share between execution data and batch
  auto ptsbeTrace = buildPTSBETrace(traceCtx.kernelTrace, noiseModel);
  auto recordLayout = buildRecordLayout(ptsbeTrace);

  std::optional<PTSBEExecutionData> executionData;
  if (options.return_execution_data) {
    executionData = PTSBEExecutionData{};
    executionData->instructions = ptsbeTrace;
  }

  // Stage 3: Build PTSBatch with trajectory generation and shot allocation.
  // buildPTSBatchFromTrace sets includeSequentialData from the options,
  // forcing it on when the trace has mid-circuit measurement.
  auto batch = buildPTSBatchFromTrace(std::move(ptsbeTrace), options, shots);
  cudaq::info("[ptsbe] Allocated {} shots across {} trajectories",
              batch.totalShots(), batch.trajectories.size());

  // Named registers survive as record-site names whenever per-shot records
  // are produced (sequential data requested, or forced on by mid-circuit
  // measurement, which batch.includeSequentialData captures). Warn only when
  // records are unavailable.
  if (!batch.includeSequentialData)
    warnNamedRegisters(kernelName, traceCtx);

  // Stage 4: Execute PTSBE with life-cycle management
  auto perTrajectoryResults = samplePTSBEWithLifecycle(batch);

  // Stage 5: Aggregate per-trajectory results. Flat pooling equals the
  // root-weighted estimator f_hat = sum_u(d_u * fbar_u) / D whenever
  // num_root_draws is set, because validateFrontierAllocation enforces
  // N_u / total = d_u / D exactly.
  sample_result result(aggregateResults(perTrajectoryResults));
  result.set_record_layout(std::move(recordLayout));

  // Stage 6: Attach trajectories and set execution data on result if requested
  if (executionData) {
    populateExecutionDataTrajectories(*executionData,
                                      std::move(batch.trajectories),
                                      std::move(perTrajectoryResults));
    result.set_execution_data(std::move(*executionData));
  }

  cudaq::info("[ptsbe] Complete: {} unique bitstrings from {} shots",
              result.size(), result.get_total_shots());
  return result;
}

/// @brief Capture kernel trace and construct PTSBatch (for testing)
///
/// Helper function that captures trace and builds PTSBatch without dispatching.
/// Used by tests to verify trace capture and batch construction independently
/// of execution. Builds the PTSBE trace with an empty noise model (no Noise
/// entries). To build with a noise model, use buildPTSBETrace and
/// buildPTSBatchFromTrace.
///
/// @tparam QuantumKernel Quantum kernel type
/// @tparam Args Kernel argument types
/// @param kernel Quantum kernel to trace
/// @param args Kernel arguments
/// @return PTSBatch with trace, empty trajectories, and measureQubits
/// @throws std::runtime_error if conditional feedback detected
template <typename QuantumKernel, typename... Args>
PTSBatch tracePTSBatch(QuantumKernel &&kernel, Args &&...args) {
  ExecutionContext traceCtx("tracer");
  auto &platform = get_platform();
  platform.with_execution_context(
      traceCtx, [&]() { kernel(std::forward<Args>(args)...); });
  cleanupTracerQubits(traceCtx.kernelTrace);

  auto kernelName = cudaq::getKernelName(kernel);
  validatePTSBEKernel(kernelName, traceCtx);

  static const cudaq::noise_model kEmptyNoiseModel;
  PTSBatch batch;
  batch.trace = buildPTSBETrace(traceCtx.kernelTrace, kEmptyNoiseModel);
  batch.measureQubits = extractMeasureQubits(batch.trace);
  return batch;
}

/// @brief Return type for asynchronous PTSBE sampling
using async_sample_result = std::future<sample_result>;

/// @brief Run PTSBE sampling with asynchronous dispatch
///
/// Uses the get_state_async pattern: enqueues a void QuantumTask on the
/// platform's QPU execution queue with a self-managed promise/future pair.
/// This preserves the full ptsbe::sample_result type (including execution
/// data) without slicing through KernelExecutionTask.
///
/// @tparam KernelFunctor Wrapped kernel functor type
/// @param wrappedKernel Functor that invokes the quantum kernel
/// @param platform Reference to the quantum platform
/// @param kernelName Name of the kernel (for diagnostics and conditional
///        feedback detection)
/// @param shots Number of shots for trajectory allocation
/// @param options PTSBE configuration options
/// @param qpu_id The QPU ID to execute on
/// @param noise Optional noise model. Copied into the asynchronous task to
///        ensure proper lifetime. When absent, executes without noise.
/// @return future resolving to ptsbe::sample_result
/// @throws std::runtime_error if platform is remote (PTSBE is local-only)
template <typename KernelFunctor>
async_sample_result
runSamplingAsyncPTSBE(KernelFunctor &&wrappedKernel, quantum_platform &platform,
                      const std::string &kernelName, std::size_t shots,
                      const PTSBEOptions &options = PTSBEOptions{},
                      std::size_t qpu_id = 0,
                      std::optional<noise_model> noise = std::nullopt) {
  // Validate upfront so exceptions are thrown in calling thread
  validatePTSBEPreconditions(platform, qpu_id);

  std::promise<sample_result> promise;
  auto future = promise.get_future();

  const bool hasNoise = noise.has_value() && !noise->empty();

  QuantumTask task = cudaq::detail::make_copyable_function(
      [p = std::move(promise), shots, kernelName, &platform, options,
       kernel = std::forward<KernelFunctor>(wrappedKernel),
       noise = std::move(noise), hasNoise]() mutable {
        try {
          if (hasNoise)
            platform.set_noise(&noise.value());
          auto result =
              runSamplingPTSBE(kernel, platform, kernelName, shots, options);
          if (hasNoise)
            platform.reset_noise();
          p.set_value(std::move(result));
        } catch (...) {
          if (hasNoise)
            platform.reset_noise();
          p.set_exception(std::current_exception());
        }
      });

  platform.enqueueAsyncTask(qpu_id, task);
  return future;
}

} // namespace cudaq::ptsbe::detail

namespace cudaq::ptsbe {

/// @brief Public return type for asynchronous PTSBE sampling
using async_sample_result = std::future<sample_result>;

/// @brief Sample options for PTSBE execution
///
/// @param shots Number of shots to run for the given kernel
/// @param noise Noise model (required for PTSBE)
/// @param ptsbe PTSBE-specific configuration (execution data, strategy, etc.)
struct sample_options {
  std::size_t shots = 1000;
  cudaq::noise_model noise;
  PTSBEOptions ptsbe;
};

/// @brief Sample the given quantum kernel with PTSBE using a noise model
///
/// @param noise The noise model (required for PTSBE)
/// @param shots The number of shots to collect
/// @param kernel The kernel expression, must contain final measurements
/// @param args The variadic concrete arguments for evaluation of the kernel
/// @return ptsbe::sample_result with optional execution data
template <typename QuantumKernel, typename... Args>
sample_result sample(const cudaq::noise_model &noise, std::size_t shots,
                     QuantumKernel &&kernel, Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  platform.set_noise(&noise);

  sample_result result = detail::runSamplingPTSBE(
      [&]() mutable { kernel(std::forward<Args>(args)...); }, platform,
      kernelName, shots);

  platform.reset_noise();
  return result;
}

/// @brief Sample the given quantum kernel with PTSBE using sample_options
///
/// @param options PTSBE sample options (shots, noise, PTSBE configuration)
/// @param kernel The kernel expression, must contain final measurements
/// @param args The variadic concrete arguments for evaluation of the kernel
/// @return ptsbe::sample_result with measurement counts and optional execution
///         data
template <typename QuantumKernel, typename... Args>
sample_result sample(const sample_options &options, QuantumKernel &&kernel,
                     Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  platform.set_noise(&options.noise);

  sample_result result = detail::runSamplingPTSBE(
      [&]() mutable { kernel(std::forward<Args>(args)...); }, platform,
      kernelName, options.shots, options.ptsbe);

  platform.reset_noise();
  return result;
}

/// @brief Asynchronously sample with PTSBE using a noise model
///
/// @param noise The noise model (required for PTSBE)
/// @param shots The number of shots to collect
/// @param kernel The kernel expression, must contain final measurements
/// @param args The variadic concrete arguments for evaluation of the kernel
/// @return future resolving to ptsbe::sample_result
template <typename QuantumKernel, typename... Args>
async_sample_result sample_async(const cudaq::noise_model &noise,
                                 std::size_t shots, QuantumKernel &&kernel,
                                 Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);

  return detail::runSamplingAsyncPTSBE(
      [&]() mutable { kernel(std::forward<Args>(args)...); }, platform,
      kernelName, shots, PTSBEOptions{}, /*qpu_id=*/0, noise);
}

/// @brief Asynchronously sample with PTSBE using sample_options
///
/// @param options PTSBE sample options (shots, noise, PTSBE configuration)
/// @param kernel The kernel expression, must contain final measurements
/// @param args The variadic concrete arguments for evaluation of the kernel
/// @return future resolving to ptsbe::sample_result
template <typename QuantumKernel, typename... Args>
async_sample_result sample_async(const sample_options &options,
                                 QuantumKernel &&kernel, Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);

  return detail::runSamplingAsyncPTSBE(
      [&]() mutable { kernel(std::forward<Args>(args)...); }, platform,
      kernelName, options.shots, options.ptsbe, /*qpu_id=*/0, options.noise);
}

/// @brief Sample with PTSBE over a set of argument packs (broadcast)
///
/// For each element in the ArgumentSet, runs ptsbe::sample() and collects
/// the results. PTSBE is simulator-only so no multi-QPU distribution is used.
///
/// @param options PTSBE sample options (shots, noise, PTSBE configuration)
/// @param kernel The kernel expression, must contain final measurements
/// @param params ArgumentSet with one vector per kernel parameter
/// @return Vector of ptsbe::sample_result, one per argument set element
template <typename QuantumKernel, typename... Args>
std::vector<sample_result> sample(const sample_options &options,
                                  QuantumKernel &&kernel,
                                  ArgumentSet<Args...> &params) {
  auto N = std::get<0>(params).size();
  std::vector<sample_result> results;
  results.reserve(N);

  for (std::size_t i = 0; i < N; i++) {
    auto result = std::apply(
        [&](auto &...vecs) { return sample(options, kernel, vecs[i]...); },
        params);
    results.push_back(std::move(result));
  }
  return results;
}

} // namespace cudaq::ptsbe
