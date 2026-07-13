/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/algorithms/draw.h"
#include "cudaq/ptsbe/PTSBEExecutionData.h"
#include "cudaq/ptsbe/PTSBESample.h"
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace cudaq::ptsbe;

namespace {

auto measureResetKernel = []() __qpu__ {
  cudaq::qubit q;
  h(q);
  mz(q);
  reset(q);
  x(q);
  mz(q);
};

auto bareResetKernel = []() __qpu__ {
  cudaq::qubit q;
  h(q);
  reset(q);
  x(q);
  mz(q);
};

auto interveningOpKernel = []() __qpu__ {
  cudaq::qubit q;
  h(q);
  mz(q);
  x(q);
  reset(q);
  mz(q);
};

auto crossQubitKernel = []() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  mz(q[0]);
  x(q[1]);
  reset(q[0]);
  mz(q[0]);
  mz(q[1]);
};

auto namedMcmKernel = []() __qpu__ {
  cudaq::qubit q;
  h(q);
  auto b = mz(q);
  reset(q);
  x(q);
  mz(q);
};

auto terminalMeasureResetKernel = []() __qpu__ {
  cudaq::qubit q;
  h(q);
  mz(q);
  reset(q);
};

auto mixedTerminalKernel = []() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  x<cudaq::ctrl>(q[0], q[1]);
  mz(q[0]);
  reset(q[0]);
  mz(q[1]);
};

auto terminalOnlyKernel = []() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  x<cudaq::ctrl>(q[0], q[1]);
  mz(q);
};

auto terminalPerQubitMzKernel = []() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  x<cudaq::ctrl>(q[0], q[1]);
  mz(q[0]);
  mz(q[1]);
};

template <typename QuantumKernel>
PTSBETrace traceKernel(QuantumKernel &&kernel,
                       const cudaq::noise_model &noiseModel) {
  cudaq::ExecutionContext traceCtx("tracer");
  auto &platform = cudaq::get_platform();
  platform.with_execution_context(traceCtx, [&]() { kernel(); });
  detail::cleanupTracerQubits(traceCtx.kernelTrace);
  return detail::buildPTSBETrace(traceCtx.kernelTrace, noiseModel);
}

template <typename QuantumKernel>
PTSBETrace traceKernel(QuantumKernel &&kernel) {
  static const cudaq::noise_model kEmptyNoiseModel;
  return traceKernel(std::forward<QuantumKernel>(kernel), kEmptyNoiseModel);
}

std::vector<std::size_t> recordIndicesInOrder(const PTSBETrace &trace) {
  std::vector<std::size_t> indices;
  for (const auto &inst : trace)
    if (inst.record_index.has_value())
      indices.push_back(*inst.record_index);
  return indices;
}

} // namespace

// mz(q); reset(q) on the same qubit with no intervening op fuses into a
// single MeasureReset site. Program order around the fused site is kept:
// Gate(h), MeasureReset(q), Gate(x), Measurement(q).
CUDAQ_TEST(McmTraceTest, MeasureThenResetFusesIntoMeasureResetSite) {
  auto trace = traceKernel(measureResetKernel);

  ASSERT_EQ(trace.size(), 4u);
  EXPECT_EQ(trace[0].type, TraceInstructionType::Gate);
  EXPECT_EQ(trace[0].name, "h");

  EXPECT_EQ(trace[1].type, TraceInstructionType::MeasureReset);
  ASSERT_EQ(trace[1].targets.size(), 1u);
  EXPECT_EQ(trace[1].targets[0], 0u);

  EXPECT_EQ(trace[2].type, TraceInstructionType::Gate);
  EXPECT_EQ(trace[2].name, "x");

  EXPECT_EQ(trace[3].type, TraceInstructionType::Measurement);
  ASSERT_EQ(trace[3].targets.size(), 1u);
  EXPECT_EQ(trace[3].targets[0], 0u);
}

// Record indices are assigned once in buildPTSBETrace: dense, ordered, and
// covering both mid-circuit and terminal measurement sites. Non-measurement
// entries carry no record index.
CUDAQ_TEST(McmTraceTest, RecordIndicesDenseOrderedIncludingTerminal) {
  auto trace = traceKernel(measureResetKernel);

  for (const auto &inst : trace) {
    const bool isMeasurementSite =
        inst.type == TraceInstructionType::Measurement ||
        inst.type == TraceInstructionType::MeasureReset;
    EXPECT_EQ(inst.record_index.has_value(), isMeasurementSite);
  }

  EXPECT_EQ(recordIndicesInOrder(trace), (std::vector<std::size_t>{0, 1}));
  ASSERT_TRUE(trace[1].record_index.has_value());
  EXPECT_EQ(*trace[1].record_index, 0u);
  ASSERT_TRUE(trace[3].record_index.has_value());
  EXPECT_EQ(*trace[3].record_index, 1u);
}

// A reset that is not immediately preceded by a measurement of the same
// qubit stays a first-class Reset site and gets no record index.
CUDAQ_TEST(McmTraceTest, BareResetStaysResetSite) {
  auto trace = traceKernel(bareResetKernel);

  ASSERT_EQ(trace.size(), 4u);
  EXPECT_EQ(trace[0].type, TraceInstructionType::Gate);
  EXPECT_EQ(trace[1].type, TraceInstructionType::Reset);
  ASSERT_EQ(trace[1].targets.size(), 1u);
  EXPECT_EQ(trace[1].targets[0], 0u);
  EXPECT_FALSE(trace[1].record_index.has_value());
  EXPECT_EQ(trace[2].type, TraceInstructionType::Gate);
  EXPECT_EQ(trace[3].type, TraceInstructionType::Measurement);
  ASSERT_TRUE(trace[3].record_index.has_value());
  EXPECT_EQ(*trace[3].record_index, 0u);
}

// An intervening op on the same qubit between mz and reset prevents fusion:
// the measurement and the reset remain separate sites.
CUDAQ_TEST(McmTraceTest, InterveningOpOnSameQubitPreventsFusion) {
  auto trace = traceKernel(interveningOpKernel);

  ASSERT_EQ(trace.size(), 5u);
  EXPECT_EQ(trace[0].type, TraceInstructionType::Gate);
  EXPECT_EQ(trace[1].type, TraceInstructionType::Measurement);
  EXPECT_EQ(trace[2].type, TraceInstructionType::Gate);
  EXPECT_EQ(trace[2].name, "x");
  EXPECT_EQ(trace[3].type, TraceInstructionType::Reset);
  EXPECT_EQ(trace[4].type, TraceInstructionType::Measurement);

  EXPECT_EQ(countInstructions(trace, TraceInstructionType::MeasureReset), 0u);
  EXPECT_EQ(recordIndicesInOrder(trace), (std::vector<std::size_t>{0, 1}));
}

// An intervening op on a DIFFERENT qubit does not block fusion. This is the
// surface-code shape: data-qubit ops interleave between an ancilla's mz and
// its reset.
CUDAQ_TEST(McmTraceTest, OpOnOtherQubitDoesNotPreventFusion) {
  auto trace = traceKernel(crossQubitKernel);

  ASSERT_EQ(trace.size(), 5u);
  EXPECT_EQ(trace[0].type, TraceInstructionType::Gate);
  EXPECT_EQ(trace[0].name, "h");

  EXPECT_EQ(trace[1].type, TraceInstructionType::MeasureReset);
  ASSERT_EQ(trace[1].targets.size(), 1u);
  EXPECT_EQ(trace[1].targets[0], 0u);

  EXPECT_EQ(trace[2].type, TraceInstructionType::Gate);
  EXPECT_EQ(trace[2].name, "x");
  ASSERT_EQ(trace[2].targets.size(), 1u);
  EXPECT_EQ(trace[2].targets[0], 1u);

  EXPECT_EQ(trace[3].type, TraceInstructionType::Measurement);
  EXPECT_EQ(trace[3].targets[0], 0u);
  EXPECT_EQ(trace[4].type, TraceInstructionType::Measurement);
  EXPECT_EQ(trace[4].targets[0], 1u);

  EXPECT_EQ(countInstructions(trace, TraceInstructionType::Reset), 0u);
  EXPECT_EQ(recordIndicesInOrder(trace), (std::vector<std::size_t>{0, 1, 2}));
}

CUDAQ_TEST(McmTraceTest, HasMidCircuitMeasurementTrueForMcmKernel) {
  auto trace = traceKernel(measureResetKernel);
  EXPECT_TRUE(hasMidCircuitMeasurement(trace));
}

CUDAQ_TEST(McmTraceTest, HasMidCircuitMeasurementFalseForTerminalOnlyKernel) {
  auto trace = traceKernel(terminalOnlyKernel);
  EXPECT_FALSE(hasMidCircuitMeasurement(trace));

  // Terminal-only kernels still get dense record indices.
  EXPECT_EQ(recordIndicesInOrder(trace), (std::vector<std::size_t>{0, 1}));
}

// Readout noise on terminal measurements must not flag the trace as
// mid-circuit: measurement noise is emitted BEFORE its Measurement, so
// nothing follows a terminal measurement on its qubit and the recorded bit
// reflects the noisy state.
CUDAQ_TEST(McmTraceTest, TerminalKernelWithMzNoiseIsNotMidCircuit) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("mz", cudaq::bit_flip_channel(0.1));
  auto trace = traceKernel(terminalPerQubitMzKernel, noise);

  EXPECT_FALSE(hasMidCircuitMeasurement(trace));

  // [Gate(h), Gate(x), Noise(q0), Measurement(q0), Noise(q1), Measurement(q1)]
  ASSERT_EQ(trace.size(), 6u);
  EXPECT_EQ(trace[2].type, TraceInstructionType::Noise);
  EXPECT_EQ(trace[2].targets, (std::vector<std::size_t>{0}));
  EXPECT_EQ(trace[3].type, TraceInstructionType::Measurement);
  EXPECT_EQ(trace[3].targets, (std::vector<std::size_t>{0}));
  EXPECT_EQ(trace[4].type, TraceInstructionType::Noise);
  EXPECT_EQ(trace[4].targets, (std::vector<std::size_t>{1}));
  EXPECT_EQ(trace[5].type, TraceInstructionType::Measurement);
  EXPECT_EQ(trace[5].targets, (std::vector<std::size_t>{1}));

  EXPECT_EQ(recordIndicesInOrder(trace), (std::vector<std::size_t>{0, 1}));
}

// Measurement noise emitted before the Measurement must not block
// measure-then-reset fusion.
CUDAQ_TEST(McmTraceTest, MzNoiseDoesNotBlockMeasureResetFusion) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("mz", cudaq::bit_flip_channel(0.1));
  auto trace = traceKernel(measureResetKernel, noise);

  EXPECT_TRUE(hasMidCircuitMeasurement(trace));
  EXPECT_EQ(countInstructions(trace, TraceInstructionType::MeasureReset), 1u);
  EXPECT_EQ(countInstructions(trace, TraceInstructionType::Reset), 0u);
}

// A kernel whose last ops on a qubit are mz then reset fuses into a TERMINAL
// MeasureReset site: the trace is not mid-circuit and the fused qubit must
// still be sampled terminally (the trailing reset cannot affect the recorded
// bit).
CUDAQ_TEST(McmTraceTest, TerminalMeasureResetIsNotMidCircuit) {
  auto trace = traceKernel(terminalMeasureResetKernel);

  ASSERT_EQ(trace.size(), 2u);
  EXPECT_EQ(trace[0].type, TraceInstructionType::Gate);
  EXPECT_EQ(trace[1].type, TraceInstructionType::MeasureReset);
  EXPECT_FALSE(hasMidCircuitMeasurement(trace));

  auto measureQubits = detail::terminalMeasureQubits(trace);
  EXPECT_EQ(measureQubits, (std::vector<std::size_t>{0}));
}

// End-to-end: sampling a terminal mz;reset kernel must produce the same
// counts as the kernel without the trailing reset, not silently drop the
// fused measurement's record.
CUDAQ_TEST(McmTraceTest, SamplingTerminalMeasureResetKeepsRecord) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  const std::size_t shots = 1000;
  auto result = cudaq::ptsbe::sample(noise, shots, terminalMeasureResetKernel);

  EXPECT_EQ(result.get_total_shots(), shots);
  for (const auto &[bits, count] : result.to_map())
    EXPECT_EQ(bits.size(), 1u);
  EXPECT_GT(result.count("0"), 0u);
  EXPECT_GT(result.count("1"), 0u);
}

// Mixed fused-terminal and plain-terminal measurements: both bits appear in
// the counts, in measurement order. The Bell pair pins the correlation.
CUDAQ_TEST(McmTraceTest, SamplingMixedTerminalMeasureResetKeepsAllBits) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  const std::size_t shots = 1000;
  auto result = cudaq::ptsbe::sample(noise, shots, mixedTerminalKernel);

  EXPECT_EQ(result.get_total_shots(), shots);
  std::size_t correlated = 0;
  for (const auto &[bits, count] : result.to_map()) {
    EXPECT_EQ(bits.size(), 2u);
    if (bits == "00" || bits == "11")
      correlated += count;
  }
  EXPECT_GT(correlated, shots * 9 / 10);
}

// Sampling a trace with mid-circuit measurement replays sites in program
// order: h; mz; reset; x; mz yields two-bit records whose second bit is
// always 1 (reset to |0>, then x), whatever Pauli error the h noise selects.
CUDAQ_TEST(McmTraceTest, SamplingMcmKernelReplaysSites) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  const std::size_t shots = 20;
  auto result = cudaq::ptsbe::sample(noise, shots, measureResetKernel);

  EXPECT_EQ(result.get_total_shots(), shots);
  for (const auto &[bits, count] : result.to_map()) {
    ASSERT_EQ(bits.size(), 2u);
    EXPECT_EQ(bits[1], '1') << "reset+x did not pin the second bit: " << bits;
  }
}

// A named mid-circuit measurement (auto b = mz(q)) must be accepted.
// Validation may only reject kernels whose MLIR metadata flags true
// conditional feedback; a populated registerNames list (what named
// measurement tracing records) is not feedback.
CUDAQ_TEST(McmTraceTest, ValidationAcceptsNamedMidCircuitMeasurement) {
  cudaq::ExecutionContext traceCtx("tracer");
  auto &platform = cudaq::get_platform();
  platform.with_execution_context(traceCtx, [&]() { namedMcmKernel(); });
  detail::cleanupTracerQubits(traceCtx.kernelTrace);

  // MLIR-mode tracing records the measurement's register name.
  traceCtx.registerNames.push_back("b");

  EXPECT_NO_THROW(detail::validatePTSBEKernel("namedMcmKernel", traceCtx));
}

// Non-PTSBE Trace consumers must keep working once the tracer records
// resets: cudaq::contrib::draw on a reset-containing kernel still renders.
CUDAQ_TEST(McmTraceTest, DrawRendersResetContainingKernel) {
  std::string drawing;
  EXPECT_NO_THROW(drawing = cudaq::contrib::draw(bareResetKernel));
  EXPECT_FALSE(drawing.empty());
}
