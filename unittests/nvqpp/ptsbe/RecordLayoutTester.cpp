/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/PTSBEExecutionData.h"
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/ptsbe/PTSBESampleResult.h"
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cudaq::ptsbe;

namespace {

// h(q0); mz(q0); x(q1); reset(q0); mz(q0); mz(q1) traces to
// [Gate(h), MeasureReset(q0,rec0), Gate(x,q1), Measurement(q0,rec1),
//  Measurement(q1,rec2)]: one mid-circuit fused site plus two terminal sites.
auto crossQubitKernel = []() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  mz(q[0]);
  x(q[1]);
  reset(q[0]);
  mz(q[0]);
  mz(q[1]);
};

// h; mz; reset fuses into a TERMINAL MeasureReset site: resets and terminal
// are independent flags and both hold here.
auto terminalMeasureResetKernel = []() __qpu__ {
  cudaq::qubit q;
  h(q);
  mz(q);
  reset(q);
};

// h; mz; reset; mz: the first record bit is 50/50, the second always 0.
auto measureResetMeasureKernel = []() __qpu__ {
  cudaq::qubit q;
  h(q);
  mz(q);
  reset(q);
  mz(q);
};

template <typename QuantumKernel>
PTSBETrace traceKernel(QuantumKernel &&kernel) {
  cudaq::ExecutionContext traceCtx("tracer");
  auto &platform = cudaq::get_platform();
  platform.with_execution_context(traceCtx, [&]() { kernel(); });
  detail::cleanupTracerQubits(traceCtx.kernelTrace);
  static const cudaq::noise_model kEmptyNoiseModel;
  return detail::buildPTSBETrace(traceCtx.kernelTrace, kEmptyNoiseModel);
}

} // namespace

// ============================================================================
// LAYOUT CONTENTS (buildRecordLayout from a trace)
// ============================================================================

// One RecordSite per measured qubit across Measurement/MeasureReset sites,
// record indices dense and ordered, resets true exactly for MeasureReset
// entries, terminal true only for sites with no later op on the qubit.
CUDAQ_TEST(RecordLayoutTest, OneSitePerMeasuredQubitDenseOrdered) {
  auto trace = traceKernel(crossQubitKernel);
  auto layout = detail::buildRecordLayout(trace);

  ASSERT_EQ(layout.size(), 3u);
  for (std::size_t i = 0; i < layout.size(); ++i)
    EXPECT_EQ(layout[i].record_index, i) << "record indices not dense at " << i;

  EXPECT_EQ(layout[0].qubit, 0u);
  EXPECT_TRUE(layout[0].resets);
  EXPECT_FALSE(layout[0].terminal)
      << "mid-circuit MeasureReset flagged terminal";

  EXPECT_EQ(layout[1].qubit, 0u);
  EXPECT_FALSE(layout[1].resets);
  EXPECT_TRUE(layout[1].terminal);

  EXPECT_EQ(layout[2].qubit, 1u);
  EXPECT_FALSE(layout[2].resets);
  EXPECT_TRUE(layout[2].terminal);
}

// A multi-target Measurement expands to one RecordSite per target qubit at
// record_index + k, matching the per-shot record writers.
CUDAQ_TEST(RecordLayoutTest, MultiTargetMeasurementExpandsPerQubit) {
  PTSBETrace trace = {
      {TraceInstructionType::Gate, "h", {0}, {}, {}},
      {TraceInstructionType::Measurement,
       "mz",
       {0, 1},
       {},
       {},
       std::nullopt,
       0},
  };
  auto layout = detail::buildRecordLayout(trace);

  ASSERT_EQ(layout.size(), 2u);
  EXPECT_EQ(layout[0].record_index, 0u);
  EXPECT_EQ(layout[0].qubit, 0u);
  EXPECT_EQ(layout[1].record_index, 1u);
  EXPECT_EQ(layout[1].qubit, 1u);
  for (const auto &site : layout) {
    EXPECT_FALSE(site.resets);
    EXPECT_TRUE(site.terminal);
  }
}

// A measurement site without an assigned record_index is a trace-construction
// bug: buildRecordLayout must throw rather than invent an index that could
// silently disagree with the record bits the replay writers emit.
CUDAQ_TEST(RecordLayoutTest, MissingRecordIndexThrows) {
  PTSBETrace trace = {
      {TraceInstructionType::Gate, "h", {0}, {}, {}},
      {TraceInstructionType::Measurement, "mz", {0}, {}, {}},
  };
  EXPECT_THROW(detail::buildRecordLayout(trace), std::runtime_error);
}

// resets and terminal both hold for a trailing fused measure-then-reset.
CUDAQ_TEST(RecordLayoutTest, TerminalMeasureResetSiteTerminalAndResets) {
  auto trace = traceKernel(terminalMeasureResetKernel);
  auto layout = detail::buildRecordLayout(trace);

  ASSERT_EQ(layout.size(), 1u);
  EXPECT_EQ(layout[0].record_index, 0u);
  EXPECT_EQ(layout[0].qubit, 0u);
  EXPECT_TRUE(layout[0].resets);
  EXPECT_TRUE(layout[0].terminal);
}

// register_name on a trace measurement is carried onto its record site;
// unnamed sites carry nullopt.
CUDAQ_TEST(RecordLayoutTest, RegisterNameCarriedFromTrace) {
  PTSBETrace trace = {
      {TraceInstructionType::Gate, "h", {0}, {}, {}},
      {TraceInstructionType::MeasureReset,
       "mz",
       {0},
       {},
       {},
       std::nullopt,
       0,
       std::string("syndrome")},
      {TraceInstructionType::Gate, "x", {0}, {}, {}},
      {TraceInstructionType::Measurement, "mz", {0}, {}, {}, std::nullopt, 1},
  };
  auto layout = detail::buildRecordLayout(trace);

  ASSERT_EQ(layout.size(), 2u);
  ASSERT_TRUE(layout[0].register_name.has_value());
  EXPECT_EQ(*layout[0].register_name, "syndrome");
  EXPECT_FALSE(layout[1].register_name.has_value());
}

// ============================================================================
// RESULT CARRIES THE LAYOUT; RECORDS ARE FIXED WIDTH
// ============================================================================

// End-to-end through ptsbe::sample: the result exposes record_layout()
// matching the trace's sites, and every sequential_data() string has width
// equal to the number of record bits.
CUDAQ_TEST(RecordLayoutTest, ResultCarriesLayoutAndFixedWidthRecords) {
  cudaq::set_random_seed(42);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.05));

  sample_options options;
  options.shots = 100;
  options.noise = noise;
  options.ptsbe.include_sequential_data = true;

  auto result = sample(options, crossQubitKernel);
  EXPECT_EQ(result.get_total_shots(), options.shots);

  const auto &layout = result.record_layout();
  ASSERT_EQ(layout.size(), 3u);
  for (std::size_t i = 0; i < layout.size(); ++i)
    EXPECT_EQ(layout[i].record_index, i);
  EXPECT_TRUE(layout[0].resets);
  EXPECT_FALSE(layout[0].terminal);
  EXPECT_TRUE(layout[1].terminal);
  EXPECT_TRUE(layout[2].terminal);

  auto records = result.sequential_data();
  ASSERT_EQ(records.size(), options.shots);
  for (const auto &record : records)
    EXPECT_EQ(record.size(), layout.size())
        << "record width differs from layout: " << record;
}

// ============================================================================
// COLUMN SLICE MARGINAL CONSISTENT WITH COUNTS
// ============================================================================

// Slicing the resets-site column out of the per-shot records must reproduce
// the marginal implied by the full-record counts exactly, and the marginal
// itself sits near 0.5 (h before the measurement; generous tolerance).
CUDAQ_TEST(RecordLayoutTest, RecordColumnMarginalConsistentWithCounts) {
  cudaq::set_random_seed(42);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  sample_options options;
  options.shots = 1000;
  options.noise = noise;
  options.ptsbe.include_sequential_data = true;

  auto result = sample(options, measureResetMeasureKernel);
  EXPECT_EQ(result.get_total_shots(), options.shots);

  const auto &layout = result.record_layout();
  ASSERT_EQ(layout.size(), 2u);
  ASSERT_TRUE(layout[0].resets);

  const std::size_t column = layout[0].record_index;
  auto records = result.sequential_data();
  ASSERT_EQ(records.size(), options.shots);

  std::size_t onesFromSlice = 0;
  for (const auto &record : records) {
    ASSERT_EQ(record.size(), layout.size());
    if (record[column] == '1')
      ++onesFromSlice;
    EXPECT_EQ(record[layout[1].record_index], '0')
        << "reset did not zero the second record bit: " << record;
  }

  std::size_t onesFromCounts = 0;
  for (const auto &[bits, count] : result.to_map()) {
    ASSERT_EQ(bits.size(), layout.size());
    if (bits[column] == '1')
      onesFromCounts += count;
  }
  EXPECT_EQ(onesFromSlice, onesFromCounts)
      << "column slice disagrees with counts marginal";

  EXPECT_GT(onesFromSlice, options.shots * 35 / 100);
  EXPECT_LT(onesFromSlice, options.shots * 65 / 100);
}
