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

// Dense/ordered record indices with per-site resets/terminal flags exposed
// end-to-end are pinned by the Python record tests
// (test_record_layout_exposed_with_site_fields,
// test_sequential_data_numpy_shape in python/tests/ptsbe/test_records.py); the
// removed OneSitePerMeasuredQubitDenseOrdered and
// ResultCarriesLayoutAndFixedWidthRecords cases duplicated that layout contract
// from the C++ side.

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
