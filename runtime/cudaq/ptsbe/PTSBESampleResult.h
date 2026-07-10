/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "PTSBEExecutionData.h"
#include "common/SampleResult.h"
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::ptsbe {

/// @brief One bit of the per-shot measurement record.
///
/// Describes the measurement site that produced the bit at position
/// `record_index` in every sequential-data string. Sites appear in
/// record-index order, one per measured target qubit.
struct RecordSite {
  /// @brief Position of this site's bit within each per-shot record
  std::size_t record_index;

  /// @brief Qubit measured at this site
  std::size_t qubit;

  /// @brief True when the site resets the qubit after measuring (a fused
  /// MeasureReset site)
  bool resets;

  /// @brief True when no later instruction in the trace touches the qubit
  bool terminal;

  /// @brief Measurement register name from the kernel, if named
  std::optional<std::string> register_name;
};

/// @brief PTSBE-specific result type returned by `ptsbe::sample()`
///    which may contain execution data.
class sample_result : public cudaq::sample_result {
private:
  std::optional<PTSBEExecutionData> executionData_;
  std::optional<std::vector<RecordSite>> recordLayout_;

public:
  sample_result() = default;

  /// @brief Construct from a base sample_result (move)
  explicit sample_result(cudaq::sample_result &&base);

  /// @brief Construct from a base sample_result with execution data
  sample_result(cudaq::sample_result &&base, PTSBEExecutionData executionData);

  /// @brief Check if execution data is available
  bool has_execution_data() const;

  /// @brief Get execution data
  /// @throws std::runtime_error if execution data not available
  const PTSBEExecutionData &execution_data() const;

  /// @brief Attach execution data to this result
  void set_execution_data(PTSBEExecutionData executionData);

  /// @brief Check if the per-shot record layout is available
  bool has_record_layout() const;

  /// @brief Get the per-shot record layout in record-index order.
  ///
  /// When the trace contains mid-circuit measurement, every sequential-data
  /// string is a fixed-width record with one bit per layout site, and counts
  /// are distributions over full records.
  ///
  /// @throws std::runtime_error if the record layout is not available
  const std::vector<RecordSite> &record_layout() const;

  /// @brief Attach the per-shot record layout to this result
  void set_record_layout(std::vector<RecordSite> layout);
};

} // namespace cudaq::ptsbe
