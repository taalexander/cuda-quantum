/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "KernelExecution.h"
#include "SampleResult.h"
#include "cudaq_json.h"
#include <string>
#include <vector>

namespace cudaq {

/// @brief One measured result's place in the user-visible output. The bit index
/// is the returned-bitstring position the result occupies, the name is the
/// register it belongs to, and the position is its user-visible output order.
struct ResultOutputEntry {
  std::size_t bitIndex = 0;
  std::string outputName;
  std::size_t outputPosition = 0;
};

/// @brief The single result representation consumed by reconstruction: the
/// per-result map read from enriched output_names. Replaces the typed execution
/// result layout.
struct ResultOutputMap {
  std::vector<ResultOutputEntry> outputs;
};

/// @brief Build the result map from enriched output_names metadata. Each
/// output-location tuple is [qubitNum, registerName, outputPosition]; the bit
/// index is the qubit number, the name is the register name, and the position
/// is read from the third tuple element. An old compiler that omits the third
/// element falls back to the result index, matching the non-mapped reference
/// order.
ResultOutputMap
makeResultOutputMapFromEnrichedOutputNames(const nlohmann::json &outputNames);
ResultOutputMap
makeResultOutputMapFromEnrichedOutputNames(const cudaq_json &outputNames);

/// Reconstruct a sample_result from per-shot flat bitstrings and a bit-index
/// result map derived from enriched output_names. Each shot string is indexed
/// by the bit positions in the map. Preserves per-shot sequential data in the
/// returned result. Throws std::invalid_argument when a bitstring is shorter
/// than a mapped bit index.
sample_result reconstructSampleResultFromFlatBitstringShots(
    const std::vector<std::string> &shots, const ResultOutputMap &resultMap);

/// Reconstruct a sample_result from a counts dictionary and a bit-index result
/// map derived from enriched output_names. Each bitstring in counts is indexed
/// by the bit positions in the map. Throws std::invalid_argument when a
/// bitstring is shorter than a mapped bit index.
sample_result
reconstructSampleResultFromLocalMeasurements(const CountsDictionary &counts,
                                             const ResultOutputMap &resultMap);

} // namespace cudaq
