/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ResultReconstruction.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

struct OutputTarget {
  std::string outputName;
  std::size_t globalPosition = 0;
  std::size_t localPosition = 0;
  std::size_t bitIndex = 0;
};

struct OutputLayout {
  std::vector<OutputTarget> targets;
  std::unordered_map<std::string, std::size_t> outputWidths;
  std::size_t globalWidth = 0;
};

} // namespace

static OutputLayout buildOutputLayout(const cudaq::ResultOutputMap &resultMap) {
  OutputLayout layout;
  std::unordered_map<std::string, std::vector<std::size_t>> targetIdsByOutput;
  std::unordered_set<std::size_t> globalPositions;

  for (const auto &entry : resultMap.outputs) {
    if (!globalPositions.insert(entry.outputPosition).second)
      throw std::invalid_argument(
          "result output map contains duplicate output positions");
    auto targetId = layout.targets.size();
    layout.targets.push_back(
        {entry.outputName, entry.outputPosition, 0, entry.bitIndex});
    targetIdsByOutput[entry.outputName].push_back(targetId);
    layout.globalWidth = std::max(layout.globalWidth, entry.outputPosition + 1);
  }

  for (auto &[outputName, targetIds] : targetIdsByOutput) {
    std::sort(targetIds.begin(), targetIds.end(),
              [&](std::size_t lhs, std::size_t rhs) {
                return layout.targets[lhs].globalPosition <
                       layout.targets[rhs].globalPosition;
              });
    for (std::size_t localPosition = 0; localPosition < targetIds.size();
         ++localPosition)
      layout.targets[targetIds[localPosition]].localPosition = localPosition;
    layout.outputWidths[outputName] = targetIds.size();
  }

  return layout;
}

static void validateBit(char bit) {
  if (bit != '0' && bit != '1')
    throw std::invalid_argument("returned measurement bit must be '0' or '1'");
}

static std::unordered_map<std::string, std::string>
makeInitialNamedBits(const OutputLayout &layout) {
  std::unordered_map<std::string, std::string> namedBits;
  for (const auto &[name, width] : layout.outputWidths)
    namedBits.emplace(name, std::string(width, '0'));
  return namedBits;
}

static cudaq::CountsDictionary
makeCounts(const std::vector<std::string> &shots) {
  cudaq::CountsDictionary counts;
  for (const auto &bits : shots)
    counts[bits]++;
  return counts;
}

cudaq::ResultOutputMap cudaq::makeResultOutputMapFromEnrichedOutputNames(
    const nlohmann::json &outputNames) {
  ResultOutputMap resultMap;
  if (outputNames.is_null() || outputNames.empty())
    return resultMap;

  resultMap.outputs.reserve(outputNames.at(0).size());
  std::size_t resultIndex = 0;
  for (const auto &entry : outputNames.at(0)) {
    const auto &outputLocation = entry.at(1);
    auto bitIndex = outputLocation.at(0).get<std::size_t>();
    auto outputName = outputLocation.at(1).get<std::string>();
    // The third tuple element is the user-visible output position. An old
    // compiler omits it, in which case fall back to the result index.
    std::size_t outputPosition = resultIndex;
    if (outputLocation.size() > 2)
      outputPosition = outputLocation.at(2).get<std::size_t>();
    resultMap.outputs.push_back(
        {bitIndex, std::move(outputName), outputPosition});
    ++resultIndex;
  }
  return resultMap;
}

cudaq::ResultOutputMap cudaq::makeResultOutputMapFromEnrichedOutputNames(
    const cudaq_json &outputNames) {
  return makeResultOutputMapFromEnrichedOutputNames(outputNames.get());
}

cudaq::sample_result cudaq::reconstructSampleResultFromFlatBitstringShots(
    const std::vector<std::string> &shots, const ResultOutputMap &resultMap) {
  if (resultMap.outputs.empty())
    return sample_result(ExecutionResult{makeCounts(shots)});

  auto layout = buildOutputLayout(resultMap);
  std::vector<std::string> globalSequentialData;
  globalSequentialData.reserve(shots.size());
  std::unordered_map<std::string, std::vector<std::string>> namedSequentialData;
  for (const auto &[name, width] : layout.outputWidths)
    namedSequentialData[name].reserve(shots.size());

  for (const auto &shot : shots) {
    std::string globalBits(layout.globalWidth, '0');
    auto namedBits = makeInitialNamedBits(layout);

    for (const auto &target : layout.targets) {
      if (target.bitIndex >= shot.size())
        throw std::invalid_argument(
            "flat bitstring shot is shorter than a mapped bit index");
      char bit = shot[target.bitIndex];
      validateBit(bit);
      globalBits[target.globalPosition] = bit;
      namedBits[target.outputName][target.localPosition] = bit;
    }

    globalSequentialData.push_back(std::move(globalBits));
    for (auto &[name, bits] : namedBits)
      namedSequentialData[name].push_back(std::move(bits));
  }

  std::vector<ExecutionResult> executionResults;
  executionResults.emplace_back(makeCounts(globalSequentialData),
                                GlobalRegisterName);
  executionResults.back().sequentialData = std::move(globalSequentialData);
  for (auto &[name, sequentialData] : namedSequentialData) {
    executionResults.emplace_back(makeCounts(sequentialData), std::move(name));
    executionResults.back().sequentialData = std::move(sequentialData);
  }

  return sample_result(executionResults);
}

cudaq::sample_result cudaq::reconstructSampleResultFromLocalMeasurements(
    const CountsDictionary &counts, const ResultOutputMap &resultMap) {
  if (resultMap.outputs.empty())
    return sample_result(ExecutionResult{counts});

  auto layout = buildOutputLayout(resultMap);
  CountsDictionary globalCounts;
  std::unordered_map<std::string, CountsDictionary> namedCounts;

  for (const auto &[bitstring, count] : counts) {
    if (count == 0)
      continue;

    std::string globalBits(layout.globalWidth, '0');
    auto namedBits = makeInitialNamedBits(layout);

    for (const auto &target : layout.targets) {
      if (target.bitIndex >= bitstring.size())
        throw std::invalid_argument(
            "bitstring is shorter than a mapped bit index");
      char bit = bitstring[target.bitIndex];
      validateBit(bit);
      globalBits[target.globalPosition] = bit;
      namedBits[target.outputName][target.localPosition] = bit;
    }

    globalCounts[globalBits] += count;
    for (const auto &[name, bits] : namedBits)
      namedCounts[name][bits] += count;
  }

  std::vector<ExecutionResult> executionResults;
  executionResults.emplace_back(globalCounts, GlobalRegisterName);
  for (auto &[name, countsByBits] : namedCounts)
    executionResults.emplace_back(std::move(countsByBits), std::move(name));

  return sample_result(executionResults);
}
