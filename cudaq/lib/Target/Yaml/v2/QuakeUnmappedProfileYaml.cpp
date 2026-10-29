/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Yaml/v2/QuakeUnmappedProfileYaml.h"

cudaq::config::QuakeUnmappedProfile
cudaq::config::v2::ProfileYaml<cudaq::config::QuakeUnmappedProfile>::to(
    const Yaml &yaml) {
  const auto &options = yaml.options.value();
  return cudaq::config::QuakeUnmappedProfile{
      .basisGates = options.basisGates.value(),
      .approximationTolerance = options.approximationTolerance.value(),
      .costModel = options.costModel};
}

cudaq::config::v2::ProfileYaml<cudaq::config::QuakeUnmappedProfile>::Yaml
cudaq::config::v2::ProfileYaml<cudaq::config::QuakeUnmappedProfile>::from(
    const cudaq::config::QuakeUnmappedProfile &profile) {
  return Yaml{.options = Yaml::Options{.basisGates = profile.basisGates,
                                       .approximationTolerance =
                                           profile.approximationTolerance,
                                       .costModel = profile.costModel}};
}
