/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Yaml/v2/QuakeReferenceSemanticsProfileYaml.h"

cudaq::config::QuakeReferenceSemanticsProfile cudaq::config::v2::ProfileYaml<
    cudaq::config::QuakeReferenceSemanticsProfile>::to(const Yaml &yaml) {
  const auto &options = yaml.options.value();
  return cudaq::config::QuakeReferenceSemanticsProfile{
      .basisGates = options.basisGates.value(),
      .measurementType = options.measurementType.value(),
      .resetAfterMeasure = options.resetAfterMeasure.value()};
}

cudaq::config::v2::ProfileYaml<
    cudaq::config::QuakeReferenceSemanticsProfile>::Yaml
cudaq::config::v2::ProfileYaml<cudaq::config::QuakeReferenceSemanticsProfile>::
    from(const cudaq::config::QuakeReferenceSemanticsProfile &profile) {
  return Yaml{
      .options = Yaml::Options{.basisGates = profile.basisGates,
                               .measurementType = profile.measurementType,
                               .resetAfterMeasure = profile.resetAfterMeasure}};
}
