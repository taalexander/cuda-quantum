# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import cudaq


@pytest.fixture
def measure_reset_kernel():

    @cudaq.kernel
    def measure_reset():
        q = cudaq.qvector(2)
        h(q[0])
        b = mz(q[0])
        reset(q[0])
        x(q[0])
        mz(q)

    return measure_reset


@pytest.fixture
def feedback_kernel():

    @cudaq.kernel
    def feedback():
        q = cudaq.qvector(2)
        h(q[0])
        if mz(q[0]):
            x(q[1])
        mz(q)

    return feedback


def test_measure_reset_kernel_samples_with_replay(measure_reset_kernel):
    # Validation accepts mid-circuit measurement and reset (no "conditional
    # feedback" rejection) and site-ordered replay executes the kernel.
    # Records are [mid mz(q0), terminal mz(q0), terminal mz(q1)]: after
    # reset(q0); x(q0) the second bit is always 1, and q1 stays 0.
    cudaq.set_target("qpp-cpu")
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("h", cudaq.DepolarizationChannel(0.01))

    shots = 50
    result = cudaq.ptsbe.sample(measure_reset_kernel,
                                noise_model=noise,
                                shots_count=shots)

    assert sum(result.count(bits) for bits in result) == shots
    for bits in result:
        assert len(bits) == 3
        assert bits[1] == '1'
        assert bits[2] == '0'


def test_feedback_kernel_still_rejected(feedback_kernel):
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("h", cudaq.DepolarizationChannel(0.01))

    with pytest.raises(RuntimeError, match="conditional feedback|measurement"):
        cudaq.ptsbe.sample(feedback_kernel, noise_model=noise)
