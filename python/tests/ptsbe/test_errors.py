# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import cudaq


def test_ptsbe_zero_shots_raises_no_results(depol_noise, bell_kernel):
    result = cudaq.ptsbe.sample(bell_kernel,
                                noise_model=depol_noise,
                                shots_count=0)
    with pytest.raises(RuntimeError, match="no results"):
        sum(result.count(bs) for bs in result)


def test_ptsbe_wrong_kernel_args_raises(depol_noise, bell_kernel):
    with pytest.raises(RuntimeError, match="Invalid number of arguments"):
        cudaq.ptsbe.sample(bell_kernel, 1.0, 2.0, noise_model=depol_noise)


def test_ptsbe_broadcast_empty_args_returns_empty_list(rotation_kernel):
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("ry", cudaq.DepolarizationChannel(0.01))
    results = cudaq.ptsbe.sample(
        rotation_kernel,
        [],
        noise_model=noise,
        shots_count=10,
    )
    assert results == []


def test_ptsbe_sample_non_integer_shots_raises(depol_noise, bell_kernel):
    with pytest.raises((RuntimeError, TypeError)):
        cudaq.ptsbe.sample(
            bell_kernel,
            noise_model=depol_noise,
            shots_count=10.5,
        )


def test_ptsbe_sample_rejects_negative_shots(depol_noise, bell_kernel):
    with pytest.raises(RuntimeError, match="shots_count"):
        cudaq.ptsbe.sample(bell_kernel, noise_model=depol_noise, shots_count=-1)


def test_ptsbe_sample_rejects_zero_max_trajectories(depol_noise, bell_kernel):
    with pytest.raises(RuntimeError, match="max_trajectories"):
        cudaq.ptsbe.sample(
            bell_kernel,
            noise_model=depol_noise,
            max_trajectories=0,
        )


def test_ptsbe_sample_rejects_negative_max_trajectories(depol_noise,
                                                        bell_kernel):
    with pytest.raises(RuntimeError, match="max_trajectories"):
        cudaq.ptsbe.sample(
            bell_kernel,
            noise_model=depol_noise,
            max_trajectories=-5,
        )


def test_ptsbe_sample_wrong_arity_single_argument_raises(
        depol_noise, bell_kernel):
    with pytest.raises(RuntimeError, match="Invalid number of arguments"):
        cudaq.ptsbe.sample(bell_kernel, 42, noise_model=depol_noise)


def test_mcm_kernel_rejected(depol_noise, mcm_kernel):
    with pytest.raises(RuntimeError, match="conditional feedback|measurement"):
        cudaq.ptsbe.sample(mcm_kernel, noise_model=depol_noise)


@pytest.mark.parametrize("knob", [
    "shots_count", "max_trajectories", "max_shots_per_path", "max_live_states"
])
def test_ptsbe_sample_rejects_bool_for_int_knobs(depol_noise, bell_kernel,
                                                 knob):
    # bool is a subclass of int in Python; the integer knobs must reject it
    # rather than silently reading True as 1.
    with pytest.raises(RuntimeError, match=knob):
        cudaq.ptsbe.sample(bell_kernel, noise_model=depol_noise, **{knob: True})


@pytest.mark.parametrize("flag",
                         ["allow_non_unitary", "include_sequential_data"])
def test_ptsbe_sample_rejects_non_bool_flags(depol_noise, bell_kernel, flag):
    with pytest.raises(RuntimeError, match=flag):
        cudaq.ptsbe.sample(bell_kernel, noise_model=depol_noise, **{flag: 1})
