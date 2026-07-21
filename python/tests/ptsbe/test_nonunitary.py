# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import math

import pytest
import cudaq

requires_gpu_cusvsim = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target("nvidia")),
    reason="requires a GPU and the nvidia (cusvsim) target")


@cudaq.kernel
def x_kernel():
    q = cudaq.qvector(1)
    x(q[0])
    mz(q)


@cudaq.kernel
def zero_damping_kernel():
    q = cudaq.qvector(1)
    cudaq.apply_noise(cudaq.AmplitudeDampingChannel, 0.2, q[0])
    mz(q)


@cudaq.kernel
def one_damping_kernel():
    q = cudaq.qvector(1)
    x(q[0])
    cudaq.apply_noise(cudaq.AmplitudeDampingChannel, 0.2, q[0])
    mz(q)


@cudaq.kernel
def plus_damping_kernel():
    q = cudaq.qvector(1)
    h(q[0])
    cudaq.apply_noise(cudaq.AmplitudeDampingChannel, 0.2, q[0])
    mz(q)


@cudaq.kernel
def three_qubit_x_kernel():
    q = cudaq.qvector(3)
    x(q[0])
    x(q[1])
    x(q[2])
    mz(q)


@cudaq.kernel
def three_qubit_x_packed() -> int:
    q = cudaq.qvector(3)
    x(q[0])
    x(q[1])
    x(q[2])
    r = 0
    if mz(q[0]):
        r += 4
    if mz(q[1]):
        r += 2
    if mz(q[2]):
        r += 1
    return r


def amplitude_damping_on_x(gamma):
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.AmplitudeDampingChannel(gamma))
    return noise


def probability(result, bits):
    total = result.get_total_shots()
    assert total > 0
    return result.count(bits) / total


@requires_gpu_cusvsim
@pytest.mark.parametrize("gamma", [0.2, 0.05])
def test_single_qubit_amplitude_damping_auto_capacity(gamma):
    # x prepares |1>; amplitude damping decays |1> -> |0> with probability
    # gamma, so the surviving population P("1") = 1 - gamma. allow_non_unitary
    # with max_live_states UNSET must pick a frontier capacity automatically
    # (the auto-capacity contract) rather than erroring.
    cudaq.set_target("nvidia", option="fp64")
    cudaq.set_random_seed(42)
    shots = 2000

    result = cudaq.ptsbe.sample(x_kernel,
                                shots_count=shots,
                                noise_model=amplitude_damping_on_x(gamma),
                                allow_non_unitary=True)

    assert result.get_total_shots() == shots

    expected = 1.0 - gamma
    tolerance = 6.0 * math.sqrt(expected * (1.0 - expected) / shots)
    assert abs(probability(result, "1") - expected) < tolerance


@requires_gpu_cusvsim
def test_counts_only_default_no_sequential_data():
    # The counts-only default carries no per-shot data; the per-shot record
    # list is populated only when include_sequential_data=True.
    cudaq.set_target("nvidia", option="fp64")
    cudaq.set_random_seed(42)
    shots = 500
    noise = amplitude_damping_on_x(0.2)

    counts_only = cudaq.ptsbe.sample(x_kernel,
                                     shots_count=shots,
                                     noise_model=noise,
                                     allow_non_unitary=True)
    assert counts_only.get_sequential_data() == []
    assert counts_only.get_total_shots() == shots

    with_records = cudaq.ptsbe.sample(x_kernel,
                                      shots_count=shots,
                                      noise_model=noise,
                                      allow_non_unitary=True,
                                      include_sequential_data=True)
    records = with_records.get_sequential_data()
    assert len(records) == shots
    assert all(len(record) == 1 for record in records)


@requires_gpu_cusvsim
def test_three_qubit_amplitude_damping_matches_density_matrix():
    # Cross-check the batched (cusvsim) non-unitary path against an exact
    # density-matrix reference. The reference distribution comes from
    # cudaq.run packed-integer records on density-matrix-cpu, not from
    # explicit_measurements sampling.
    gamma = 0.2
    shots = 4000
    noise = amplitude_damping_on_x(gamma)

    cudaq.set_target("density-matrix-cpu")
    cudaq.set_random_seed(7)
    packed = cudaq.run(three_qubit_x_packed,
                       shots_count=shots,
                       noise_model=noise)
    reference = {}
    for value in packed:
        bits = format(value, "03b")
        reference[bits] = reference.get(bits, 0) + 1

    cudaq.set_target("nvidia", option="fp64")
    cudaq.set_random_seed(7)
    result = cudaq.ptsbe.sample(three_qubit_x_kernel,
                                shots_count=shots,
                                noise_model=noise,
                                allow_non_unitary=True)

    assert result.get_total_shots() == shots

    outcomes = {format(value, "03b") for value in range(8)}
    tolerance = 0.04
    for bits in outcomes:
        p_ptsbe = result.count(bits) / shots
        p_reference = reference.get(bits, 0) / shots
        assert abs(p_ptsbe - p_reference) < tolerance, (
            f"{bits}: ptsbe={p_ptsbe:.4f} reference={p_reference:.4f}")


@requires_gpu_cusvsim
def test_importance_public_result_and_count_contract(monkeypatch):
    cudaq.set_target("nvidia", option="fp64")
    cudaq.set_random_seed(20260710)
    shots = 512
    noise = amplitude_damping_on_x(0.2)

    monkeypatch.setenv("CUDAQ_PTSBE_NONUNITARY_MODE", "frontier")
    exact = cudaq.ptsbe.sample(x_kernel,
                               shots_count=shots,
                               noise_model=noise,
                               allow_non_unitary=True,
                               max_live_states=64,
                               max_shots_per_path=0)

    monkeypatch.setenv("CUDAQ_PTSBE_NONUNITARY_MODE", "importance")
    monkeypatch.setenv("CUDAQ_PTSBE_IMPORTANCE_NORMALIZATION", "site")
    monkeypatch.setenv("CUDAQ_PTSBE_IMPORTANCE_RESAMPLER",
                       "residual_stratified")
    candidate = cudaq.ptsbe.sample(x_kernel,
                                   shots_count=shots,
                                   noise_model=noise,
                                   allow_non_unitary=True,
                                   max_live_states=64)
    sequential = cudaq.ptsbe.sample(x_kernel,
                                    shots_count=shots,
                                    noise_model=noise,
                                    allow_non_unitary=True,
                                    max_live_states=64,
                                    include_sequential_data=True)

    assert type(candidate) is type(exact)
    assert candidate.get_total_shots() == shots
    assert all(
        isinstance(candidate.count(bits), int) and candidate.count(bits) >= 0
        for bits in candidate)
    assert candidate.get_sequential_data() == []
    records = sequential.get_sequential_data()
    assert len(records) == shots
    histogram = {}
    for record in records:
        histogram[record] = histogram.get(record, 0) + 1
    assert histogram == {bits: sequential.count(bits) for bits in sequential}


@requires_gpu_cusvsim
@pytest.mark.parametrize(
    "kernel,expected_one",
    [
        (zero_damping_kernel, 0.0),
        (one_damping_kernel, 0.8),
        (plus_damping_kernel, 0.4),
    ],
)
def test_compressed_importance_amplitude_damping_states(kernel, expected_one,
                                                        monkeypatch):
    cudaq.set_target("nvidia", option="fp64")
    cudaq.set_random_seed(20260720)
    shots = 4096
    proposals = 512

    monkeypatch.setenv("CUDAQ_PTSBE_NONUNITARY_MODE", "importance")
    monkeypatch.setenv("CUDAQ_PTSBE_IMPORTANCE_PROPOSALS", str(proposals))
    monkeypatch.setenv("CUDAQ_PTSBE_IMPORTANCE_NORMALIZATION", "site")
    monkeypatch.setenv("CUDAQ_PTSBE_IMPORTANCE_RESAMPLER",
                       "residual_stratified")
    result = cudaq.ptsbe.sample(kernel,
                                shots_count=shots,
                                allow_non_unitary=True,
                                max_live_states=128)

    assert result.get_total_shots() == shots
    tolerance = 6.0 * math.sqrt(
        max(expected_one * (1.0 - expected_one), 1.0 / proposals) / proposals)
    assert abs(probability(result, "1") - expected_one) < tolerance


def test_importance_rejects_execution_data_before_dispatch(monkeypatch):
    monkeypatch.setenv("CUDAQ_PTSBE_NONUNITARY_MODE", "importance")
    with pytest.raises(RuntimeError, match="return_execution_data"):
        cudaq.ptsbe.sample(x_kernel,
                           shots_count=8,
                           noise_model=amplitude_damping_on_x(0.2),
                           allow_non_unitary=True,
                           max_live_states=4,
                           return_execution_data=True)
