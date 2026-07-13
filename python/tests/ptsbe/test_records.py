# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import numpy as np
import pytest
import cudaq

requires_gpu_cusvsim = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target("nvidia")),
    reason="requires a GPU and the nvidia (cusvsim) target")


@pytest.fixture
def named_mcm_kernel():

    @cudaq.kernel
    def named_mcm():
        q = cudaq.qvector(2)
        h(q[0])
        s = mz(q[0])
        reset(q[0])
        x(q[0])
        mz(q)

    return named_mcm


@pytest.fixture
def repetition_code_kernel():
    # Two data qubits (q0, q1) and one ancilla (q2). h(q0) puts the data
    # parity q0 XOR q1 = q0 in superposition; the cx pair accumulates that
    # parity onto the ancilla, which is measured and reset mid-circuit.

    @cudaq.kernel
    def repetition_code():
        q = cudaq.qvector(3)
        h(q[0])
        x.ctrl(q[0], q[2])
        x.ctrl(q[1], q[2])
        mz(q[2])
        reset(q[2])
        mz(q)

    return repetition_code


def records_as_array(result):
    records = result.get_sequential_data()
    return np.array([[int(bit) for bit in record] for record in records],
                    dtype=np.uint8)


def test_record_layout_exposed_with_site_fields(named_mcm_kernel):
    # Records: [mid mz(q0) fused with reset, terminal mz(q0), terminal
    # mz(q1)]. The layout exposes record_index, qubit, resets, terminal,
    # and register_name for each site.
    shots = 50
    result = cudaq.ptsbe.sample(named_mcm_kernel,
                                shots_count=shots,
                                include_sequential_data=True)

    layout = result.record_layout
    assert len(layout) == 3
    for i, site in enumerate(layout):
        assert site.record_index == i

    assert layout[0].qubit == 0
    assert layout[0].resets is True
    assert layout[0].terminal is False
    assert layout[0].register_name == "s"

    assert layout[1].qubit == 0
    assert layout[1].resets is False
    assert layout[1].terminal is True
    assert layout[1].register_name is None

    assert layout[2].qubit == 1
    assert layout[2].resets is False
    assert layout[2].terminal is True
    assert layout[2].register_name is None


def test_mcm_counts_only_by_default(named_mcm_kernel):
    # Mid-circuit results default to counts over unique full records; the
    # per-shot list is opt-in via include_sequential_data.
    shots = 60
    result = cudaq.ptsbe.sample(named_mcm_kernel, shots_count=shots)

    assert result.get_sequential_data() == []
    layout = result.record_layout
    assert len(layout) == 3
    counts = {bits: result.count(bits) for bits in result}
    assert sum(counts.values()) == shots
    assert all(len(bits) == len(layout) for bits in counts)


def test_sequential_data_numpy_shape(named_mcm_kernel):
    shots = 40
    result = cudaq.ptsbe.sample(named_mcm_kernel,
                                shots_count=shots,
                                include_sequential_data=True)

    arr = records_as_array(result)
    assert arr.shape == (shots, len(result.record_layout))


def test_repetition_code_ancilla_column_matches_syndrome(
        repetition_code_kernel):
    # The mid-circuit ancilla record is the parity q0 XOR q1. Measuring the
    # ancilla collapses the data, so per shot the ancilla column equals the
    # XOR of the terminal data columns exactly; its marginal is near 0.5
    # (h on q0, generous tolerance); the reset ancilla reads 0 terminally.
    shots = 400
    result = cudaq.ptsbe.sample(repetition_code_kernel,
                                shots_count=shots,
                                include_sequential_data=True)

    layout = result.record_layout
    assert len(layout) == 4

    ancilla_sites = [s for s in layout if s.resets]
    assert len(ancilla_sites) == 1
    assert ancilla_sites[0].qubit == 2

    terminal_cols = {s.qubit: s.record_index for s in layout if s.terminal}
    assert set(terminal_cols) == {0, 1, 2}

    assert_repetition_code_record_invariants(result, shots)


def layout_tuples(result):
    return [(s.record_index, s.qubit, s.resets, s.terminal, s.register_name)
            for s in result.record_layout]


def assert_repetition_code_record_invariants(result, shots):
    # Exact per-shot invariants: the ancilla column equals the terminal
    # data parity, and the reset ancilla reads 0 terminally.
    layout = result.record_layout
    syndrome_col = next(s.record_index for s in layout if s.resets)
    terminal_cols = {s.qubit: s.record_index for s in layout if s.terminal}

    arr = records_as_array(result)
    assert arr.shape == (shots, len(layout))

    syndrome = arr[:, syndrome_col]
    data_parity = arr[:, terminal_cols[0]] ^ arr[:, terminal_cols[1]]
    assert np.array_equal(syndrome, data_parity)
    assert np.all(arr[:, terminal_cols[2]] == 0)

    marginal = syndrome.mean()
    assert 0.35 < marginal < 0.65


@requires_gpu_cusvsim
def test_batched_gpu_records_round_trip_matches_generic(repetition_code_kernel,
                                                        monkeypatch):
    # Records must round-trip identically through the generic per-shot
    # replay and the batched (cusvsim, GPU) executor: same layout, same
    # record width, and the same exact per-shot invariants on both paths.
    shots = 400
    cpu_generic = cudaq.ptsbe.sample(repetition_code_kernel,
                                     shots_count=shots,
                                     include_sequential_data=True)

    cudaq.set_target("nvidia", option="fp64")
    cudaq.set_random_seed(42)
    batched = cudaq.ptsbe.sample(repetition_code_kernel,
                                 shots_count=shots,
                                 include_sequential_data=True)

    monkeypatch.setenv("CUDAQ_PTSBE_FORCE_GENERIC", "1")
    gpu_generic = cudaq.ptsbe.sample(repetition_code_kernel,
                                     shots_count=shots,
                                     include_sequential_data=True)

    assert layout_tuples(batched) == layout_tuples(cpu_generic)
    assert layout_tuples(batched) == layout_tuples(gpu_generic)

    assert_repetition_code_record_invariants(batched, shots)
    assert_repetition_code_record_invariants(gpu_generic, shots)


@requires_gpu_cusvsim
def test_batched_gpu_named_mcm_layout_and_shape(named_mcm_kernel):
    cudaq.set_target("nvidia", option="fp64")
    cudaq.set_random_seed(42)
    shots = 40
    result = cudaq.ptsbe.sample(named_mcm_kernel,
                                shots_count=shots,
                                include_sequential_data=True)

    layout = result.record_layout
    assert len(layout) == 3
    assert layout[0].register_name == "s"
    assert layout[0].resets is True

    arr = records_as_array(result)
    assert arr.shape == (shots, len(layout))

    # reset(q0) then x(q0) leaves q0 in |1> deterministically
    assert np.all(arr[:, 1] == 1)


def test_no_named_register_warning(named_mcm_kernel, capfd):
    # Named measurements become record-site names; producing records must
    # not emit the "named measurement results" stderr warning.
    result = cudaq.ptsbe.sample(named_mcm_kernel,
                                shots_count=20,
                                include_sequential_data=True)
    assert len(result.get_sequential_data()) == 20

    captured = capfd.readouterr()
    assert "named measurement results" not in captured.err
