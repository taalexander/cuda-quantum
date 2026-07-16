# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Terminal-only unitary cross-root batching regression gate.

The frontier executor is the single PTSBE executor. For terminal-only unitary
noise (no mid-circuit measurement, reset, non-unitary Kraus, or unitary-branch
fan-out) every pre-sampled root is branch-free, so the frontier must evolve a
chunk of independent roots in one launch (cross-root batching) rather than one
statevector per launch. Before the branch-free chunk handler is folded into the
frontier, forcing a terminal-only workload onto the frontier (an explicit
``max_live_states``) walks the roots one wave at a time and is far slower than
the default path, which still reaches the flat cross-root batcher.

These tests pin the post-fold contract:

  * the forced-frontier terminal-only path draws the same distribution as the
    default path (behavior preserving), and
  * its wall clock matches the default path within a generous factor at high
    shots (no cross-root-batching regression).

The wall-clock gate is red until the fold lands: today the forced-frontier run
serializes and is many times slower than the default flat run. The reference
target is the committed path-4 curve (~17x generic / ~23x batched vs standard
trajectory sampling at 10M shots).
"""
import time

import numpy as np
import pytest
import cudaq

requires_gpu_cusvsim = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target("nvidia")),
    reason="requires a GPU and the nvidia (cusvsim) target")

XEB_GATE_IDS = [0, 1, 2]


def _generate_gate_choices(n, depth, rng):
    choices = np.empty((n, depth), dtype=int)
    choices[:, 0] = rng.choice(XEB_GATE_IDS, size=n)
    for d in range(1, depth):
        for i in range(n):
            prev = choices[i, d - 1]
            choices[i, d] = rng.choice([g for g in XEB_GATE_IDS if g != prev])
    return choices


def _apply_xeb_gate(kernel, qubit, gate_id):
    if gate_id == 0:
        kernel.rx(np.pi / 2, qubit)
    elif gate_id == 1:
        kernel.ry(np.pi / 2, qubit)
    else:
        kernel.rz(np.pi / 4, qubit)
        kernel.rx(np.pi / 2, qubit)
        kernel.rz(-np.pi / 4, qubit)


def _build_terminal_only_xeb(n, depth, gate_choices):
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(n)
    for d in range(depth):
        for i in range(n):
            _apply_xeb_gate(kernel, q[i], int(gate_choices[i, d]))
        for i in range(n - 1):
            kernel.cx(q[i], q[i + 1])
    kernel.mz(q)
    return kernel


def _terminal_only_noise(p1=0.001, p2=0.01):
    # Unitary-mixture (Pauli) gate noise only, and no measurement channel, so
    # the whole workload is terminal-only and branch-free after pre-sampling.
    noise = cudaq.NoiseModel()
    pauli1 = cudaq.Pauli1([p1 / 3] * 3)
    noise.add_all_qubit_channel("rx", pauli1)
    noise.add_all_qubit_channel("ry", pauli1)
    noise.add_all_qubit_channel("cx", cudaq.Pauli2([p2 / 15] * 15))
    return noise


def _tvd(lhs, rhs):
    total_l = lhs.get_total_shots()
    total_r = rhs.get_total_shots()
    keys = set(lhs) | set(rhs)
    return 0.5 * sum(
        abs(lhs.count(k) / total_l - rhs.count(k) / total_r) for k in keys)


@requires_gpu_cusvsim
def test_forced_frontier_terminal_only_matches_default_distribution():
    # Behavior preservation: forcing the terminal-only workload onto the
    # frontier (max_live_states set, full-multiplicity per root) must draw the
    # same record distribution as the default flat path. This holds before and
    # after the fold and guards the fold against changing the statistics.
    cudaq.set_target("nvidia", option="fp64")
    cudaq.set_random_seed(7)
    n, depth = 7, 6
    rng = np.random.default_rng(2024)
    gate_choices = _generate_gate_choices(n, depth, rng)
    kernel = _build_terminal_only_xeb(n, depth, gate_choices)
    noise = _terminal_only_noise()
    shots = 200_000
    max_traj = 400

    default = cudaq.ptsbe.sample(
        kernel,
        noise_model=noise,
        shots_count=shots,
        sampling_strategy=cudaq.ptsbe.ProbabilisticSamplingStrategy(seed=1),
        max_trajectories=max_traj)
    frontier = cudaq.ptsbe.sample(
        kernel,
        noise_model=noise,
        shots_count=shots,
        sampling_strategy=cudaq.ptsbe.ProbabilisticSamplingStrategy(seed=1),
        max_trajectories=max_traj,
        max_shots_per_path=0,
        max_live_states=64)

    assert default.get_total_shots() == shots
    assert frontier.get_total_shots() == shots
    # Two independent high-shot estimates of the same terminal distribution over
    # D = 2^7 outcomes agree to well under 5% total variation.
    assert _tvd(default, frontier) < 0.05
    cudaq.reset_target()


@requires_gpu_cusvsim
def test_forced_frontier_terminal_only_has_no_batching_regression():
    # Cross-root batching wall-clock gate (red until the fold lands). The
    # terminal-only workload forced onto the frontier must not be dramatically
    # slower than the default flat path. Today the forced-frontier run walks the
    # pre-sampled roots one statevector per launch (~27x slower on the committed
    # XEB curve), so this assertion fails; after the branch-free chunk handler
    # cross-root batches those roots the two paths run at parity.
    cudaq.set_target("nvidia", option="fp64")
    cudaq.set_random_seed(11)
    n, depth = 8, 8
    rng = np.random.default_rng(600)
    gate_choices = _generate_gate_choices(n, depth, rng)
    kernel = _build_terminal_only_xeb(n, depth, gate_choices)
    noise = _terminal_only_noise()
    shots = 1_000_000
    max_traj = 500

    def run(**kwargs):
        return cudaq.ptsbe.sample(
            kernel,
            noise_model=noise,
            shots_count=shots,
            sampling_strategy=cudaq.ptsbe.ProbabilisticSamplingStrategy(seed=1),
            max_trajectories=max_traj,
            **kwargs)

    # Warm up both entry points (JIT and backend init) before timing.
    run()
    run(max_shots_per_path=0, max_live_states=64)

    t0 = time.perf_counter()
    default = run()
    t_default = time.perf_counter() - t0

    t0 = time.perf_counter()
    frontier = run(max_shots_per_path=0, max_live_states=64)
    t_frontier = time.perf_counter() - t0

    assert default.get_total_shots() == shots
    assert frontier.get_total_shots() == shots
    assert _tvd(default, frontier) < 0.05

    # Parity within a generous factor. The default flat path and the folded
    # branch-free frontier both cross-root batch, so their wall clocks track.
    # The pre-fold serial frontier is many times slower and trips this bound.
    tolerance = 3.0
    assert t_frontier <= tolerance * t_default, (
        f"terminal-only frontier path regressed vs the flat cross-root batcher: "
        f"frontier {t_frontier:.3f}s vs default {t_default:.3f}s "
        f"(ratio {t_frontier / t_default:.1f}x, tolerance {tolerance:.1f}x)")
    cudaq.reset_target()
