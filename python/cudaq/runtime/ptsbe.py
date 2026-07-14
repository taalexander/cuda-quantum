# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)
from cudaq.runtime.sample import (_detail_check_conditionals_on_measure,
                                  AsyncSampleResult)
from cudaq.util import trace
from .utils import __isBroadcast, __createArgumentSet

from cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime.ptsbe import *


def _validate_ptsbe_args(kernel,
                         args,
                         shots_count,
                         noise_model,
                         max_trajectories,
                         max_shots_per_path,
                         num_root_draws=None,
                         max_paths_per_root=None,
                         max_live_states=None,
                         allow_non_unitary=False,
                         include_sequential_data=False):
    """Validate arguments common to `sample` and `sample_async`."""
    decorator = kernel
    if not isa_kernel_decorator(decorator):
        decorator = mk_decorator(decorator)

    if decorator.qkeModule is None:
        raise RuntimeError(
            "Unsupported target / Invalid kernel for `ptsbe.sample`: "
            "missing module")

    if decorator.formal_arity() != len(args):
        raise RuntimeError(
            "Invalid number of arguments passed to ptsbe.sample. " +
            str(len(args)) + " given and " + str(decorator.formal_arity()) +
            " expected.")

    # bool is a subclass of int in Python, so isinstance(True, int) is True.
    # Reject bool explicitly for the integer knobs to catch e.g.
    # max_live_states=True silently meaning 1.
    def _is_int(value):
        return isinstance(value, int) and not isinstance(value, bool)

    if (not _is_int(shots_count)) or (shots_count < 0):
        raise RuntimeError(
            "Invalid `shots_count`. Must be a non-negative integer.")

    if max_trajectories is not None:
        if (not _is_int(max_trajectories)) or (max_trajectories < 1):
            raise RuntimeError(
                "Invalid `max_trajectories`. Must be a positive integer.")

    if max_shots_per_path is not None:
        if (not _is_int(max_shots_per_path)) or (max_shots_per_path < 0):
            raise RuntimeError(
                "Invalid `max_shots_per_path`. Must be a non-negative "
                "integer.")

    for name, value in (("num_root_draws", num_root_draws),
                        ("max_paths_per_root", max_paths_per_root),
                        ("max_live_states", max_live_states)):
        if value is not None:
            if (not _is_int(value)) or (value < 1):
                raise RuntimeError("Invalid `" + name +
                                   "`. Must be a positive integer.")

    for name, value in (("allow_non_unitary", allow_non_unitary),
                        ("include_sequential_data", include_sequential_data)):
        if not isinstance(value, bool):
            raise RuntimeError("Invalid `" + name + "`. Must be a bool.")

    _detail_check_conditionals_on_measure(decorator)

    return decorator


@trace.traced
def sample(kernel,
           *args,
           shots_count=1000,
           noise_model=None,
           max_trajectories=None,
           sampling_strategy=None,
           shot_allocation=None,
           return_execution_data=False,
           include_sequential_data=False,
           max_shots_per_path=None,
           num_root_draws=None,
           max_paths_per_root=None,
           max_live_states=None,
           allow_non_unitary=False):
    """
    Sample using Pre-Trajectory Sampling with Batch Execution (`PTSBE`).

    Pre-samples noise realizations (trajectories) and batches circuit
    executions by unique noise configuration, enabling efficient noisy
    sampling of many shots.

    When called with list arguments (broadcast mode), executes the kernel
    for each set of arguments and returns a list of results.

    Args:
      kernel: The quantum kernel to execute.
      shots_count (int): Number of measurement shots. Defaults to 1000.
      noise_model: Optional noise model for gate-based noise. Noise can also
          be specified inside the kernel via ``cudaq.apply_noise()``; both
          can be used together.
      max_trajectories (int or ``None``): Maximum unique trajectories to
          generate. ``None`` means use the number of shots. Note for large
          shot counts setting a maximum is recommended to get the benefits
          of PTS.
      sampling_strategy (``PTSSamplingStrategy`` or ``None``): Strategy for
          trajectory generation. ``None`` uses the default probabilistic
          sampling strategy.
      shot_allocation (``ShotAllocationStrategy`` or ``None``): Strategy for
          allocating shots across trajectories. ``None`` uses the default
          proportional (weight-based) allocation.
      return_execution_data (bool): Include circuit structure, trajectory
          specifications, and per-trajectory measurement outcomes in the
          returned result. Defaults to ``False``.
      include_sequential_data (bool): Populate per-shot sequential bitstring
          data on the result. Defaults to ``False``. For kernels with
          mid-circuit measurement or reset the default result is counts
          over unique full records (a lossless sufficient statistic for
          the per-shot table); set this to ``True`` for one record string
          per shot via ``result.get_sequential_data()``.
      max_shots_per_path (int or ``None``): Maximum shots sharing one replay
          of a trajectory (one replay path). ``None`` (default) selects
          automatically: 1 when the kernel
          contains mid-circuit measurement or reset or when any frontier
          knob (``num_root_draws``, ``max_paths_per_root``,
          ``max_live_states``) is set, unlimited otherwise. 0 forces
          unlimited. The environment variable
          ``CUDAQ_PTSBE_MAX_SHOTS_PER_PATH`` takes precedence.
      num_root_draws (int or ``None``): Fixed number of independent root
          draws performed before deduplication. Shot allocation becomes the
          exact root-weight split; configurations that cannot satisfy the
          flat-result integer conditions raise errors rather than being
          silently adjusted. ``None`` keeps strategy-controlled budgeting.
      max_paths_per_root (int or ``None``): Maximum replay paths sampled for
          one root. Configurations requiring more paths raise errors.
          ``None`` means unbounded.
      max_live_states (int or ``None``): Requested live frontier width, the
          maximum number of statevectors that stay live in one path group of
          the branching frontier executor. The resident allocation rounds this
          up to the next power of two; the executor reports the requested width
          and rounded capacity in its frontier metrics. ``None`` lets the
          executor pick a width automatically: kernels with mid-circuit
          measurement, reset, or admitted non-unitary Kraus channels use one
          live state per replay path bounded by device memory.
      allow_non_unitary (bool): Admit general (non-unitary) Kraus channels.
          Admitted channels keep their raw operators and branch during
          replay at their true state-dependent probabilities. Requires a
          batched simulator backend. With ``max_live_states`` unset the
          executor selects the frontier width automatically, so no capacity
          knob is required. Defaults to ``False``.

    Returns:
      ``SampleResult``: Measurement results. Returns a list of results
          in broadcast mode.

    Record semantics with mid-circuit measurement:
      Every measurement site (mid-circuit or terminal) contributes one bit
      per measured qubit to a fixed-width per-shot record. Each
      ``get_sequential_data()`` string is one full record in record-index
      order, and the counts distribution is over full records rather than
      terminal bits alone. ``result.record_layout`` lists one ``RecordSite``
      per record bit with its ``record_index``, ``qubit``, ``resets`` and
      ``terminal`` flags, and the kernel's measurement ``register_name``
      when named.

    Raises:
      RuntimeError: If the kernel is invalid or arguments are invalid.
    """
    decorator = _validate_ptsbe_args(kernel, args, shots_count, noise_model,
                                     max_trajectories, max_shots_per_path,
                                     num_root_draws, max_paths_per_root,
                                     max_live_states, allow_non_unitary,
                                     include_sequential_data)

    if noise_model is None:
        noise_model = cudaq_runtime.NoiseModel()

    if __isBroadcast(decorator, *args):
        argSets = __createArgumentSet(*args)
        results = []
        for argSet in argSets:
            processedArgs, module = decorator.prepare_call(*argSet)
            compiled = decorator.cachedCompiledModule()
            result = cudaq_runtime.ptsbe.sample_impl(
                decorator.uniqName, module, compiled, shots_count, noise_model,
                max_trajectories, sampling_strategy, shot_allocation,
                return_execution_data, include_sequential_data,
                max_shots_per_path, num_root_draws, max_paths_per_root,
                max_live_states, allow_non_unitary, *processedArgs)
            results.append(result)
        return results

    processedArgs, module = decorator.prepare_call(*args)
    compiled = decorator.cachedCompiledModule()
    return cudaq_runtime.ptsbe.sample_impl(
        decorator.uniqName, module, compiled, shots_count, noise_model,
        max_trajectories, sampling_strategy, shot_allocation,
        return_execution_data, include_sequential_data, max_shots_per_path,
        num_root_draws, max_paths_per_root, max_live_states, allow_non_unitary,
        *processedArgs)


@trace.traced
def sample_async(kernel,
                 *args,
                 shots_count=1000,
                 noise_model=None,
                 max_trajectories=None,
                 sampling_strategy=None,
                 shot_allocation=None,
                 return_execution_data=False,
                 include_sequential_data=False,
                 max_shots_per_path=None,
                 num_root_draws=None,
                 max_paths_per_root=None,
                 max_live_states=None,
                 allow_non_unitary=False):
    """
    Asynchronously sample using PTSBE. Returns a future whose result
    can be retrieved via ``.get()``.

    Args:
      kernel: The quantum kernel to execute.
      shots_count (int): Number of measurement shots. Defaults to 1000.
      noise_model: Optional noise model for gate-based noise; noise can also
          be specified in the kernel via ``cudaq.apply_noise()``.
      max_trajectories (int or ``None``): Maximum unique trajectories.
      sampling_strategy (``PTSSamplingStrategy`` or ``None``): Strategy for
          trajectory generation.
      shot_allocation (``ShotAllocationStrategy`` or ``None``): Strategy for
          allocating shots across trajectories.
      return_execution_data (bool): Include execution data in the result.
      include_sequential_data (bool): Populate per-shot sequential data.
          Defaults to ``False``: mid-circuit kernels then return counts
          over unique full records; see ``sample`` for the record
          semantics and ``record_layout``.
      max_shots_per_path (int or ``None``): Maximum shots sharing one replay
          of a trajectory (one replay path).
          ``None`` selects automatically; 0 forces unlimited. The environment
          variable ``CUDAQ_PTSBE_MAX_SHOTS_PER_PATH`` takes precedence.
      num_root_draws (int or ``None``): Fixed number of independent root
          draws before deduplication; see ``sample``.
      max_paths_per_root (int or ``None``): Maximum replay paths per root;
          see ``sample``.
      max_live_states (int or ``None``): Requested live frontier width per
          path group; ``None`` lets the executor pick automatically, including
          for non-unitary channels. See ``sample``.
      allow_non_unitary (bool): Admit general (non-unitary) Kraus channels;
          no capacity knob is required when ``max_live_states`` is unset. See
          ``sample``.

    Returns:
      ``AsyncPTSBESampleResult``: A future whose ``.get()`` returns the
          ``SampleResult``.

    Raises:
      RuntimeError: If the kernel is invalid or arguments are invalid.
    """
    decorator = _validate_ptsbe_args(kernel, args, shots_count, noise_model,
                                     max_trajectories, max_shots_per_path,
                                     num_root_draws, max_paths_per_root,
                                     max_live_states, allow_non_unitary,
                                     include_sequential_data)

    if noise_model is None:
        noise_model = cudaq_runtime.NoiseModel()

    processedArgs, module = decorator.prepare_call(*args)

    impl = cudaq_runtime.ptsbe.sample_async_impl(
        decorator.uniqName, module, shots_count, noise_model, max_trajectories,
        sampling_strategy, shot_allocation, return_execution_data,
        include_sequential_data, max_shots_per_path, num_root_draws,
        max_paths_per_root, max_live_states, allow_non_unitary, *processedArgs)

    return AsyncSampleResult(impl, module)
