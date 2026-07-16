# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Convergence gate: one PTSBE executor, no unitary_noise_as_branch flag.

These tests pin the post-convergence public interface. Mode selection between
the flat degenerate configuration (terminal-only unitary, B=1) and the
prefix-sharing tree (mid-circuit measurement or non-unitary noise with B > 1)
is internal and automatic. There is no user-facing tree-vs-flat choice, so the
``unitary_noise_as_branch`` option must not appear anywhere on the public
surface. They fail while the flag still exists and pass once convergence lands.
"""
import inspect
from pathlib import Path

import pytest
import cudaq
import cudaq.runtime.ptsbe as ptsbe_module

# The removed public option name. Kept as a single literal so the intent of
# each assertion is unambiguous.
REMOVED_OPTION = "unitary_noise_as_branch"

# cuda-quantum repository root, derived from this test's location:
# .../cuda-quantum/python/tests/ptsbe/test_convergence.py
CUDAQ_ROOT = Path(__file__).resolve().parents[3]

# Public-surface source files that carried the flag before convergence.
PUBLIC_SURFACE_FILES = [
    CUDAQ_ROOT / "runtime" / "cudaq" / "ptsbe" / "PTSBEOptions.h",
    CUDAQ_ROOT / "python" / "cudaq" / "runtime" / "ptsbe.py",
    CUDAQ_ROOT / "python" / "runtime" / "cudaq" / "algorithms" /
    "py_sample_ptsbe.cpp",
]


def test_sample_signature_has_no_unitary_flag():
    params = inspect.signature(ptsbe_module.sample).parameters
    assert REMOVED_OPTION not in params, (
        "ptsbe.sample must not expose the unitary_noise_as_branch flag; "
        "mode selection is internal and automatic after convergence.")


def test_sample_async_signature_has_no_unitary_flag():
    params = inspect.signature(ptsbe_module.sample_async).parameters
    assert REMOVED_OPTION not in params, (
        "ptsbe.sample_async must not expose the unitary_noise_as_branch flag.")


def test_validate_helper_has_no_unitary_flag_param():
    params = inspect.signature(ptsbe_module._validate_ptsbe_args).parameters
    assert REMOVED_OPTION not in params, (
        "_validate_ptsbe_args must not carry the unitary_noise_as_branch "
        "parameter after convergence.")


def test_sample_rejects_unitary_flag_kwarg(depol_noise, bell_kernel):
    with pytest.raises(TypeError):
        cudaq.ptsbe.sample(bell_kernel,
                           noise_model=depol_noise,
                           **{REMOVED_OPTION: True})


def test_sample_async_rejects_unitary_flag_kwarg(depol_noise, bell_kernel):
    with pytest.raises(TypeError):
        cudaq.ptsbe.sample_async(bell_kernel,
                                 noise_model=depol_noise,
                                 **{REMOVED_OPTION: True})


@pytest.mark.parametrize("surface_file",
                         PUBLIC_SURFACE_FILES,
                         ids=[f.name for f in PUBLIC_SURFACE_FILES])
def test_public_surface_source_has_no_unitary_flag(surface_file):
    # Guard against source drift back to a user-facing flag. The degenerate
    # flat path is the frontier's B=1 configuration, not a separate option.
    assert surface_file.exists(), f"missing public-surface file: {surface_file}"
    text = surface_file.read_text()
    assert REMOVED_OPTION not in text, (
        f"{surface_file} still references {REMOVED_OPTION}; the flag and the "
        "separate flat executor must be removed at convergence.")
