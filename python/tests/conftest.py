# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""
Shared pytest fixtures and markers for CUDA-Q Python tests.
"""

import sys
import platform
import pytest

# On macOS ARM64, C++ exceptions from JIT-compiled code cannot be caught.
# The process aborts instead of raising a catchable exception.
# See Building.md macOS Limitations and https://github.com/llvm/llvm-project/issues/49036
skip_macos_arm64_jit_exception = pytest.mark.skipif(
    sys.platform == 'darwin' and platform.machine() == 'arm64',
    reason="JIT exception handling broken on macOS ARM64 (llvm-project#49036)")
