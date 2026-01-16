#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# Run this script to validate CUDA-Q applications on macOS (CPU-only).
# This script tests both C++ applications (via ctest) and Python notebooks.
#
# Prerequisites:
#   - C++ tests require a CMake build directory (for CI reproducibility)
#   - Python tests require cudaq module and jupyter/nbconvert
#
# Options:
#   -c: Run C++ applications only
#   -p: Run Python notebooks only
#   -v: Verbose mode
#   -h: Show help
#
# Examples:
#   bash scripts/validate_applications_macos.sh       # Run all tests
#   bash scripts/validate_applications_macos.sh -c    # C++ only
#   bash scripts/validate_applications_macos.sh -p    # Python only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

run_cpp=true
run_python=true
verbose=false

while getopts "cpvh" opt; do
    case $opt in
        c) run_python=false ;;
        p) run_cpp=false ;;
        v) verbose=true ;;
        h)
            echo "Usage: $0 [-c] [-p] [-v] [-h]"
            echo "  -c  Run C++ applications only"
            echo "  -p  Run Python notebooks only"
            echo "  -v  Verbose mode"
            echo "  -h  Show this help"
            exit 0
            ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

echo "=============================================="
echo "CUDA-Q Application Validation (macOS/CPU)"
echo "=============================================="

cpp_failed=0

# ============================================
# C++ Applications (via ctest, same as CI)
# ============================================
if $run_cpp; then
    echo ""
    echo -e "${YELLOW}Testing C++ Applications...${NC}"
    echo "----------------------------------------------"
    
    BUILD_DIR="$REPO_ROOT/build"
    if [ ! -f "$BUILD_DIR/CTestTestfile.cmake" ]; then
        echo -e "${RED}Error: Build directory not found or not configured.${NC}"
        echo ""
        echo "C++ tests require a CMake build directory for CI reproducibility."
        echo "Please build CUDA-Q first:"
        echo ""
        echo "  ./scripts/build_cudaq.sh"
        echo ""
        exit 1
    fi
    
    echo "  Using ctest..."
    echo ""
    
    # Run the nvqpp_* doc tests (CPU-compatible applications)
    # --timeout 300: fail tests taking longer than 5 minutes
    # --verbose: show timing for each test
    cpp_start_time=$(date +%s)
    ctest --test-dir "$BUILD_DIR" --output-on-failure --verbose --timeout 300 \
        -R "nvqpp_(PhaseEstimation|Grover|QAOA|VQEH2|AmplitudeEstimation|IterativePhaseEstimation|RandomWalkPhaseEstimation)"
    ctest_status=$?
    cpp_end_time=$(date +%s)
    cpp_elapsed=$((cpp_end_time - cpp_start_time))
    
    echo ""
    if [ $ctest_status -eq 0 ]; then
        echo -e "${GREEN}All C++ tests passed${NC} (${cpp_elapsed}s total)"
    else
        echo -e "${RED}Some C++ tests failed${NC} (${cpp_elapsed}s total)"
        cpp_failed=1
    fi
fi

# ============================================
# Python Notebooks
# ============================================
if $run_python; then
    echo ""
    echo -e "${YELLOW}Testing Python Notebooks...${NC}"
    echo "----------------------------------------------"
    
    # Check if required tools are available
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: python3 not found${NC}"
        exit 1
    fi
    
    if ! python3 -c "import cudaq" 2>/dev/null; then
        echo -e "${RED}Error: cudaq module not found${NC}"
        exit 1
    fi
    
    if ! command -v jupyter &> /dev/null; then
        echo -e "${YELLOW}Warning: jupyter not found, installing...${NC}"
        pip install notebook nbconvert
    fi
    
    # Get available backends
    echo "  Detecting available backends..."
    backends=$(python3 -c "import cudaq; print(' '.join(cudaq.get_targets()))" 2>/dev/null || echo "qpp-cpu")
    echo "  Available backends: $backends"
    echo ""
    
    # Run notebook validation
    NOTEBOOK_DIR="$REPO_ROOT/docs/sphinx/applications/python"
    VALIDATION_SCRIPT="$REPO_ROOT/docs/notebook_validation.py"
    
    if [ -f "$VALIDATION_SCRIPT" ]; then
        echo "  Running notebook validation..."
        echo ""
        py_start_time=$(date +%s)
        echo "$backends" | python3 "$VALIDATION_SCRIPT"
        py_status=$?
        py_end_time=$(date +%s)
        py_elapsed=$((py_end_time - py_start_time))
        
        echo ""
        if [ $py_status -eq 0 ]; then
            echo -e "${GREEN}Notebook validation completed successfully${NC} (${py_elapsed}s total)"
        else
            echo -e "${RED}Some notebooks failed${NC} (${py_elapsed}s total)"
        fi
    else
        echo -e "${RED}Error: notebook_validation.py not found${NC}"
        exit 1
    fi
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "Validation Complete"
echo "=============================================="

# Exit with error if any tests failed
if [ $cpp_failed -gt 0 ]; then
    exit 1
fi
