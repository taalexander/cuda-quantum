# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_TARGET_DEFINITIONS CompilerPasses.td)
add_cudaq_doc(CompilerPasses CompilerPassesBody -gen-pass-doc)

set(COMPILER_PASSES_BODY "${CUDAQ_BINARY_DIR}/docs/CompilerPassesBody.md")
set(COMPILER_PASSES_DOC "${CUDAQ_BINARY_DIR}/docs/CompilerPasses.md")
set(COMPILER_PASSES_DOC_SCRIPT
    "${CUDAQ_SOURCE_DIR}/cmake/modules/GenerateCompilerPassDocs.cmake")
add_custom_command(
  OUTPUT "${COMPILER_PASSES_DOC}"
  COMMAND "${CMAKE_COMMAND}"
          "-DINPUT_FILE=${COMPILER_PASSES_BODY}"
          "-DOUTPUT_FILE=${COMPILER_PASSES_DOC}"
          -P "${COMPILER_PASSES_DOC_SCRIPT}"
  DEPENDS "${COMPILER_PASSES_BODY}" "${COMPILER_PASSES_DOC_SCRIPT}"
  VERBATIM)
add_custom_target(CompilerPassesDocGen DEPENDS "${COMPILER_PASSES_DOC}")
add_dependencies(cudaq-doc CompilerPassesDocGen)

add_subdirectory(CodeGen)
add_subdirectory(Dialect)
add_subdirectory(Transforms)
