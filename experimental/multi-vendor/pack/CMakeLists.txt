# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(therock_multi_vendor_module_packs LANGUAGES NONE)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
foreach(_required THEROCK_MODULE_TOOLS_DIR THEROCK_MODULE_KPACK_PYTHON_DIR
    THEROCK_MODULE_TARGETS_FILE THEROCK_MODULE_BUILD_ROOT)
  if(NOT ${_required})
    message(FATAL_ERROR "Set ${_required} for module pack assembly")
  endif()
endforeach()

# Native module subproject stages exist before this child is configured.
file(GLOB_RECURSE _payload_inputs CONFIGURE_DEPENDS LIST_DIRECTORIES FALSE
  "${THEROCK_MODULE_BUILD_ROOT}/*.hsaco"
  "${THEROCK_MODULE_BUILD_ROOT}/*.cubin"
  "${THEROCK_MODULE_BUILD_ROOT}/*.ptx"
  "${THEROCK_MODULE_BUILD_ROOT}/*.spirv"
)
set(_outputs)
foreach(_module saxpy relu)
  list(APPEND _outputs
    "${CMAKE_CURRENT_BINARY_DIR}/packs/${_module}/catalog.json"
    "${CMAKE_CURRENT_BINARY_DIR}/packs/${_module}/validation-${_module}.kpack"
  )
endforeach()
add_custom_command(OUTPUT ${_outputs}
  COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${THEROCK_MODULE_KPACK_PYTHON_DIR}"
    "${Python3_EXECUTABLE}" "${THEROCK_MODULE_TOOLS_DIR}/assemble_multi_vendor_modules.py"
    --targets-file "${THEROCK_MODULE_TARGETS_FILE}"
    --module-build-root "${THEROCK_MODULE_BUILD_ROOT}"
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/packs"
  DEPENDS ${_payload_inputs} "${THEROCK_MODULE_TARGETS_FILE}"
    "${THEROCK_MODULE_TOOLS_DIR}/assemble_multi_vendor_modules.py"
    "${THEROCK_MODULE_TOOLS_DIR}/_therock_utils/payload_catalog.py"
    "${THEROCK_MODULE_TOOLS_DIR}/_therock_utils/gpu_targets.py"
    "${THEROCK_MODULE_KPACK_PYTHON_DIR}/rocm_kpack/kpack.py"
    "${THEROCK_MODULE_KPACK_PYTHON_DIR}/rocm_kpack/compression.py"
  COMMENT "Assembling verified multi-vendor module packs"
  VERBATIM
)
add_custom_target(module-packs ALL DEPENDS ${_outputs})
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/packs/" DESTINATION share/therock/packs)
