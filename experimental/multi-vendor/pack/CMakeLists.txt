# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
include("${CMAKE_CURRENT_LIST_DIR}/../../../tests/multi_vendor/input_guard.cmake")
project(therock_multi_vendor_module_packs LANGUAGES NONE)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
therock_multi_vendor_initialize_inputs()
foreach(_required THEROCK_MODULE_TOOLS_DIR THEROCK_MODULE_KPACK_PYTHON_DIR
    THEROCK_MODULE_TARGETS_FILE THEROCK_MODULE_BUILD_ROOT)
  if(NOT ${_required})
    message(FATAL_ERROR "Set ${_required} for module pack assembly")
  endif()
endforeach()

# The runtime source map is shared by the parent fingerprint and this child.
# Scan bytes on every child build; content-only writes preserve no-op outputs.
set(_runtime_helper "${THEROCK_MODULE_TOOLS_DIR}/stage_multi_vendor_runtime.py")
set(_runtime_root "${CMAKE_CURRENT_BINARY_DIR}/runtime")
set(_runtime_args --tools-dir "${THEROCK_MODULE_TOOLS_DIR}"
  --kpack-python-dir "${THEROCK_MODULE_KPACK_PYTHON_DIR}")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${_runtime_helper}" list ${_runtime_args}
  OUTPUT_VARIABLE _runtime_listing OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
set(_runtime_sources)
set(_runtime_outputs)
foreach(_field source_files output_files)
  string(JSON _length LENGTH "${_runtime_listing}" "${_field}")
  math(EXPR _last "${_length} - 1")
  foreach(_index RANGE 0 ${_last})
    string(JSON _path GET "${_runtime_listing}" "${_field}" ${_index})
    if(_field STREQUAL "source_files")
      list(APPEND _runtime_sources "${_path}")
    else()
      list(APPEND _runtime_outputs "${_runtime_root}/${_path}")
    endif()
  endforeach()
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${THEROCK_MODULE_TOOLS_DIR}/multi_vendor_runtime_sources.json")
add_custom_target(module-runtime ALL
  COMMAND "${Python3_EXECUTABLE}" "${_runtime_helper}" build ${_runtime_args}
    --output-dir "${_runtime_root}"
  DEPENDS ${_runtime_sources} ${_therock_input_dependencies}
  BYPRODUCTS ${_runtime_outputs}
  COMMENT "Materializing the standalone multi-vendor Python runtime"
  VERBATIM)

# Native module subproject stages exist before this child is configured.
file(GLOB_RECURSE _payload_inputs CONFIGURE_DEPENDS LIST_DIRECTORIES FALSE
  "${THEROCK_MODULE_BUILD_ROOT}/*.hsaco"
  "${THEROCK_MODULE_BUILD_ROOT}/*.cubin"
  "${THEROCK_MODULE_BUILD_ROOT}/*.ptx"
  "${THEROCK_MODULE_BUILD_ROOT}/*.spirv"
  "${THEROCK_MODULE_BUILD_ROOT}/*/stage/bin/*/runner-contract.json"
  "${THEROCK_MODULE_BUILD_ROOT}/*/stage/bin/*/therock_module_validation"
)
# Resolve target slugs through the same canonical parser used by assembly.
# A direct pack-child build cannot rely on the parent's native BUILD_DEPS.
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${THEROCK_MODULE_KPACK_PYTHON_DIR}"
    "${Python3_EXECUTABLE}" -c
    "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); from assemble_multi_vendor_modules import read_targets; print(';'.join(t.slug for t in read_targets(Path(sys.argv[2]))))"
    "${THEROCK_MODULE_TOOLS_DIR}" "${THEROCK_MODULE_TARGETS_FILE}"
  OUTPUT_VARIABLE _target_slugs OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${THEROCK_MODULE_TARGETS_FILE}")
# Keep expected runner paths as explicit dependencies even if a staged file is
# deleted. A glob alone would silently remove that dependency when reconfigured.
foreach(_slug IN LISTS _target_slugs)
  list(APPEND _payload_inputs
    "${THEROCK_MODULE_BUILD_ROOT}/${_slug}/stage/bin/${_slug}/runner-contract.json"
    "${THEROCK_MODULE_BUILD_ROOT}/${_slug}/stage/bin/${_slug}/therock_module_validation")
endforeach()
list(REMOVE_DUPLICATES _payload_inputs)
set(_outputs "${CMAKE_CURRENT_BINARY_DIR}/packs/runners.json")
foreach(_module saxpy relu)
  list(APPEND _outputs
    "${CMAKE_CURRENT_BINARY_DIR}/packs/${_module}/catalog.json"
    "${CMAKE_CURRENT_BINARY_DIR}/packs/${_module}/validation-${_module}.kpack"
  )
endforeach()
set(_provenance_command)
set(_receipt_check_command)
set(_stage_receipts)
if(THEROCK_MULTI_VENDOR_INPUT_PROVENANCE)
  foreach(_slug IN LISTS _target_slugs)
    list(APPEND _stage_receipts
      "${THEROCK_MODULE_BUILD_ROOT}/${_slug}/stage/bin/${_slug}/input-build-receipt.json")
  endforeach()
  string(REPLACE ";" "\\;" _receipt_list_arg "${_stage_receipts}")
  list(APPEND _receipt_check_command COMMAND "${CMAKE_COMMAND}"
    "-DTHEROCK_MULTI_VENDOR_CHECK_PROVENANCE=${THEROCK_MULTI_VENDOR_INPUT_PROVENANCE}"
    "-DTHEROCK_MULTI_VENDOR_CHECK_RECEIPTS=${_receipt_list_arg}"
    -P "${CMAKE_CURRENT_LIST_DIR}/../../../tests/multi_vendor/input_guard.cmake")
  list(APPEND _outputs "${CMAKE_CURRENT_BINARY_DIR}/packs/input-provenance.json")
  list(APPEND _provenance_command COMMAND "${CMAKE_COMMAND}" -E copy
    "${THEROCK_MULTI_VENDOR_INPUT_PROVENANCE}" "${CMAKE_CURRENT_BINARY_DIR}/packs/input-provenance.json")
endif()
add_custom_command(OUTPUT ${_outputs}
  ${_receipt_check_command}
  COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${THEROCK_MODULE_KPACK_PYTHON_DIR}"
    "${Python3_EXECUTABLE}" "${THEROCK_MODULE_TOOLS_DIR}/assemble_multi_vendor_modules.py"
    --targets-file "${THEROCK_MODULE_TARGETS_FILE}"
    --module-build-root "${THEROCK_MODULE_BUILD_ROOT}"
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/packs"
  ${_provenance_command}
  DEPENDS ${_therock_input_dependencies} ${_stage_receipts} ${_payload_inputs} "${THEROCK_MODULE_TARGETS_FILE}"
    "${THEROCK_MODULE_TOOLS_DIR}/assemble_multi_vendor_modules.py"
    "${THEROCK_MODULE_TOOLS_DIR}/_therock_utils/payload_catalog.py"
    "${THEROCK_MODULE_TOOLS_DIR}/_therock_utils/module_contract.py"
    "${THEROCK_MODULE_TOOLS_DIR}/_therock_utils/runner_registry.py"
    "${THEROCK_MODULE_TOOLS_DIR}/_therock_utils/gpu_targets.py"
    "${THEROCK_MODULE_KPACK_PYTHON_DIR}/rocm_kpack/kpack.py"
    "${THEROCK_MODULE_KPACK_PYTHON_DIR}/rocm_kpack/compression.py"
  COMMENT "Assembling verified multi-vendor module packs"
  VERBATIM
)
add_custom_target(module-packs ALL DEPENDS ${_outputs})
add_dependencies(module-packs module-runtime)
set(_installed_outputs
  "share/therock/packs/input-provenance.json"
  "share/therock/packs/runners.json")
foreach(_module saxpy relu)
  list(APPEND _installed_outputs
    "share/therock/packs/${_module}/catalog.json"
    "share/therock/packs/${_module}/validation-${_module}.kpack")
endforeach()
# Verify the completed runtime BEFORE prepare_install removes old pack receipts.
# Direct cmake --install never repins Python sources or rebuilds missing files.
set(_runtime_command)
foreach(_argument "${Python3_EXECUTABLE}" "${_runtime_helper}")
  _therock_multi_vendor_quote(_quoted "${_argument}")
  string(APPEND _runtime_command " ${_quoted}")
endforeach()
set(_runtime_install_args)
foreach(_argument ${_runtime_args} --output-dir "${_runtime_root}")
  _therock_multi_vendor_quote(_quoted "${_argument}")
  string(APPEND _runtime_install_args " ${_quoted}")
endforeach()
install(CODE "execute_process(COMMAND ${_runtime_command} verify ${_runtime_install_args}
  COMMAND_ERROR_IS_FATAL ANY)")

therock_multi_vendor_prepare_install(
  RECEIPT "${CMAKE_CURRENT_BINARY_DIR}/packs/input-provenance.json"
  FILES ${_installed_outputs})
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/packs/" DESTINATION share/therock/packs
  PATTERN "input-provenance.json" EXCLUDE)
# The Python completion manifest is published after its files; the pack input
# receipt below remains the last publication for the complete child stage.
install(CODE "execute_process(COMMAND ${_runtime_command} install ${_runtime_install_args}
  --destination \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}\"
  COMMAND_ERROR_IS_FATAL ANY)")
string(JSON _runtime_output_count LENGTH "${_runtime_listing}" output_files)
math(EXPR _runtime_last "${_runtime_output_count} - 1")
foreach(_index RANGE 0 ${_runtime_last})
  string(JSON _relative GET "${_runtime_listing}" output_files ${_index})
  _therock_multi_vendor_quote(_quoted "${_relative}")
  install(CODE "set(_runtime_relative ${_quoted})
list(APPEND CMAKE_INSTALL_MANIFEST_FILES \"\${CMAKE_INSTALL_PREFIX}/\${_runtime_relative}\")")
endforeach()
if(THEROCK_MULTI_VENDOR_INPUT_PROVENANCE)
  therock_multi_vendor_install_receipt("${CMAKE_CURRENT_BINARY_DIR}/packs/input-provenance.json"
    share/therock/packs)
endif()
