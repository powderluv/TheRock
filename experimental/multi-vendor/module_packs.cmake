# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Native module builds share target identities and artifacts with the HIP
# profile, while NVIDIA uses CUDA directly instead of the HIP header adapter.
set(_native_projects)
set(_native_keys)
set(_module_build_root "${CMAKE_CURRENT_BINARY_DIR}/modules")
set(_kpack_python "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/shared/kpack/python")
set(_runtime_helper "${THEROCK_SOURCE_DIR}/build_tools/stage_multi_vendor_runtime.py")
set(_runtime_source_map "${THEROCK_SOURCE_DIR}/build_tools/multi_vendor_runtime_sources.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${_runtime_helper}" "${_runtime_source_map}")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${_runtime_helper}" list
    --tools-dir "${THEROCK_SOURCE_DIR}/build_tools" --kpack-python-dir "${_kpack_python}"
  OUTPUT_VARIABLE _runtime_listing OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
string(JSON _runtime_source_count LENGTH "${_runtime_listing}" source_files)
math(EXPR _runtime_source_last "${_runtime_source_count} - 1")
set(_runtime_sources)
foreach(_index RANGE 0 ${_runtime_source_last})
  string(JSON _source GET "${_runtime_listing}" source_files ${_index})
  list(APPEND _runtime_sources "${_source}")
endforeach()
set(_module_descriptor "")
foreach(_key IN LISTS THEROCK_MULTI_VENDOR_TARGET_KEYS)
  set(_vendor "${THEROCK_MULTI_VENDOR_${_key}_VENDOR}")
  set(_sdk_args)
  set(_native_sources kernels.cpp loader.cpp module_service_hip_cuda.h module_service_blas.h)
  set(_enable_sgemm OFF)
  if(THEROCK_ENABLE_MULTI_VENDOR_SGEMM AND _vendor MATCHES "^(amd|nvidia)$")
    set(_enable_sgemm ON)
  endif()
  if(_vendor STREQUAL "amd")
    set(_sdk "${THEROCK_MULTI_VENDOR_AMD_ROOT}")
    set(_compiler_target "${THEROCK_MULTI_VENDOR_${_key}_COMPILER_TARGET}")
    _therock_multi_vendor_compiler(_compiler amd "${_sdk}" clang++)
  elseif(_vendor STREQUAL "intel")
    set(_sdk "${THEROCK_MULTI_VENDOR_LEVEL_ZERO_ROOT}")
    if(NOT _sdk)
      message(FATAL_ERROR "Intel native modules require THEROCK_MULTI_VENDOR_LEVEL_ZERO_ROOT")
    endif()
    set(_compiler_target "${THEROCK_MULTI_VENDOR_${_key}_PROCESSOR}")
    _therock_multi_vendor_compiler(_compiler spirv "${THEROCK_MULTI_VENDOR_AMD_ROOT}" clang)
    set(_native_sources kernels.cl level_zero_loader.cpp module_service_level_zero.h)
    list(APPEND _sdk_args
      "-DTHEROCK_MODULE_SPIRV_TRANSLATOR=${THEROCK_MULTI_VENDOR_RESOLVED_SPIRV_TRANSLATOR}"
      "-DTHEROCK_MODULE_SPIRV_VALIDATOR=${THEROCK_MULTI_VENDOR_RESOLVED_SPIRV_VALIDATOR}"
      "-DTHEROCK_MODULE_INTEL_DEVICE_ID=${THEROCK_MULTI_VENDOR_INTEL_DEVICE_ID}"
    )
  else()
    set(_sdk "${THEROCK_MULTI_VENDOR_CUDA_ROOT}")
    set(_compiler_target "${THEROCK_MULTI_VENDOR_${_key}_HIP_ARCHITECTURE}")
    _therock_multi_vendor_compiler(_compiler nvidia "${_sdk}" nvcc)
  endif()
  set(_native_project "native-modules-${_key}")
  therock_cmake_subproject_declare("${_native_project}"
    EXTERNAL_SOURCE_DIR "${THEROCK_SOURCE_DIR}/tests/multi_vendor/modules"
    BINARY_DIR "${_module_build_root}/${_key}"
    DISABLE_AMDGPU_TARGETS
    BUILD_GUARDS multi-vendor-verify-inputs
    EXTRA_DEPENDS "${_therock_input_report}" "${_therock_input_guard}"
      "${THEROCK_SOURCE_DIR}/build_tools/configure_module_contract.py"
      "${THEROCK_SOURCE_DIR}/build_tools/_therock_utils/module_contract.py"
      "${THEROCK_SOURCE_DIR}/build_tools/_therock_utils/sgemm_contract.py"
    NO_INSTALL_RPATH
    FPRINT_SOURCE_HASH
    CMAKE_ARGS
      "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
      "-DTHEROCK_MODULE_TOOLS_DIR=${THEROCK_SOURCE_DIR}/build_tools"
      "-DTHEROCK_MODULE_BACKEND=${_vendor}"
      "-DTHEROCK_MODULE_ENABLE_SGEMM=${_enable_sgemm}"
      "-DTHEROCK_MODULE_TARGET=${_compiler_target}"
      "-DTHEROCK_MODULE_SDK_ROOT=${_sdk}"
      "-DTHEROCK_MODULE_COMPILER=${_compiler}"
      "-DTHEROCK_MODULE_INSTALL_SUBDIR=${_key}"
      "-DTHEROCK_MODULE_DEVICE_INDEX=${THEROCK_MULTI_VENDOR_DEVICE_INDEX}"
      "-DBUILD_TESTING=ON"
      "-DTHEROCK_MULTI_VENDOR_INPUT_GUARD=${_therock_input_guard}"
      "-DTHEROCK_MULTI_VENDOR_INPUT_PROVENANCE=${_therock_input_report}"
      ${_sdk_args}
  )
  foreach(_source IN LISTS _native_sources)
    target_sources("${_native_project}" PRIVATE
      "${THEROCK_SOURCE_DIR}/tests/multi_vendor/modules/${_source}")
  endforeach()
  target_sources("${_native_project}" PRIVATE
    "${THEROCK_SOURCE_DIR}/tests/multi_vendor/input_guard.cmake"
    "${THEROCK_SOURCE_DIR}/tests/multi_vendor/modules/module_contract_options.h"
    "${THEROCK_SOURCE_DIR}/tests/multi_vendor/modules/module_session_options.h"
    "${THEROCK_SOURCE_DIR}/tests/multi_vendor/modules/module_service_protocol.h"
    "${THEROCK_SOURCE_DIR}/tests/multi_vendor/modules/device_inventory.h"
    "${THEROCK_SOURCE_DIR}/build_tools/configure_module_contract.py"
    "${THEROCK_SOURCE_DIR}/build_tools/_therock_utils/module_contract.py"
    "${THEROCK_SOURCE_DIR}/build_tools/_therock_utils/sgemm_contract.py")
  therock_cmake_subproject_activate("${_native_project}")
  _therock_multi_vendor_register_inputs("${_native_project}")
  list(APPEND _native_projects "${_native_project}")
  list(APPEND _native_keys "${_key}")
  string(APPEND _module_descriptor
    "[components.test.\"experimental/multi-vendor/modules/${_key}/stage\"]\n"
    "include = [\"bin/${_key}/**\"]\n\n"
  )
endforeach()
if(NOT _native_projects)
  return()
endif()

therock_cmake_subproject_declare(multi-vendor-module-packs
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pack"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/module-packs"
  DISABLE_AMDGPU_TARGETS
  BUILD_GUARDS multi-vendor-verify-inputs
  NO_MERGE_COMPILE_COMMANDS
  CMAKE_ARGS
    "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
    "-DTHEROCK_MODULE_TOOLS_DIR=${THEROCK_SOURCE_DIR}/build_tools"
    "-DTHEROCK_MODULE_KPACK_PYTHON_DIR=${_kpack_python}"
    "-DTHEROCK_MODULE_TARGETS_FILE=${CMAKE_CURRENT_BINARY_DIR}/gpu_targets.json"
    "-DTHEROCK_MODULE_BUILD_ROOT=${_module_build_root}"
    "-DTHEROCK_MULTI_VENDOR_INPUT_GUARD=${_therock_input_guard}"
    "-DTHEROCK_MULTI_VENDOR_INPUT_PROVENANCE=${_therock_input_report}"
  EXTRA_DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/gpu_targets.json"
    "${_therock_input_report}" "${_therock_input_guard}"
  BUILD_DEPS ${_native_projects}
)
target_sources(multi-vendor-module-packs PRIVATE
  ${_runtime_sources}
  "${THEROCK_SOURCE_DIR}/tests/multi_vendor/input_guard.cmake"
  "${THEROCK_SOURCE_DIR}/build_tools/assemble_multi_vendor_modules.py"
  "${THEROCK_SOURCE_DIR}/build_tools/_therock_utils/payload_catalog.py"
  "${THEROCK_SOURCE_DIR}/build_tools/_therock_utils/runner_registry.py"
  "${THEROCK_SOURCE_DIR}/build_tools/_therock_utils/module_contract.py"
  "${THEROCK_SOURCE_DIR}/build_tools/_therock_utils/gpu_targets.py"
  "${_kpack_python}/rocm_kpack/kpack.py"
  "${_kpack_python}/rocm_kpack/compression.py"
)
therock_cmake_subproject_activate(multi-vendor-module-packs)
_therock_multi_vendor_register_inputs(multi-vendor-module-packs)
add_dependencies(therock-build-tests multi-vendor-module-packs)
string(APPEND _module_descriptor
  "[components.test.\"experimental/multi-vendor/module-packs/stage\"]\n"
  "include = [\"share/therock/packs/**\", \"share/therock/python/**\", \"share/therock/examples/**\"]\n"
)
set(_descriptor_path "${CMAKE_CURRENT_BINARY_DIR}/artifact-modules.toml")
file(WRITE "${_descriptor_path}" "${_module_descriptor}")
list(JOIN _native_keys "-" _native_bundle)
therock_provide_artifact(multi-vendor-modules
  BUILD_GUARDS multi-vendor-verify-inputs
  DESCRIPTOR "${_descriptor_path}"
  DISTRIBUTION multi-vendor-modules
  DIST_BUNDLE_NAME "${_native_bundle}"
  COMPONENTS test
  SUBPROJECT_DEPS ${_native_projects} multi-vendor-module-packs
)

set(_module_dist "${THEROCK_BINARY_DIR}/dist/multi-vendor-modules")
set(_module_receipts "${_module_dist}/share/therock/packs/input-provenance.json")
foreach(_key IN LISTS _native_keys)
  list(APPEND _module_receipts "${_module_dist}/bin/${_key}/input-build-receipt.json")
endforeach()
add_test(NAME multi-vendor-module-receipts
  COMMAND "${CMAKE_COMMAND}"
    "-DTHEROCK_MULTI_VENDOR_CHECK_PROVENANCE=${_therock_input_report}"
    "-DTHEROCK_MULTI_VENDOR_CHECK_RECEIPTS=${_module_receipts}"
    -P "${THEROCK_SOURCE_DIR}/tests/multi_vendor/input_guard.cmake")
set_tests_properties(multi-vendor-module-receipts PROPERTIES
  FIXTURES_SETUP multi-vendor-module-receipts FIXTURES_REQUIRED multi-vendor-inputs
  LABELS "multi-vendor;provenance")
add_test(NAME multi-vendor-device-discovery
  COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_kpack_python}"
    "${Python3_EXECUTABLE}" "${THEROCK_SOURCE_DIR}/build_tools/dispatch_multi_vendor_modules.py"
    --dist-root "${_module_dist}" list)
set_tests_properties(multi-vendor-device-discovery PROPERTIES
  LABELS "multi-vendor;discovery"
  FIXTURES_REQUIRED "multi-vendor-inputs;multi-vendor-module-receipts"
  RUN_SERIAL TRUE TIMEOUT 180)
foreach(_key IN LISTS _native_keys)
  set(_vendor "${THEROCK_MULTI_VENDOR_${_key}_VENDOR}")
  set(_identity_args)
  if(_vendor STREQUAL "amd")
    set(_formats hsaco)
  elseif(_vendor STREQUAL "intel")
    set(_formats spirv)
    if(NOT THEROCK_MULTI_VENDOR_INTEL_DEVICE_ID STREQUAL "")
      list(APPEND _identity_args --expect-device-id "${THEROCK_MULTI_VENDOR_INTEL_DEVICE_ID}")
    endif()
  else()
    set(_formats cubin ptx)
  endif()
  set(_session_formats ${_formats})
  if(_vendor STREQUAL "nvidia")
    list(APPEND _session_formats mixed)
  endif()
  foreach(_format IN LISTS _session_formats)
    set(_saxpy_format "${_format}")
    set(_relu_format "${_format}")
    if(_format STREQUAL "mixed")
      set(_saxpy_format cubin)
      set(_relu_format ptx)
    endif()
    set(_test "packed-client-${_key}-${_format}")
    add_test(NAME "${_test}"
      COMMAND "${Python3_EXECUTABLE}" -I
        "${_module_dist}/share/therock/examples/packed_session_client.py"
        --dist-root "${_module_dist}"
        --target "${THEROCK_MULTI_VENDOR_${_key}_ID}"
        --format "${_format}"
        --device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}"
        ${_identity_args}
    )
    set_tests_properties("${_test}" PROPERTIES
      LABELS "multi-vendor;packed-module;module-client;gpu;${_vendor}"
      FIXTURES_REQUIRED "multi-vendor-inputs;multi-vendor-module-receipts"
      RUN_SERIAL TRUE TIMEOUT 180)
    set(_test "packed-service-${_key}-${_format}")
    add_test(NAME "${_test}"
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_kpack_python}"
        "${Python3_EXECUTABLE}" "${THEROCK_SOURCE_DIR}/build_tools/dispatch_multi_vendor_modules.py"
        --dist-root "${_module_dist}" run-service
        --target "${THEROCK_MULTI_VENDOR_${_key}_ID}"
        --module validation/saxpy "${_saxpy_format}" therock_module_saxpy
        --module validation/relu "${_relu_format}" therock_module_relu
        --module validation/saxpy "${_saxpy_format}" therock_module_saxpy
        --device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}"
        ${_identity_args}
    )
    set_tests_properties("${_test}" PROPERTIES
      LABELS "multi-vendor;packed-module;module-service;gpu;${_vendor}"
      FIXTURES_REQUIRED "multi-vendor-inputs;multi-vendor-module-receipts"
      RUN_SERIAL TRUE TIMEOUT 180)
    set(_test "packed-pipeline-${_key}-${_format}")
    add_test(NAME "${_test}"
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_kpack_python}"
        "${Python3_EXECUTABLE}" "${THEROCK_SOURCE_DIR}/build_tools/dispatch_multi_vendor_modules.py"
        --dist-root "${_module_dist}" run-pipeline
        --target "${THEROCK_MULTI_VENDOR_${_key}_ID}"
        --module validation/saxpy "${_saxpy_format}" therock_module_saxpy
        --module validation/relu "${_relu_format}" therock_module_relu
        --module validation/saxpy "${_saxpy_format}" therock_module_saxpy
        --device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}"
        ${_identity_args}
    )
    set_tests_properties("${_test}" PROPERTIES
      LABELS "multi-vendor;packed-module;module-pipeline;gpu;${_vendor}"
      FIXTURES_REQUIRED "multi-vendor-inputs;multi-vendor-module-receipts"
      RUN_SERIAL TRUE TIMEOUT 180)
    set(_test "packed-session-${_key}-${_format}")
    add_test(NAME "${_test}"
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_kpack_python}"
        "${Python3_EXECUTABLE}" "${THEROCK_SOURCE_DIR}/build_tools/dispatch_multi_vendor_modules.py"
        --dist-root "${_module_dist}" run-batch
        --target "${THEROCK_MULTI_VENDOR_${_key}_ID}"
        --module validation/saxpy "${_saxpy_format}" therock_module_saxpy
        --module validation/relu "${_relu_format}" therock_module_relu
        --device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}"
        ${_identity_args}
    )
    set_tests_properties("${_test}" PROPERTIES
      LABELS "multi-vendor;packed-module;module-session;gpu;${_vendor}"
      FIXTURES_REQUIRED "multi-vendor-inputs;multi-vendor-module-receipts"
      RUN_SERIAL TRUE TIMEOUT 180)
  endforeach()
  if(THEROCK_ENABLE_MULTI_VENDOR_SGEMM AND _vendor MATCHES "^(amd|nvidia)$")
    set(_sgemm_format hsaco)
    if(_vendor STREQUAL "nvidia")
      set(_sgemm_format mixed)
    endif()
    set(_test "sgemm-client-${_key}")
    add_test(NAME "${_test}"
      COMMAND "${Python3_EXECUTABLE}" -I
        "${_module_dist}/share/therock/python/therock_multi_vendor/sgemm_example.py"
        --dist-root "${_module_dist}" --target "${THEROCK_MULTI_VENDOR_${_key}_ID}"
        --format "${_sgemm_format}" --device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}")
    set_tests_properties("${_test}" PROPERTIES
      LABELS "multi-vendor;sgemm;module-client;gpu;${_vendor}"
      FIXTURES_REQUIRED "multi-vendor-inputs;multi-vendor-module-receipts"
      RUN_SERIAL TRUE TIMEOUT 180)
  endif()
  foreach(_module saxpy relu)
    foreach(_format IN LISTS _formats)
      set(_test "packed-module-${_key}-${_module}-${_format}")
      add_test(NAME "${_test}"
        COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_kpack_python}"
          "${Python3_EXECUTABLE}" "${THEROCK_SOURCE_DIR}/build_tools/dispatch_multi_vendor_modules.py"
          --dist-root "${_module_dist}" run
          --module "validation/${_module}"
          --target "${THEROCK_MULTI_VENDOR_${_key}_ID}"
          --format "${_format}"
          --entry-point "therock_module_${_module}"
          --require-capability cross-queue-events
          --device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}"
          ${_identity_args}
      )
      set_tests_properties("${_test}" PROPERTIES
        LABELS "multi-vendor;packed-module;gpu;${_vendor}"
        FIXTURES_REQUIRED "multi-vendor-inputs;multi-vendor-module-receipts"
        RUN_SERIAL TRUE TIMEOUT 180)
    endforeach()
  endforeach()
endforeach()

# A single Python consumer keeps one worker per vendor alive and transfers
# intermediate values through host memory. Each exact pair remains a separate
# qualification target; registering an SM90 case does not qualify that hardware.
foreach(_amd_key IN LISTS _native_keys)
  if(NOT THEROCK_MULTI_VENDOR_${_amd_key}_VENDOR STREQUAL "amd")
    continue()
  endif()
  foreach(_nvidia_key IN LISTS _native_keys)
    if(NOT THEROCK_MULTI_VENDOR_${_nvidia_key}_VENDOR STREQUAL "nvidia")
      continue()
    endif()
    if(THEROCK_ENABLE_MULTI_VENDOR_SGEMM)
      set(_test "sgemm-client-pair-${_amd_key}-${_nvidia_key}")
      add_test(NAME "${_test}"
        COMMAND "${Python3_EXECUTABLE}" -I
          "${_module_dist}/share/therock/python/therock_multi_vendor/sgemm_example.py"
          --dist-root "${_module_dist}"
          --target "${THEROCK_MULTI_VENDOR_${_amd_key}_ID}" --format hsaco
          --device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}"
          --peer-target "${THEROCK_MULTI_VENDOR_${_nvidia_key}_ID}" --peer-format mixed
          --peer-device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}")
      set_tests_properties("${_test}" PROPERTIES
        LABELS "multi-vendor;sgemm;module-client;multi-vendor-client;gpu;amd;nvidia"
        FIXTURES_REQUIRED "multi-vendor-inputs;multi-vendor-module-receipts"
        RUN_SERIAL TRUE TIMEOUT 180)
    endif()
    set(_test "packed-client-pair-${_amd_key}-${_nvidia_key}")
    add_test(NAME "${_test}"
      COMMAND "${Python3_EXECUTABLE}" -I
        "${_module_dist}/share/therock/examples/packed_session_client.py"
        --dist-root "${_module_dist}"
        --target "${THEROCK_MULTI_VENDOR_${_amd_key}_ID}" --format hsaco
        --device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}"
        --peer-target "${THEROCK_MULTI_VENDOR_${_nvidia_key}_ID}" --peer-format mixed
        --peer-device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}"
    )
    set_tests_properties("${_test}" PROPERTIES
      LABELS "multi-vendor;packed-module;module-client;multi-vendor-client;gpu;amd;nvidia"
      FIXTURES_REQUIRED "multi-vendor-inputs;multi-vendor-module-receipts"
      RUN_SERIAL TRUE TIMEOUT 180)
  endforeach()
endforeach()

# Exercise selective delivery from the same complete build. Each case repacks
# only its exact targets, relocates the export, and runs the installed consumer
# from an unrelated working directory. Exporting never queries vendor runtimes.
foreach(_key IN LISTS _native_keys)
  set(_vendor "${THEROCK_MULTI_VENDOR_${_key}_VENDOR}")
  if(_vendor STREQUAL "amd")
    set(_format hsaco)
  elseif(_vendor STREQUAL "nvidia")
    set(_format mixed)
  else()
    set(_format spirv)
  endif()
  set(_identity_args)
  if(_vendor STREQUAL "intel" AND THEROCK_MULTI_VENDOR_INTEL_DEVICE_ID)
    list(APPEND _identity_args --expect-device-id "${THEROCK_MULTI_VENDOR_INTEL_DEVICE_ID}")
  endif()
  set(_test "exported-client-${_key}")
  add_test(NAME "${_test}"
    COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_kpack_python}"
      "${Python3_EXECUTABLE}" -B
      "${THEROCK_SOURCE_DIR}/tests/multi_vendor/modules/exported_session_client.py"
      --source-dist "${_module_dist}" --target "${THEROCK_MULTI_VENDOR_${_key}_ID}"
      -- --target "${THEROCK_MULTI_VENDOR_${_key}_ID}" --format "${_format}"
      --device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}" ${_identity_args}
  )
  set_tests_properties("${_test}" PROPERTIES
    LABELS "multi-vendor;selective-export;module-client;gpu;${_vendor}"
    FIXTURES_REQUIRED "multi-vendor-inputs;multi-vendor-module-receipts"
    RUN_SERIAL TRUE TIMEOUT 180)
endforeach()
foreach(_amd_key IN LISTS _native_keys)
  if(NOT THEROCK_MULTI_VENDOR_${_amd_key}_VENDOR STREQUAL "amd")
    continue()
  endif()
  foreach(_nvidia_key IN LISTS _native_keys)
    if(NOT THEROCK_MULTI_VENDOR_${_nvidia_key}_VENDOR STREQUAL "nvidia")
      continue()
    endif()
    set(_test "exported-client-pair-${_amd_key}-${_nvidia_key}")
    add_test(NAME "${_test}"
      COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_kpack_python}"
        "${Python3_EXECUTABLE}" -B
        "${THEROCK_SOURCE_DIR}/tests/multi_vendor/modules/exported_session_client.py"
        --source-dist "${_module_dist}"
        --target "${THEROCK_MULTI_VENDOR_${_amd_key}_ID}"
        --target "${THEROCK_MULTI_VENDOR_${_nvidia_key}_ID}"
        -- --target "${THEROCK_MULTI_VENDOR_${_amd_key}_ID}" --format hsaco
        --device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}"
        --peer-target "${THEROCK_MULTI_VENDOR_${_nvidia_key}_ID}" --peer-format mixed
        --peer-device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}"
    )
    set_tests_properties("${_test}" PROPERTIES
      LABELS "multi-vendor;selective-export;module-client;gpu;amd;nvidia"
      FIXTURES_REQUIRED "multi-vendor-inputs;multi-vendor-module-receipts"
      RUN_SERIAL TRUE TIMEOUT 180)
  endforeach()
endforeach()
