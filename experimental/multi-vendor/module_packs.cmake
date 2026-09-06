# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Native module builds share target identities and artifacts with the HIP
# profile, while NVIDIA uses CUDA directly instead of the HIP header adapter.
set(_native_projects)
set(_native_keys)
set(_module_build_root "${CMAKE_CURRENT_BINARY_DIR}/modules")
set(_kpack_python "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/shared/kpack/python")
set(_module_descriptor "")
foreach(_key IN LISTS THEROCK_MULTI_VENDOR_TARGET_KEYS)
  set(_vendor "${THEROCK_MULTI_VENDOR_${_key}_VENDOR}")
  set(_sdk_args)
  set(_native_sources kernels.cpp loader.cpp)
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
    set(_native_sources kernels.cl level_zero_loader.cpp)
    list(APPEND _sdk_args
      "-DTHEROCK_MODULE_SPIRV_TRANSLATOR=${THEROCK_MULTI_VENDOR_SPIRV_TRANSLATOR}"
      "-DTHEROCK_MODULE_SPIRV_VALIDATOR=${THEROCK_MULTI_VENDOR_SPIRV_VALIDATOR}"
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
    NO_INSTALL_RPATH
    FPRINT_SOURCE_HASH
    CMAKE_ARGS
      "-DTHEROCK_MODULE_BACKEND=${_vendor}"
      "-DTHEROCK_MODULE_TARGET=${_compiler_target}"
      "-DTHEROCK_MODULE_SDK_ROOT=${_sdk}"
      "-DTHEROCK_MODULE_COMPILER=${_compiler}"
      "-DTHEROCK_MODULE_INSTALL_SUBDIR=${_key}"
      "-DTHEROCK_MODULE_DEVICE_INDEX=${THEROCK_MULTI_VENDOR_DEVICE_INDEX}"
      "-DBUILD_TESTING=ON"
      ${_sdk_args}
  )
  foreach(_source IN LISTS _native_sources)
    target_sources("${_native_project}" PRIVATE
      "${THEROCK_SOURCE_DIR}/tests/multi_vendor/modules/${_source}")
  endforeach()
  therock_cmake_subproject_activate("${_native_project}")
  # Imported SDK/compiler content identities are not yet captured.
  set_property(TARGET "${_native_project}" PROPERTY THEROCK_FPRINT "")
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
  NO_MERGE_COMPILE_COMMANDS
  CMAKE_ARGS
    "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
    "-DTHEROCK_MODULE_TOOLS_DIR=${THEROCK_SOURCE_DIR}/build_tools"
    "-DTHEROCK_MODULE_KPACK_PYTHON_DIR=${_kpack_python}"
    "-DTHEROCK_MODULE_TARGETS_FILE=${CMAKE_CURRENT_BINARY_DIR}/gpu_targets.json"
    "-DTHEROCK_MODULE_BUILD_ROOT=${_module_build_root}"
  EXTRA_DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/gpu_targets.json"
  BUILD_DEPS ${_native_projects}
)
target_sources(multi-vendor-module-packs PRIVATE
  "${THEROCK_SOURCE_DIR}/build_tools/assemble_multi_vendor_modules.py"
  "${THEROCK_SOURCE_DIR}/build_tools/_therock_utils/payload_catalog.py"
  "${THEROCK_SOURCE_DIR}/build_tools/_therock_utils/gpu_targets.py"
  "${_kpack_python}/rocm_kpack/kpack.py"
  "${_kpack_python}/rocm_kpack/compression.py"
)
therock_cmake_subproject_activate(multi-vendor-module-packs)
set_property(TARGET multi-vendor-module-packs PROPERTY THEROCK_FPRINT "")
add_dependencies(therock-build-tests multi-vendor-module-packs)
string(APPEND _module_descriptor
  "[components.test.\"experimental/multi-vendor/module-packs/stage\"]\n"
  "include = [\"share/therock/packs/**\"]\n"
)
set(_descriptor_path "${CMAKE_CURRENT_BINARY_DIR}/artifact-modules.toml")
file(WRITE "${_descriptor_path}" "${_module_descriptor}")
list(JOIN _native_keys "-" _native_bundle)
therock_provide_artifact(multi-vendor-modules
  DESCRIPTOR "${_descriptor_path}"
  DISTRIBUTION multi-vendor-modules
  DIST_BUNDLE_NAME "${_native_bundle}"
  COMPONENTS test
  SUBPROJECT_DEPS ${_native_projects} multi-vendor-module-packs
)

set(_module_dist "${THEROCK_BINARY_DIR}/dist/multi-vendor-modules")
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
  foreach(_module saxpy relu)
    foreach(_format IN LISTS _formats)
      set(_test "packed-module-${_key}-${_module}-${_format}")
      add_test(NAME "${_test}"
        COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_kpack_python}"
          "${Python3_EXECUTABLE}" "${THEROCK_SOURCE_DIR}/build_tools/validate_multi_vendor_modules.py"
          --catalog "${_module_dist}/share/therock/packs/saxpy/catalog.json"
          --catalog "${_module_dist}/share/therock/packs/relu/catalog.json"
          --module "validation/${_module}"
          --target "${THEROCK_MULTI_VENDOR_${_key}_ID}"
          --format "${_format}"
          --entry-point "therock_module_${_module}"
          --runner "${_module_dist}/bin/${_key}/therock_module_validation"
          --device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}"
          ${_identity_args}
      )
      set_tests_properties("${_test}" PROPERTIES
        LABELS "multi-vendor;packed-module;gpu;${_vendor}"
        RUN_SERIAL TRUE TIMEOUT 180)
    endforeach()
  endforeach()
endforeach()
