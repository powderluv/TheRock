# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(Python3_VERSION VERSION_LESS 3.10)
  message(FATAL_ERROR "The multi-vendor-hip profile requires Python 3.10 or newer")
endif()

set(THEROCK_MULTI_VENDOR_TARGETS "" CACHE STRING
  "Canonical vendor:backend:processor target IDs, separated by semicolons")
set(THEROCK_MULTI_VENDOR_AMD_ROOT "/opt/rocm" CACHE PATH "Installed AMD ROCm SDK prefix")
set(THEROCK_MULTI_VENDOR_CUDA_ROOT "/usr/local/cuda" CACHE PATH "Installed NVIDIA CUDA toolkit prefix")
set(THEROCK_MULTI_VENDOR_INTEL_ROOT "" CACHE PATH "Installed chipStar SDK prefix")
set(THEROCK_MULTI_VENDOR_AMD_COMPILER "" CACHE FILEPATH "AMD HIP clang++ compiler (empty: discover in SDK)")
set(THEROCK_MULTI_VENDOR_NVIDIA_COMPILER "" CACHE FILEPATH "NVIDIA nvcc compiler (empty: discover in CUDA toolkit)")
set(THEROCK_MULTI_VENDOR_INTEL_COMPILER "" CACHE FILEPATH "chipStar-compatible clang++ compiler (empty: discover in SDK)")
set(THEROCK_MULTI_VENDOR_DEVICE_INDEX "0" CACHE STRING "Backend device index used by validation")
set(THEROCK_MULTI_VENDOR_LEVEL_ZERO_ROOT "" CACHE PATH "Installed Level Zero headers and loader prefix for native modules")
set(THEROCK_MULTI_VENDOR_SPIRV_COMPILER "" CACHE FILEPATH "Clang SPIR64 frontend (empty: discover in AMD SDK)")
set(THEROCK_MULTI_VENDOR_SPIRV_TRANSLATOR "" CACHE FILEPATH "Matching LLVM-to-SPIR-V translator (empty: discover beside frontend)")
set(THEROCK_MULTI_VENDOR_SPIRV_VALIDATOR "" CACHE FILEPATH "SPIRV-Tools spirv-val (empty: discover in Level Zero SDK or PATH)")
set(THEROCK_MULTI_VENDOR_INTEL_DEVICE_ID "" CACHE STRING "Expected Intel PCI device ID for native modules (B70 defaults to 0xe223)")

if(NOT THEROCK_MULTI_VENDOR_TARGETS)
  message(FATAL_ERROR "The multi-vendor-hip profile requires THEROCK_MULTI_VENDOR_TARGETS")
endif()
if(THEROCK_AMDGPU_TARGETS OR THEROCK_AMDGPU_FAMILIES)
  message(FATAL_ERROR "Use THEROCK_MULTI_VENDOR_TARGETS for the multi-vendor-hip profile")
endif()
if(THEROCK_FLAG_KPACK_SPLIT_ARTIFACTS)
  message(FATAL_ERROR "The multi-vendor-hip profile requires THEROCK_FLAG_KPACK_SPLIT_ARTIFACTS=OFF")
endif()
if(THEROCK_SANITIZER)
  message(FATAL_ERROR "The multi-vendor-hip profile does not yet qualify sanitizer builds")
endif()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${THEROCK_SOURCE_DIR}/build_tools/configure_multi_vendor.py"
  "${THEROCK_SOURCE_DIR}/build_tools/_therock_utils/gpu_targets.py"
)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${THEROCK_SOURCE_DIR}/build_tools/configure_multi_vendor.py"
    --targets ${THEROCK_MULTI_VENDOR_TARGETS}
    --output-dir "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND_ERROR_IS_FATAL ANY
)
include("${CMAKE_CURRENT_BINARY_DIR}/targets.cmake")
message(STATUS "Experimental multi-vendor target selection: ${THEROCK_MULTI_VENDOR_TARGETS}")
message(STATUS "gpu_targets.json records requested targets, not build or hardware qualification")

function(_therock_multi_vendor_compiler output_var vendor sdk_root compiler_name)
  string(TOUPPER "${vendor}" _upper_vendor)
  set(_cache_var "THEROCK_MULTI_VENDOR_${_upper_vendor}_COMPILER")
  if(${_cache_var})
    if(NOT EXISTS "${${_cache_var}}")
      message(FATAL_ERROR "${_cache_var} does not exist: ${${_cache_var}}")
    endif()
    set(_compiler "${${_cache_var}}")
  else()
    find_program(_discovered_compiler NAMES "${compiler_name}"
      PATHS "${sdk_root}/bin" "${sdk_root}/lib/llvm/bin" "${sdk_root}/llvm/bin"
      NO_DEFAULT_PATH NO_CACHE
    )
    if(NOT _discovered_compiler)
      message(FATAL_ERROR "Set ${_cache_var} to an installed ${vendor} HIP compiler")
    endif()
    set(_compiler "${_discovered_compiler}")
  endif()
  set(${output_var} "${_compiler}" PARENT_SCOPE)
endfunction()

set(_needs_nvidia OFF)
foreach(_key IN LISTS THEROCK_MULTI_VENDOR_TARGET_KEYS)
  set(_vendor "${THEROCK_MULTI_VENDOR_${_key}_VENDOR}")
  if(_vendor STREQUAL "nvidia")
    if(CMAKE_VERSION VERSION_LESS 3.28)
      message(FATAL_ERROR "NVIDIA HIP requires CMake 3.28 or newer")
    endif()
    set(_needs_nvidia ON)
  elseif(_vendor STREQUAL "intel" AND THEROCK_BUILD_TESTING
      AND THEROCK_ENABLE_MULTI_VENDOR_VALIDATION AND CMAKE_VERSION VERSION_LESS 4.3)
    message(FATAL_ERROR "Intel chipStar SPIR-V HIP requires CMake 4.3 or newer")
  endif()
endforeach()

if(_needs_nvidia AND NOT THEROCK_ENABLE_HIP_NVIDIA_SDK)
  message(FATAL_ERROR "Selected NVIDIA targets require THEROCK_ENABLE_HIP_NVIDIA_SDK=ON")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/input_provenance.cmake")

if(_needs_nvidia)
  # CLR already supports installing the HIP common and HIPNV headers without
  # compiling its AMD runtime. CUDA remains an externally installed dependency.
  therock_cmake_subproject_declare(hip-nvidia-sdk
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/clr"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hip-nvidia-sdk"
    DISABLE_AMDGPU_TARGETS
    BUILD_GUARDS multi-vendor-verify-inputs
    NO_MERGE_COMPILE_COMMANDS
    CMAKE_ARGS
      "-DCLR_BUILD_HIP=ON"
      "-DCLR_BUILD_OCL=OFF"
      "-DHIP_PLATFORM=nvidia"
      "-DHIP_COMMON_DIR=${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/hip"
      "-DHIPNV_DIR=${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/hipother/hipnv"
      "-DHIPCC_BIN_DIR=${CMAKE_CURRENT_BINARY_DIR}/absent-hipcc"
      "-DCMAKE_DISABLE_FIND_PACKAGE_Git=ON"
      "-DDOXYGEN_EXE:FILEPATH="
      "-D__HIP_ENABLE_PCH=OFF"
      "-DBUILD_TESTING=OFF"
  )
  # These files are copied/configured by the header-only SDK. Register them
  # with the outer graph so an edit rebuilds/restages the SDK and its consumers.
  file(GLOB_RECURSE _hip_nvidia_sdk_sources CONFIGURE_DEPENDS LIST_DIRECTORIES FALSE
    "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/clr/CMakeLists.txt"
    "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/clr/hipamd/CMakeLists.txt"
    "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/clr/hipamd/*.cmake"
    "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/clr/hipamd/*.in"
    "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/clr/hipamd/include/*"
    "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/hip/include/*"
    "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/hip/cmake/*"
    "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/hip/VERSION"
    "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/hipother/hipnv/include/*"
  )
  target_sources(hip-nvidia-sdk PRIVATE ${_hip_nvidia_sdk_sources})
  therock_cmake_subproject_provide_package(hip-nvidia-sdk hip lib/cmake/hip)
  therock_cmake_subproject_activate(hip-nvidia-sdk)
  _therock_multi_vendor_register_inputs(hip-nvidia-sdk)
  therock_provide_artifact(hip-nvidia-sdk
    TARGET_NEUTRAL
    BUILD_GUARDS multi-vendor-verify-inputs
    DESCRIPTOR artifact-hip-nvidia-sdk.toml
    DISTRIBUTION hip-nvidia
    COMPONENTS dev
    SUBPROJECT_DEPS hip-nvidia-sdk
  )
  therock_cmake_subproject_dist_dir(_nvidia_hip_root hip-nvidia-sdk)
endif()

if(THEROCK_BUILD_TESTING AND THEROCK_ENABLE_MULTI_VENDOR_MODULES)
  include("${CMAKE_CURRENT_LIST_DIR}/module_packs.cmake")
endif()

if(NOT THEROCK_BUILD_TESTING OR NOT THEROCK_ENABLE_MULTI_VENDOR_VALIDATION)
  _therock_multi_vendor_finalize_inputs()
  return()
endif()

set(_validation_projects)
# Metadata has one owner in the aggregate artifact, separate from backend bins.
set(_metadata_stage "${CMAKE_CURRENT_BINARY_DIR}/metadata-stage")
file(MAKE_DIRECTORY "${_metadata_stage}/share/therock/multi-vendor")
configure_file("${CMAKE_CURRENT_BINARY_DIR}/gpu_targets.json"
  "${_metadata_stage}/share/therock/multi-vendor/gpu_targets.json" COPYONLY)
file(SHA256 "${CMAKE_CURRENT_BINARY_DIR}/gpu_targets.json" _metadata_sha256)
configure_file("${_therock_input_report}"
  "${_metadata_stage}/share/therock/multi-vendor/hip-input-provenance.json" COPYONLY)
set(_validation_artifact_sections
  "# Target selection SHA256: ${_metadata_sha256}\n"
  "[components.test.\"experimental/multi-vendor/metadata-stage\"]\n"
  "include = [\"share/therock/multi-vendor/gpu_targets.json\", \"share/therock/multi-vendor/hip-input-provenance.json\"]\n\n"
)
# set() forms a CMake list; concatenate the descriptor fragments as plain TOML.
list(JOIN _validation_artifact_sections "" _validation_artifact_sections)
message(STATUS "Validation artifact cache reuse is disabled: imported-input locks do not cover the full build environment")
foreach(_key IN LISTS THEROCK_MULTI_VENDOR_TARGET_KEYS)
  set(_vendor "${THEROCK_MULTI_VENDOR_${_key}_VENDOR}")
  set(_processor "${THEROCK_MULTI_VENDOR_${_key}_PROCESSOR}")
  set(_hip_platform "${THEROCK_MULTI_VENDOR_${_key}_HIP_PLATFORM}")
  set(_hip_arch "${THEROCK_MULTI_VENDOR_${_key}_HIP_ARCHITECTURE}")
  set(_build_deps)
  set(_sdk_args)
  set(_expected_arch "${_processor}")
  if(_vendor STREQUAL "nvidia")
    # CUDA reports a major/minor ISA, not compiler-only feature suffixes such
    # as sm_90a. Keep those suffixes in compilation and target identity only.
    string(REGEX MATCH "^sm_[0-9]+" _expected_arch "${_processor}")
    set(_hip_root "${_nvidia_hip_root}")
    _therock_multi_vendor_compiler(_compiler nvidia "${THEROCK_MULTI_VENDOR_CUDA_ROOT}" nvcc)
    list(APPEND _build_deps hip-nvidia-sdk)
    list(APPEND _sdk_args "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${_hip_root}")
  elseif(_vendor STREQUAL "amd")
    set(_hip_root "${THEROCK_MULTI_VENDOR_AMD_ROOT}")
    _therock_multi_vendor_compiler(_compiler amd "${_hip_root}" clang++)
    list(APPEND _sdk_args "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${_hip_root}")
  elseif(_vendor STREQUAL "intel")
    if(NOT THEROCK_MULTI_VENDOR_INTEL_ROOT)
      message(FATAL_ERROR "Intel targets require THEROCK_MULTI_VENDOR_INTEL_ROOT and a chipStar SDK")
    endif()
    set(_hip_root "${THEROCK_MULTI_VENDOR_INTEL_ROOT}")
    _therock_multi_vendor_compiler(_compiler intel "${_hip_root}" clang++)
    list(APPEND _sdk_args "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${_hip_root}")
    set(_expected_arch spirv)
    if(_processor STREQUAL "xe2-b70")
      list(APPEND _sdk_args "-DTHEROCK_MULTI_VENDOR_EXPECT_DEVICE_NAME=B70")
    endif()
  endif()
  if(NOT _vendor STREQUAL "nvidia")
    if(NOT EXISTS "${_hip_root}/include/hip/hip_runtime.h")
      message(FATAL_ERROR "The ${_vendor} SDK lacks include/hip/hip_runtime.h: ${_hip_root}")
    endif()
    # Imported runtimes stay in their SDKs. Preserve their explicit runpath in
    # the installed validation binaries instead of redistributing their files.
    list(APPEND _sdk_args
      "-DCMAKE_PREFIX_PATH=${_hip_root}"
      "-DCMAKE_INSTALL_RPATH=${_hip_root}/lib"
    )
  endif()
  if(_hip_arch)
    list(APPEND _sdk_args "-DCMAKE_HIP_ARCHITECTURES=${_hip_arch}")
  endif()
  # An Intel product identity never becomes a compiler architecture flag.
  set(_target "hip-validation-${_key}")
  set(_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/validation/${_key}")
  therock_cmake_subproject_declare("${_target}"
    EXTERNAL_SOURCE_DIR "${THEROCK_SOURCE_DIR}/tests/multi_vendor"
    BINARY_DIR "${_binary_dir}"
    DISABLE_AMDGPU_TARGETS
    BUILD_GUARDS multi-vendor-verify-inputs
    EXTRA_DEPENDS "${_therock_input_report}" "${_therock_input_guard}"
    NO_INSTALL_RPATH
    FPRINT_SOURCE_HASH
    CMAKE_ARGS
      "-DBUILD_TESTING=ON"
      "-DTHEROCK_MULTI_VENDOR_INPUT_GUARD=${_therock_input_guard}"
      "-DTHEROCK_MULTI_VENDOR_INPUT_PROVENANCE=${_therock_input_report}"
      "-DTHEROCK_MULTI_VENDOR_BACKEND=${_vendor}"
      "-DCMAKE_HIP_PLATFORM=${_hip_platform}"
      "-DCMAKE_HIP_COMPILER=${_compiler}"
      "-DTHEROCK_MULTI_VENDOR_HIP_ROOT=${_hip_root}"
      "-DTHEROCK_MULTI_VENDOR_INSTALL_SUBDIR=${_key}"
      "-DTHEROCK_MULTI_VENDOR_DEVICE_INDEX=${THEROCK_MULTI_VENDOR_DEVICE_INDEX}"
      "-DTHEROCK_MULTI_VENDOR_EXPECT_VENDOR=${_vendor}"
      "-DTHEROCK_MULTI_VENDOR_EXPECT_ARCH=${_expected_arch}"
      ${_sdk_args}
    BUILD_DEPS ${_build_deps}
  )
  target_sources("${_target}" PRIVATE
    "${THEROCK_SOURCE_DIR}/tests/multi_vendor/validation.hip"
    "${THEROCK_SOURCE_DIR}/tests/multi_vendor/input_guard.cmake")
  therock_cmake_subproject_activate("${_target}")
  _therock_multi_vendor_register_inputs("${_target}")
  list(APPEND _validation_projects "${_target}")
  add_dependencies(therock-build-tests "${_target}")
  add_test(NAME "multi-vendor-hip-receipt-${_key}"
    COMMAND "${CMAKE_COMMAND}"
      "-DTHEROCK_MULTI_VENDOR_CHECK_PROVENANCE=${_therock_input_report}"
      "-DTHEROCK_MULTI_VENDOR_CHECK_RECEIPTS=${_binary_dir}/stage/bin/${_key}/input-build-receipt.json"
      -P "${THEROCK_SOURCE_DIR}/tests/multi_vendor/input_guard.cmake")
  set_tests_properties("multi-vendor-hip-receipt-${_key}" PROPERTIES
    FIXTURES_SETUP "multi-vendor-hip-receipt-${_key}" FIXTURES_REQUIRED multi-vendor-inputs
    LABELS "multi-vendor;provenance")
  add_test(NAME "multi-vendor-${_key}"
    COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${_binary_dir}/build"
      --output-on-failure --no-tests=error -C "$<CONFIG>"
  )
  set_tests_properties("multi-vendor-${_key}" PROPERTIES
    LABELS "multi-vendor;gpu;${_vendor}"
    FIXTURES_REQUIRED "multi-vendor-inputs;multi-vendor-hip-receipt-${_key}"
    RUN_SERIAL TRUE
    TIMEOUT 600
  )
  string(APPEND _validation_artifact_sections
    "[components.test.\"experimental/multi-vendor/validation/${_key}/stage\"]\n"
    "include = [\"bin/${_key}/**\"]\n\n"
  )
endforeach()

configure_file(artifact-multi-vendor-validation.toml.in
  "${CMAKE_CURRENT_BINARY_DIR}/artifact-multi-vendor-validation.toml" @ONLY)
list(JOIN THEROCK_MULTI_VENDOR_TARGET_KEYS "-" _validation_bundle)
therock_provide_artifact(multi-vendor-validation
  BUILD_GUARDS multi-vendor-verify-inputs
  DESCRIPTOR "${CMAKE_CURRENT_BINARY_DIR}/artifact-multi-vendor-validation.toml"
  DISTRIBUTION multi-vendor-validation
  DIST_BUNDLE_NAME "${_validation_bundle}"
  COMPONENTS test
  SUBPROJECT_DEPS ${_validation_projects}
)

_therock_multi_vendor_finalize_inputs()
