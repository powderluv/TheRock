# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(therock_native_module_validation LANGUAGES CXX)
include(CTest)

set(THEROCK_MODULE_BACKEND "" CACHE STRING "Native module backend: amd, nvidia, or intel")
set_property(CACHE THEROCK_MODULE_BACKEND PROPERTY STRINGS amd nvidia intel)
set(THEROCK_MODULE_TARGET "" CACHE STRING "Compiler architecture or Intel product identity, e.g. gfx1201, 120, xe2-b70")
set(THEROCK_MODULE_SDK_ROOT "" CACHE PATH "Installed ROCm, CUDA, or Level Zero SDK prefix")
set(THEROCK_MODULE_COMPILER "" CACHE FILEPATH "Device compiler: clang++ for AMD, nvcc for NVIDIA, clang for Intel SPIR-V")
set(THEROCK_MODULE_INSTALL_SUBDIR "" CACHE STRING "Target slug for installed binaries and payloads")
set(THEROCK_MODULE_DEVICE_INDEX "0" CACHE STRING "Device index for native module tests")
set(THEROCK_MODULE_SPIRV_TRANSLATOR "" CACHE FILEPATH "LLVM-to-SPIR-V translator matching the Intel frontend LLVM version")
set(THEROCK_MODULE_SPIRV_VALIDATOR "" CACHE FILEPATH "SPIRV-Tools spirv-val executable")
set(THEROCK_MODULE_INTEL_DEVICE_ID "" CACHE STRING "Expected Intel PCI device ID (B70 defaults to 0xe223)")
if(NOT THEROCK_MODULE_INSTALL_SUBDIR MATCHES "^[A-Za-z0-9][A-Za-z0-9-]*$")
  message(FATAL_ERROR "Set THEROCK_MODULE_INSTALL_SUBDIR to a nonempty target slug")
endif()
if(NOT THEROCK_MODULE_SDK_ROOT)
  message(FATAL_ERROR "Set THEROCK_MODULE_SDK_ROOT to the installed backend SDK")
endif()

# Imported SDK paths can change between configurations. Internal discoveries
# must follow the selected prefix rather than a prior find_* cache entry.
foreach(_discovered _hip_include _hip_runtime _level_zero_include _level_zero_loader
    _compiler _spirv_translator _spirv_validator)
  unset(${_discovered} CACHE)
  unset(${_discovered})
endforeach()

if(THEROCK_MODULE_BACKEND STREQUAL "intel")
  add_executable(therock_module_validation level_zero_loader.cpp)
else()
  add_executable(therock_module_validation loader.cpp)
endif()
target_compile_features(therock_module_validation PRIVATE cxx_std_17)
set_target_properties(therock_module_validation PROPERTIES CXX_EXTENSIONS NO)

if(THEROCK_MODULE_BACKEND STREQUAL "nvidia")
  if(NOT THEROCK_MODULE_TARGET MATCHES "^[1-9][0-9]+[a-z]?$")
    message(FATAL_ERROR "NVIDIA THEROCK_MODULE_TARGET must be a CUDA architecture such as 120")
  endif()
  set(CUDAToolkit_ROOT "${THEROCK_MODULE_SDK_ROOT}")
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(therock_module_validation PRIVATE THEROCK_MODULE_NVIDIA=1)
  target_link_libraries(therock_module_validation PRIVATE CUDA::cuda_driver)
  set(_compiler_name nvcc)
  set(_formats cubin ptx)
  string(REGEX MATCH "^[0-9]+" _runtime_arch "${THEROCK_MODULE_TARGET}")
  string(PREPEND _runtime_arch "sm_")
elseif(THEROCK_MODULE_BACKEND STREQUAL "amd")
  if(NOT THEROCK_MODULE_TARGET MATCHES "^gfx[0-9a-f]+(:sramecc[+-]|:xnack[+-])*$")
    message(FATAL_ERROR "AMD THEROCK_MODULE_TARGET must be a gfx target ID")
  endif()
  find_path(_hip_include hip/hip_runtime_api.h
    PATHS "${THEROCK_MODULE_SDK_ROOT}/include" NO_DEFAULT_PATH NO_CACHE REQUIRED)
  find_library(_hip_runtime amdhip64
    PATHS "${THEROCK_MODULE_SDK_ROOT}/lib" "${THEROCK_MODULE_SDK_ROOT}/lib64"
    NO_DEFAULT_PATH NO_CACHE REQUIRED)
  target_include_directories(therock_module_validation PRIVATE "${_hip_include}")
  target_compile_definitions(therock_module_validation PRIVATE
    THEROCK_MODULE_AMD=1 __HIP_PLATFORM_AMD__=1)
  target_link_libraries(therock_module_validation PRIVATE "${_hip_runtime}")
  get_filename_component(_hip_library_dir "${_hip_runtime}" DIRECTORY)
  set_target_properties(therock_module_validation PROPERTIES INSTALL_RPATH "${_hip_library_dir}")
  set(_compiler_name clang++)
  set(_formats hsaco)
  string(REGEX MATCH "^gfx[0-9a-f]+" _runtime_arch "${THEROCK_MODULE_TARGET}")
elseif(THEROCK_MODULE_BACKEND STREQUAL "intel")
  if(NOT THEROCK_MODULE_TARGET MATCHES "^[a-z0-9]+(-[a-z0-9]+)*$")
    message(FATAL_ERROR "Intel THEROCK_MODULE_TARGET must be a product identity such as xe2-b70")
  endif()
  if(THEROCK_MODULE_TARGET STREQUAL "xe2-b70" AND THEROCK_MODULE_INTEL_DEVICE_ID STREQUAL "")
    set(THEROCK_MODULE_INTEL_DEVICE_ID "0xe223")
  endif()
  if(NOT THEROCK_MODULE_INTEL_DEVICE_ID MATCHES "^(0[xX][0-9A-Fa-f]+|[0-9]+)$")
    message(FATAL_ERROR "Set THEROCK_MODULE_INTEL_DEVICE_ID to the expected Intel PCI device ID")
  endif()
  # Bound the input before CMake's signed integer arithmetic can overflow.
  string(REGEX REPLACE "^0[xX]" "" _device_digits "${THEROCK_MODULE_INTEL_DEVICE_ID}")
  string(REGEX REPLACE "^0+" "" _device_digits "${_device_digits}")
  string(LENGTH "${_device_digits}" _device_digits_length)
  if(_device_digits_length GREATER 5)
    message(FATAL_ERROR "THEROCK_MODULE_INTEL_DEVICE_ID must be a nonzero 16-bit ID")
  endif()
  math(EXPR _intel_device_id "${THEROCK_MODULE_INTEL_DEVICE_ID}")
  if(_intel_device_id LESS 1 OR _intel_device_id GREATER 65535)
    message(FATAL_ERROR "THEROCK_MODULE_INTEL_DEVICE_ID must be a nonzero 16-bit ID")
  endif()
  if(THEROCK_MODULE_TARGET STREQUAL "xe2-b70" AND NOT _intel_device_id EQUAL 57891)
    message(FATAL_ERROR "The xe2-b70 target requires Intel device ID 0xe223")
  endif()
  find_path(_level_zero_include level_zero/ze_api.h
    PATHS "${THEROCK_MODULE_SDK_ROOT}/include" NO_DEFAULT_PATH NO_CACHE REQUIRED)
  find_library(_level_zero_loader ze_loader
    PATHS "${THEROCK_MODULE_SDK_ROOT}/lib" "${THEROCK_MODULE_SDK_ROOT}/lib64"
      "${THEROCK_MODULE_SDK_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
    NO_DEFAULT_PATH NO_CACHE REQUIRED)
  target_include_directories(therock_module_validation PRIVATE "${_level_zero_include}")
  target_link_libraries(therock_module_validation PRIVATE "${_level_zero_loader}")
  get_filename_component(_level_zero_library_dir "${_level_zero_loader}" DIRECTORY)
  set_target_properties(therock_module_validation PROPERTIES
    INSTALL_RPATH "${_level_zero_library_dir}")
  set(_compiler_name clang)
  set(_formats spirv)
  set(_runtime_arch spirv)
  set(_identity_args --expect-device-id "${THEROCK_MODULE_INTEL_DEVICE_ID}")
else()
  message(FATAL_ERROR "THEROCK_MODULE_BACKEND must be amd, nvidia, or intel")
endif()

if(THEROCK_MODULE_COMPILER)
  if(NOT EXISTS "${THEROCK_MODULE_COMPILER}")
    message(FATAL_ERROR "THEROCK_MODULE_COMPILER does not exist: ${THEROCK_MODULE_COMPILER}")
  endif()
  set(_compiler "${THEROCK_MODULE_COMPILER}")
else()
  find_program(_compiler NAMES "${_compiler_name}"
    PATHS "${THEROCK_MODULE_SDK_ROOT}/bin" "${THEROCK_MODULE_SDK_ROOT}/lib/llvm/bin"
      "${THEROCK_MODULE_SDK_ROOT}/llvm/bin"
    NO_DEFAULT_PATH NO_CACHE REQUIRED)
endif()

if(THEROCK_MODULE_BACKEND STREQUAL "intel")
  get_filename_component(_compiler_dir "${_compiler}" DIRECTORY)
  if(THEROCK_MODULE_SPIRV_TRANSLATOR)
    set(_spirv_translator "${THEROCK_MODULE_SPIRV_TRANSLATOR}")
  else()
    find_program(_spirv_translator NAMES amd-llvm-spirv llvm-spirv
      PATHS "${_compiler_dir}" NO_DEFAULT_PATH NO_CACHE REQUIRED)
  endif()
  if(THEROCK_MODULE_SPIRV_VALIDATOR)
    set(_spirv_validator "${THEROCK_MODULE_SPIRV_VALIDATOR}")
  else()
    find_program(_spirv_validator NAMES spirv-val
      HINTS "${THEROCK_MODULE_SDK_ROOT}/bin" NO_CACHE REQUIRED)
  endif()
  foreach(_tool _spirv_translator _spirv_validator)
    if(NOT EXISTS "${${_tool}}")
      message(FATAL_ERROR "SPIR-V tool does not exist: ${${_tool}}")
    endif()
  endforeach()
endif()

set(_payloads)
foreach(_module saxpy relu)
  set(_relu 0)
  if(_module STREQUAL "relu")
    set(_relu 1)
  endif()
  foreach(_format IN LISTS _formats)
    set(_payload "${CMAKE_CURRENT_BINARY_DIR}/${_module}.${_format}")
    if(THEROCK_MODULE_BACKEND STREQUAL "intel")
      # Portable OpenCL kernel ABI: this product identity is never an ISA flag.
      # The translator must accept bitcode from the configured Clang version.
      set(_bitcode "${CMAKE_CURRENT_BINARY_DIR}/${_module}.bc")
      add_custom_command(OUTPUT "${_payload}"
        COMMAND "${_compiler}" -target spir64-unknown-unknown -x cl -cl-std=CL1.2
          -Xclang -finclude-default-header -O2 -emit-llvm -c
          "-DTHEROCK_MODULE_RELU=${_relu}"
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl" -o "${_bitcode}"
        COMMAND "${_spirv_translator}" --spirv-max-version=1.0
          "${_bitcode}" -o "${_payload}.unvalidated"
        COMMAND "${_spirv_validator}" --target-env opencl1.2 "${_payload}.unvalidated"
        COMMAND "${CMAKE_COMMAND}" -E rename "${_payload}.unvalidated" "${_payload}"
        DEPENDS kernels.cl
        BYPRODUCTS "${_bitcode}"
        COMMENT "Compiling and validating portable SPIR-V ${_module} for ${THEROCK_MODULE_TARGET}"
        VERBATIM
      )
    else()
      if(THEROCK_MODULE_BACKEND STREQUAL "nvidia")
        if(_format STREQUAL "ptx")
          set(_arch "compute_${THEROCK_MODULE_TARGET}")
        else()
          set(_arch "sm_${THEROCK_MODULE_TARGET}")
        endif()
        set(_device_args --${_format} "--gpu-architecture=${_arch}" -x cu)
      else()
        # clang is the compiler, not the hipcc wrapper. --offload-device-only is
        # its driver spelling for producing a linked AMD GPU code object.
        set(_device_args --offload-device-only --no-gpu-bundle-output
          "--offload-arch=${THEROCK_MODULE_TARGET}" -x hip
          "--rocm-path=${THEROCK_MODULE_SDK_ROOT}" "--hip-path=${THEROCK_MODULE_SDK_ROOT}")
      endif()
      add_custom_command(OUTPUT "${_payload}"
        COMMAND "${_compiler}" ${_device_args} -O2 -std=c++17
          "-DTHEROCK_MODULE_RELU=${_relu}"
          "${CMAKE_CURRENT_SOURCE_DIR}/kernels.cpp" -o "${_payload}"
        DEPENDS kernels.cpp
        COMMENT "Compiling native ${_module}.${_format} for ${THEROCK_MODULE_TARGET}"
        VERBATIM
      )
    endif()
    list(APPEND _payloads "${_payload}")
    if(BUILD_TESTING)
      set(_test "module-${THEROCK_MODULE_BACKEND}-${_module}-${_format}")
      add_test(NAME "${_test}" COMMAND therock_module_validation
        --payload "${_payload}" --symbol "therock_module_${_module}"
        --device "${THEROCK_MODULE_DEVICE_INDEX}" --expect-arch "${_runtime_arch}"
        ${_identity_args})
      set_tests_properties("${_test}" PROPERTIES
        LABELS "multi-vendor;native-module;gpu;${THEROCK_MODULE_BACKEND}"
        RUN_SERIAL TRUE TIMEOUT 180)
    endif()
  endforeach()
endforeach()
add_custom_target(therock_module_payloads ALL DEPENDS ${_payloads})
install(TARGETS therock_module_validation
  RUNTIME DESTINATION "bin/${THEROCK_MODULE_INSTALL_SUBDIR}")
install(FILES ${_payloads}
  DESTINATION "share/therock/modules/${THEROCK_MODULE_INSTALL_SUBDIR}")
