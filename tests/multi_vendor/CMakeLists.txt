# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

# This is a standalone SDK consumer, configured once per backend. Compiler and
# architecture selection belongs to the caller, not to the kernel source.
set(THEROCK_MULTI_VENDOR_BACKEND "" CACHE STRING "HIP backend: amd, nvidia, or intel")
set_property(CACHE THEROCK_MULTI_VENDOR_BACKEND PROPERTY STRINGS amd nvidia intel)
if(THEROCK_MULTI_VENDOR_BACKEND STREQUAL "amd")
  set(_hip_platform amd)
elseif(THEROCK_MULTI_VENDOR_BACKEND STREQUAL "nvidia")
  if(CMAKE_VERSION VERSION_LESS 3.28)
    message(FATAL_ERROR "The NVIDIA HIP language requires CMake 3.28 or newer")
  endif()
  set(_hip_platform nvidia)
elseif(THEROCK_MULTI_VENDOR_BACKEND STREQUAL "intel")
  if(CMAKE_VERSION VERSION_LESS 4.3)
    message(FATAL_ERROR "The chipStar SPIR-V HIP language requires CMake 4.3 or newer")
  endif()
  set(_hip_platform spirv)
else()
  message(FATAL_ERROR "Set THEROCK_MULTI_VENDOR_BACKEND to amd, nvidia, or intel")
endif()
if(DEFINED CMAKE_HIP_PLATFORM AND NOT CMAKE_HIP_PLATFORM STREQUAL _hip_platform)
  message(FATAL_ERROR "CMAKE_HIP_PLATFORM conflicts with THEROCK_MULTI_VENDOR_BACKEND")
endif()
set(CMAKE_HIP_PLATFORM "${_hip_platform}" CACHE STRING "HIP platform")

set(THEROCK_MULTI_VENDOR_HIP_ROOT "" CACHE PATH "Installed HIP SDK prefix")
# CMake needs this prefix during HIP compiler discovery, before project().
if(THEROCK_MULTI_VENDOR_HIP_ROOT AND NOT DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT)
  set(CMAKE_HIP_COMPILER_ROCM_ROOT "${THEROCK_MULTI_VENDOR_HIP_ROOT}" CACHE PATH
    "HIP compiler runtime SDK prefix")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/input_guard.cmake")
project(therock_multi_vendor_validation LANGUAGES HIP)
include(CTest)
therock_multi_vendor_initialize_inputs()

set(THEROCK_MULTI_VENDOR_INSTALL_SUBDIR "" CACHE STRING
  "Target slug appended to bin/ so independently compiled backends can coexist")
if(THEROCK_MULTI_VENDOR_INSTALL_SUBDIR AND
    NOT THEROCK_MULTI_VENDOR_INSTALL_SUBDIR MATCHES "^[A-Za-z0-9][A-Za-z0-9-]*$")
  message(FATAL_ERROR "THEROCK_MULTI_VENDOR_INSTALL_SUBDIR must be a target slug")
endif()

set(THEROCK_MULTI_VENDOR_DEVICE_INDEX "0" CACHE STRING "HIP runtime device index")
set(THEROCK_MULTI_VENDOR_EXPECT_VENDOR "${THEROCK_MULTI_VENDOR_BACKEND}" CACHE STRING
  "Required GPU vendor reported by the validation executable")
set(THEROCK_MULTI_VENDOR_EXPECT_ARCH "" CACHE STRING
  "Required runtime ISA: gfx name, sm name, or spirv (not Intel hardware identity)")
set(THEROCK_MULTI_VENDOR_EXPECT_DEVICE_NAME "" CACHE STRING
  "Required case-insensitive substring of the runtime device name")

add_executable(therock_multi_vendor_validation validation.hip)
therock_multi_vendor_guard_consumer(therock_multi_vendor_validation)
set_target_properties(therock_multi_vendor_validation PROPERTIES
  HIP_STANDARD 17
  HIP_STANDARD_REQUIRED YES
  HIP_EXTENSIONS NO
)
string(TOUPPER "${THEROCK_MULTI_VENDOR_BACKEND}" _backend_upper)
target_compile_definitions(therock_multi_vendor_validation PRIVATE
  "THEROCK_MULTI_VENDOR_${_backend_upper}=1"
  "THEROCK_MULTI_VENDOR_BACKEND_NAME=\"${THEROCK_MULTI_VENDOR_BACKEND}\""
)

# The NVIDIA HIP implementation is a header adapter over the CUDA runtime.
# CMake selects/links CUDA, while this consumer needs the installed HIP headers.
if(THEROCK_MULTI_VENDOR_BACKEND STREQUAL "nvidia")
  find_path(_hip_include_dir hip/hip_runtime.h
    HINTS "${THEROCK_MULTI_VENDOR_HIP_ROOT}" "${CMAKE_HIP_COMPILER_ROCM_ROOT}"
    PATH_SUFFIXES include
    REQUIRED
  )
  target_include_directories(therock_multi_vendor_validation PRIVATE "${_hip_include_dir}")
  target_compile_definitions(therock_multi_vendor_validation PRIVATE __HIP_PLATFORM_NVIDIA__=1)
elseif(THEROCK_MULTI_VENDOR_HIP_ROOT)
  target_include_directories(therock_multi_vendor_validation PRIVATE
    "${THEROCK_MULTI_VENDOR_HIP_ROOT}/include")
endif()

if(BUILD_TESTING)
  set(_test_args --device "${THEROCK_MULTI_VENDOR_DEVICE_INDEX}")
  foreach(_field VENDOR ARCH DEVICE_NAME)
    if(NOT THEROCK_MULTI_VENDOR_EXPECT_${_field} STREQUAL "")
      string(TOLOWER "${_field}" _arg)
      string(REPLACE "_" "-" _arg "${_arg}")
      list(APPEND _test_args "--expect-${_arg}" "${THEROCK_MULTI_VENDOR_EXPECT_${_field}}")
    endif()
  endforeach()
  foreach(_case core streams reduction)
    set(_test "multi-vendor-${THEROCK_MULTI_VENDOR_BACKEND}-${_case}")
    add_test(NAME "${_test}" COMMAND therock_multi_vendor_validation --case "${_case}" ${_test_args})
    therock_multi_vendor_guard_test("${_test}")
    set_tests_properties("${_test}" PROPERTIES
      LABELS "multi-vendor;gpu;${THEROCK_MULTI_VENDOR_BACKEND}"
      TIMEOUT 180
      RUN_SERIAL TRUE
    )
    if(THEROCK_MULTI_VENDOR_BACKEND STREQUAL "intel")
      # Reject an OpenCL CPU fallback: Intel qualification is a Level Zero GPU run.
      set_tests_properties("${_test}" PROPERTIES
        ENVIRONMENT "CHIP_BE=level0;CHIP_DEVICE_TYPE=gpu")
    endif()
  endforeach()
endif()

therock_multi_vendor_prepare_install(
  RECEIPT "${CMAKE_CURRENT_BINARY_DIR}/input-build-receipt.json"
  FILES
  "bin/${THEROCK_MULTI_VENDOR_INSTALL_SUBDIR}/therock_multi_vendor_validation${CMAKE_EXECUTABLE_SUFFIX}"
  "bin/${THEROCK_MULTI_VENDOR_INSTALL_SUBDIR}/input-build-receipt.json")
install(TARGETS therock_multi_vendor_validation
  RUNTIME DESTINATION "bin/${THEROCK_MULTI_VENDOR_INSTALL_SUBDIR}")

therock_multi_vendor_record_inputs(therock_validation_input_receipt
  DEPENDS therock_multi_vendor_validation
  INSTALL_DESTINATION "bin/${THEROCK_MULTI_VENDOR_INSTALL_SUBDIR}")
