# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT WIN32)
  # Configure roctracer if on a supported operating system (Linux).
  # RCCL has deprecated dependencies on roctracer. We apply a patch to redirect
  # naked linking against `-lroctx64` to an explicitly found version of the library.
  # See: https://github.com/ROCm/TheRock/issues/364
  #
  # Note: roctracer is optional (disabled on some platforms, e.g. darwin) and
  # controlled by THEROCK_FLAG_INCLUDE_PROFILER.  Only wire up the patch when
  # the library is actually available in the super-project's library path so
  # that a missing roctx64 does not produce a fatal configure error.
  list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
  include(therock_subproject_utils)
  find_library(_therock_legacy_roctx64 roctx64)
  if(_therock_legacy_roctx64)
    cmake_language(DEFER CALL therock_patch_linked_lib OLD_LIBRARY "roctx64" NEW_TARGET "${_therock_legacy_roctx64}")
  endif()
endif()

# Enable BUILD_ADDRESS_SANITIZER when THEROCK_SANITIZER is ASAN
# This enables RCCL's LTO optimization bypass for faster ASAN link times
if(THEROCK_SANITIZER STREQUAL "ASAN")
  set(BUILD_ADDRESS_SANITIZER ON)
  message(STATUS "Enabling BUILD_ADDRESS_SANITIZER for RCCL
(THEROCK_SANITIZER=${THEROCK_SANITIZER})")
endif()
