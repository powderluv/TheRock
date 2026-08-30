# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# therock_explicit_finders.cmake
# Overrides find_library / find_path within a sub-project such that it will only
# find explicit things from the super-project. Unlike find_package, CMake does
# not have a resolver protocol for overriding these, so we rely on the dodgy
# behavior of defining a function with the same name and signature.
#
# In general, we discourage naked use of find_library and find_path, but this
# is available to include in pre hooks of projects that have not been upgraded
# in order to at least make these features safe.
#
# Because this is such a deprecated feature, we just have a static list of
# libraries and paths that we know are part of the super-project. If any of
# these are found, we use our dedicated logic. Otherwise, fall back.
#
# Since the way this works is not perfect, we make it available opt-in. Include
# it for a sub-project that finds ROCm libraries in this way via:
#       CMAKE_INCLUDES
#         therock_explicit_finders.cmake

# Library names that MUST come from the super-project; fatal if not found in
# CMAKE_LIBRARY_PATH.
set(THEROCK_SUPER_PROJECT_FIND_LIBRARY_NAMES
  hsa-amd-aqlprofile64
  hsa-amd-aqlprofile
)

# Library names that PREFER to come from the super-project but gracefully return
# NOTFOUND when not present in CMAKE_LIBRARY_PATH (e.g. component disabled on
# this platform).
set(THEROCK_SUPER_PROJECT_FIND_LIBRARY_NAMES_OPTIONAL
  roctx64
)

# Paths that MUST come from the super-project; fatal if not found in
# THEROCK_SUPERPROJECT_INCLUDE_DIRS.
set(THEROCK_SUPER_PROJECT_FIND_PATHS
  "half/half.hpp"
)

# Paths that PREFER to come from the super-project but gracefully return
# NOTFOUND (rather than fatal-error) when not present in the scoped include
# dirs.  This is appropriate for optional components (e.g. roctracer/roctx)
# that are disabled on some platforms: the consuming library already has a
# graceful "not found" fallback, so we should not override that with a fatal.
set(THEROCK_SUPER_PROJECT_FIND_PATHS_OPTIONAL
  "roctracer/roctx.h"
)

message(STATUS "Including TheRock explicit overrides for find_library and find_path")
message(STATUS "  Libraries that must resolve from super-project: ${THEROCK_SUPER_PROJECT_FIND_LIBRARY_NAMES}")
message(STATUS "  Libraries preferred from super-project (optional): ${THEROCK_SUPER_PROJECT_FIND_LIBRARY_NAMES_OPTIONAL}")
message(STATUS "  Paths that must resolve from super-project: ${THEROCK_SUPER_PROJECT_FIND_PATHS}")
message(STATUS "  Paths preferred from super-project (optional): ${THEROCK_SUPER_PROJECT_FIND_PATHS_OPTIONAL}")

# Override CMake built-in find_library.
function(find_library var name)
  if("${name}" STREQUAL "NAMES")
    # Parse NAMES form.
    cmake_parse_arguments(PARSE_ARGV 1 ARG
      ""
      ""
      "NAMES"
    )
  else()
    # Parse single name form with optional paths.
    set(ARG_NAMES "${name}")
    set(ARG_UNPARSED_ARGUMENTS "PATHS" ${ARGN})
  endif()

  set(_is_superproject FALSE)
  set(_is_superproject_optional FALSE)
  foreach(_name ${ARG_NAMES})
    if("${_name}" IN_LIST THEROCK_SUPER_PROJECT_FIND_LIBRARY_NAMES)
      # Resolve from super-project (required; fatal if not found).
      set(_is_superproject TRUE)
      _find_library(_found_library NAMES "${_name}"
        NO_CACHE NO_DEFAULT_PATH NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH NO_CMAKE_INSTALL_PREFIX
        PATHS ${CMAKE_LIBRARY_PATH}
      )
      if(_found_library)
        message(STATUS "Resolving super-project find_library(${var} NAMES ${ARG_NAMES}): ${_found_library}")
        set("${var}" "${_found_library}" PARENT_SCOPE)
        return()
      endif()
    elseif("${_name}" IN_LIST THEROCK_SUPER_PROJECT_FIND_LIBRARY_NAMES_OPTIONAL)
      # Resolve from super-project preferably, but fall through gracefully when
      # the library is not available (e.g. the component is disabled on this
      # platform).
      set(_is_superproject_optional TRUE)
      _find_library(_found_library NAMES "${_name}"
        NO_CACHE NO_DEFAULT_PATH NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH NO_CMAKE_INSTALL_PREFIX
        PATHS ${CMAKE_LIBRARY_PATH}
      )
      if(_found_library)
        message(STATUS "Resolving super-project find_library(${var} NAMES ${ARG_NAMES}): ${_found_library}")
        set("${var}" "${_found_library}" PARENT_SCOPE)
        return()
      endif()
    endif()
  endforeach()
  if(_is_superproject)
    message(FATAL_ERROR "Could not find super-project find_library(NAMES ${ARG_NAMES}) in ${CMAKE_LIBRARY_PATH}")
  endif()
  if(_is_superproject_optional)
    # Not found in super-project lib dirs (component likely disabled on this
    # platform); return NOTFOUND so the consumer can apply its own fallback.
    message(STATUS "Optional super-project find_library(${var} NAMES ${ARG_NAMES}): not found (component disabled)")
    set("${var}" "${var}-NOTFOUND" PARENT_SCOPE)
    return()
  endif()

  # System fallback.
  # Note that in the system resolution case, the native version only sets a cache variable,
  # relying on scope fallback for a local. Some things absolutely depend on this, so we
  # preserve it here.
  _find_library("${var}" NAMES ${ARG_NAMES} ${ARG_UNPARSED_ARGUMENTS})
  message(STATUS "Resolving system find_library(${var} NAMES ${ARG_NAMES} ${ARG_UNPARSED_ARGUMENTS}): ${${var}}")
endfunction()


# Override CMake built-in find_path
function(find_path var name)
  if("${name}" STREQUAL "NAMES")
    # Parse NAMES form.
    cmake_parse_arguments(PARSE_ARGV 1 ARG
      ""
      ""
      "NAMES"
    )
  else()
    # Parse single name form with optional paths.
    set(ARG_NAMES "${name}")
    set(ARG_UNPARSED_ARGUMENTS "PATHS" ${ARGN})
  endif()

  set(_is_superproject FALSE)
  set(_is_superproject_optional FALSE)
  foreach(_name ${ARG_NAMES})
    if("${_name}" IN_LIST THEROCK_SUPER_PROJECT_FIND_PATHS)
      # Resolve from super-project (required; fatal if not found).
      set(_is_superproject TRUE)
      _find_path(_found_path NAMES "${_name}"
        NO_CACHE NO_DEFAULT_PATH NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH NO_CMAKE_INSTALL_PREFIX
        # See therock_subproject.cmake where we set this in _init.cmake.
        # We use a dedicated var for this so as to be sure it does not contain
        # system or project specific includes.
        PATHS ${THEROCK_SUPERPROJECT_INCLUDE_DIRS}
      )
      if(_found_path)
        message(STATUS "Resolving super-project find_path(${var} NAMES ${ARG_NAMES}): ${_found_path}")
        set("${var}" "${_found_path}" PARENT_SCOPE)
        return()
      endif()
    elseif("${_name}" IN_LIST THEROCK_SUPER_PROJECT_FIND_PATHS_OPTIONAL)
      # Resolve from super-project preferably, but fall through gracefully when
      # the path is not available (e.g. the component is disabled on this
      # platform).
      set(_is_superproject_optional TRUE)
      _find_path(_found_path NAMES "${_name}"
        NO_CACHE NO_DEFAULT_PATH NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH NO_CMAKE_INSTALL_PREFIX
        PATHS ${THEROCK_SUPERPROJECT_INCLUDE_DIRS}
      )
      if(_found_path)
        message(STATUS "Resolving super-project find_path(${var} NAMES ${ARG_NAMES}): ${_found_path}")
        set("${var}" "${_found_path}" PARENT_SCOPE)
        return()
      endif()
    endif()
  endforeach()
  if(_is_superproject)
    message(FATAL_ERROR "Could not find super-project find_path(NAMES ${ARG_NAMES}) in ${THEROCK_SUPERPROJECT_INCLUDE_DIRS}")
  endif()
  if(_is_superproject_optional)
    # Not found in super-project include dirs (component likely disabled on this
    # platform); return NOTFOUND so the consumer can apply its own fallback.
    message(STATUS "Optional super-project find_path(${var} NAMES ${ARG_NAMES}): not found (component disabled)")
    set("${var}" "${var}-NOTFOUND" PARENT_SCOPE)
    return()
  endif()

  # System fallback.
  # Note that in the system resolution case, the native version only sets a cache variable,
  # relying on scope fallback for a local. Some things absolutely depend on this, so we
  # preserve it here.
  _find_path("${var}" NAMES ${ARG_NAMES} ${ARG_UNPARSED_ARGUMENTS})
  message(STATUS "Resolving system find_path(${var} NAMES ${ARG_NAMES} ${ARG_UNPARSED_ARGUMENTS}): ${${var}}")
endfunction()
