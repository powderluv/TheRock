# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Imported inputs are content-locked, but do not cover the full host toolchain
# closure. Keep artifact fingerprints disabled for these experimental consumers.
set(THEROCK_MULTI_VENDOR_INPUT_LOCK "" CACHE FILEPATH
  "Expected imported SDK/tool lock (empty: create an immutable lock in this build)")
set(_therock_input_tool "${THEROCK_SOURCE_DIR}/build_tools/multi_vendor_inputs.py")
set(_therock_input_cache "${CMAKE_CURRENT_BINARY_DIR}/input-digest-cache.json")
set(_therock_input_report "${CMAKE_CURRENT_BINARY_DIR}/input-provenance.json")
set(_therock_input_guard "${CMAKE_CURRENT_BINARY_DIR}/verify-inputs.cmake")
set(_therock_input_lock "${THEROCK_MULTI_VENDOR_INPUT_LOCK}")
set(_therock_input_action verify)
if(_therock_input_lock STREQUAL "")
  set(_therock_input_action capture)
  set(_therock_input_lock "${CMAKE_CURRENT_BINARY_DIR}/inputs.lock.json")
else()
  get_filename_component(_therock_input_lock "${_therock_input_lock}" ABSOLUTE
    BASE_DIR "${THEROCK_SOURCE_DIR}")
endif()
set_property(GLOBAL PROPERTY THEROCK_MULTI_VENDOR_INPUT_ARGS "")
set_property(GLOBAL PROPERTY THEROCK_MULTI_VENDOR_INPUT_NAMES "")
set_property(GLOBAL PROPERTY THEROCK_MULTI_VENDOR_INPUT_PROJECTS "")

function(_therock_multi_vendor_input name kind path)
  if(path STREQUAL "")
    message(FATAL_ERROR "Imported input ${name} requires a nonempty path")
  endif()
  if(path MATCHES ";")
    message(FATAL_ERROR "Imported input paths cannot contain CMake list separators: ${path}")
  endif()
  get_property(_names GLOBAL PROPERTY THEROCK_MULTI_VENDOR_INPUT_NAMES)
  if(name IN_LIST _names)
    get_property(_previous GLOBAL PROPERTY "THEROCK_MULTI_VENDOR_INPUT_${name}")
    if(NOT _previous STREQUAL "${kind};${path}")
      message(FATAL_ERROR "Conflicting imported input ${name}")
    endif()
    return()
  endif()
  set_property(GLOBAL APPEND PROPERTY THEROCK_MULTI_VENDOR_INPUT_NAMES "${name}")
  set_property(GLOBAL PROPERTY "THEROCK_MULTI_VENDOR_INPUT_${name}" "${kind};${path}")
  set_property(GLOBAL APPEND PROPERTY THEROCK_MULTI_VENDOR_INPUT_ARGS --input "${name}" "${kind}" "${path}")
endfunction()

# Record the selected tools as well as SDK trees. Host libraries, environment,
# compiler-discovered system headers and driver state remain outside this scope.
_therock_multi_vendor_input(cmake-tool file "${CMAKE_COMMAND}")
_therock_multi_vendor_input(build-tool file "${CMAKE_MAKE_PROGRAM}")
_therock_multi_vendor_input(python-tool file "${Python3_EXECUTABLE}")
_therock_multi_vendor_input(host-c-tool file "${CMAKE_C_COMPILER}")
_therock_multi_vendor_input(host-cxx-tool file "${CMAKE_CXX_COMPILER}")
if(THEROCK_BUILD_TESTING AND (THEROCK_ENABLE_MULTI_VENDOR_VALIDATION OR THEROCK_ENABLE_MULTI_VENDOR_MODULES))
  foreach(_key IN LISTS THEROCK_MULTI_VENDOR_TARGET_KEYS)
    set(_vendor "${THEROCK_MULTI_VENDOR_${_key}_VENDOR}")
    if(_vendor STREQUAL "amd")
      _therock_multi_vendor_input(amd-sdk tree "${THEROCK_MULTI_VENDOR_AMD_ROOT}")
      _therock_multi_vendor_compiler(_compiler amd "${THEROCK_MULTI_VENDOR_AMD_ROOT}" clang++)
      _therock_multi_vendor_input(amd-compiler file "${_compiler}")
    elseif(_vendor STREQUAL "nvidia")
      _therock_multi_vendor_input(cuda-sdk tree "${THEROCK_MULTI_VENDOR_CUDA_ROOT}")
      _therock_multi_vendor_compiler(_compiler nvidia "${THEROCK_MULTI_VENDOR_CUDA_ROOT}" nvcc)
      _therock_multi_vendor_input(nvidia-compiler file "${_compiler}")
    elseif(_vendor STREQUAL "intel")
      if(THEROCK_ENABLE_MULTI_VENDOR_VALIDATION)
        _therock_multi_vendor_input(chipstar-sdk tree "${THEROCK_MULTI_VENDOR_INTEL_ROOT}")
        _therock_multi_vendor_compiler(_compiler intel "${THEROCK_MULTI_VENDOR_INTEL_ROOT}" clang++)
        _therock_multi_vendor_input(chipstar-compiler file "${_compiler}")
      endif()
      if(THEROCK_ENABLE_MULTI_VENDOR_MODULES)
        if(THEROCK_ENABLE_MULTI_VENDOR_INTEL_SGEMM)
          # Include sibling compiler, oneMKL, TBB, and UR runtime components.
          # Component prefixes alone do not cover the imported SDK closure.
          _therock_multi_vendor_input(oneapi-sdk tree "${THEROCK_MULTI_VENDOR_ONEAPI_ROOT}")
          _therock_multi_vendor_input(sycl-compiler file "${THEROCK_MULTI_VENDOR_RESOLVED_SYCL_COMPILER}")
        endif()
        _therock_multi_vendor_input(level-zero-sdk tree "${THEROCK_MULTI_VENDOR_LEVEL_ZERO_ROOT}")
        _therock_multi_vendor_compiler(_compiler spirv "${THEROCK_MULTI_VENDOR_AMD_ROOT}" clang)
        _therock_multi_vendor_input(spirv-compiler file "${_compiler}")
        get_filename_component(_compiler_dir "${_compiler}" DIRECTORY)
        set(THEROCK_MULTI_VENDOR_RESOLVED_SPIRV_TRANSLATOR "${THEROCK_MULTI_VENDOR_SPIRV_TRANSLATOR}")
        if(NOT THEROCK_MULTI_VENDOR_RESOLVED_SPIRV_TRANSLATOR)
          unset(THEROCK_MULTI_VENDOR_RESOLVED_SPIRV_TRANSLATOR)
          unset(THEROCK_MULTI_VENDOR_RESOLVED_SPIRV_TRANSLATOR CACHE)
          find_program(THEROCK_MULTI_VENDOR_RESOLVED_SPIRV_TRANSLATOR NAMES amd-llvm-spirv llvm-spirv
            PATHS "${_compiler_dir}" NO_DEFAULT_PATH NO_CACHE REQUIRED)
        endif()
        set(THEROCK_MULTI_VENDOR_RESOLVED_SPIRV_VALIDATOR "${THEROCK_MULTI_VENDOR_SPIRV_VALIDATOR}")
        if(NOT THEROCK_MULTI_VENDOR_RESOLVED_SPIRV_VALIDATOR)
          unset(THEROCK_MULTI_VENDOR_RESOLVED_SPIRV_VALIDATOR)
          unset(THEROCK_MULTI_VENDOR_RESOLVED_SPIRV_VALIDATOR CACHE)
          find_program(THEROCK_MULTI_VENDOR_RESOLVED_SPIRV_VALIDATOR NAMES spirv-val
            HINTS "${THEROCK_MULTI_VENDOR_LEVEL_ZERO_ROOT}/bin" NO_CACHE REQUIRED)
        endif()
        _therock_multi_vendor_input(spirv-translator file "${THEROCK_MULTI_VENDOR_RESOLVED_SPIRV_TRANSLATOR}")
        _therock_multi_vendor_input(spirv-validator file "${THEROCK_MULTI_VENDOR_RESOLVED_SPIRV_VALIDATOR}")
        # Capture the installed ROCm compiler resources when that SDK supplies
        # the SPIR-V frontend, including Intel-only target selections.
        if(EXISTS "${THEROCK_MULTI_VENDOR_AMD_ROOT}")
          file(REAL_PATH "${THEROCK_MULTI_VENDOR_AMD_ROOT}" _amd_sdk_real)
          file(REAL_PATH "${_compiler}" _compiler_real)
          cmake_path(IS_PREFIX _amd_sdk_real "${_compiler_real}" NORMALIZE _uses_amd_sdk)
          if(_uses_amd_sdk)
            _therock_multi_vendor_input(amd-sdk tree "${THEROCK_MULTI_VENDOR_AMD_ROOT}")
          endif()
        endif()
      endif()
    endif()
  endforeach()
endif()
get_property(_therock_input_arguments GLOBAL PROPERTY THEROCK_MULTI_VENDOR_INPUT_ARGS)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${_therock_input_tool}"
  "${THEROCK_SOURCE_DIR}/build_tools/_therock_utils/input_provenance.py"
  "${_therock_input_lock}")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${_therock_input_tool}" ${_therock_input_action}
  ${_therock_input_arguments} --lock "${_therock_input_lock}"
  --cache "${_therock_input_cache}" --report "${_therock_input_report}"
  COMMAND_ERROR_IS_FATAL ANY)
add_custom_target(multi-vendor-verify-inputs
  COMMAND "${CMAKE_COMMAND}" -P "${_therock_input_guard}"
  COMMENT "Verifying locked multi-vendor SDK and compiler inputs"
  VERBATIM)
add_custom_target(multi-vendor-verify-inputs-full
  COMMAND "${CMAKE_COMMAND}" -DTHEROCK_MULTI_VENDOR_VERIFY_FULL=ON -P "${_therock_input_guard}"
  COMMENT "Fully hashing locked multi-vendor SDK and compiler inputs"
  VERBATIM)
if(THEROCK_BUILD_TESTING)
  add_test(NAME multi-vendor-inputs COMMAND "${CMAKE_COMMAND}" -P "${_therock_input_guard}")
  set_tests_properties(multi-vendor-inputs PROPERTIES
    FIXTURES_SETUP multi-vendor-inputs LABELS "multi-vendor;provenance")
endif()

function(_therock_multi_vendor_register_inputs target)
  # A prebuilt stage needs provenance for its original build, not merely a
  # matching SDK today. Until that import contract exists, reject these markers.
  get_target_property(_stage "${target}" THEROCK_STAGE_DIR)
  if(EXISTS "${_stage}.prebuilt")
    message(FATAL_ERROR "Experimental multi-vendor inputs cannot qualify prebuilt stage: ${_stage}.prebuilt")
  endif()
  set_property(GLOBAL APPEND PROPERTY THEROCK_MULTI_VENDOR_INPUT_PROJECTS "${target}")
  set_property(TARGET "${target}" PROPERTY THEROCK_FPRINT "")
endfunction()

# Quote generated CMake literals even when a path contains bracket delimiters.
function(_therock_multi_vendor_quote out value)
  set(_equals "==")
  string(FIND "${value}" "]${_equals}]" _closing)
  while(NOT _closing EQUAL -1)
    string(APPEND _equals "=")
    string(FIND "${value}" "]${_equals}]" _closing)
  endwhile()
  set("${out}" "[${_equals}[${value}]${_equals}]" PARENT_SCOPE)
endfunction()

function(_therock_multi_vendor_finalize_inputs)
  get_property(_projects GLOBAL PROPERTY THEROCK_MULTI_VENDOR_INPUT_PROJECTS)
  set(_script "# Generated multi-vendor imported-input verification.\n")
  foreach(_project IN LISTS _projects)
    get_target_property(_stage "${_project}" THEROCK_STAGE_DIR)
    _therock_multi_vendor_quote(_quoted_stage "${_stage}.prebuilt")
    _therock_multi_vendor_quote(_quoted_error "Unqualified prebuilt stage: ${_stage}.prebuilt")
    string(APPEND _script "if(EXISTS ${_quoted_stage})\n  message(FATAL_ERROR ${_quoted_error})\nendif()\n")
  endforeach()
  string(APPEND _script "if(THEROCK_MULTI_VENDOR_VERIFY_FULL)\n  set(_verification_extra --full)\nendif()\nexecute_process(COMMAND\n")
  foreach(_arg IN ITEMS "${Python3_EXECUTABLE}" "${_therock_input_tool}" verify)
    _therock_multi_vendor_quote(_quoted_arg "${_arg}")
    string(APPEND _script "  ${_quoted_arg}\n")
  endforeach()
  foreach(_arg IN LISTS _therock_input_arguments)
    _therock_multi_vendor_quote(_quoted_arg "${_arg}")
    string(APPEND _script "  ${_quoted_arg}\n")
  endforeach()
  foreach(_arg IN ITEMS --lock "${_therock_input_lock}" --cache "${_therock_input_cache}"
      --report "${_therock_input_report}")
    _therock_multi_vendor_quote(_quoted_arg "${_arg}")
    string(APPEND _script "  ${_quoted_arg}\n")
  endforeach()
  string(APPEND _script "  \${_verification_extra}\n  COMMAND_ERROR_IS_FATAL ANY)\n")
  set(_temporary "${_therock_input_guard}.tmp")
  file(WRITE "${_temporary}" "${_script}")
  configure_file("${_temporary}" "${_therock_input_guard}" COPYONLY)
  file(REMOVE "${_temporary}")
endfunction()
