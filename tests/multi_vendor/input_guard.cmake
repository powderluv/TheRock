# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Receipts bind completed outputs to the exact imported-input observation.
# They are local build evidence, not signatures or a full reproducibility claim.
function(therock_multi_vendor_verify_receipts provenance)
  if(NOT EXISTS "${provenance}" OR IS_DIRECTORY "${provenance}")
    message(FATAL_ERROR "Imported-input provenance is missing: ${provenance}")
  endif()
  file(SHA256 "${provenance}" _expected)
  foreach(_receipt IN LISTS ARGN)
    if(NOT EXISTS "${_receipt}" OR IS_DIRECTORY "${_receipt}")
      message(FATAL_ERROR "Imported-input build receipt is missing: ${_receipt}. Build and stage the consumer first.")
    endif()
    file(SHA256 "${_receipt}" _actual)
    if(NOT _actual STREQUAL _expected)
      message(FATAL_ERROR "Stale imported-input build receipt: ${_receipt}. Rebuild and stage the consumer with the selected input lock.")
    endif()
  endforeach()
endfunction()

if(CMAKE_SCRIPT_MODE_FILE)
  if(NOT THEROCK_MULTI_VENDOR_CHECK_PROVENANCE OR NOT THEROCK_MULTI_VENDOR_CHECK_RECEIPTS)
    message(FATAL_ERROR "Receipt checking requires provenance and at least one receipt")
  endif()
  therock_multi_vendor_verify_receipts("${THEROCK_MULTI_VENDOR_CHECK_PROVENANCE}"
    ${THEROCK_MULTI_VENDOR_CHECK_RECEIPTS})
  return()
endif()

# Include before project() to reject changed inputs before compiler discovery.
# Standalone consumers can omit both paths and retain their existing behavior.
set(THEROCK_MULTI_VENDOR_INPUT_GUARD "" CACHE FILEPATH "Parent imported-input verification script")
set(THEROCK_MULTI_VENDOR_INPUT_PROVENANCE "" CACHE FILEPATH "Parent imported-input provenance and identity")
set(_therock_input_dependencies)
if(THEROCK_MULTI_VENDOR_INPUT_GUARD)
  if(NOT EXISTS "${THEROCK_MULTI_VENDOR_INPUT_GUARD}" OR
      NOT EXISTS "${THEROCK_MULTI_VENDOR_INPUT_PROVENANCE}")
    message(FATAL_ERROR "Imported-input guard and provenance files must exist")
  endif()
  include("${THEROCK_MULTI_VENDOR_INPUT_GUARD}")
endif()

# Call after project() and include(CTest), once BUILD_TESTING has its default.
function(therock_multi_vendor_initialize_inputs)
  if(THEROCK_MULTI_VENDOR_INPUT_GUARD)
    add_custom_target(therock-verify-inputs
      COMMAND "${CMAKE_COMMAND}" -P "${THEROCK_MULTI_VENDOR_INPUT_GUARD}"
      COMMENT "Verifying imported inputs before consumer build"
      VERBATIM)
    set(_therock_input_dependencies therock-verify-inputs
      "${THEROCK_MULTI_VENDOR_INPUT_PROVENANCE}" PARENT_SCOPE)
    if(BUILD_TESTING)
      add_test(NAME imported-inputs COMMAND "${CMAKE_COMMAND}" -P "${THEROCK_MULTI_VENDOR_INPUT_GUARD}")
      set_tests_properties(imported-inputs PROPERTIES
        FIXTURES_SETUP imported-inputs LABELS "multi-vendor;provenance")
    endif()
  endif()
endfunction()

function(therock_multi_vendor_guard_consumer target)
  if(TARGET therock-verify-inputs)
    add_dependencies("${target}" therock-verify-inputs)
    get_target_property(_sources "${target}" SOURCES)
    set_property(SOURCE ${_sources} APPEND PROPERTY OBJECT_DEPENDS
      "${THEROCK_MULTI_VENDOR_INPUT_PROVENANCE}")
  endif()
endfunction()

# Quote a literal for generated install code; paths may contain CMake syntax.
function(_therock_multi_vendor_quote output value)
  string(REPLACE "\\" "\\\\" _quoted "${value}")
  string(REPLACE "\"" "\\\"" _quoted "${_quoted}")
  string(REPLACE "$" "\\$" _quoted "${_quoted}")
  set(${output} "\"${_quoted}\"" PARENT_SCOPE)
endfunction()

# CMake's regular install can skip same-size files with coarse equal timestamps.
# Verify the current inputs and completed build before clearing this consumer's
# known destinations. A failed/partial install cannot leave a valid receipt.
function(therock_multi_vendor_prepare_install)
  cmake_parse_arguments(PARSE_ARGV 0 ARG "" "RECEIPT" "FILES")
  if(NOT TARGET therock-verify-inputs)
    return()
  endif()
  if(NOT ARG_RECEIPT OR NOT ARG_FILES OR ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Guarded install requires a build RECEIPT and destination FILES")
  endif()
  _therock_multi_vendor_quote(_guard "${THEROCK_MULTI_VENDOR_INPUT_GUARD}")
  _therock_multi_vendor_quote(_provenance "${THEROCK_MULTI_VENDOR_INPUT_PROVENANCE}")
  _therock_multi_vendor_quote(_receipt "${ARG_RECEIPT}")
  _therock_multi_vendor_quote(_helper "${CMAKE_CURRENT_FUNCTION_LIST_FILE}")
  set(_code "include(${_guard})\nset(THEROCK_MULTI_VENDOR_CHECK_PROVENANCE ${_provenance})\nset(THEROCK_MULTI_VENDOR_CHECK_RECEIPTS ${_receipt})\ninclude(${_helper})\n")
  foreach(_relative IN LISTS ARG_FILES)
    if(IS_ABSOLUTE "${_relative}" OR _relative MATCHES "(^|/)\\.\\.(/|$)")
      message(FATAL_ERROR "Guarded install paths must be relative to the install prefix")
    endif()
    _therock_multi_vendor_quote(_literal "${_relative}")
    string(APPEND _code "set(_receipt_relative ${_literal})\nfile(REMOVE \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/\${_receipt_relative}\")\n")
  endforeach()
  install(CODE "${_code}")
endfunction()

function(therock_multi_vendor_install_receipt receipt destination)
  get_filename_component(_name "${receipt}" NAME)
  _therock_multi_vendor_quote(_source "${receipt}")
  _therock_multi_vendor_quote(_destination "${destination}")
  _therock_multi_vendor_quote(_filename "${_name}")
  install(CODE "
set(_receipt_destination ${_destination})
set(_receipt_filename ${_filename})
if(NOT IS_ABSOLUTE \"\${_receipt_destination}\")
  set(_receipt_destination \"\${CMAKE_INSTALL_PREFIX}/\${_receipt_destination}\")
endif()
file(MAKE_DIRECTORY \"\$ENV{DESTDIR}\${_receipt_destination}\")
file(COPY_FILE ${_source} \"\$ENV{DESTDIR}\${_receipt_destination}/\${_receipt_filename}\" ONLY_IF_DIFFERENT)
file(CHMOD \"\$ENV{DESTDIR}\${_receipt_destination}/\${_receipt_filename}\"
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
list(APPEND CMAKE_INSTALL_MANIFEST_FILES \"\${_receipt_destination}/\${_receipt_filename}\")
")
endfunction()

# Register once per child, after its other install rules. DEPENDS must contain
# every executable and device payload represented by this receipt. Target-only
# builds may omit the receipt; a normal all-build completes it before staging.
function(therock_multi_vendor_record_inputs target)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "INSTALL_DESTINATION" "DEPENDS")
  if(NOT TARGET therock-verify-inputs)
    return()
  endif()
  if(NOT ARG_DEPENDS OR NOT ARG_INSTALL_DESTINATION)
    message(FATAL_ERROR "Input receipts require output DEPENDS and INSTALL_DESTINATION")
  endif()
  set(_receipt "${CMAKE_CURRENT_BINARY_DIR}/input-build-receipt.json")
  add_custom_command(OUTPUT "${_receipt}"
    COMMAND "${CMAKE_COMMAND}" -E copy
      "${THEROCK_MULTI_VENDOR_INPUT_PROVENANCE}" "${_receipt}"
    DEPENDS ${ARG_DEPENDS} ${_therock_input_dependencies}
    COMMENT "Recording completed imported-input consumer build"
    VERBATIM)
  add_custom_target("${target}" ALL DEPENDS "${_receipt}")
  # Publish the installed receipt only after the executable/payload install rules.
  therock_multi_vendor_install_receipt("${_receipt}" "${ARG_INSTALL_DESTINATION}")
  if(BUILD_TESTING)
    add_test(NAME imported-build-receipt
      COMMAND "${CMAKE_COMMAND}"
        "-DTHEROCK_MULTI_VENDOR_CHECK_PROVENANCE=${THEROCK_MULTI_VENDOR_INPUT_PROVENANCE}"
        "-DTHEROCK_MULTI_VENDOR_CHECK_RECEIPTS=${_receipt}"
        -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}")
    set_tests_properties(imported-build-receipt PROPERTIES
      FIXTURES_SETUP imported-build-receipt FIXTURES_REQUIRED imported-inputs
      LABELS "multi-vendor;provenance")
  endif()
endfunction()

function(therock_multi_vendor_guard_test name)
  if(TARGET therock-verify-inputs)
    set_tests_properties("${name}" PROPERTIES
      FIXTURES_REQUIRED "imported-inputs;imported-build-receipt")
  endif()
endfunction()
