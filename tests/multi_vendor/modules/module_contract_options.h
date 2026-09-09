// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "module_contract_data.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace therock::module_validation {

static_assert(module_contract::kContractVersion == 1,
              "This native adapter implements contract version 1 only");
static_assert(std::string_view(module_contract::kContractAbi) ==
                  "therock.validation.f32-vector",
              "This native adapter implements the f32-vector fixture only");

static_assert(CHAR_BIT == 8, "The module contract requires eight-bit bytes");
static_assert(sizeof(void *) * CHAR_BIT == module_contract::kPointerBits,
              "The module contract requires 64-bit host pointers");
static_assert(module_contract::kPointerBits == 64,
              "This adapter only implements the 64-bit pointer contract");
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24,
              "The module contract requires IEEE 754 binary32");
static_assert(sizeof(uint32_t) == 4 && sizeof(unsigned) == 4,
              "The module contract requires a 32-bit unsigned count");
static_assert(alignof(void *) == 8 && alignof(float) == 4 &&
                  alignof(uint32_t) == 4 && alignof(unsigned) == 4,
              "The native binding requires pointer/f32/u32 alignments 8/4/4");
static_assert(module_contract::kGroupSize == 128,
              "This adapter implements a fixed 128-item workgroup");

// This reports compiled adapter behavior only. No payload is opened and no
// driver API is called; observed hardware capability remains a launch-time
// check.
inline bool describe_contract(int argc, char **argv) {
  bool requested = false;
  for (int i = 1; i < argc; ++i) {
    requested = requested || std::string(argv[i]) == "--describe-contract";
  }
  if (!requested) {
    return false;
  }
  if (argc != 2) {
    throw std::runtime_error("--describe-contract must be used alone");
  }
  std::puts(module_contract::kRunnerDescription);
  return true;
}

struct ContractOptions {
  std::string abi;
  std::string version;
  std::string sha256;
  std::string payload_format;

  bool consume(const std::string &argument, const std::string &value) {
    if (argument == "--launch-abi") {
      abi = value;
    } else if (argument == "--launch-abi-version") {
      version = value;
    } else if (argument == "--launch-contract-sha256") {
      sha256 = value;
    } else if (argument == "--payload-format") {
      payload_format = value;
    } else {
      return false;
    }
    return true;
  }

  void validate(std::initializer_list<const char *> formats) const {
    if (abi.empty() || version.empty() || sha256.empty() ||
        payload_format.empty()) {
      throw std::runtime_error(
          "Payload execution requires --launch-abi, --launch-abi-version, "
          "--launch-contract-sha256, and --payload-format");
    }
    if (abi != module_contract::kContractAbi) {
      throw std::runtime_error("Unsupported --launch-abi " + abi);
    }
    if (version != std::to_string(module_contract::kContractVersion)) {
      throw std::runtime_error("Unsupported --launch-abi-version " + version);
    }
    if (sha256 != module_contract::kContractSha256) {
      throw std::runtime_error("Mismatched --launch-contract-sha256");
    }
    for (const char *format : formats) {
      if (payload_format == format) {
        return;
      }
    }
    throw std::runtime_error("Unsupported --payload-format " + payload_format);
  }

  void validate_inspected_format(const std::string &format) const {
    if (payload_format != format) {
      throw std::runtime_error("Declared --payload-format " + payload_format +
                               " does not match inspected format " + format);
    }
  }
};

} // namespace therock::module_validation
