// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "module_contract_options.h"

#include <cmath>
#include <initializer_list>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace therock::module_validation {

struct ModuleRequest {
  std::string payload_format;
  std::string symbol;
  std::string payload;
};

inline ModuleRequest parse_module_request(int &i, int argc, char **argv) {
  if (argc - i <= 3) {
    throw std::runtime_error("--module requires FORMAT SYMBOL PATH");
  }
  ModuleRequest request{argv[i + 1], argv[i + 2], argv[i + 3]};
  i += 3;
  return request;
}

inline std::vector<ModuleRequest>
finalize_module_requests(const ContractOptions &contract,
                         const std::string &payload, const std::string &symbol,
                         const std::vector<ModuleRequest> &batch,
                         const std::unordered_set<std::string> &seen,
                         std::initializer_list<const char *> formats,
                         bool allow_repeated_requests = false) {
  if (allow_repeated_requests && batch.empty()) {
    throw std::runtime_error("--pipeline requires at least one --module");
  }
  std::vector<ModuleRequest> requests;
  if (batch.empty()) {
    // Preserve the legacy flag validation order and diagnostics.
    contract.validate(formats);
    if (payload.empty()) {
      throw std::runtime_error("--payload is required");
    }
    requests.push_back({contract.payload_format, symbol, payload});
  } else {
    if (seen.count("--payload") || seen.count("--symbol") ||
        seen.count("--payload-format")) {
      throw std::runtime_error(
          "--module cannot be combined with --payload, --symbol, or "
          "--payload-format");
    }
    if (batch.size() > 32) {
      throw std::runtime_error("A module session accepts at most 32 requests");
    }
    if (contract.abi.empty() || contract.version.empty() ||
        contract.sha256.empty()) {
      throw std::runtime_error(
          "Module sessions require --launch-abi, --launch-abi-version, and "
          "--launch-contract-sha256");
    }
    requests = batch;
  }
  std::set<std::tuple<std::string, std::string, std::string>> identities;
  for (const auto &request : requests) {
    ContractOptions request_contract = contract;
    request_contract.payload_format = request.payload_format;
    request_contract.validate(formats);
    if (request.payload.empty()) {
      throw std::runtime_error("Module payload path cannot be empty");
    }
    if (request.symbol != "therock_module_saxpy" &&
        request.symbol != "therock_module_relu") {
      throw std::runtime_error("No CPU reference is defined for symbol " +
                               request.symbol);
    }
    if (!allow_repeated_requests &&
        !identities
             .emplace(request.payload_format, request.symbol, request.payload)
             .second) {
      throw std::runtime_error(
          "Duplicate module request: " + request.payload_format + " " +
          request.symbol + " " + request.payload);
    }
  }
  return requests;
}

// Compare two permitted FP32 implementations: the stored CPU stage reference
// and device arithmetic, either of which may contract multiply/add into FMA.
// Eight epsilons conservatively cover both sides' multiply, add, and optional
// subtraction roundoff. Propagate earlier input uncertainty through alpha and
// the rounding term; ReLU is nonexpansive. Operand magnitudes preserve the
// bound under cancellation, unlike a relative tolerance on the result alone.
inline double pipeline_error_bound(double input, double y, double alpha,
                                   bool relu, double previous_error) {
  const double propagated = std::abs(alpha) * previous_error;
  const double scale =
      std::abs(alpha * input) + std::abs(y) + (relu ? 0.125 : 0.0) + propagated;
  return propagated + 8.0 * std::numeric_limits<float>::epsilon() * scale +
         1.0e-7;
}

} // namespace therock::module_validation
