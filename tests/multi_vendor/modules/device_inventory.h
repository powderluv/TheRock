// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace therock::module_validation {

inline bool list_devices_requested(int argc, char **argv) {
  bool requested = false;
  for (int i = 1; i < argc; ++i) {
    requested = requested || std::string_view(argv[i]) == "--list-devices";
  }
  if (requested && argc != 2) {
    throw std::runtime_error("--list-devices must be used alone");
  }
  return requested;
}

inline void validate_device_uuid(std::string_view uuid) {
  if (uuid.size() != 32 ||
      !std::all_of(uuid.begin(), uuid.end(),
                   [](char c) {
                     return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                   }) ||
      std::all_of(uuid.begin(), uuid.end(), [](char c) { return c == '0'; })) {
    throw std::runtime_error(
        "Device UUID must be 32 lowercase hex characters and nonzero");
  }
}

// Preserve the 16 bytes returned by the driver in native order. No byte
// swapping, PCI fallback, or invented UUID is permitted.
template <typename Byte, size_t N>
std::string device_uuid_hex(const Byte (&bytes)[N]) {
  static_assert(N == 16 && sizeof(Byte) == 1,
                "The device inventory requires a 16-byte driver UUID");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(32);
  for (Byte byte : bytes) {
    const auto value = static_cast<unsigned char>(byte);
    result += hex[value >> 4];
    result += hex[value & 0xf];
  }
  validate_device_uuid(result);
  return result;
}

inline void require_device_uuid(std::string_view expected,
                                std::string_view observed) {
  if (!expected.empty() && expected != observed) {
    throw std::runtime_error(
        "Device UUID mismatch: expected=" + std::string(expected) +
        " observed=" + std::string(observed));
  }
}

// Runtime names are bounded arrays, not necessarily terminated C strings.
template <size_t N> std::string device_name(const char (&name)[N]) {
  return std::string(name, std::find(name, name + N, '\0'));
}

inline std::string json_string(std::string_view value) {
  constexpr char hex[] = "0123456789abcdef";
  std::string result = "\"";
  for (size_t i = 0; i < value.size(); ++i) {
    const auto byte = static_cast<unsigned char>(value[i]);
    if (byte == '"' || byte == '\\') {
      result += '\\';
      result += static_cast<char>(byte);
    } else if (byte < 0x20) {
      result += "\\u00";
      result += hex[byte >> 4];
      result += hex[byte & 0xf];
    } else if (byte < 0x80) {
      result += static_cast<char>(byte);
    } else {
      // Preserve valid UTF-8 exactly, rejecting malformed runtime strings
      // instead of emitting invalid JSON or silently changing device names.
      size_t length = 0;
      uint32_t codepoint = 0;
      uint32_t minimum = 0;
      if (byte >= 0xc2 && byte <= 0xdf) {
        length = 2;
        codepoint = byte & 0x1f;
        minimum = 0x80;
      } else if (byte >= 0xe0 && byte <= 0xef) {
        length = 3;
        codepoint = byte & 0x0f;
        minimum = 0x800;
      } else if (byte >= 0xf0 && byte <= 0xf4) {
        length = 4;
        codepoint = byte & 0x07;
        minimum = 0x10000;
      }
      if (length == 0 || length > value.size() - i) {
        throw std::runtime_error("Device string is not valid UTF-8");
      }
      for (size_t j = 1; j < length; ++j) {
        const auto continuation = static_cast<unsigned char>(value[i + j]);
        if ((continuation & 0xc0) != 0x80) {
          throw std::runtime_error("Device string is not valid UTF-8");
        }
        codepoint = (codepoint << 6) | (continuation & 0x3f);
      }
      if (codepoint < minimum || codepoint > 0x10ffff ||
          (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
        throw std::runtime_error("Device string is not valid UTF-8");
      }
      result.append(value.substr(i, length));
      i += length - 1;
    }
  }
  result += '"';
  return result;
}

inline uint64_t positive_device_limit(int64_t value) {
  if (value <= 0) {
    throw std::runtime_error("Device reports a nonpositive compute limit");
  }
  return static_cast<uint64_t>(value);
}

struct DeviceInventoryEntry {
  uint64_t index = 0;
  std::string name;
  std::string device_uuid;
  std::optional<std::string> architecture;
  uint32_t vendor_id = 0;
  std::optional<uint32_t> device_id;
  uint64_t max_threads_per_block = 0;
  std::array<uint64_t, 3> max_block_dimensions{};
  std::array<uint64_t, 3> max_grid_dimensions{};
  uint64_t total_memory_bytes = 0;
};

inline std::string dimensions_json(const std::array<uint64_t, 3> &values) {
  return "[" + std::to_string(values[0]) + "," + std::to_string(values[1]) +
         "," + std::to_string(values[2]) + "]";
}

inline std::string
device_inventory_json(std::string_view vendor, std::string_view backend,
                      const std::vector<DeviceInventoryEntry> &devices) {
  std::string result =
      "{\"schema_version\":2,\"kind\":\"native-device-inventory\","
      "\"scope\":\"observed-runtime\",\"vendor\":" +
      json_string(vendor) + ",\"backend\":" + json_string(backend) +
      ",\"devices\":[";
  bool first = true;
  std::unordered_set<std::string> uuids;
  for (const auto &device : devices) {
    validate_device_uuid(device.device_uuid);
    if (!uuids.insert(device.device_uuid).second) {
      throw std::runtime_error("Duplicate device UUID in native inventory: " +
                               device.device_uuid);
    }
    if (device.name.empty() ||
        (device.architecture && device.architecture->empty()) ||
        device.vendor_id == 0 || device.max_threads_per_block == 0 ||
        device.total_memory_bytes == 0 ||
        std::find(device.max_block_dimensions.begin(),
                  device.max_block_dimensions.end(),
                  0) != device.max_block_dimensions.end() ||
        std::find(device.max_grid_dimensions.begin(),
                  device.max_grid_dimensions.end(),
                  0) != device.max_grid_dimensions.end()) {
      throw std::runtime_error(
          "Device reports incomplete inventory properties");
    }
    if (!first) {
      result += ',';
    }
    first = false;
    result +=
        "{\"index\":" + std::to_string(device.index) +
        ",\"name\":" + json_string(device.name) +
        ",\"device_uuid\":" + json_string(device.device_uuid) +
        ",\"architecture\":" +
        (device.architecture ? json_string(*device.architecture) : "null") +
        ",\"vendor_id\":" + std::to_string(device.vendor_id) +
        ",\"device_id\":" +
        (device.device_id ? std::to_string(*device.device_id) : "null") +
        ",\"limits\":{\"max_threads_per_block\":" +
        std::to_string(device.max_threads_per_block) +
        ",\"max_block_dimensions\":" +
        dimensions_json(device.max_block_dimensions) +
        ",\"max_grid_dimensions\":" +
        dimensions_json(device.max_grid_dimensions) +
        ",\"total_memory_bytes\":" + std::to_string(device.total_memory_bytes) +
        "}}";
  }
  result += "]}";
  return result;
}

} // namespace therock::module_validation
