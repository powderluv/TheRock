// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "device_inventory.h"
#include "module_contract_options.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace therock::module_service {
constexpr uint32_t kMaxCount = 65539;
constexpr uint32_t kMaxCapacity = 65556;
constexpr uint32_t kMaxFrame = 1024 * 1024;
struct Config {
  int device = 0;
  uint32_t device_id = 0;
  std::string expected_arch;
  std::string device_uuid;
};

struct CommandError : std::runtime_error {
  CommandError(uint32_t code, const std::string &message)
      : std::runtime_error(message), status(code) {}
  uint32_t status;
};
inline void require(bool condition, const std::string &message,
                    uint32_t status = 1) {
  if (!condition)
    throw CommandError(status, message);
}
inline bool requested(int argc, char **argv) {
  bool found = false;
  for (int i = 1; i < argc; ++i)
    found = found || std::string(argv[i]) == "--serve";
  if (found && argc != 2)
    throw std::runtime_error("--serve must be used alone");
  return found;
}
using Bytes = std::vector<uint8_t>;
inline void put32(Bytes &bytes, uint32_t value) {
  for (int i = 0; i < 4; ++i)
    bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
}
inline void put64(Bytes &bytes, uint64_t value) {
  put32(bytes, static_cast<uint32_t>(value));
  put32(bytes, static_cast<uint32_t>(value >> 32));
}
inline void put_string(Bytes &bytes, const std::string &value) {
  put32(bytes, static_cast<uint32_t>(value.size()));
  bytes.insert(bytes.end(), value.begin(), value.end());
}
inline void put_float(Bytes &bytes, float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  put32(bytes, bits);
}
inline bool valid_utf8(const std::string &value) {
  for (size_t i = 0; i < value.size();) {
    const unsigned char first = static_cast<unsigned char>(value[i++]);
    if (first < 0x80)
      continue;
    uint32_t point = 0;
    unsigned remaining = 0;
    uint32_t minimum = 0;
    if (first >= 0xc2 && first <= 0xdf) {
      point = first & 0x1f;
      remaining = 1;
      minimum = 0x80;
    } else if (first >= 0xe0 && first <= 0xef) {
      point = first & 0x0f;
      remaining = 2;
      minimum = 0x800;
    } else if (first >= 0xf0 && first <= 0xf4) {
      point = first & 7;
      remaining = 3;
      minimum = 0x10000;
    } else
      return false;
    if (remaining > value.size() - i)
      return false;
    while (remaining--) {
      const unsigned char next = static_cast<unsigned char>(value[i++]);
      if ((next & 0xc0) != 0x80)
        return false;
      point = (point << 6) | (next & 0x3f);
    }
    if (point < minimum || point > 0x10ffff ||
        (point >= 0xd800 && point <= 0xdfff))
      return false;
  }
  return true;
}
class Reader {
public:
  explicit Reader(const Bytes &bytes) : bytes_(bytes) {}
  uint32_t u32() {
    require(bytes_.size() - offset_ >= 4, "Truncated request field");
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i)
      value |= uint32_t(bytes_[offset_++]) << (8 * i);
    return value;
  }
  uint64_t u64() {
    const uint64_t low = u32();
    return low | (uint64_t(u32()) << 32);
  }
  float f32() {
    const uint32_t bits = u32();
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  std::string string() {
    const uint32_t length = u32();
    require(length <= 4096 && length <= bytes_.size() - offset_,
            "Invalid string length");
    std::string value(bytes_.begin() + offset_,
                      bytes_.begin() + offset_ + length);
    offset_ += length;
    require(value.find('\0') == std::string::npos,
            "Embedded NUL in request string");
    require(valid_utf8(value), "Request strings must be UTF-8");
    return value;
  }
  void end() const {
    require(offset_ == bytes_.size(), "Trailing request bytes");
  }

private:
  const Bytes &bytes_;
  size_t offset_ = 0;
};
inline bool read_exact(int fd, uint8_t *data, size_t size,
                       bool allow_eof = false) {
  size_t offset = 0;
  while (offset < size) {
    const ssize_t count = ::read(fd, data + offset, size - offset);
    if (count < 0 && errno == EINTR)
      continue;
    if (count == 0 && offset == 0 && allow_eof)
      return false;
    require(count > 0, "Truncated or unreadable service frame", 2);
    offset += static_cast<size_t>(count);
  }
  return true;
}
inline void write_exact(int fd, const Bytes &bytes) {
  size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count =
        ::write(fd, bytes.data() + offset, bytes.size() - offset);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      throw std::runtime_error("Cannot write service response");
    offset += static_cast<size_t>(count);
  }
}
struct Frame {
  uint16_t opcode = 0;
  uint32_t id = 0;
  Bytes payload;
};
inline bool read_frame(Frame &frame, uint32_t expected_id) {
  Bytes header(16);
  if (!read_exact(STDIN_FILENO, header.data(), header.size(), true))
    return false;
  Reader reader(header);
  const uint32_t magic = reader.u32();
  const uint32_t version_opcode = reader.u32();
  frame.opcode = static_cast<uint16_t>(version_opcode >> 16);
  frame.id = reader.u32();
  const uint32_t length = reader.u32();
  require(magic == 0x534d5254, "Invalid service frame magic", 2);
  require((version_opcode & 0xffff) == 1,
          "Unsupported service protocol version", 5);
  require(frame.id == expected_id && frame.id != 0,
          "Invalid service request sequence", 2);
  require(length <= kMaxFrame, "Service frame exceeds 1 MiB", 2);
  require(frame.opcode >= 1 && frame.opcode <= 11, "Unknown service opcode", 2);
  frame.payload.resize(length);
  read_exact(STDIN_FILENO, frame.payload.data(), length);
  return true;
}
inline void respond(int fd, const Frame &frame, uint32_t status,
                    const std::string &error, const Bytes &result = {}) {
  Bytes body;
  put32(body, status);
  std::string message =
      valid_utf8(error) ? error : "Native error text is not UTF-8";
  if (message.size() > 4096) {
    message.resize(4096);
    while (!valid_utf8(message))
      message.pop_back();
  }
  put_string(body, message);
  body.insert(body.end(), result.begin(), result.end());
  Bytes header;
  put32(header, 0x534d5254);
  put32(header, 1u | (uint32_t(frame.opcode | 0x8000) << 16));
  put32(header, frame.id);
  put32(header, static_cast<uint32_t>(body.size()));
  write_exact(fd, header);
  write_exact(fd, body);
}

template <typename Backend> class State {
  struct BufferEntry {
    std::unique_ptr<typename Backend::Buffer> object;
    uint32_t capacity;
  };

public:
  ~State() { close(); }
  void close() noexcept {
    if (backend_) {
      // This precedes ALL map destruction, including EOF and malformed frames.
      // Unknown completion exits the isolated worker without freeing resources.
      backend_->synchronize();
      buffers_.clear();
      modules_.clear();
      backend_.reset();
    }
  }
  Bytes dispatch(const Frame &frame) {
    Reader reader(frame.payload);
    Bytes result;
    if (frame.opcode == 1) {
      reader.end();
      require(!hello_, "HELLO was already completed", 4);
      hello_ = true;
      put_string(result, module_contract::kRunnerDescription);
      return result;
    }
    require(hello_, "HELLO is required first", 4);
    if (frame.opcode == 11) {
      reader.end();
      close();
      require(Backend::cleanup_ok(), "Native resource cleanup failed", 3);
      return result;
    }
    if (frame.opcode == 2) {
      require(!backend_, "Only one OPEN is allowed per connection", 4);
      Config config;
      const uint32_t device = reader.u32();
      config.device_id = reader.u32();
      config.expected_arch = reader.string();
      config.device_uuid = reader.string();
      const std::string abi = reader.string();
      const uint32_t version = reader.u32();
      const std::string hash = reader.string();
      reader.end();
      require(device <= uint32_t(std::numeric_limits<int>::max()),
              "Invalid device ordinal");
      require(config.device_id <= 65535, "Invalid device ID");
      require(!config.expected_arch.empty(),
              "Expected architecture is required");
      try {
        module_validation::validate_device_uuid(config.device_uuid);
      } catch (const std::exception &error) {
        throw CommandError(1, error.what());
      }
      require(abi == module_contract::kContractAbi &&
                  version == module_contract::kContractVersion &&
                  hash == module_contract::kContractSha256,
              "Unsupported service launch contract", 5);
      config.device = static_cast<int>(device);
      backend_ = std::make_unique<Backend>(config);
      return result;
    }
    require(bool(backend_), "OPEN is required before device operations", 4);
    if (frame.opcode == 3) {
      const uint32_t capacity = reader.u32();
      reader.end();
      require(capacity > 0 && capacity <= kMaxCapacity,
              "Invalid buffer capacity");
      require(buffers_.size() < 64, "Service buffer limit is 64");
      const uint64_t handle = new_handle();
      buffers_.emplace(handle,
                       BufferEntry{backend_->allocate(capacity), capacity});
      put64(result, handle);
    } else if (frame.opcode == 4 || frame.opcode == 6) {
      const uint64_t handle = reader.u64();
      reader.end();
      if (frame.opcode == 4)
        (void)buffer(handle);
      else
        (void)module(handle);
      backend_->synchronize();
      if (frame.opcode == 4)
        buffers_.erase(handle);
      else
        modules_.erase(handle);
      require(Backend::cleanup_ok(), "Native resource cleanup failed", 3);
    } else if (frame.opcode == 5) {
      const std::string format = reader.string();
      const std::string symbol = reader.string();
      const std::string path = reader.string();
      reader.end();
      require(Backend::supports_format(format),
              "Unsupported service payload format");
      require(symbol == "therock_module_saxpy" ||
                  symbol == "therock_module_relu",
              "Unsupported service entry point");
      require(!path.empty(), "Module payload path is empty");
      require(modules_.size() < 32, "Service module limit is 32");
      backend_->synchronize();
      const uint64_t handle = new_handle();
      modules_.emplace(handle, backend_->load(format, symbol, path));
      put64(result, handle);
    } else if (frame.opcode == 7 || frame.opcode == 8) {
      const uint64_t handle = reader.u64();
      const uint32_t offset = reader.u32();
      const uint32_t count = reader.u32();
      auto &entry = buffer(handle);
      require(count > 0 && uint64_t(offset) + count <= entry.capacity,
              "Buffer transfer is out of bounds");
      if (frame.opcode == 7) {
        std::vector<float> values(count);
        for (float &value : values)
          value = reader.f32();
        reader.end();
        backend_->write(*entry.object, offset, values);
      } else {
        reader.end();
        const auto values = backend_->read(*entry.object, offset, count);
        require(values.size() == count, "Backend read returned wrong count", 3);
        for (float value : values)
          put_float(result, value);
      }
    } else if (frame.opcode == 9) {
      const uint64_t kernel = reader.u64(), x = reader.u64(), y = reader.u64(),
                     output = reader.u64();
      const float alpha = reader.f32();
      const uint32_t count = reader.u32();
      reader.end();
      auto &function = module(kernel);
      auto &bx = buffer(x);
      auto &by = buffer(y);
      auto &bo = buffer(output);
      require(output != x && output != y, "Output must not alias an input");
      require(std::isfinite(alpha) && count > 0 && count <= kMaxCount,
              "Invalid launch scalar or count");
      require(count <= bx.capacity && count <= by.capacity &&
                  count <= bo.capacity,
              "Launch exceeds buffer capacity");
      backend_->launch(function, *bx.object, *by.object, *bo.object, alpha,
                       count);
    } else if (frame.opcode == 10) {
      reader.end();
      backend_->synchronize();
    }
    require(Backend::cleanup_ok(), "Native resource cleanup failed", 3);
    return result;
  }

private:
  uint64_t new_handle() {
    require(next_handle_ != std::numeric_limits<uint64_t>::max(),
            "Handle space exhausted", 4);
    return next_handle_++;
  }
  BufferEntry &buffer(uint64_t handle) {
    const auto found = buffers_.find(handle);
    require(found != buffers_.end(),
            "Unknown, released, or wrong-type buffer handle");
    return found->second;
  }
  typename Backend::Module &module(uint64_t handle) {
    const auto found = modules_.find(handle);
    require(found != modules_.end(),
            "Unknown, released, or wrong-type module handle");
    return *found->second;
  }
  bool hello_ = false;
  uint64_t next_handle_ = 1;
  std::unique_ptr<Backend> backend_;
  std::unordered_map<uint64_t, BufferEntry> buffers_;
  std::unordered_map<uint64_t, std::unique_ptr<typename Backend::Module>>
      modules_;
};

template <typename Backend> int serve() {
  // Preserve the protocol channel before redirecting legacy native diagnostics.
  const int output = ::dup(STDOUT_FILENO);
  if (output < 0 || ::dup2(STDERR_FILENO, STDOUT_FILENO) < 0)
    throw std::runtime_error("Cannot establish service protocol channel");
  State<Backend> state;
  uint32_t expected_id = 1;
  int result = 0;
  while (true) {
    Frame frame;
    try {
      if (!read_frame(frame, expected_id))
        break;
      require(expected_id != std::numeric_limits<uint32_t>::max(),
              "Request sequence exhausted", 2);
      ++expected_id;
      const auto response = state.dispatch(frame);
      respond(output, frame, 0, "", response);
      if (frame.opcode == 11)
        break;
    } catch (const CommandError &error) {
      const bool fatal =
          error.status == 2 || error.status == 3 || error.status == 5;
      if (fatal)
        state.close();
      respond(output, frame, error.status, error.what());
      if (fatal) {
        result = 1;
        break;
      }
    } catch (const std::exception &error) {
      state.close();
      respond(output, frame, 3, error.what());
      result = 1;
      break;
    }
  }
  state.close();
  ::close(output);
  return Backend::cleanup_ok() ? result : 1;
}
} // namespace therock::module_service
