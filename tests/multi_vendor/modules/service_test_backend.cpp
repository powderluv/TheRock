// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CPU-only backend for exercising the real framed worker server. No SDK is
// linked. Deferred work intentionally references live buffers and modules.
#include "module_service_protocol.h"

#include <algorithm>
#include <cstdlib>
#include <functional>

namespace {
size_t pending_work = 0;
void audit(const char *event) { std::fprintf(stderr, "CPU %s\n", event); }
void require_drained() {
  if (pending_work != 0) {
    audit("DESTRUCTION_WHILE_PENDING");
    std::fflush(stderr);
    std::_Exit(93);
  }
}

class CpuBackend {
public:
  struct Buffer {
    explicit Buffer(uint32_t count) : values(count) {}
    ~Buffer() {
      require_drained();
      audit("DESTROY_BUFFER");
    }
    std::vector<float> values;
  };
  struct Module {
    explicit Module(const std::string &name) : symbol(name) {}
    ~Module() {
      require_drained();
      audit("DESTROY_MODULE");
    }
    std::string symbol;
  };
  explicit CpuBackend(const therock::module_service::Config &config) {
    audit("OPEN");
    if (config.device != 3 || config.device_id != 0 ||
        config.expected_arch != "sm_120" ||
        config.device_uuid != std::string(32, '1')) {
      throw std::runtime_error("CPU backend received incorrect OPEN fields");
    }
    // The server must redirect ordinary backend stdout away from binary frames.
    std::puts("CPU BACKEND_STDOUT");
  }
  ~CpuBackend() {
    require_drained();
    audit("DESTROY_BACKEND");
  }
  static bool supports_format(const std::string &format) {
    return format == "cubin" || format == "ptx";
  }
  std::unique_ptr<Buffer> allocate(uint32_t capacity) {
    audit("ALLOCATE");
    return std::make_unique<Buffer>(capacity);
  }
  std::unique_ptr<Module> load(const std::string &, const std::string &symbol,
                               const std::string &path) {
    audit("LOAD");
    if (path == "throw-load") {
      throw std::runtime_error("Injected CPU backend load failure");
    }
    return std::make_unique<Module>(symbol);
  }
  void write(Buffer &buffer, uint32_t offset,
             const std::vector<float> &values) {
    audit("WRITE");
    synchronize();
    std::copy(values.begin(), values.end(), buffer.values.begin() + offset);
  }
  std::vector<float> read(Buffer &buffer, uint32_t offset, uint32_t count) {
    audit("READ");
    synchronize();
    return {buffer.values.begin() + offset,
            buffer.values.begin() + offset + count};
  }
  void launch(Module &module, Buffer &x, Buffer &y, Buffer &output, float alpha,
              uint32_t count) {
    audit("LAUNCH");
    work_.push_back([&module, &x, &y, &output, alpha, count] {
      audit("EXECUTE");
      for (uint32_t i = 0; i < count; ++i) {
        float value = alpha * x.values[i] + y.values[i];
        if (module.symbol == "therock_module_relu") {
          value = std::max(0.0f, value - 0.125f);
        }
        output.values[i] = value;
      }
    });
    ++pending_work;
    if (alpha == 13.0f) {
      throw std::runtime_error("Injected CPU failure after enqueue");
    }
  }
  void synchronize() noexcept {
    audit("SYNCHRONIZE");
    for (auto &operation : work_) {
      operation();
    }
    work_.clear();
    pending_work = 0;
  }
  static bool cleanup_ok() { return true; }

private:
  std::vector<std::function<void()>> work_;
};
} // namespace

int main(int argc, char **argv) {
  try {
    if (!therock::module_service::requested(argc, argv)) {
      throw std::runtime_error("CPU test backend requires --serve");
    }
    return therock::module_service::serve<CpuBackend>();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "CPU ERROR %s\n", error.what());
    return 1;
  }
}
