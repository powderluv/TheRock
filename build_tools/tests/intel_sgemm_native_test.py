# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU control-flow tests for the real Intel oneMKL resource owner.

Minimal SYCL/oneMKL substitutes model native identity, queued arithmetic, and
failures. These tests do not establish SDK ABI or Intel hardware compatibility.
"""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _therock_utils.sgemm_contract import parse_sgemm_provider_json


_HARNESS = r"""
#include "module_service_protocol.h"
#include <array>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#define THEROCK_MODULE_ENABLE_SGEMM 1
#define SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO 1
#define SYCL_EXT_ONEAPI_QUEUE_EMPTY 1
#define SYCL_EXT_ONEAPI_PROD 1
using ze_driver_handle_t = int;
using ze_device_handle_t = int;
using ze_context_handle_t = int;
constexpr uint64_t kWaitTimeoutNs = 1000000;
bool cleanup_failed = false;
std::string scenario;

void event(const std::string &value) {
  std::fprintf(stderr, "EVENT %s\n", value.c_str());
  std::fflush(stderr);
}
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

namespace sycl {
enum class backend { ext_oneapi_level_zero, other };
enum class aspect { usm_device_allocations };
namespace usm { enum class alloc { unknown, host, device, shared }; }
namespace ext::oneapi::level_zero { enum class ownership { keep, transfer }; }
using exception_list = std::vector<std::exception_ptr>;
using async_handler = std::function<void(exception_list)>;
struct platform {
  int native;
  bool operator!=(const platform &other) const { return native != other.native; }
};
struct DeviceState {
  explicit DeviceState(int id) : native(id) { event("CREATE_SYCL_DEVICE"); }
  ~DeviceState() { event("DESTROY_SYCL_DEVICE"); }
  int native;
};
struct device {
  explicit device(int id) : state(std::make_shared<DeviceState>(id)) {}
  platform get_platform() const { return {scenario == "device-platform" ? 2 : 1}; }
  bool is_gpu() const { return scenario != "device-cpu"; }
  bool has(aspect) const { return scenario != "device-usm"; }
  bool operator!=(const device &other) const {
    return native() != other.native();
  }
  int native() const { return override_native ? override_native : state->native; }
  std::shared_ptr<DeviceState> state;
  int override_native = 0;
};
struct ContextInput {
  int native;
  std::vector<device> devices;
  ext::oneapi::level_zero::ownership ownership;
};
struct ContextState {
  explicit ContextState(ContextInput input) : input(std::move(input)) {
    event("CREATE_SYCL_CONTEXT");
  }
  ~ContextState() {
    event(input.ownership == ext::oneapi::level_zero::ownership::keep
              ? "DESTROY_SYCL_CONTEXT_KEEP" : "DESTROY_NATIVE_CONTEXT_WRONG");
  }
  ContextInput input;
};
struct context {
  explicit context(ContextInput input)
      : state(std::make_shared<ContextState>(std::move(input))) {}
  int native() const {
    return override_native ? override_native : state->input.native;
  }
  std::shared_ptr<ContextState> state;
  int override_native = 0;
};
template <backend> platform make_platform(int driver) {
  event("MAKE_PLATFORM");
  return {driver};
}
template <backend> device make_device(int native) {
  return device(scenario == "device-native" ? native + 1 : native);
}
template <backend> context make_context(ContextInput input, async_handler) {
  require(input.ownership == ext::oneapi::level_zero::ownership::keep,
          "Native context ownership must remain with the worker");
  require(input.devices.size() == 1 && input.devices[0].native() == 10,
          "Wrong device in borrowed context");
  event("BORROW_CONTEXT_KEEP");
  if (scenario == "context-construct") throw std::runtime_error("context construction failed");
  if (scenario == "context-native") input.native += 1;
  return context(std::move(input));
}
template <backend> int get_native(const device &value) { return value.native(); }
template <backend> int get_native(const context &value) { return value.native(); }
namespace property::queue { struct in_order {}; }
struct property_list {
  explicit property_list(property::queue::in_order) {}
};
class queue {
public:
  queue(context ctx, device dev, async_handler handler, property_list)
      : context_(std::move(ctx)), device_(std::move(dev)), handler_(std::move(handler)) {
    event("CREATE_SYCL_QUEUE_IN_ORDER");
    if (scenario == "queue-construct") throw std::runtime_error("queue construction failed");
  }
  ~queue() {
    event(pending.empty() ? "DESTROY_SYCL_QUEUE" : "DESTROY_IN_FLIGHT_QUEUE");
  }
  backend get_backend() const {
    return scenario == "queue-backend" ? backend::other : backend::ext_oneapi_level_zero;
  }
  context get_context() const {
    auto result = context_;
    if (scenario == "queue-context") result.override_native = 101;
    return result;
  }
  device get_device() const {
    auto result = device_;
    if (scenario == "queue-device") result.override_native = 11;
    return result;
  }
  void ext_oneapi_prod() {
    event("QUEUE_PROD");
    if (scenario == "prod-failure" && !pending.empty())
      throw std::runtime_error("queue progress failed");
  }
  bool ext_oneapi_empty() {
    event("QUERY_QUEUE");
    if (pending.empty()) return true;
    if (scenario == "query-failure") throw std::runtime_error("queue query failed");
    if (scenario == "query-unknown") throw 42;
    if (scenario == "timeout") return false;
    if (!queried_pending_) {
      queried_pending_ = true;
      event("QUEUED_NOT_COMPLETE");
      return false;
    }
    for (const auto &operation : pending) operation();
    pending.clear();
    queried_pending_ = false;
    if (scenario == "async-failure")
      asynchronous_.push_back(std::make_exception_ptr(std::runtime_error("deferred BLAS error")));
    event("QUEUE_COMPLETE");
    return true;
  }
  void throw_asynchronous() {
    if (scenario == "init-async" && !initial_async_sent_) {
      initial_async_sent_ = true;
      asynchronous_.push_back(std::make_exception_ptr(std::runtime_error("initial async error")));
    }
    if (!asynchronous_.empty()) {
      event("DELIVER_ASYNC_ERROR");
      handler_(asynchronous_);
      asynchronous_.clear();
    }
  }
  std::vector<std::function<void()>> pending;
private:
  context context_;
  device device_;
  async_handler handler_;
  exception_list asynchronous_;
  bool queried_pending_ = false;
  bool initial_async_sent_ = false;
};
struct Allocation {
  int context;
  int device;
  usm::alloc kind;
};
std::map<const void *, Allocation> allocations;
usm::alloc get_pointer_type(const void *base, const context &ctx) {
  const auto found = allocations.find(base);
  if (found == allocations.end() || found->second.context != ctx.native())
    return usm::alloc::unknown;
  return found->second.kind;
}
device get_pointer_device(const void *base, const context &ctx) {
  require(get_pointer_type(base, ctx) == usm::alloc::device, "Unknown allocation");
  device result = ctx.state->input.devices[0];
  result.override_native = allocations.at(base).device;
  return result;
}
} // namespace sycl

struct MKLVersion { int MajorVersion, UpdateVersion, PatchVersion; };
void mkl_get_version(MKLVersion *version) {
  event("OBSERVE_MKL_VERSION");
  *version = {2026, 0, 1};
  if (scenario == "invalid-version") version->MajorVersion = 0;
}
namespace oneapi::mkl {
enum class transpose { nontrans, trans };
namespace blas {
enum class compute_mode { standard, reduced };
namespace column_major {
void gemm(sycl::queue &queue, transpose ta, transpose tb,
          int64_t m, int64_t n, int64_t k, float alpha,
          const float *a, int64_t lda, const float *b, int64_t ldb,
          float beta, float *c, int64_t ldc, compute_mode mode) {
  require(ta == transpose::nontrans && tb == transpose::nontrans, "Wrong transpose");
  require(mode == compute_mode::standard, "Reduced precision was allowed");
  event("SUBMIT_STANDARD_NN");
  require(c[0] == 8, "Output changed before completion");
  queue.pending.push_back([=]() {
    event("EXECUTE_SGEMM");
    for (int64_t column = 0; column < n; ++column) {
      for (int64_t row = 0; row < m; ++row) {
        float product = 0;
        for (int64_t inner = 0; inner < k; ++inner)
          product += a[row + inner * lda] * b[inner + column * ldb];
        const auto index = row + column * ldc;
        c[index] = alpha * product + beta * c[index];
      }
    }
  });
  event("SUBMITTED_OUTPUT_UNCHANGED");
  if (scenario == "partial-submit") throw std::runtime_error("partial BLAS submission failed");
}
} // namespace column_major
} // namespace blas
} // namespace oneapi::mkl

#include "module_service_onemkl.h"

struct NativeResources {
  ~NativeResources() {
    event("DESTROY_NATIVE_BUFFERS");
    event("DESTROY_NATIVE_CONTEXT");
  }
};
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  scenario = argv[1];
  NativeResources native;
  std::array<float, 16> a, b, c;
  a.fill(-77); b.fill(-77); c.fill(-77);
  a[1] = 1; a[2] = 2; a[4] = 3; a[5] = 4;
  b[2] = 5; b[3] = 6; b[6] = 7; b[7] = 8;
  c[3] = c[4] = c[8] = c[9] = 8;
  const auto original_a = a, original_b = b;
  for (const auto *pointer : {a.data(), b.data(), c.data()})
    sycl::allocations[pointer] = {100, 10, sycl::usm::alloc::device};
  if (scenario == "a-shared") sycl::allocations[a.data()].kind = sycl::usm::alloc::shared;
  if (scenario == "a-host") sycl::allocations[a.data()].kind = sycl::usm::alloc::host;
  if (scenario == "b-context") sycl::allocations[b.data()].context = 101;
  if (scenario == "c-device") sycl::allocations[c.data()].device = 11;
  const therock::module_service::SgemmRequest request{1, 2, 3, 2, 2, 2, 3, 4, 5, 2, 0.5};
  {
    ServiceOneMkl provider(1, 10, 100);
    try {
      if (scenario != "unnegotiated") {
        const auto info = provider.info();
        std::cout << info << std::endl;
        require(provider.info() == info, "Provider information was not cached");
      }
      provider.sgemm(scenario == "null-a" ? nullptr : a.data(), b.data(), c.data(), request);
      event("SGEMM_RETURNED");
    } catch (const std::exception &error) {
      event(std::string("CAUGHT:") + error.what());
      if (scenario.find("device-") == 0 || scenario.find("context-") == 0 ||
          scenario.find("queue-") == 0 || scenario == "invalid-version" || scenario == "init-async") {
        try { provider.info(); }
        catch (const std::exception &retry) { event(std::string("RETRY:") + retry.what()); }
      }
    }
    event(std::string("INPUTS_") + (a == original_a && b == original_b ? "UNCHANGED" : "CORRUPTED"));
    const auto expected = std::array<float, 16>{-77, -77, -77, 50, 72, -77, -77, -77, 66, 96, -77, -77, -77, -77, -77, -77};
    event(c == expected ? "RESULT_AND_GUARDS_CORRECT" : "NO_COMPLETE_RESULT");
  }
  event(std::string("CLEANUP_FAILED=") + (cleanup_failed ? "1" : "0"));
  return 0;
}
"""


class IntelSgemmNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        compiler = shutil.which("c++")
        if compiler is None or sys.platform == "win32":
            raise unittest.SkipTest("A POSIX host and C++ compiler are required")
        temporary = tempfile.TemporaryDirectory(prefix="therock-intel-sgemm-native-")
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name)
        repository = Path(__file__).resolve().parents[2]
        modules = repository / "tests/multi_vendor/modules"
        subprocess.run(
            [
                sys.executable,
                str(repository / "build_tools/configure_module_contract.py"),
                "--vendor",
                "intel",
                "--enable-sgemm",
                "--header",
                str(root / "module_contract_data.h"),
                "--description",
                str(root / "description.json"),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        source = root / "intel_sgemm_test.cpp"
        source.write_text(_HARNESS)
        cls.executable = root / "intel_sgemm_test"
        result = subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pthread",
                "-I",
                str(root),
                "-I",
                str(modules),
                str(source),
                "-o",
                str(cls.executable),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(result.stderr)

    def run_case(
        self, scenario: str
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        result = subprocess.run(
            [str(self.executable), scenario], capture_output=True, text=True, timeout=10
        )
        events = [
            line.removeprefix("EVENT ")
            for line in result.stderr.splitlines()
            if line.startswith("EVENT ")
        ]
        return result, events

    def assert_safe_cleanup(self, events: list[str]) -> None:
        self.assertNotIn("DESTROY_IN_FLIGHT_QUEUE", events)
        self.assertNotIn("DESTROY_NATIVE_CONTEXT_WRONG", events)
        self.assertEqual(events.count("DESTROY_NATIVE_CONTEXT"), 1)
        self.assertEqual(events.count("DESTROY_NATIVE_BUFFERS"), 1)
        if "DESTROY_SYCL_QUEUE" in events:
            self.assertLess(
                events.index("DESTROY_SYCL_QUEUE"),
                events.index("DESTROY_SYCL_CONTEXT_KEEP"),
            )
        if "DESTROY_SYCL_CONTEXT_KEEP" in events:
            self.assertLess(
                events.index("DESTROY_SYCL_CONTEXT_KEEP"),
                events.index("DESTROY_SYCL_DEVICE"),
            )
        if "DESTROY_SYCL_DEVICE" in events:
            self.assertLess(
                events.index("DESTROY_SYCL_DEVICE"),
                events.index("DESTROY_NATIVE_CONTEXT"),
            )
        if "EXECUTE_SGEMM" in events:
            self.assertLess(
                events.index("EXECUTE_SGEMM"), events.index("DESTROY_NATIVE_BUFFERS")
            )

    def test_standard_sgemm_borrows_context_and_completes_queued_arithmetic(
        self,
    ) -> None:
        result, events = self.run_case("success")
        self.assertEqual(result.returncode, 0, result.stderr)
        info = parse_sgemm_provider_json(result.stdout.strip(), vendor="intel")
        self.assertEqual(info.provider, "onemkl")
        self.assertEqual(info.library_version, "2026.0.1")
        self.assertEqual(events.count("MAKE_PLATFORM"), 1)
        self.assertEqual(events.count("OBSERVE_MKL_VERSION"), 1)
        self.assertEqual(events.count("BORROW_CONTEXT_KEEP"), 1)
        self.assertIn("CREATE_SYCL_QUEUE_IN_ORDER", events)
        ordered = [
            "SUBMIT_STANDARD_NN",
            "SUBMITTED_OUTPUT_UNCHANGED",
            "QUEUED_NOT_COMPLETE",
            "EXECUTE_SGEMM",
            "QUEUE_COMPLETE",
            "SGEMM_RETURNED",
        ]
        self.assertEqual(sorted(ordered, key=events.index), ordered)
        self.assertIn("RESULT_AND_GUARDS_CORRECT", events)
        self.assertIn("INPUTS_UNCHANGED", events)
        self.assertIn("CLEANUP_FAILED=0", events)
        self.assert_safe_cleanup(events)

    def test_exact_native_identity_and_partial_initialization_cleanup(self) -> None:
        for scenario in (
            "device-platform",
            "device-native",
            "device-cpu",
            "device-usm",
            "context-native",
            "queue-backend",
            "queue-context",
            "queue-device",
            "context-construct",
            "queue-construct",
            "invalid-version",
        ):
            with self.subTest(scenario=scenario):
                result, events = self.run_case(scenario)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(
                    any(event.startswith("CAUGHT:") for event in events), result.stderr
                )
                self.assertIn("RETRY:oneMKL initialization previously failed", events)
                self.assertNotIn("SUBMIT_STANDARD_NN", events)
                self.assertEqual(result.stdout, "")
                self.assertEqual(events.count("MAKE_PLATFORM"), 1)
                self.assertEqual(events.count("DESTROY_SYCL_DEVICE"), 1)
                self.assert_safe_cleanup(events)

    def test_unnegotiated_call_does_not_initialize_provider(self) -> None:
        result, events = self.run_case("unnegotiated")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CAUGHT:SGEMM provider has not been negotiated", events)
        self.assertNotIn("MAKE_PLATFORM", events)
        self.assertNotIn("SUBMIT_STANDARD_NN", events)
        self.assert_safe_cleanup(events)

    def test_usm_allocations_require_exact_native_context_device_and_device_kind(
        self,
    ) -> None:
        for scenario in ("null-a", "a-shared", "a-host", "b-context", "c-device"):
            with self.subTest(scenario=scenario):
                result, events = self.run_case(scenario)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(
                    any(
                        "requires a native device allocation" in event
                        for event in events
                    )
                )
                self.assertNotIn("SUBMIT_STANDARD_NN", events)
                self.assertNotIn("EXECUTE_SGEMM", events)
                self.assertIn("INPUTS_UNCHANGED", events)
                self.assert_safe_cleanup(events)

    def test_partial_submission_drains_before_exception_and_resource_cleanup(
        self,
    ) -> None:
        result, events = self.run_case("partial-submit")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CAUGHT:partial BLAS submission failed", events)
        self.assertLess(
            events.index("EXECUTE_SGEMM"),
            events.index("CAUGHT:partial BLAS submission failed"),
        )
        self.assertIn("RESULT_AND_GUARDS_CORRECT", events)
        self.assertNotIn("SGEMM_RETURNED", events)
        self.assertIn("CLEANUP_FAILED=0", events)
        self.assert_safe_cleanup(events)

    def test_async_failures_are_preserved_after_completion_and_reported_in_cleanup(
        self,
    ) -> None:
        for scenario, message in (
            ("async-failure", "deferred BLAS error"),
            ("init-async", "initial async error"),
        ):
            with self.subTest(scenario=scenario):
                result, events = self.run_case(scenario)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("DELIVER_ASYNC_ERROR", events)
                self.assertIn(f"CAUGHT:{message}", events)
                self.assertNotIn("SGEMM_RETURNED", events)
                self.assertIn("CLEANUP_FAILED=1", events)
                self.assertIn("FAIL cleanup=oneMKL", result.stderr)
                if scenario == "async-failure":
                    self.assertLess(
                        events.index("QUEUE_COMPLETE"),
                        events.index("DELIVER_ASYNC_ERROR"),
                    )
                    self.assertIn("RESULT_AND_GUARDS_CORRECT", events)
                self.assert_safe_cleanup(events)

    def test_unknown_queue_completion_fail_stops_without_freeing_live_resources(
        self,
    ) -> None:
        for scenario, reason in (
            ("query-failure", "queue query failed"),
            ("query-unknown", "queue completion query failed"),
            ("prod-failure", "queue progress failed"),
            ("timeout", "queue deadline exceeded"),
        ):
            with self.subTest(scenario=scenario):
                result, events = self.run_case(scenario)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("completion=unknown", result.stderr)
                self.assertIn(reason, result.stderr)
                self.assertIn("SUBMITTED_OUTPUT_UNCHANGED", events)
                self.assertNotIn("EXECUTE_SGEMM", events)
                self.assertNotIn("SGEMM_RETURNED", events)
                self.assertFalse(
                    any(event.startswith("DESTROY_") for event in events), result.stderr
                )
                self.assertFalse(
                    any(event.startswith("CAUGHT:") for event in events), result.stderr
                )


if __name__ == "__main__":
    unittest.main()
