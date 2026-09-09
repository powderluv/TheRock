# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU tests for native inventory serialization and real runner CLI gates.

The compiled-runner tests require THEROCK_NATIVE_MODULE_BUILD_ROOT and Linux.
An interposed shim handles initialization and zero-device responses entirely on
CPU; forbidden initialization exits immediately for exclusive-mode tests.
"""

import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest


class DeviceInventorySerializationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("A C++ compiler is required for serializer tests")
        temporary = tempfile.TemporaryDirectory(prefix="therock-inventory-json-")
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / "serialize.cpp"
        source.write_text(
            r"""#include "device_inventory.h"
#include <iostream>
#include <iterator>
int main(int argc, char **argv) {
  using namespace therock::module_validation;
  try {
    if (argc > 1) {
      DeviceInventoryEntry device;
      device.index = 7;
      const unsigned char uuid_bytes[16] = {0, 1, 2, 3, 4, 5, 6, 7,
                                            8, 9, 10, 11, 12, 13, 254, 255};
      device.device_uuid = device_uuid_hex(uuid_bytes);
      device.name = "GPU \"雪\"\\\n";
      device.vendor_id = 0x8086;
      device.device_id = 0xe223;
      device.max_threads_per_block = 1024;
      device.max_block_dimensions = {1024, 1024, 64};
      device.max_grid_dimensions = {2147483647, 65535, 65535};
      device.total_memory_bytes = 48ULL * 1024 * 1024 * 1024;
      if (std::string(argv[1]) == "invalid") {
        device.max_grid_dimensions[1] = 0;
      }
      if (std::string(argv[1]) == "zero-uuid") {
        const char zero[16]{};
        device.device_uuid = device_uuid_hex(zero);
      }
      if (std::string(argv[1]) == "duplicate-uuid") {
        DeviceInventoryEntry second = device;
        second.index = 8;
        std::cout << device_inventory_json("intel", "level-zero", {device, second});
      } else {
        std::cout << device_inventory_json("intel", "level-zero", {device});
      }
    } else {
      const std::string input(std::istreambuf_iterator<char>(std::cin), {});
      std::cout << json_string(input);
    }
  } catch (const std::exception &error) {
    std::cerr << error.what();
    return 1;
  }
}
"""
        )
        cls.executable = root / "serialize"
        include = Path(__file__).resolve().parents[2] / "tests/multi_vendor/modules"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(include),
                str(source),
                "-o",
                str(cls.executable),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

    def test_all_ascii_control_characters_round_trip(self) -> None:
        value = bytes(range(128))
        result = subprocess.run(
            [str(self.executable)], input=value, capture_output=True, timeout=5
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), value.decode("ascii"))
        self.assertFalse(any(byte < 0x20 for byte in result.stdout))

    def test_valid_utf8_names_round_trip(self) -> None:
        value = 'GPU "雪" \U0001f988 \u0080 \u07ff \u0800 \U0010ffff'
        result = subprocess.run(
            [str(self.executable)],
            input=value.encode("utf-8"),
            capture_output=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), value)

    def test_malformed_utf8_is_rejected_without_partial_json(self) -> None:
        for value in (
            b"\x80",
            b"\xc0\x80",
            b"\xe0\x80\x80",
            b"\xed\xa0\x80",
            b"\xf0\x80\x80\x80",
            b"\xf4\x90\x80\x80",
            b"\xf5\x80\x80\x80",
            b"\xe2\x82",
            b"\xe2bad",
        ):
            with self.subTest(value=value):
                result = subprocess.run(
                    [str(self.executable)],
                    input=value,
                    capture_output=True,
                    timeout=5,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b"")
                self.assertIn(b"not valid UTF-8", result.stderr)

    def test_inventory_schema_preserves_unknown_architecture_and_large_memory(
        self,
    ) -> None:
        result = subprocess.run(
            [str(self.executable), "inventory"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        inventory = json.loads(result.stdout)
        self.assertEqual(
            inventory,
            {
                "schema_version": 2,
                "kind": "native-device-inventory",
                "scope": "observed-runtime",
                "vendor": "intel",
                "backend": "level-zero",
                "devices": [
                    {
                        "index": 7,
                        "name": 'GPU "雪"\\\n',
                        "device_uuid": "000102030405060708090a0b0c0dfeff",
                        "architecture": None,
                        "vendor_id": 0x8086,
                        "device_id": 0xE223,
                        "limits": {
                            "max_threads_per_block": 1024,
                            "max_block_dimensions": [1024, 1024, 64],
                            "max_grid_dimensions": [2147483647, 65535, 65535],
                            "total_memory_bytes": 48 * 1024 * 1024 * 1024,
                        },
                    }
                ],
            },
        )

    def test_incomplete_limits_are_rejected_without_partial_json(self) -> None:
        result = subprocess.run(
            [str(self.executable), "invalid"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("incomplete inventory properties", result.stderr)

    def test_zero_and_duplicate_driver_uuids_are_rejected(self) -> None:
        for argument, message in (
            ("zero-uuid", "must be 32 lowercase hex characters and nonzero"),
            ("duplicate-uuid", "Duplicate device UUID"),
        ):
            with self.subTest(argument=argument):
                result = subprocess.run(
                    [str(self.executable), argument],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertIn(message, result.stderr)


class NativeDeviceInventoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = os.environ.get("THEROCK_NATIVE_MODULE_BUILD_ROOT")
        if not root or not sys.platform.startswith("linux"):
            raise unittest.SkipTest(
                "Set THEROCK_NATIVE_MODULE_BUILD_ROOT on Linux for CPU runner gates"
            )
        cls.runners = sorted(Path(root).glob("*/build/therock_module_validation"))
        if not cls.runners:
            raise AssertionError(f"No compiled native runners found beneath {root}")
        compiler = shutil.which("cc")
        if compiler is None:
            raise AssertionError("A C compiler is required for the GPU API shim")
        temporary = tempfile.TemporaryDirectory(prefix="therock-native-inventory-")
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        source = cls.root / "inventory_api_shim.c"
        source.write_text(
            r"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int mode(void) {
  const char *value = getenv("THEROCK_INVENTORY_CPU_SHIM_MODE");
  if (value && !strcmp(value, "empty")) return 1;
  if (value && !strcmp(value, "error")) return 2;
  if (value && !strcmp(value, "identity")) return 3;
  if (value && !strcmp(value, "zero-uuid")) return 4;
  fputs("FORBIDDEN_GPU_API_CALL\n", stderr);
  fflush(stderr);
  _Exit(91);
}
static int initialized(void) { return mode() == 2 ? 3 : 0; }
static int forbidden_resource(void) {
  fputs("FORBIDDEN_GPU_RESOURCE_CALL\n", stderr);
  fflush(stderr);
  _Exit(92);
}
static int device_uuid(void *bytes) {
  unsigned char uuid[16] = {0, 1, 2, 3, 4, 5, 6, 7,
                           8, 9, 10, 11, 12, 13, 254, 255};
  if (mode() == 4) memset(uuid, 0, sizeof(uuid));
  memcpy(bytes, uuid, sizeof(uuid));
  return initialized();
}
int cuInit(unsigned flags) { return initialized(); }
int cuDeviceGetCount(int *count) {
  *count = mode() >= 3 ? 1 : 0;
  return initialized();
}
int cuDeviceGet(int *device, int index) {
  *device = index;
  return initialized();
}
int cuDeviceGetUuid_v2(void *uuid, int device) { return device_uuid(uuid); }
int cuDeviceGetUuid(void *uuid, int device) { return device_uuid(uuid); }
int cuDeviceGetAttribute(int *value, int attribute, int device) {
  return forbidden_resource();
}
int cuDevicePrimaryCtxRetain(void **context, int device) {
  return forbidden_resource();
}
int hipInit(unsigned flags) { return initialized(); }
int hipGetDeviceCount(int *count) {
  *count = mode() >= 3 ? 1 : 0;
  return initialized();
}
int hipDeviceGetUuid(void *uuid, int device) { return device_uuid(uuid); }
int hipSetDevice(int device) { return forbidden_resource(); }
#ifndef THEROCK_CPU_INTEL_UUID_SHIM
int zeInit(unsigned flags) { return mode() == 1 ? 0 : 0x78000001; }
int zeDriverGet(unsigned *count, void *drivers) {
  *count = 0;
  return mode() == 1 ? 0 : 0x78000001;
}
#else
#include <level_zero/ze_api.h>
#include <stdint.h>
ze_result_t zeInit(ze_init_flags_t flags) {
  return mode() == 2 ? ZE_RESULT_ERROR_UNINITIALIZED : ZE_RESULT_SUCCESS;
}
ze_result_t zeDriverGet(uint32_t *count, ze_driver_handle_t *drivers) {
  *count = mode() >= 3 ? 1 : 0;
  if (drivers && *count) drivers[0] = (ze_driver_handle_t)(uintptr_t)1;
  return ZE_RESULT_SUCCESS;
}
ze_result_t zeDriverGetProperties(ze_driver_handle_t driver,
                                 ze_driver_properties_t *properties) {
  properties->uuid.id[0] = 1;
  return ZE_RESULT_SUCCESS;
}
ze_result_t zeDeviceGet(ze_driver_handle_t driver, uint32_t *count,
                       ze_device_handle_t *devices) {
  *count = 1;
  if (devices) devices[0] = (ze_device_handle_t)(uintptr_t)1;
  return ZE_RESULT_SUCCESS;
}
ze_result_t zeDeviceGetProperties(ze_device_handle_t device,
                                 ze_device_properties_t *properties) {
  properties->type = ZE_DEVICE_TYPE_GPU;
  properties->vendorId = 0x8086;
  properties->deviceId = 0xe223;
  device_uuid(properties->uuid.id);
  return ZE_RESULT_SUCCESS;
}
ze_result_t zeDriverGetApiVersion(ze_driver_handle_t driver,
                                 ze_api_version_t *version) {
  return (ze_result_t)forbidden_resource();
}
ze_result_t zeContextCreate(ze_driver_handle_t driver,
                           const ze_context_desc_t *description,
                           ze_context_handle_t *context) {
  return (ze_result_t)forbidden_resource();
}
#endif
"""
        )
        shim = cls.root / "inventory_api_shim.so"
        subprocess.run(
            [compiler, "-shared", "-fPIC", str(source), "-o", str(shim)],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        cls.environment = dict(os.environ)
        previous = cls.environment.get("LD_PRELOAD", "")
        cls.environment["LD_PRELOAD"] = str(shim) + (":" + previous if previous else "")
        cls.descriptions = {}
        for runner in cls.runners:
            result = subprocess.run(
                [str(runner), "--describe-contract"],
                env=cls.environment,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            cls.descriptions[runner] = json.loads(result.stdout)
        cls.intel_environments = {}
        for runner, description in cls.descriptions.items():
            if description["vendor"] != "intel":
                continue
            cache = runner.parent / "CMakeCache.txt"
            if not cache.exists():
                continue
            roots = [
                line.split("=", 1)[1]
                for line in cache.read_text().splitlines()
                if line.startswith("THEROCK_MODULE_SDK_ROOT:PATH=")
            ]
            if not roots:
                continue
            include = Path(roots[0]) / "include"
            intel_shim = (
                cls.root / f"intel_uuid_api_shim_{len(cls.intel_environments)}.so"
            )
            subprocess.run(
                [
                    compiler,
                    "-shared",
                    "-fPIC",
                    "-DTHEROCK_CPU_INTEL_UUID_SHIM=1",
                    "-I",
                    str(include),
                    str(source),
                    "-o",
                    str(intel_shim),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            cls.intel_environments[runner] = cls.environment | {
                "LD_PRELOAD": str(intel_shim) + (":" + previous if previous else "")
            }

    def invoke(
        self, runner: Path, *arguments: str, mode: str = "trap"
    ) -> subprocess.CompletedProcess[str]:
        base = self.intel_environments.get(runner, self.environment)
        environment = base | {"THEROCK_INVENTORY_CPU_SHIM_MODE": mode}
        return subprocess.run(
            [str(runner), *arguments],
            env=environment,
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_discovery_conflicts_are_rejected_before_driver_access(self) -> None:
        for runner in self.runners:
            for extra in (
                ("--payload", str(self.root / "missing")),
                ("--device", "0"),
                ("--help",),
                ("--describe-contract",),
                ("--list-devices",),
                ("--unknown",),
            ):
                for arguments in (
                    ("--list-devices", *extra),
                    (*extra, "--list-devices"),
                ):
                    with self.subTest(runner=runner, arguments=arguments):
                        result = self.invoke(runner, *arguments)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertNotEqual(result.returncode, 91, result.stderr)
                        self.assertNotIn("FORBIDDEN_GPU_API_CALL", result.stderr)
                        self.assertIn("must be used alone", result.stderr)
                        self.assertEqual(result.stdout, "")

    def test_exclusive_discovery_reaches_the_driver_entrypoint(self) -> None:
        for runner in self.runners:
            with self.subTest(runner=runner):
                result = self.invoke(runner, "--list-devices")
                self.assertEqual(result.returncode, 91, result.stderr)
                self.assertIn("FORBIDDEN_GPU_API_CALL", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_enumeration_errors_do_not_publish_empty_success(self) -> None:
        for runner in self.runners:
            with self.subTest(runner=runner):
                result = self.invoke(runner, "--list-devices", mode="error")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("FAIL backend=", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_successful_empty_device_lists_are_distinct_from_no_intel_driver(
        self,
    ) -> None:
        for runner, description in self.descriptions.items():
            with self.subTest(runner=runner):
                result = self.invoke(runner, "--list-devices", mode="empty")
                if description["vendor"] == "intel":
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn("found no GPU drivers", result.stderr)
                    self.assertEqual(result.stdout, "")
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stderr, "")
                    self.assertEqual(
                        json.loads(result.stdout),
                        {
                            "schema_version": 2,
                            "kind": "native-device-inventory",
                            "scope": "observed-runtime",
                            "vendor": description["vendor"],
                            "backend": description["backend"],
                            "devices": [],
                        },
                    )

    def test_description_remains_offline(self) -> None:
        for runner, description in self.descriptions.items():
            with self.subTest(runner=runner):
                result = self.invoke(runner, "--describe-contract")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, "")
                self.assertEqual(json.loads(result.stdout), description)
                self.assertNotIn("devices", description)

    def contract_arguments(self, runner: Path) -> list[str]:
        description = self.descriptions[runner]
        return [
            "--launch-abi",
            description["contract"]["abi"],
            "--launch-abi-version",
            str(description["contract"]["version"]),
            "--launch-contract-sha256",
            description["contract_sha256"],
            "--payload-format",
            description["payload_types"][0],
        ]

    def test_malformed_expected_uuids_fail_before_payload_or_driver_access(
        self,
    ) -> None:
        for runner in self.runners:
            for uuid in (
                "",
                "0" * 32,
                "a" * 31,
                "a" * 33,
                "A" * 32,
                "g" * 32,
                " a" + "0" * 30,
                "00010203-0405-0607-0809-0a0b0c0dfeff",
            ):
                with self.subTest(runner=runner, uuid=uuid):
                    result = self.invoke(
                        runner,
                        "--payload",
                        str(self.root / "missing"),
                        *self.contract_arguments(runner),
                        "--expect-device-uuid",
                        uuid,
                    )
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn("Device UUID must be", result.stderr)
                    self.assertNotIn("FORBIDDEN_GPU", result.stderr)
                    self.assertNotIn("Cannot read payload", result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_duplicate_expected_uuid_flag_is_rejected_before_driver(self) -> None:
        for runner in self.runners:
            with self.subTest(runner=runner):
                result = self.invoke(
                    runner,
                    "--payload",
                    str(self.root / "missing"),
                    *self.contract_arguments(runner),
                    "--expect-device-uuid",
                    "1" * 32,
                    "--expect-device-uuid",
                    "1" * 32,
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("Duplicate argument --expect-device-uuid", result.stderr)
                self.assertNotIn("FORBIDDEN_GPU", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_valid_expected_uuid_preserves_payload_validation_order(self) -> None:
        for runner in self.runners:
            with self.subTest(runner=runner):
                result = self.invoke(
                    runner,
                    "--payload",
                    str(self.root / "missing"),
                    *self.contract_arguments(runner),
                    "--expect-device-uuid",
                    "1" * 32,
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("Cannot read payload", result.stderr)
                self.assertNotIn("FORBIDDEN_GPU", result.stderr)

    def test_mismatched_or_unavailable_driver_uuid_stops_before_context(self) -> None:
        for runner, description in self.descriptions.items():
            if (
                description["vendor"] == "intel"
                and runner not in self.intel_environments
            ):
                continue
            if description["vendor"] == "intel":
                payload_bytes = struct.pack(
                    "<6I", 0x07230203, 0x10000, 0, 1, 0, 0x10000
                )
            else:
                payload_bytes = bytearray(64)
                payload_bytes[:7] = b"\x7fELF\x02\x01\x01"
                payload_bytes[18:20] = struct.pack(
                    "<H", 224 if description["vendor"] == "amd" else 190
                )
            payload = self.root / "identity-test-payload"
            payload.write_bytes(payload_bytes)
            for mode, message in (
                ("identity", "Device UUID mismatch:"),
                ("zero-uuid", "Device UUID must be"),
            ):
                with self.subTest(runner=runner, mode=mode):
                    result = self.invoke(
                        runner,
                        "--payload",
                        str(payload),
                        *self.contract_arguments(runner),
                        "--expect-device-uuid",
                        "1" * 32,
                        mode=mode,
                    )
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn(message, result.stderr)
                    self.assertNotIn("FORBIDDEN_GPU", result.stderr)
                    self.assertEqual(result.stdout, "")
                    if mode == "identity":
                        self.assertIn("expected=" + "1" * 32, result.stderr)
                        self.assertIn(
                            "observed=000102030405060708090a0b0c0dfeff", result.stderr
                        )

    def test_help_documents_runtime_discovery(self) -> None:
        for runner in self.runners:
            with self.subTest(runner=runner):
                result = self.invoke(runner, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--list-devices", result.stdout)


if __name__ == "__main__":
    unittest.main()
