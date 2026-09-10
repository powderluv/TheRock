# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the installed provider-only consumer with independent CPU workers."""

import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sgemm_example_test import CpuSession, float32

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY / "build_tools/therock_multi_vendor"
AMD = "amd:hip:gfx1201"
NVIDIA = "nvidia:cuda:sm_120"
AMD_UUID = "a" * 32
NVIDIA_UUID = "b" * 32

# Execute the installed entry point in an isolated process. The public runtime
# below substitutes independent FP32 workers, while bootstrap/origins, numerical
# fixtures, argument parsing, reporting, and peer exchange are production code.
_COMMAND = r"""
import json, runpy, sys, tempfile
example = runpy.run_path(sys.argv[1])
def no_temporary_payloads(*args, **kwargs):
    raise AssertionError("Provider-only consumer created temporary payloads")
tempfile.TemporaryDirectory = no_temporary_payloads
tempfile.mkdtemp = no_temporary_payloads
assert example["main"](sys.argv[2:]) == 0
runtime = sys.modules["therock_multi_vendor"]
print(json.dumps(runtime.observations(), sort_keys=True))
"""

_RUNTIME = r"""
import json
from .cpu_session import ProviderOnlySession
calls = []
sessions = []
def open_sgemm_session(root, target, *, device_index=None, device_uuid=None,
                       expected_device_id=None):
    calls.append({
        "target": target,
        "device_index": device_index,
        "device_uuid": device_uuid,
        "prior_worker_live": any(not session.closed for session in sessions),
    })
    if expected_device_id is not None:
        calls[-1]["expected_device_id"] = expected_device_id
    (root.parent / "opened-workers.json").write_text(json.dumps(calls))
    session = ProviderOnlySession(target, device_index, device_uuid)
    sessions.append(session)
    return session

def forbidden(*args, **kwargs):
    raise AssertionError("Provider-only consumer requested a packed module API")
ModuleRequest = forbidden
open_session = forbidden

def observations():
    return {
        "calls": calls,
        "sessions": [{
            "closed": session.closed,
            "trace": session.trace,
            "drains": session.drains,
            "close_pending": session.close_pending,
            "foreign_rejections": session.foreign_rejections,
            "buffers_remaining": len(session.buffers),
        } for session in sessions],
    }
"""

_SESSION = r"""
class ProviderOnlySession(CpuSession):
    def __init__(self, target, index, uuid):
        super().__init__()
        vendor = target.split(":")[0]
        index = 0 if index is None else index
        uuid = uuid or ("a" if vendor == "amd" else "b") * 32
        self.target = SimpleNamespace(canonical_id=target)
        self.device = SimpleNamespace(
            device_uuid=uuid,
            record=lambda: {"vendor": vendor, "device_index": index, "device_uuid": uuid},
        )
        self.runner_sha256 = ("c" if vendor == "amd" else "d") * 64
        self.provider = SimpleNamespace(record=lambda: {
            "vendor": vendor,
            "provider": {"amd": "rocblas", "nvidia": "cublas", "intel": "onemkl"}[vendor],
            "library_version": "cpu-test-observed-version",
        })

    def module(self, *args, **kwargs):
        raise AssertionError("Provider-only consumer loaded a packed module")

    def launch(self, *args, **kwargs):
        raise AssertionError("Provider-only consumer launched a kernel")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.closed:
            self.close()
        return False
"""


class SgemmProviderExampleTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(
            prefix="therock-sgemm-provider-example-"
        )
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "installed distribution"
        self.python_root = self.root / "share/therock/python"
        self.package = self.python_root / "therock_multi_vendor"
        self.package.mkdir(parents=True)
        for package in ("_therock_utils", "rocm_kpack"):
            directory = self.python_root / package
            directory.mkdir()
            (directory / "__init__.py").write_text("# installed test package\n")
        (self.package / "__init__.py").write_text(_RUNTIME)
        (self.package / "cpu_session.py").write_text(
            "from array import array\nimport struct\nfrom types import SimpleNamespace\n"
            + inspect.getsource(float32)
            + "\n"
            + inspect.getsource(CpuSession)
            + _SESSION
        )
        for filename in ("sgemm_example.py", "sgemm_provider_example.py"):
            shutil.copyfile(EXAMPLES / filename, self.package / filename)
        self.script = self.package / "sgemm_provider_example.py"
        self.helper = self.root / "share/therock/examples/packed_session_client.py"
        self.helper.parent.mkdir(parents=True)
        shutil.copyfile(
            REPOSITORY / "tests/multi_vendor/modules/packed_session_client.py",
            self.helper,
        )
        self.arguments = ["--dist-root", str(self.root), "--target", AMD]

    def run_example(
        self, *arguments: str, isolated: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                *(["-I"] if isolated else []),
                "-c",
                _COMMAND,
                str(self.script),
                *self.arguments,
                *arguments,
            ],
            cwd=self.base,
            capture_output=True,
            text=True,
        )

    def records(self, *arguments: str) -> tuple[dict[str, object], dict[str, object]]:
        result = self.run_example(*arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 2, result.stdout)
        return json.loads(lines[0]), json.loads(lines[1])

    def assert_session_report(self, record: dict[str, object], target: str) -> None:
        self.assertEqual(record["target"], target)
        self.assertEqual(record["modules_loaded"], 0)
        self.assertEqual(record["buffer_handles"], 4)
        self.assertNotIn("kernel_interop", record)
        self.assertEqual(record["provider"]["vendor"], target.split(":")[0])
        self.assertEqual(
            record["provider"]["library_version"], "cpu-test-observed-version"
        )
        self.assertEqual(len(record["checks"]), 7)
        self.assertEqual(
            [check["m"] for check in record["checks"]], [1, 5, 17, 7, 256, 1, 256]
        )
        for check in record["checks"]:
            self.assertEqual(check["guards"], "pass")
            self.assertTrue(check["inputs_unchanged"])
            self.assertLessEqual(
                check["maximum_absolute_error"], check["maximum_absolute_bound"]
            )

    def test_single_provider_needs_no_module_format_or_payloads(self) -> None:
        before = {path.relative_to(self.root) for path in self.root.rglob("*")}
        report, observed = self.records()
        self.assertEqual(report["kind"], "installed-sgemm-provider-client")
        self.assertEqual(report["modules_loaded"], 0)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["isolated_python"])
        self.assertEqual(len(report["sessions"]), 1)
        self.assert_session_report(report["sessions"][0], AMD)
        self.assertEqual(report["sessions"][0]["device_uuid"], AMD_UUID)
        self.assertEqual(
            observed["calls"],
            [
                {
                    "target": AMD,
                    "device_index": None,
                    "device_uuid": None,
                    "prior_worker_live": False,
                }
            ],
        )
        session = observed["sessions"][0]
        self.assertTrue(session["closed"])
        self.assertEqual(session["buffers_remaining"], 0)
        self.assertEqual(session["drains"], [["sgemm"]] * 7)
        self.assertEqual(session["trace"].count("sgemm"), 7)
        self.assertFalse(
            any(operation.startswith("launch") for operation in session["trace"])
        )
        origins = report["runtime_origins"]
        self.assertIn("therock_multi_vendor.sgemm_example", origins)
        self.assertTrue(
            all(
                Path(path).is_relative_to(self.python_root) for path in origins.values()
            )
        )
        self.assertEqual(
            before, {path.relative_to(self.root) for path in self.root.rglob("*")}
        )

    def test_paired_workers_forward_device_selectors_and_survive_peer_close(
        self,
    ) -> None:
        report, observed = self.records(
            "--device",
            "2",
            "--peer-target",
            NVIDIA,
            "--peer-device-uuid",
            NVIDIA_UUID,
        )
        self.assertEqual(len(report["sessions"]), 2)
        self.assert_session_report(report["sessions"][0], AMD)
        self.assert_session_report(report["sessions"][1], NVIDIA)
        self.assertEqual(
            observed["calls"],
            [
                {
                    "target": AMD,
                    "device_index": 2,
                    "device_uuid": None,
                    "prior_worker_live": False,
                },
                {
                    "target": NVIDIA,
                    "device_index": None,
                    "device_uuid": NVIDIA_UUID,
                    "prior_worker_live": True,
                },
            ],
        )
        self.assertEqual(
            report["host_transfer_directions"], ["target-to-peer", "peer-to-target"]
        )
        self.assertEqual(report["foreign_handle_rejections"], 2)
        self.assertTrue(report["first_close_with_queued_sgemm"])
        self.assertEqual(report["peer_after_first_close"], "pass")
        self.assertEqual(
            [record["foreign_rejections"] for record in observed["sessions"]], [1, 1]
        )
        self.assertEqual(observed["sessions"][0]["close_pending"], 1)
        self.assertTrue(all(record["closed"] for record in observed["sessions"]))
        self.assertEqual(
            [record["buffers_remaining"] for record in observed["sessions"]], [0, 0]
        )

    def test_primary_uuid_and_peer_index_selectors_are_forwarded(self) -> None:
        report, observed = self.records(
            "--device-uuid", AMD_UUID, "--peer-target", NVIDIA, "--peer-device", "3"
        )
        self.assertEqual(observed["calls"][0]["device_index"], None)
        self.assertEqual(observed["calls"][0]["device_uuid"], AMD_UUID)
        self.assertEqual(observed["calls"][1]["device_index"], 3)
        self.assertEqual(observed["calls"][1]["device_uuid"], None)
        self.assertEqual(report["sessions"][0]["device_uuid"], AMD_UUID)
        self.assertEqual(report["sessions"][1]["device"]["device_index"], 3)

    def test_intel_provider_forwards_expected_pci_id(self) -> None:
        self.arguments = [
            "--dist-root",
            str(self.root),
            "--target",
            "intel:level-zero:xe2-b70",
        ]
        report, observed = self.records("--expect-device-id", "0xe223")
        self.assertEqual(observed["calls"][0]["expected_device_id"], 0xE223)
        self.assert_session_report(report["sessions"][0], "intel:level-zero:xe2-b70")
        self.assertEqual(report["sessions"][0]["provider"]["provider"], "onemkl")
        self.assertEqual(report["modules_loaded"], 0)
        self.assertTrue(observed["sessions"][0]["closed"])

    def test_invalid_cli_fails_before_opening_a_worker(self) -> None:
        for arguments in (
            ("--format", "hsaco"),
            ("--peer-format", "cubin"),
            ("--device", "-1"),
            ("--expect-device-id", "-1"),
            ("--expect-device-id", "0xinvalid"),
            ("--peer-device", "1"),
            ("--peer-device-uuid", NVIDIA_UUID),
            ("--peer-target", AMD),
            ("--device", "2", "--device-uuid", AMD_UUID),
            (
                "--peer-target",
                NVIDIA,
                "--peer-device",
                "3",
                "--peer-device-uuid",
                NVIDIA_UUID,
            ),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_example(*arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertFalse((self.base / "opened-workers.json").exists())

    def test_bootstrap_rejects_nonisolated_python(self) -> None:
        result = self.run_example(isolated=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("python -I", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse((self.base / "opened-workers.json").exists())

    def test_bootstrap_rejects_escaped_installed_helper(self) -> None:
        outside = self.base / "outside-helper.py"
        shutil.move(self.helper, outside)
        self.helper.symlink_to(outside)
        result = self.run_example()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("escape", result.stderr.lower())
        self.assertEqual(result.stdout, "")
        self.assertFalse((self.base / "opened-workers.json").exists())

    def test_bootstrap_rejects_escaped_runtime_package(self) -> None:
        outside = self.base / "outside-runtime.py"
        initializer = self.package / "__init__.py"
        shutil.move(initializer, outside)
        initializer.symlink_to(outside)
        result = self.run_example()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not found in the installed runtime", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse((self.base / "opened-workers.json").exists())

    def test_bootstrap_rejects_escaped_numerical_fixture(self) -> None:
        outside = self.base / "outside-numerics.py"
        numerics = self.package / "sgemm_example.py"
        shutil.move(numerics, outside)
        numerics.symlink_to(outside)
        result = self.run_example()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("escape", result.stderr.lower())
        self.assertEqual(result.stdout, "")
        self.assertFalse((self.base / "opened-workers.json").exists())


if __name__ == "__main__":
    unittest.main()
