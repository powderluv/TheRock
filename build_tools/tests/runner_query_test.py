# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU coverage for metadata-query startup and transport failures."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _therock_utils import runner_query


@unittest.skipUnless(os.name == "posix", "Native queries require POSIX")
class RunnerQueryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="therock-runner-query-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runner = self.root / "runner.py"
        self.runner.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(20)\n")
        self.runner.chmod(0o755)
        self.processes = []
        original = subprocess.Popen

        def start(*args, **kwargs):
            process = original(*args, **kwargs)
            self.processes.append(process)
            return process

        patch = mock.patch.object(runner_query.subprocess, "Popen", side_effect=start)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.check_reaped)

    def check_reaped(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
                process.wait()
                self.fail("Metadata query leaked a worker")

    def test_unsupported_query_cannot_start_worker(self):
        with self.assertRaisesRegex(ValueError, "Unsupported runner query"):
            runner_query.query_runner(self.runner, "--serve")
        self.assertEqual(self.processes, [])

    def test_selector_creation_failure_cannot_start_worker(self):
        with mock.patch.object(
            runner_query.selectors, "DefaultSelector", side_effect=OSError("fd limit")
        ):
            with self.assertRaisesRegex(OSError, "fd limit"):
                runner_query.describe_runner(self.runner)
        self.assertEqual(self.processes, [])

    def test_registration_failure_reaps_started_worker(self):
        selector = runner_query.selectors.DefaultSelector()
        with mock.patch.object(
            runner_query.selectors, "DefaultSelector", return_value=selector
        ), mock.patch.object(
            selector, "register", side_effect=OSError("registration failed")
        ):
            with self.assertRaisesRegex(OSError, "registration failed"):
                runner_query.describe_runner(self.runner)
        self.assertEqual(len(self.processes), 1)
        self.assertIsNotNone(self.processes[0].poll())
        self.assertIsNone(selector.get_map())

    def test_timeout_reaps_worker(self):
        with self.assertRaisesRegex(ValueError, "timed out"):
            runner_query.describe_runner(self.runner, timeout=0.1)
        self.assertIsNotNone(self.processes[0].poll())

    def test_query_error_preserves_worker_stderr(self):
        self.runner.write_text(
            f"#!{sys.executable}\nimport sys\nprint('fixture driver unavailable', file=sys.stderr)\nsys.exit(23)\n"
        )
        with self.assertRaisesRegex(ValueError, "23: fixture driver unavailable"):
            runner_query.query_runner(self.runner, "--list-devices")
        self.assertIsNotNone(self.processes[0].poll())


if __name__ == "__main__":
    unittest.main()
