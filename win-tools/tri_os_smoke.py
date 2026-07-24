#!/usr/bin/env python3
"""Portable curated PyTorch smoke runner for the tri-OS gfx1201 lite:: effort.

Runs a curated subset of TheRock's smoke-tests/pytorch_smoke_test.py and prints a
machine-parseable pass/fail line so the same invocation yields a real count on
macOS (eGPU/DEXT) and Windows (win11-gpu VFIO).

Two modes:
  * default (single process): pytest.main over the whole slice in one process;
    the authoritative count is parsed from a JUnit XML (the lite:: C++ driver's
    bring-up logging interleaves with pytest stdout and mangles it).
  * SMOKE_ISOLATE=1: run each collected test id in its OWN subprocess. A GPU
    fault/crash in one test becomes a clean FAIL for that test (and a fresh
    process resets clr's per-process global gpu_error_ latch), instead of
    aborting the whole run. Needed once a test can hard-crash (0xC0000005).

Config via env (all optional):
  SMOKE_FILE          path to pytorch_smoke_test.py (or argv[1]).
  SMOKE_SELECT        pytest -k expression. Default = rocm_available + the
                      matmul/elementwise/transpose GPU ops; skips conv (MIOpen)
                      and the CPU BLAS/LAPACK config tests.
  SMOKE_TIMEOUT       faulthandler dump timeout seconds (default 20).
  SMOKE_JUNIT         path for the JUnit XML report (default: temp dir).
  SMOKE_ISOLATE       if set, per-test-subprocess mode.
  SMOKE_TEST_TIMEOUT  per-test subprocess timeout seconds (default 180).
"""
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

SMOKE_FILE = os.environ.get("SMOKE_FILE") or (sys.argv[1] if len(sys.argv) > 1 else "")
if not SMOKE_FILE or not os.path.isfile(SMOKE_FILE):
    print(f"[tri_os_smoke] SMOKE_FILE not found: {SMOKE_FILE!r}", file=sys.stderr)
    sys.exit(2)

# Run subprocesses from the smoke file's directory so pytest resolves the
# collected (rootdir-relative) node IDs regardless of the launcher's CWD (e.g.
# run-torch-egpu.sh cd's to /tmp, where the relative id is "file not found" -> rc=4).
SMOKE_DIR = os.path.dirname(os.path.abspath(SMOKE_FILE)) or "."
SELECT = os.environ.get("SMOKE_SELECT", "test_rocm_available or TestMatrixOperations")
TIMEOUT = os.environ.get("SMOKE_TIMEOUT", "20")
JUNIT = os.environ.get("SMOKE_JUNIT") or os.path.join(tempfile.gettempdir(), "tri_os_smoke_junit.xml")


def _verdict_from_junit(path):
    """Return (verdict, msg) for a single-testcase junit file, or (None, '')."""
    if not os.path.isfile(path):
        return None, ""
    try:
        root = ET.parse(path).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        tc = suite.find("testcase") if suite is not None else None
        if tc is None:
            return "EMPTY", ""
        if tc.find("failure") is not None:
            return "FAIL", (tc.find("failure").get("message", "") or "")
        if tc.find("error") is not None:
            return "ERROR", (tc.find("error").get("message", "") or "")
        if tc.find("skipped") is not None:
            return "SKIP", (tc.find("skipped").get("message", "") or "")
        return "PASS", ""
    except Exception as e:  # noqa: BLE001
        return "?", str(e)


def run_single_process():
    try:
        os.remove(JUNIT)
    except OSError:
        pass
    import pytest

    args = ["-v", "-s", "-p", "no:cacheprovider", "-p", "faulthandler",
            "-o", f"faulthandler_timeout={TIMEOUT}", "-k", SELECT,
            f"--junitxml={JUNIT}", SMOKE_FILE]
    print(f"[tri_os_smoke] python={sys.executable}", flush=True)
    print(f"[tri_os_smoke] pytest {' '.join(args)}", flush=True)
    rc = pytest.main(args)
    print("[tri_os_smoke] ===== JUNIT SUMMARY =====", flush=True)
    try:
        root = ET.parse(JUNIT).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        total = int(suite.get("tests", "0"))
        fails = int(suite.get("failures", "0"))
        errs = int(suite.get("errors", "0"))
        skips = int(suite.get("skipped", "0"))
        passed = total - fails - errs - skips
        for tc in suite.iter("testcase"):
            name = f"{tc.get('classname', '')}::{tc.get('name', '')}"
            if tc.find("failure") is not None:
                v = "FAIL"
            elif tc.find("error") is not None:
                v = "ERROR"
            elif tc.find("skipped") is not None:
                v = "SKIP"
            else:
                v = "PASS"
            print(f"[tri_os_smoke] {v:5} {name}", flush=True)
        print(f"[tri_os_smoke] RESULT passed={passed} failed={fails} errors={errs} "
              f"skipped={skips} total={total} pytest_rc={int(rc)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[tri_os_smoke] RESULT pytest_rc={int(rc)} (junit parse failed: {e})", flush=True)
    return int(rc)


def run_isolated():
    """Run each collected test id in its own subprocess (crash-tolerant)."""
    per_timeout = int(os.environ.get("SMOKE_TEST_TIMEOUT", "180"))
    collect = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", "-k", SELECT, SMOKE_FILE],
        capture_output=True, text=True, cwd=SMOKE_DIR)
    ids = [ln.strip() for ln in collect.stdout.splitlines()
           if "::" in ln and not ln.startswith(("=", "warning", "ERROR", "no tests"))]
    if not ids:
        print(f"[tri_os_smoke] no tests collected. stderr tail:\n{collect.stderr[-800:]}", flush=True)
        return 5
    print(f"[tri_os_smoke] ISOLATE python={sys.executable} tests={len(ids)} timeout={per_timeout}s", flush=True)
    passed = failed = crashed = 0
    for tid in ids:
        px = os.path.join(tempfile.gettempdir(),
                          "tri_" + tid.replace("::", "__").replace("/", "_").replace("\\", "_") + ".xml")
        try:
            os.remove(px)
        except OSError:
            pass
        try:
            r = subprocess.run(
                [sys.executable, "-u", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                 f"--junitxml={px}", tid],
                timeout=per_timeout, capture_output=True, text=True, cwd=SMOKE_DIR)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -999
        v, _ = _verdict_from_junit(px)
        if v is None:  # no junit written -> the process died before pytest finished
            v = "CRASH" if rc not in (0, 1) else ("PASS" if rc == 0 else "FAIL")
        if v == "PASS":
            passed += 1
        elif v in ("SKIP",):
            pass
        elif v in ("CRASH",) or rc == -999 or (rc not in (0, 1) and v in ("?", "EMPTY")):
            crashed += 1
            failed += 1
        else:
            failed += 1
        print(f"[tri_os_smoke] {v:6} rc={rc} {tid}", flush=True)
    total = len(ids)
    print(f"[tri_os_smoke] RESULT passed={passed} failed={failed} crashed={crashed} "
          f"total={total} (isolated)", flush=True)
    return 0 if failed == 0 else 1


if os.environ.get("SMOKE_ISOLATE"):
    sys.exit(run_isolated())
sys.exit(run_single_process())
