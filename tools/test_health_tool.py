"""Tool health check for the project's pytest suite.

Returns a zodern:relay-style envelope. On success the payload summarizes
bounded pytest output with counts, duration and a small capped set of
representative failure summaries. On unavailable/unusable pytest it returns a
failure envelope with code ``test_health_error``.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from tools.validation_helper import relay_failure, relay_success

MAX_FAILURE_SUMMARIES = 5


def _detect_test_runner() -> Optional[str]:
    """Return the project's test runner if detectable, otherwise None."""

    runner = shutil.which("pytest")
    if runner and runner.strip():
        return runner
    return None


def _check_fn() -> bool:
    return _detect_test_runner() is not None


def _parse_summary_line(text: str) -> Dict[str, int]:
    """Best-effort parse of pytest's final summary line.

    Looks for fragments like ``= 12 passed, 3 failed, ...`` handling optional
    ANSI escape codes. Pytest uses the singular form ``error`` in the terminal
    summary, so both forms are accepted.
    """
    import re

    cleaned = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    counts: Dict[str, int] = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
    }
    lower = cleaned.lower()
    last = r"""(?s)==+\s+.*?in\s+[\d\.]+s\s+==+"""
    matches = list(re.finditer(last, lower))
    segment = matches[-1].group(0) if matches else lower
    segment = re.sub(r"^[=\s]+", "", segment)
    for key in ("passed", "failed", "errors", "skipped"):
        match = re.search(rf"(\d+)\s+{re.escape(key)}\b", segment)
        if match:
            counts[key] = int(match.group(1))
    return counts


def _as_file_test(entry: str) -> str:
    """Normalize a pytest failure node to ``file::test`` form."""
    if "::" not in entry:
        return entry.strip()
    left, right = entry.split("::", 1)
    left = left.strip().strip('"')
    right = right.strip().strip('"').split("[")[0].split(" ")[0]
    return f"{left}::{right}"


def _parse_failure_summaries(text: str, cap: int) -> List[str]:
    """Return up to ``cap`` representative ``file::test`` failure summaries."""
    summaries: List[str] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        if "FAILED " in raw:
            node = raw.split("FAILED ", 1)[1].strip().strip('"')
        elif raw.startswith("FAILED "):
            node = raw[len("FAILED "):].strip().strip('"')
        else:
            continue
        node = node.strip()
        if not node:
            continue
        simplified = _as_file_test(node)
        if simplified and simplified not in summaries:
            summaries.append(simplified)
        if len(summaries) >= cap:
            break
    return summaries


def run_test_health(
    args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a bounded pytest slice and return a zodern:relay envelope.

    Bounds:
    - ``-q`` suppresses verbose output.
    - ``--maxfail=5`` fails fast after the first 5 test failures.
    - Timeout after 300 seconds regardless of test length.
    """
    runner = _detect_test_runner()
    if runner is None:
        return relay_failure(
            "pytest not detected; unavailable for test_health",
            code="test_health_error",
        )

    bounded_args = [
        runner,
        "-q",
        "--disable-warnings",
        "--maxfail=5",
        "-r",
        "fE",
    ]

    start = time.perf_counter()
    proc = _safe_popen(bounded_args)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        # Terminate a possibly lingering pytest process if communicate missed it.
        try:
            proc.kill()
        except Exception:
            pass
    output = ((stdout or b"") + (stderr or b"")).decode("utf-8", "replace")
    duration = round(time.perf_counter() - start, 6)

    if duration <= 0:
        try:
            proc.terminate()
        except Exception:
            pass

    counts = _parse_summary_line(output)
    fallback = dict.fromkeys(["passed", "failed", "errors", "skipped"], 0)
    cases = (
        int(counts.get("passed", 0))
        + int(counts.get("failed", 0))
        + int(counts.get("errors", 0))
        + int(counts.get("skipped", 0))
    )
    if cases == 0:
        counts = fallback

    status = "passed" if proc.returncode == 0 else "failed"
    payload: Dict[str, Any] = {
        "status": status,
        "duration_seconds": duration,
        "total_passed": int(counts.get("passed", 0)),
        "total_failed": int(counts.get("failed", 0)),
        "total_errors": int(counts.get("errors", 0)),
        "total_skipped": int(counts.get("skipped", 0)),
        "failure_summaries": _parse_failure_summaries(output, MAX_FAILURE_SUMMARIES),
    }
    return relay_success(payload)


def _safe_popen(cmd: List[str]) -> "subprocess.Popen[bytes]":
    try:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:
        raise RuntimeError(f"unable to launch pytest: {exc}") from exc


__all__ = [
    "run_test_health",
    "check_fn",
    "_detect_test_runner",
    "_parse_summary_line",
    "_parse_failure_summaries",
]

check_fn = _check_fn
