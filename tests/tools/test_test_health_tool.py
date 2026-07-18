"""Tests for ``tools.test_health_tool``.

Asserts the relay envelope shape, failure summary capping, missing-runner
failure behavior and that ``duration_seconds`` is numeric.
"""

from __future__ import annotations

import types
from typing import Any, Dict

import pytest

from tools.test_health_tool import (
    _detect_test_runner,
    _parse_failure_summaries,
    _parse_summary_line,
    _safe_popen,
    check_fn,
    run_test_health,
)


@pytest.fixture()
def good_summary() -> str:
    return (
        "============================== test session starts ==============================\n"
        "collected 6 items\n"
        "\n"
        "tests/sample_test.py::test_ok PASSED\n"
        "tests/sample_test.py::test_ok2 PASSED\n"
        "tests/sample_test.py::test_fail_one FAILED\n"
        "tests/sample_test.py::test_fail_two FAILED\n"
        "tests/sample_test.py::test_skip_one SKIPPED\n"
        "\n"
        "================================== FAILURES ===================================\n"
        "\n"
        "__________________________________ test_fail_one _________________________________\n"
        "\n"
        "    def test_fail_one():\n"
        "        assert False\n"
        "\n"
        "tests/sample_test.py:4: AssertionError\n"
        "\n"
        "================================ short test summary info ========================\n"
        "FAILED tests/sample_test.py::test_fail_one - AssertionError\n"
        "FAILED tests/sample_test.py::test_fail_two - AssertionError\n"
        "SKIPPED tests/sample_test.py::test_skip_one\n"
        "= 2 passed, 2 failed, 1 skipped in 0.12s =\n"
    )


def test_success_envelope_shape(good_summary: str, monkeypatch: pytest.MonkeyPatch):
    counts = _parse_summary_line(good_summary)
    expected = {
        "passed": 2,
        "failed": 2,
        "errors": 0,
        "skipped": 1,
    }
    assert counts == expected

    summaries = _parse_failure_summaries(good_summary, cap=5)
    assert len(summaries) == 2
    assert all("::" in item for item in summaries)
    assert "tests/sample_test.py" in summaries[0]

    runner = _detect_test_runner()
    if runner is None:
        pytest.skip("pytest not installed in current environment")

    class DummyResult:
        returncode = 0 if counts["failed"] == counts["errors"] == 0 else 1
        stdout = good_summary.encode("utf-8")
        stderr = good_summary.encode("utf-8")

        def communicate(self) -> tuple[bytes, bytes]:
            return self.stdout, self.stderr

    monkeypatch.setattr(
        "tools.test_health_tool._safe_popen",
        lambda cmd: DummyResult(),
    )
    monkeypatch.setattr(
        "tools.test_health_tool._detect_test_runner",
        lambda: runner,
    )
    monkeypatch.setattr(
        "tools.test_health_tool._check_fn",
        lambda: True,
    )

    envelope = run_test_health()
    assert envelope["kind"] == "success"
    payload = envelope["data"]
    assert set(payload) >= {
        "status",
        "duration_seconds",
        "total_passed",
        "total_failed",
        "total_errors",
        "total_skipped",
        "failure_summaries",
    }
    assert payload["status"] in {"passed", "failed"}
    assert payload["duration_seconds"] >= 0
    assert isinstance(payload["duration_seconds"], (int, float))


def test_failure_summary_cap(monkeypatch: pytest.MonkeyPatch):
    text = "\n".join(
        f'FAILED tests/sample_test.py::test_fail_{i} - AssertionError'
        for i in range(20)
    )

    capped = _parse_failure_summaries(text, cap=5)
    assert len(capped) == 5
    assert capped == [
        "tests/sample_test.py::test_fail_0",
        "tests/sample_test.py::test_fail_1",
        "tests/sample_test.py::test_fail_2",
        "tests/sample_test.py::test_fail_3",
        "tests/sample_test.py::test_fail_4",
    ]


def test_missing_runner_failure_behavior(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "tools.test_health_tool._detect_test_runner",
        lambda: None,
    )
    monkeypatch.setattr(
        "tools.test_health_tool._check_fn",
        lambda: False,
    )

    envelope = run_test_health()
    assert envelope["kind"] == "failure"
    assert envelope["code"] == "test_health_error"
    assert "pytest" in envelope["error"].lower()


def test_duration_is_numeric(good_summary: str, monkeypatch: pytest.MonkeyPatch):
    runner = _detect_test_runner()
    if runner is None:
        pytest.skip("pytest not installed in current environment")

    class DummyResult:
        returncode = 0
        stdout = good_summary.encode("utf-8")
        stderr = b""

        def communicate(self) -> tuple[bytes, bytes]:
            return self.stdout, self.stderr

    monkeypatch.setattr(
        "tools.test_health_tool._safe_popen",
        lambda cmd: DummyResult(),
    )
    monkeypatch.setattr(
        "tools.test_health_tool._detect_test_runner",
        lambda: runner,
    )
    monkeypatch.setattr(
        "tools.test_health_tool._check_fn",
        lambda: True,
    )

    envelope = run_test_health()
    assert envelope["kind"] == "success"
    assert isinstance(envelope["data"]["duration_seconds"], (int, float))
    assert envelope["data"]["duration_seconds"] >= 0


def test_check_fn_matches_runner_detection():
    expected = _detect_test_runner() is not None
    assert check_fn() is expected
