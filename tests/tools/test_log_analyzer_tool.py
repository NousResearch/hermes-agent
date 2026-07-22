"""Tests for tools/log_analyzer_tool.py.

Validates the full behavior contract:
* success envelope shape
* check_fn gating for missing/unreadable log directories/files
* capping of recent log tail
* error/warning pattern aggregation
* missing-log failure behavior
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.log_analyzer_tool import (
    _ERROR_CODE,
    _LOG_FILES,
    _RECENT_TAIL_CAP,
    _bucket_pattern,
    check_log_analyzer_requirements,
    log_analyzer_tool,
    summarize_logs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_log_dir(tmp_path: Path, files: dict[str, list[str]]) -> Path:
    logs = tmp_path / "logs"
    logs.mkdir()
    for filename, lines in files.items():
        _write_lines(logs / filename, lines)
    return logs


# ---------------------------------------------------------------------------
# check_fn gating
# ---------------------------------------------------------------------------


class TestCheckLogAnalyzerRequirements:
    def test_false_when_default_path_missing(self):
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=Path("/nonexistent-hermes-home/logs"),
        ):
            assert check_log_analyzer_requirements() is False

    def test_false_when_path_resolution_fails(self):
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=None,
        ):
            assert check_log_analyzer_requirements() is False

    def test_true_when_logs_directory_exists_and_readable(self, tmp_path: Path):
        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / "agent.log").write_text("", encoding="utf-8")
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=logs,
        ):
            assert check_log_analyzer_requirements() is True

    def test_false_on_unreadable_log_file(self, tmp_path: Path):
        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / "agent.log").write_text("", encoding="utf-8")
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=logs,
        ), patch.object(
            Path, "read_text", side_effect=OSError("permission denied")
        ):
            assert check_log_analyzer_requirements() is False

    def test_true_when_directory_empty(self, tmp_path: Path):
        logs = tmp_path / "logs"
        logs.mkdir()
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=logs,
        ):
            assert check_log_analyzer_requirements() is True


# ---------------------------------------------------------------------------
# Failure behavior
# ---------------------------------------------------------------------------


class TestMissingLogFailureBehavior:
    def test_missing_logs_dir_returns_failure(self):
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=Path("/nonexistent/logs"),
        ):
            result = summarize_logs()

        assert result["kind"] == "failure"
        assert result["code"] == _ERROR_CODE
        assert "logs" in result["error"].lower() or "unavailable" in result["error"].lower()

    def test_none_logs_dir_returns_failure(self):
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=None,
        ):
            result = summarize_logs()

        assert result["kind"] == "failure"
        assert result["code"] == _ERROR_CODE


# ---------------------------------------------------------------------------
# Success envelope shape
# ---------------------------------------------------------------------------


class TestSuccessEnvelopeShape:
    def test_success_envelope_root_keys(self, tmp_path: Path):
        logs = _make_log_dir(tmp_path, {"agent.log": []})
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=logs,
        ):
            result = summarize_logs()

        assert result["kind"] == "success"
        data = result["data"]
        for field in (
            "recent_error_counts",
            "top_error_warning_patterns",
            "recent_log_tail",
        ):
            assert field in data

    def test_recent_error_counts_shape(self, tmp_path: Path):
        lines = [
            "2026-07-17 12:00:00 ERROR something failed",
            "2026-07-17 12:00:01 WARNING disk low",
            "2026-07-17 12:00:02 INFO heartbeat ok",
        ]
        logs = _make_log_dir(tmp_path, {"agent.log": lines})
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=logs,
        ):
            result = summarize_logs()

        counts = result["data"]["recent_error_counts"]
        assert counts.get("error") == 1
        assert counts.get("warning") == 1

    def test_top_patterns_return_objects(self, tmp_path: Path):
        lines = [
            "2026-07-17 12:00:00 ERROR timeouterror",
            "2026-07-17 12:00:01 ERROR timeout",
            "2026-07-17 12:00:02 WARNING something else",
        ]
        logs = _make_log_dir(tmp_path, {"errors.log": lines})
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=logs,
        ):
            result = summarize_logs()

        patterns = result["data"]["top_error_warning_patterns"]
        assert isinstance(patterns, list)
        assert patterns
        for entry in patterns:
            assert "pattern" in entry
            assert "count" in entry
            assert isinstance(entry["pattern"], str)
            assert isinstance(entry["count"], int)

    def test_tool_function_matches_summarize(self, tmp_path: Path):
        lines = [
            "2026-07-17 12:00:00 ERROR foo",
            "2026-07-17 12:00:01 WARN bar",
        ]
        logs = _make_log_dir(tmp_path, {"gateway.log": lines})
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=logs,
        ):
            result = log_analyzer_tool()

        assert result["kind"] == "success"


# ---------------------------------------------------------------------------
# Capping of recent tail
# ---------------------------------------------------------------------------


class TestRecentTailCapping:
    def test_tail_capped_to_max(self, tmp_path: Path):
        lines = [
            f"2026-07-17 12:{i:02d}:00 INFO event-{i}"
            for i in range(200)
        ]
        logs = _make_log_dir(tmp_path, {"agent.log": lines})
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=logs,
        ):
            result = summarize_logs()

        tail = result["data"]["recent_log_tail"]
        assert len(tail) <= _RECENT_TAIL_CAP
        assert tail == lines[-_RECENT_TAIL_CAP:]

    def test_empty_tail_when_no_lines(self, tmp_path: Path):
        logs = _make_log_dir(tmp_path, {"agent.log": []})
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=logs,
        ):
            result = summarize_logs()

        assert result["data"]["recent_log_tail"] == []


# ---------------------------------------------------------------------------
# Pattern aggregation
# ---------------------------------------------------------------------------


class TestPatternAggregation:
    def test_aggregates_similar_patterns(self, tmp_path: Path):
        lines = [
            "2026-07-17 12:00:00 ERROR timeout connecting to database",
            "2026-07-17 12:00:01 ERROR timeout connecting to database",
            "2026-07-17 12:00:02 ERROR connection refused by peer",
        ]
        logs = _make_log_dir(tmp_path, {"errors.log": lines})
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=logs,
        ):
            result = summarize_logs()

        patterns = {
            p["pattern"]: p["count"]
            for p in result["data"]["top_error_warning_patterns"]
        }
        assert patterns.get("timeout connecting to database") == 2

    def test_warning_patterns_included(self, tmp_path: Path):
        lines = [
            "2026-07-17 12:00:00 WARNING disk space low",
            "2026-07-17 12:00:01 WARNING disk space low",
            "2026-07-17 12:00:02 WARNING memory pressure high",
        ]
        logs = _make_log_dir(tmp_path, {"agent.log": lines})
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=logs,
        ):
            result = summarize_logs()

        patterns = {
            p["pattern"]: p["count"]
            for p in result["data"]["top_error_warning_patterns"]
        }
        assert patterns.get("disk space low") == 2
        assert patterns.get("memory pressure high") == 1

    def test_non_error_warning_lines_skipped(self, tmp_path: Path):
        lines = [
            "2026-07-17 12:00:00 INFO harmless heartbeat",
            "2026-07-17 12:00:01 DEBUG debug msg",
        ]
        logs = _make_log_dir(tmp_path, {"agent.log": lines})
        with patch(
            "tools.log_analyzer_tool._default_logs_dir",
            return_value=logs,
        ):
            result = summarize_logs()

        assert result["data"]["top_error_warning_patterns"] == []
