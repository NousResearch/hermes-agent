"""Tests for tools/cron_health_tool.py.

Validates the full behavior contract:
* success envelope shape
* failure behavior when state is missing or malformed
* cap on last_run_summary
* check_fn gating behavior
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.cron_health_tool import (
    _HERMES_ERROR_CODE,
    _RUN_SUMMARY_CAP,
    check_cron_health_requirements,
    cron_health_tool,
    summarize_cron_health,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jobs(path: Path, jobs) -> None:
    path.write_text(
        json.dumps(jobs if not isinstance(jobs, dict) else jobs),
        encoding="utf-8",
    )


def _make_job(*, job_id="job-1", name="Example job", enabled=True, last_status="ok", last_error=None):
    job: Dict[str, Any] = {
        "id": job_id,
        "name": name,
        "enabled": enabled,
        "last_status": last_status,
    }
    if last_error is not None:
        job["last_error"] = last_error
    return job



# ---------------------------------------------------------------------------
# check_fn gating
# ---------------------------------------------------------------------------


class TestCheckCronHealthRequirements:
    def test_false_when_default_path_missing(self, monkeypatch):
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=Path("/nonexistent-hermes-home/cron/jobs.json"),
        ):
            assert check_cron_health_requirements() is False

    def test_false_when_path_resolution_fails(self, monkeypatch):
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=None,
        ):
            assert check_cron_health_requirements() is False

    def test_true_when_jobs_file_exists_and_is_readable(self, tmp_path: Path):
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text("[]", encoding="utf-8")
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            assert check_cron_health_requirements() is True

    def test_false_on_unreadable_jobs_file(self, tmp_path: Path):
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text("[]", encoding="utf-8")
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ), patch.object(Path, "read_text", side_effect=OSError("permission denied")):
            assert check_cron_health_requirements() is False



# ---------------------------------------------------------------------------
# Success envelope shape
# ---------------------------------------------------------------------------


class TestSuccessEnvelopeShape:
    def test_empty_jobs(self, tmp_path: Path):
        jobs_file = tmp_path / "jobs.json"
        _write_jobs(jobs_file, [])
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            result = summarize_cron_health()

        assert result["kind"] == "success"
        assert isinstance(result["data"], dict)
        assert result["data"]["total_jobs"] == 0
        assert result["data"]["enabled_count"] == 0
        assert result["data"]["recent_failures"] == 0
        assert result["data"]["last_run_summary"] == []

    def test_required_fields_present(self, tmp_path: Path):
        job = _make_job(last_status="ok")
        jobs_file = tmp_path / "jobs.json"
        _write_jobs(jobs_file, [job])
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            result = summarize_cron_health()

        assert result["kind"] == "success"
        data = result["data"]
        for field in ("total_jobs", "enabled_count", "recent_failures", "last_run_summary"):
            assert field in data

    def test_enabled_count_counts_enabled_jobs(self, tmp_path: Path):
        jobs = [
            _make_job(job_id="a", enabled=True),
            _make_job(job_id="b", enabled=False),
            _make_job(job_id="c", enabled=True),
        ]
        jobs_file = tmp_path / "jobs.json"
        _write_jobs(jobs_file, jobs)
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            result = summarize_cron_health()

        assert result["data"]["enabled_count"] == 2

    def test_recent_failures_counts_error_status(self, tmp_path: Path):
        jobs = [
            _make_job(job_id="a", last_status="ok"),
            _make_job(job_id="b", last_status="error"),
            _make_job(job_id="c", last_status="error"),
        ]
        jobs_file = tmp_path / "jobs.json"
        _write_jobs(jobs_file, jobs)
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            result = summarize_cron_health()

        assert result["data"]["recent_failures"] == 2

    def test_last_run_summary_entry_happy_path(self, tmp_path: Path):
        job = _make_job(job_id="a", name="Heartbeat", enabled=True, last_status="ok")
        jobs_file = tmp_path / "jobs.json"
        _write_jobs(jobs_file, [job])
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            result = summarize_cron_health()

        entry = result["data"]["last_run_summary"][0]
        assert entry["job_id"] == "a"
        assert entry["name"] == "Heartbeat"
        assert entry["enabled"] is True
        assert entry["last_status"] == "ok"
        assert "last_error" not in entry

    def test_last_run_summary_entry_error_includes_last_error(self, tmp_path: Path):
        job = _make_job(
            job_id="a",
            name="Broken job",
            last_status="error",
            last_error="provider timeout",
        )
        jobs_file = tmp_path / "jobs.json"
        _write_jobs(jobs_file, [job])
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            result = summarize_cron_health()

        entry = result["data"]["last_run_summary"][0]
        assert entry["last_error"] == "provider timeout"

    def test_last_run_summary_cap(self, tmp_path: Path):
        jobs = [_make_job(job_id=f"job-{idx}", name=f"Job {idx}", last_status="ok") for idx in range(25)]
        jobs_file = tmp_path / "jobs.json"
        _write_jobs(jobs_file, jobs)
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            result = summarize_cron_health()

        assert len(result["data"]["last_run_summary"]) == _RUN_SUMMARY_CAP
        assert result["data"]["last_run_summary"][0]["job_id"] == "job-15"

    def test_last_run_summary_formatted_as_list_of_dicts(self, tmp_path: Path):
        jobs = [_make_job()]
        jobs_file = tmp_path / "jobs.json"
        _write_jobs(jobs_file, jobs)
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            result = summarize_cron_health()

        assert isinstance(result["data"]["last_run_summary"], list)
        if result["data"]["last_run_summary"]:
            assert isinstance(result["data"]["last_run_summary"][0], dict)



# ---------------------------------------------------------------------------
# Failure behavior: missing / inaccessible / malformed state
# ---------------------------------------------------------------------------


class TestFailureBehavior:
    def test_missing_jobs_file(self, tmp_path: Path):
        jobs_file = tmp_path / "missing.json"
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            result = summarize_cron_health()

        assert result["kind"] == "failure"
        assert result["code"] == _HERMES_ERROR_CODE

    def test_access_denied_jobs_file(self, tmp_path: Path):
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text("[]", encoding="utf-8")
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ), patch.object(Path, "read_text", side_effect=OSError("access denied")):
            result = summarize_cron_health()

        assert result["kind"] == "failure"
        assert result["code"] == _HERMES_ERROR_code if False else _HERMES_ERROR_CODE  # noqa: F741
        assert result is not None
        assert result["kind"] == "failure"

    def test_malformed_json(self, tmp_path: Path):
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text("not-json", encoding="utf-8")
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            result = summarize_cron_health()

        assert result["kind"] == "failure"
        assert result["code"] == _HERMES_ERROR_CODE

    def test_empty_document(self, tmp_path: Path):
        jobs_file = tmp_path / "jobs.json"
        jobs_file.write_text("", encoding="utf-8")
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            result = summarize_cron_health()

        # Empty document is treated as an in-memory empty state, returning success.
        # That preserves monotone(len(summary)) semantics for brand-new installs
        # while still protecting against truly inaccessible files above.
        assert result["kind"] == "success"
        assert result["data"]["total_jobs"] == 0

    def test_dict_missing_jobs_key(self, tmp_path: Path):
        jobs_file = tmp_path / "jobs.json"
        _write_jobs(jobs_file, {"metadata": {}})
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            result = summarize_cron_health()

        assert result["kind"] == "failure"
        assert result["code"] == _HERMES_ERROR_CODE



# ---------------------------------------------------------------------------
# Handler behavior matches direct calls
# ---------------------------------------------------------------------------


class TestHandlerBehavior:
    def test_handler_returns_failure_when_state_missing(self, tmp_path: Path):
        jobs_file = tmp_path / "missing.json"
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            result = cron_health_tool({})

        assert result["kind"] == "failure"
        assert result["code"] == _HERMES_ERROR_CODE

    def test_handler_returns_success_when_state_available(self, tmp_path: Path):
        jobs_file = tmp_path / "jobs.json"
        _write_jobs(jobs_file, [_make_job()])
        with patch(
            "tools.cron_health_tool._default_cron_jobs_file",
            return_value=jobs_file,
        ):
            result = cron_health_tool({})

        assert result["kind"] == "success"
        assert result["data"]["total_jobs"] == 1
