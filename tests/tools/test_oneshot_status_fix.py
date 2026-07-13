"""Tests for _execute_job_now one-shot status reporting fix.

Verifies that one-shot jobs (repeat=1) which are auto-removed by mark_job_run
after completion correctly report success instead of failure.
"""

from unittest.mock import patch

import pytest


@pytest.fixture()
def _job_oneshot():
    return {"id": "oneshot-1", "name": "test-oneshot", "schedule": {"kind": "once", "at": "2099-01-01T00:00:00"}, "repeat": {"times": 1, "completed": 0}}


@pytest.fixture()
def _job_recurring():
    return {"id": "recurring-1", "name": "test-recurring"}


class TestExecuteJobNowOneShot:
    """One-shot jobs auto-removed after success should report success."""

    def test_oneshot_removed_after_success(self, _job_oneshot):
        """When mark_job_run removes a one-shot, _execute_job_now should
        return success=True (the job ran and completed)."""
        with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
             patch("tools.cronjob_tools.get_job", return_value=None), \
             patch("cron.scheduler.run_one_job", return_value=True):
            from tools.cronjob_tools import _execute_job_now
            result = _execute_job_now(_job_oneshot)
            assert result["claimed"] is True
            assert result["success"] is True
            assert result["error"] is None

    def test_oneshot_agent_failure(self, _job_oneshot):
        """When the agent fails but job still exists, report failure."""
        with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
             patch("tools.cronjob_tools.get_job", return_value={"id": "oneshot-1", "last_status": "error", "last_error": "bad"}), \
             patch("cron.scheduler.run_one_job", return_value=True):
            from tools.cronjob_tools import _execute_job_now
            result = _execute_job_now(_job_oneshot)
            assert result["success"] is False
            assert result["error"] == "bad"

    def test_recurring_success(self, _job_recurring):
        """Recurring jobs that still exist after run report normally."""
        with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
             patch("tools.cronjob_tools.get_job", return_value={"id": "recurring-1", "last_status": "ok"}), \
             patch("cron.scheduler.run_one_job", return_value=True):
            from tools.cronjob_tools import _execute_job_now
            result = _execute_job_now(_job_recurring)
            assert result["success"] is True
            assert result["error"] is None

    def test_processing_failure_job_gone(self, _job_recurring):
        """When run_one_job returns False and job is gone, report failure."""
        with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
             patch("tools.cronjob_tools.get_job", return_value=None), \
             patch("cron.scheduler.run_one_job", return_value=False):
            from tools.cronjob_tools import _execute_job_now
            result = _execute_job_now(_job_recurring)
            assert result["success"] is False

    def test_claim_failed(self, _job_recurring):
        """When claim fails, report not-claimed."""
        with patch("tools.cronjob_tools.claim_job_for_fire", return_value=False), \
             patch("tools.cronjob_tools.get_job", return_value=None):
            from tools.cronjob_tools import _execute_job_now
            result = _execute_job_now(_job_recurring)
            assert result["claimed"] is False
            assert result["success"] is False
