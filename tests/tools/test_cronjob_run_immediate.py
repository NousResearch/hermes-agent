"""Tests for immediate, non-blocking manual cron execution (#41037, #52705)."""

import json
import threading
import time
from unittest.mock import patch

from tools.cronjob_tools import _execute_job_now, cronjob


_JOB = {
    "id": "job-run-1",
    "name": "manual run",
    "prompt": "hi",
    "schedule": {"kind": "cron", "expr": "0 9 * * *"},
}


class TestCronjobRunExecutesImmediately:
    def test_run_action_claims_and_dispatches_via_scheduler_pool(self):
        with (
            patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)),
            patch(
                "tools.cronjob_tools.claim_job_for_fire", return_value=True
            ) as claim,
            patch("cron.scheduler.dispatch_job", return_value=object()) as dispatch,
            patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)),
        ):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["success"] is True
        assert out["job"]["executed"] is True
        assert out["job"]["execution_pending"] is True
        assert out["job"]["execution_mode"] == "background"
        claim.assert_called_once_with("job-run-1")
        dispatch.assert_called_once_with(dict(_JOB))

    def test_run_preserves_held_claim_reason(self):
        with (
            patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)),
            patch("tools.cronjob_tools.claim_job_for_fire", return_value=False),
            patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)),
            patch("cron.scheduler.dispatch_job") as dispatch,
        ):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["job"]["executed"] is False
        assert "already being fired" in out["job"]["execution_skipped"].lower()
        dispatch.assert_not_called()

    def test_run_preserves_paused_claim_reason(self):
        paused = dict(_JOB, enabled=False, state="paused")
        with (
            patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)),
            patch("tools.cronjob_tools.claim_job_for_fire", return_value=False),
            patch("tools.cronjob_tools.get_job", return_value=paused),
            patch("cron.scheduler.dispatch_job") as dispatch,
        ):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        reason = out["job"]["execution_skipped"].lower()
        assert "paused/disabled" in reason
        assert "already being fired" not in reason
        dispatch.assert_not_called()

    def test_run_preserves_missing_claim_reason(self):
        with (
            patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)),
            patch("tools.cronjob_tools.claim_job_for_fire", return_value=False),
            patch("tools.cronjob_tools.get_job", return_value=None),
            patch("cron.scheduler.dispatch_job") as dispatch,
        ):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        reason = out["job"]["execution_skipped"].lower()
        assert "no longer exists" in reason
        assert "already being fired" not in reason
        dispatch.assert_not_called()

    def test_run_surfaces_dispatch_rejection(self):
        with (
            patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)),
            patch("tools.cronjob_tools.claim_job_for_fire", return_value=True),
            patch("cron.scheduler.dispatch_job", return_value=None),
            patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)),
        ):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["job"]["executed"] is False
        assert out["job"]["execution_success"] is False
        assert "not dispatched" in out["job"]["execution_skipped"].lower()

    def test_run_returns_before_background_job_finishes(self):
        import cron.scheduler as scheduler

        scheduler._parallel_pool = None
        scheduler._parallel_pool_max_workers = None
        scheduler._running_job_ids.clear()
        barrier = threading.Barrier(2, timeout=5)

        def slow_run_one_job(_job, **_kwargs):
            barrier.wait()
            return True

        try:
            with (
                patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)),
                patch("tools.cronjob_tools.claim_job_for_fire", return_value=True),
                patch("cron.scheduler.run_one_job", side_effect=slow_run_one_job),
                patch("cron.scheduler.create_execution", return_value={"id": "exec-1"}),
                patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)),
            ):
                start = time.monotonic()
                out = json.loads(cronjob(action="run", job_id="job-run-1"))
                elapsed = time.monotonic() - start

            assert out["job"]["executed"] is True
            assert out["job"]["execution_pending"] is True
            assert elapsed < 1.0
        finally:
            barrier.wait()
            time.sleep(0.1)
            scheduler._shutdown_parallel_pool()

    def test_execute_job_now_bails_without_claim(self):
        with (
            patch("tools.cronjob_tools.claim_job_for_fire", return_value=False),
            patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)),
            patch("cron.scheduler.dispatch_job") as dispatch,
        ):
            result = _execute_job_now(dict(_JOB))

        assert result["claimed"] is False
        assert result["started"] is False
        assert result["success"] is False
        dispatch.assert_not_called()

    def test_execute_job_now_marks_failure_on_exception(self):
        with (
            patch("tools.cronjob_tools.claim_job_for_fire", return_value=True),
            patch("cron.scheduler.dispatch_job", side_effect=RuntimeError("boom")),
            patch("tools.cronjob_tools.mark_job_run") as mark_run,
        ):
            result = _execute_job_now(dict(_JOB))

        assert result["claimed"] is True
        assert result["started"] is False
        assert result["success"] is False
        assert "boom" in result["error"]
        mark_run.assert_called_once()
