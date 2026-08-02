"""Tests for cronjob action='run' immediate execution (#41037).

Before this fix, `cronjob(action='run')` only set next_run_at=now and returned
success, relying on the scheduler ticker to actually run the job. With no
gateway/ticker active (e.g. a CLI-only Windows setup) the job never executed and
last_run_at stayed null forever. Now action='run' claims the job (at-most-once,
blocking a concurrent tick) and fires it inline via the shared run_one_job body.

#76502: the inline fire is synchronous, so while it runs it fires a heartbeat
into the calling agent's activity tracker — otherwise the gateway inactivity
watchdog kills the parent turn at ~1800s.
"""
import json
import threading
import time
from unittest.mock import patch

from tools.cronjob_tools import cronjob, _execute_job_now
from tools.environments.base import set_activity_callback


_JOB = {"id": "job-run-1", "name": "manual run", "prompt": "hi",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"}}


class TestCronjobRunExecutesImmediately:
    def test_run_action_claims_and_fires_via_run_one_job(self):
        """action='run' must claim the job then fire it through run_one_job."""
        ran = {"job": "after-run", "last_status": "ok", "last_error": None}
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=True) as m_claim, \
             patch("cron.scheduler.run_one_job", return_value=True) as m_run, \
             patch("tools.cronjob_tools.get_job", return_value=ran):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["success"] is True
        assert out["job"]["executed"] is True
        assert out["job"]["execution_success"] is True
        m_claim.assert_called_once_with("job-run-1")   # at-most-once claim taken
        m_run.assert_called_once()                       # fired via the shared body


    def test_execute_job_now_bails_without_claim(self):
        """_execute_job_now never calls run_one_job when the claim is lost."""
        with patch("tools.cronjob_tools.claim_job_for_fire", return_value=False), \
             patch("cron.scheduler.run_one_job") as m_run:
            res = _execute_job_now(dict(_JOB))
        assert res["claimed"] is False
        assert res["success"] is False
        m_run.assert_not_called()

    def test_execute_job_now_marks_failure_on_exception(self):
        """An exception during fire is captured, marked failed, not propagated."""
        with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
             patch("cron.scheduler.run_one_job", side_effect=RuntimeError("boom")), \
             patch("tools.cronjob_tools.mark_job_run") as m_mark, \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
            res = _execute_job_now(dict(_JOB))
        assert res["claimed"] is True
        assert res["success"] is False
        assert "boom" in res["error"]
        m_mark.assert_called_once()

    def test_execute_job_now_heartbeats_while_job_runs(self):
        """A manual run ticks the caller's activity tracker while the job
        executes so the gateway inactivity watchdog doesn't kill the parent
        turn (#76502)."""
        touches = []
        set_activity_callback(lambda desc: touches.append(desc))
        try:
            started = threading.Event()

            def slow_run(job):
                started.set()
                time.sleep(0.15)
                return True

            with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
                 patch("tools.cronjob_tools._CRON_RUN_HEARTBEAT_INTERVAL", 0.05), \
                 patch("cron.scheduler.run_one_job", side_effect=slow_run) as m_run, \
                 patch("tools.cronjob_tools.get_job",
                       return_value={"last_status": "ok", "last_error": None}):
                res = _execute_job_now(dict(_JOB))

            m_run.assert_called_once()
            assert res["success"] is True
            assert any("cronjob: running job" in t for t in touches), touches
        finally:
            set_activity_callback(None)

    def test_execute_job_now_without_callback_does_not_heartbeat(self):
        """No activity callback registered (direct callers, tests) → the
        heartbeat thread is never started and behavior is unchanged."""
        set_activity_callback(None)
        try:
            with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
                 patch("cron.scheduler.run_one_job", return_value=True) as m_run, \
                 patch("tools.cronjob_tools.get_job",
                       return_value={"last_status": "ok", "last_error": None}):
                res = _execute_job_now(dict(_JOB))
            assert res["success"] is True
            m_run.assert_called_once()
        finally:
            set_activity_callback(None)
