"""Tests for cronjob action='run' immediate execution (#41037).

Before this fix, `cronjob(action='run')` only set next_run_at=now and returned
success, relying on the scheduler ticker to actually run the job. With no
gateway/ticker active (e.g. a CLI-only Windows setup) the job never executed and
last_run_at stayed null forever. Now action='run' claims the job (at-most-once,
blocking a concurrent tick). CLI callers fire inline; persistent gateway
sessions dispatch through the scheduler pool so a long run cannot consume the
chat turn's idle timeout.
"""
import json
from unittest.mock import patch

from tools.cronjob_tools import cronjob, _execute_job_now, _manual_run_should_detach


_JOB = {"id": "job-run-1", "name": "manual run", "prompt": "hi",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"}}


class TestCronjobRunExecutesImmediately:
    def test_only_persistent_gateway_sessions_detach(self):
        with patch("gateway.session_context.get_session_env", return_value="telegram"), \
             patch("gateway.session_context.async_delivery_supported", return_value=True):
            assert _manual_run_should_detach() is True

        with patch("gateway.session_context.get_session_env", return_value=""), \
             patch("gateway.session_context.async_delivery_supported", return_value=True):
            assert _manual_run_should_detach() is False

        with patch("gateway.session_context.get_session_env", return_value="api"), \
             patch("gateway.session_context.async_delivery_supported", return_value=False):
            assert _manual_run_should_detach() is False

    def test_gateway_run_dispatches_without_waiting_for_completion(self):
        """Persistent gateway sessions must not block on a long cron run."""
        started = {"claimed": True, "started": True, "success": None, "error": None}
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools._manual_run_should_detach", return_value=True), \
             patch("tools.cronjob_tools._execute_job_now", return_value=started) as m_run, \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["success"] is True
        assert out["job"]["executed"] is True
        assert out["job"]["execution_started"] is True
        assert out["job"]["execution_pending"] is True
        assert "execution_success" not in out["job"]
        m_run.assert_called_once_with(dict(_JOB), wait=False)

    def test_detached_execute_submits_without_running_inline(self):
        """wait=False must return after scheduler dispatch, before completion."""
        future = object()
        with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
             patch("cron.scheduler.submit_claimed_job", return_value=future) as m_submit, \
             patch("cron.scheduler.run_one_job") as m_inline:
            result = _execute_job_now(dict(_JOB), wait=False)

        assert result == {
            "claimed": True,
            "started": True,
            "success": None,
            "error": None,
        }
        m_submit.assert_called_once_with(dict(_JOB))
        m_inline.assert_not_called()

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

    def test_run_skips_when_claim_lost(self):
        """If the scheduler already holds the fire claim, do NOT double-run."""
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=False), \
             patch("cron.scheduler.run_one_job") as m_run, \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["success"] is True
        assert out["job"]["executed"] is False
        assert out["job"]["execution_success"] is False
        assert "execution_skipped" in out["job"]
        m_run.assert_not_called()  # claim lost -> never fired

    def test_run_reports_failure_from_last_status(self):
        """A failed run is reported via the re-read job's last_status/last_error."""
        failed = {"id": "job-run-1", "last_status": "error", "last_error": "provider 500"}
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
             patch("cron.scheduler.run_one_job", return_value=True), \
             patch("tools.cronjob_tools.get_job", return_value=failed):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["job"]["executed"] is True
        assert out["job"]["execution_success"] is False
        assert out["job"]["execution_error"] == "provider 500"

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
