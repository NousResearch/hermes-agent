"""Tests for cronjob action='run' non-blocking execution (#52705).

Before #52705, `cronjob(action='run')` called ``run_one_job`` synchronously
inside the tool handler, blocking the calling agent until the entire cron
session (LLM loop, output save, delivery) completed. Now action='run' is
non-blocking:

- Gateway alive → arms the job via ``trigger_job`` (sets next_run_at=now) so
  the scheduler ticker fires it asynchronously on its next cycle.
- Gateway not alive → claims via ``claim_job_for_fire`` and spawns a daemon
  thread that calls ``run_one_job``.

Either way the tool returns immediately. The caller checks post-run status via
``cronjob(action='list')``.
"""
import json
import threading
from unittest.mock import patch, MagicMock

from tools.cronjob_tools import cronjob


_JOB = {"id": "job-run-1", "name": "manual run", "prompt": "hi",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"}}


class TestCronjobRunNonBlocking:
    """action='run' must never block the calling agent (#52705)."""

    # ── Gateway alive path ──────────────────────────────────────────────

    def test_run_uses_scheduler_path_when_gateway_alive(self):
        """When the gateway is running, action='run' arms the job via
        trigger_job (sets next_run_at=now) and does NOT call run_one_job
        synchronously."""
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools._is_gateway_active", return_value=True), \
             patch("tools.cronjob_tools.trigger_job", return_value=dict(_JOB, next_run_at="now")) as m_trigger, \
             patch("cron.scheduler.run_one_job") as m_run, \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["success"] is True
        assert out["job"]["triggered"] is True
        assert out["job"]["trigger_mode"] == "scheduled"
        m_trigger.assert_called_once_with("job-run-1")
        m_run.assert_not_called()

    def test_run_never_blocks_when_gateway_alive(self):
        """The tool must return immediately even if trigger_job would 'take
        forever' — proving no synchronous work leak."""
        import time as _time

        def slow_trigger(_jid):
            _time.sleep(0.5)  # simulate scheduling overhead
            return dict(_JOB, next_run_at="now")

        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools._is_gateway_active", return_value=True), \
             patch("tools.cronjob_tools.trigger_job", side_effect=slow_trigger), \
             patch("cron.scheduler.run_one_job") as m_run, \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["success"] is True
        m_run.assert_not_called()

    # ── Gateway not-alive path ──────────────────────────────────────────

    def test_run_spawns_thread_when_gateway_not_alive(self):
        """When the gateway is NOT running, action='run' claims the job and
        spawns a daemon thread — never calling run_one_job synchronously."""
        mock_thread = MagicMock()
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools._is_gateway_active", return_value=False), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=True) as m_claim, \
             patch("cron.scheduler.run_one_job") as m_run, \
             patch("threading.Thread", return_value=mock_thread) as m_thread_cls, \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["success"] is True
        assert out["job"]["triggered"] is True
        assert out["job"]["trigger_mode"] == "background"
        m_claim.assert_called_once_with("job-run-1")
        m_run.assert_not_called()  # NOT called synchronously
        m_thread_cls.assert_called_once()
        mock_thread.start.assert_called_once()
        # Thread must be daemon so it doesn't block process exit
        assert m_thread_cls.call_args.kwargs.get("daemon") is True

    def test_run_skips_when_claim_lost_no_thread(self):
        """In daemon-thread mode, a lost claim means no thread spawned."""
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools._is_gateway_active", return_value=False), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=False), \
             patch("threading.Thread") as m_thread, \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["success"] is True
        assert out["job"]["executed"] is False
        assert "execution_skipped" in out["job"]
        m_thread.assert_not_called()

    def test_run_no_claim_no_trigger_when_claim_lost(self):
        """Claim-lost in daemon mode: trigger_job must NOT be called either."""
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools._is_gateway_active", return_value=False), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=False), \
             patch("tools.cronjob_tools.trigger_job") as m_trigger, \
             patch("threading.Thread"), \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
            json.loads(cronjob(action="run", job_id="job-run-1"))
        m_trigger.assert_not_called()

    # ── Response contract ───────────────────────────────────────────────

    def test_run_response_no_execution_success_field(self):
        """The response must NOT include execution_success — the job hasn't
        finished when the tool returns."""
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools._is_gateway_active", return_value=True), \
             patch("tools.cronjob_tools.trigger_job", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))
        assert "execution_success" not in out["job"]
        assert "execution_error" not in out["job"]
        assert out["job"]["triggered"] is True

    def test_run_response_includes_trigger_mode(self):
        """Both paths must report which mode was used."""
        for alive, expected_mode in [(True, "scheduled"), (False, "background")]:
            mock_thread = MagicMock()
            with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
                 patch("tools.cronjob_tools._is_gateway_active", return_value=alive), \
                 patch("tools.cronjob_tools.trigger_job", return_value=dict(_JOB)), \
                 patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
                 patch("threading.Thread", return_value=mock_thread), \
                 patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
                out = json.loads(cronjob(action="run", job_id="job-run-1"))
            assert out["job"]["trigger_mode"] == expected_mode, \
                f"mode mismatch for alive={alive}"

    # ── Core invariant: NEVER synchronous run_one_job ───────────────────

    def test_run_never_calls_run_one_job_synchronously(self):
        """Regardless of gateway state, run_one_job must NOT be called inline."""
        for alive in (True, False):
            mock_thread = MagicMock()
            with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
                 patch("tools.cronjob_tools._is_gateway_active", return_value=alive), \
                 patch("tools.cronjob_tools.trigger_job", return_value=dict(_JOB)), \
                 patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
                 patch("cron.scheduler.run_one_job") as m_run, \
                 patch("threading.Thread", return_value=mock_thread), \
                 patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
                cronjob(action="run", job_id="job-run-1")
            m_run.assert_not_called(), \
                f"run_one_job called synchronously when alive={alive}"

    # ── Background worker ───────────────────────────────────────────────

    def test_background_worker_runs_job_and_handles_errors(self):
        """The daemon thread worker must call run_one_job and mark failure on
        exception."""
        from tools.cronjob_tools import _run_job_in_background

        with patch("cron.scheduler.run_one_job", side_effect=RuntimeError("boom")), \
             patch("tools.cronjob_tools.mark_job_run") as m_mark:
            _run_job_in_background(dict(_JOB))

        m_mark.assert_called_once()
        assert "boom" in m_mark.call_args[0][2]

    def test_background_worker_succeeds_silently(self):
        """The daemon thread worker runs run_one_job without error on success."""
        from tools.cronjob_tools import _run_job_in_background

        with patch("cron.scheduler.run_one_job", return_value=True), \
             patch("tools.cronjob_tools.mark_job_run") as m_mark:
            _run_job_in_background(dict(_JOB))

        # mark_job_run is NOT called on success — run_one_job handles it internally
        m_mark.assert_not_called()
