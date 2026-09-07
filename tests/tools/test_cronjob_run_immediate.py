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
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from tools.cronjob_tools import cronjob, _execute_job_now
from tools.environments.base import set_activity_callback


_JOB = {"id": "job-run-1", "name": "manual run", "prompt": "hi",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"}}


class TestCronjobRunExecutesImmediately:
    def test_run_action_claims_and_fires_via_run_one_job(self):
        """action='run' must claim the job then fire it through run_one_job."""
        ran = {"job": "after-run", "last_status": "ok", "last_error": None}
        claimed = {**_JOB, "fire_claim": {"by": "manual-owner"}}
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=claimed) as m_claim, \
             patch("cron.scheduler.run_one_job", return_value=True) as m_run, \
             patch("tools.cronjob_tools.get_job", return_value=ran):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["success"] is True
        assert out["job"]["executed"] is True
        assert out["job"]["execution_success"] is True
        m_claim.assert_called_once_with("job-run-1", return_job=True)
        m_run.assert_called_once_with(claimed, adapters=None, loop=None, extra_prompt=None)

    def test_run_reconciles_external_provider_after_claimed_execution(self):
        """A direct run must re-arm Chronos after it advances next_run_at.

        Otherwise a scheduled Chronos fire that loses its claim to this direct
        run is consumed without a successor one-shot, permanently stalling the
        recurring job.
        """
        order = []
        ran = {"id": "job-run-1", "last_status": "ok", "last_error": None}
        claimed = {**_JOB, "fire_claim": {"by": "manual-owner"}}
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=claimed), \
             patch("cron.scheduler.run_one_job",
                   side_effect=lambda *a, **kw: order.append("run") or True), \
             patch("tools.cronjob_tools.get_job", return_value=ran), \
             patch("tools.cronjob_tools._notify_provider_jobs_changed_safe",
                   side_effect=lambda: order.append("notify")) as m_notify:
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["job"]["executed"] is True
        m_notify.assert_called_once_with()
        # Reconcile only AFTER the run persisted its final state (mark_job_run
        # inside run_one_job), so the provider arms the post-run next_run_at.
        assert order == ["run", "notify"]

    def test_run_reconciles_external_provider_even_when_claimed_run_fails(self):
        """A claimed direct run advances next_run_at at claim time, so the
        provider must be reconciled even when the execution itself fails."""
        failed = {"id": "job-run-1", "last_status": "error", "last_error": "provider 500"}
        claimed = {**_JOB, "fire_claim": {"by": "manual-owner"}}
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=claimed), \
             patch("cron.scheduler.run_one_job", side_effect=RuntimeError("boom")), \
             patch("tools.cronjob_tools.mark_job_run"), \
             patch("tools.cronjob_tools.get_job", return_value=failed), \
             patch("tools.cronjob_tools._notify_provider_jobs_changed_safe") as m_notify:
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["job"]["executed"] is True
        assert out["job"]["execution_success"] is False
        m_notify.assert_called_once_with()

    def test_run_skips_when_claim_lost(self):
        """If the scheduler already holds the fire claim, do NOT double-run."""
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=False), \
             patch("cron.scheduler.run_one_job") as m_run, \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools._notify_provider_jobs_changed_safe") as m_notify:
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["success"] is True
        assert out["job"]["executed"] is False
        assert out["job"]["execution_success"] is False
        assert "execution_skipped" in out["job"]
        m_run.assert_not_called()  # claim lost -> never fired
        m_notify.assert_not_called()  # the winning scheduler owns the re-arm

    def test_run_reports_failure_from_last_status(self):
        """A failed run is reported via the re-read job's last_status/last_error."""
        failed = {"id": "job-run-1", "last_status": "error", "last_error": "provider 500"}
        claimed = {**_JOB, "fire_claim": {"by": "manual-owner"}}
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=claimed), \
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

    def test_execute_job_now_passes_live_gateway_context_to_delivery(self):
        """Manual runs must deliver on the live gateway adapter's owning loop."""
        adapters = {"matrix": object()}
        gateway_loop = object()
        runner = SimpleNamespace(adapters=adapters, _gateway_loop=gateway_loop)
        completed = {"id": "job-run-1", "last_status": "ok", "last_error": None}

        with patch("tools.cronjob_tools.claim_job_for_fire", return_value={**_JOB, "fire_claim": {"by": "manual-owner"}}), \
             patch("gateway.run._gateway_runner_ref", return_value=runner), \
             patch("cron.scheduler.run_one_job", return_value=True) as m_run, \
             patch("tools.cronjob_tools.get_job", return_value=completed):
            res = _execute_job_now(dict(_JOB))

        assert res["success"] is True
        m_run.assert_called_once_with(
            {**_JOB, "fire_claim": {"by": "manual-owner"}},
            adapters=adapters,
            loop=gateway_loop,
            extra_prompt=None,
        )

    def test_execute_job_now_remains_standalone_without_gateway(self):
        """CLI-only runs retain the standalone delivery path."""
        completed = {"id": "job-run-1", "last_status": "ok", "last_error": None}

        with patch("tools.cronjob_tools.claim_job_for_fire", return_value={**_JOB, "fire_claim": {"by": "manual-owner"}}), \
             patch.dict(sys.modules, {"gateway.run": None}), \
             patch("cron.scheduler.run_one_job", return_value=True) as m_run, \
             patch("tools.cronjob_tools.get_job", return_value=completed):
            res = _execute_job_now(dict(_JOB))

        assert res["success"] is True
        m_run.assert_called_once_with(
            {**_JOB, "fire_claim": {"by": "manual-owner"}},
            adapters=None,
            loop=None,
            extra_prompt=None,
        )

    def test_execute_job_now_marks_failure_on_exception(self):
        """An exception during fire is captured, marked failed, not propagated."""
        claimed = {**_JOB, "fire_claim": {"by": "manual-owner"}}
        with patch("tools.cronjob_tools.claim_job_for_fire", return_value=claimed), \
             patch("cron.scheduler.run_one_job", side_effect=RuntimeError("boom")), \
             patch("tools.cronjob_tools.mark_job_run") as m_mark, \
             patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)):
            res = _execute_job_now(dict(_JOB))
        assert res["claimed"] is True
        assert res["success"] is False
        assert "boom" in res["error"]
        m_mark.assert_called_once_with(
            "job-run-1",
            False,
            "boom",
            expected_fire_owner="manual-owner",
        )

    def test_execute_job_now_heartbeats_while_job_runs(self):
        """A manual run ticks the caller's activity tracker while the job
        executes so the gateway inactivity watchdog doesn't kill the parent
        turn (#76502)."""
        touches = []
        heartbeat_seen = threading.Event()

        def record(desc):
            touches.append(desc)
            heartbeat_seen.set()

        set_activity_callback(record)
        try:
            def slow_run(job, **kw):
                # Deterministic: block until at least one heartbeat has fired
                # (bounded so a broken heartbeat can't hang the test).
                assert heartbeat_seen.wait(timeout=5.0), "no heartbeat within 5s"
                return True

            with patch("tools.cronjob_tools.claim_job_for_fire", return_value={**_JOB, "fire_claim": {"by": "manual-owner"}}), \
                 patch("tools.cronjob_tools._CRON_RUN_HEARTBEAT_INTERVAL", 0.05), \
                 patch("cron.scheduler.run_one_job", side_effect=slow_run) as m_run, \
                 patch("tools.cronjob_tools.get_job",
                       return_value={"last_status": "ok", "last_error": None}):
                res = _execute_job_now(dict(_JOB))

            m_run.assert_called_once()
            assert res["success"] is True, res
            assert any("cronjob: running job" in t for t in touches), touches
        finally:
            set_activity_callback(None)

    def test_execute_job_now_without_callback_does_not_heartbeat(self):
        """No activity callback registered (direct callers, tests) → the
        heartbeat thread is never started and behavior is unchanged."""
        set_activity_callback(None)
        try:
            with patch("tools.cronjob_tools.claim_job_for_fire", return_value={**_JOB, "fire_claim": {"by": "manual-owner"}}), \
                 patch("cron.scheduler.run_one_job", return_value=True) as m_run, \
                 patch("tools.cronjob_tools.get_job",
                       return_value={"last_status": "ok", "last_error": None}), \
                 patch("tools.cronjob_tools.threading.Thread") as m_thread:
                res = _execute_job_now(dict(_JOB))
            assert res["success"] is True
            m_run.assert_called_once()
            m_thread.assert_not_called()   # heartbeat thread truly never created
        finally:
            set_activity_callback(None)

    def test_heartbeat_stops_at_ceiling_but_job_completes(self):
        """Past _CRON_RUN_HEARTBEAT_CEILING the heartbeat stops (so the
        gateway watchdog regains authority over a wedged run) while the job
        itself keeps running to completion."""
        touches = []
        first_beat = threading.Event()

        def record(desc):
            touches.append(desc)
            first_beat.set()

        set_activity_callback(record)
        try:
            def slow_run(job, **kw):
                # Ceiling=0 → the very first wake stops the loop without
                # touching. Give it a couple of cycles to prove silence.
                time.sleep(0.2)
                return True

            with patch("tools.cronjob_tools.claim_job_for_fire", return_value={**_JOB, "fire_claim": {"by": "manual-owner"}}), \
                 patch("tools.cronjob_tools._CRON_RUN_HEARTBEAT_INTERVAL", 0.05), \
                 patch("tools.cronjob_tools._CRON_RUN_HEARTBEAT_CEILING", 0.0), \
                 patch("cron.scheduler.run_one_job", side_effect=slow_run), \
                 patch("tools.cronjob_tools.get_job",
                       return_value={"last_status": "ok", "last_error": None}):
                res = _execute_job_now(dict(_JOB))
            assert res["success"] is True, res
            assert not first_beat.is_set(), touches   # heartbeat never fired
        finally:
            set_activity_callback(None)

    def test_heartbeat_survives_callback_exception(self):
        """One raising callback must not silently kill watchdog protection
        for the rest of a long job — the loop continues heartbeating."""
        calls = []
        second_beat = threading.Event()

        def flaky(desc):
            calls.append(desc)
            if len(calls) >= 2:
                second_beat.set()
            if len(calls) == 1:
                raise RuntimeError("transient")

        set_activity_callback(flaky)
        try:
            def slow_run(job, **kw):
                # Block until a heartbeat AFTER the raising one has fired.
                assert second_beat.wait(timeout=5.0), \
                    "heartbeat stopped after one callback exception"
                return True

            with patch("tools.cronjob_tools.claim_job_for_fire", return_value={**_JOB, "fire_claim": {"by": "manual-owner"}}), \
                 patch("tools.cronjob_tools._CRON_RUN_HEARTBEAT_INTERVAL", 0.05), \
                 patch("cron.scheduler.run_one_job", side_effect=slow_run), \
                 patch("tools.cronjob_tools.get_job",
                       return_value={"last_status": "ok", "last_error": None}):
                res = _execute_job_now(dict(_JOB))
            assert res["success"] is True, res
            assert len(calls) >= 2, calls
        finally:
            set_activity_callback(None)


class TestManualRunReportsDeliveryFailure:
    """#83993: a manual run whose agent succeeded but whose delivery failed
    must not come back as success=True with no error — the calling agent
    relays that result to the user."""

    def test_delivery_failed_status_is_not_success_and_surfaces_reason(self):
        refreshed = {
            "id": "job-run-1",
            "last_status": "delivery_failed",
            "last_error": None,
            "last_delivery_error": "live adapter send failed: 502 (target telegram:123)",
        }
        with patch("tools.cronjob_tools.claim_job_for_fire",
                   return_value={**_JOB, "fire_claim": {"by": "manual-owner"}}), \
             patch("cron.scheduler.run_one_job", return_value=True), \
             patch("tools.cronjob_tools.get_job", return_value=refreshed):
            res = _execute_job_now(dict(_JOB))

        assert res["claimed"] is True
        assert res["success"] is False
        assert "502" in res["error"]

    def test_plain_ok_is_still_success_with_no_error(self):
        with patch("tools.cronjob_tools.claim_job_for_fire",
                   return_value={**_JOB, "fire_claim": {"by": "manual-owner"}}), \
             patch("cron.scheduler.run_one_job", return_value=True), \
             patch("tools.cronjob_tools.get_job",
                   return_value={"id": "job-run-1", "last_status": "ok", "last_error": None,
                                 "last_delivery_error": None}):
            res = _execute_job_now(dict(_JOB))
        assert res["success"] is True
        assert res["error"] is None
