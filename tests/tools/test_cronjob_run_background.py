"""Tests for cronjob action='run' background dispatch.

A manual `cronjob(action='run')` used to execute the job synchronously on the
calling agent's tool thread — a full agent run (minutes to hours) inside ONE
tool call, uninterruptible and serial. It now dispatches through the async
delegation registry (same rail as delegate_task background mode): the tool
returns immediately with a handle and the run's outcome re-enters the
conversation as a type='async_delegation' completion event.

Sync fallbacks preserved:
  - no routable session (direct Python callers, `hermes cron run`)
  - async delivery unsupported (one-shot runners, cron child sessions)
  - dispatch pool at capacity (claim already taken — must not strand it)
"""
import json
import threading
from unittest.mock import patch

from tools.cronjob_tools import (
    _try_dispatch_background_run,
    cronjob,
)


_JOB = {"id": "job-bg-1", "name": "bg run", "prompt": "hi",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"}}


def _job(job_id):
    """Per-test job dict with a UNIQUE id.

    Background workers outlive their test (daemon executor) and hold the id
    in the scheduler's shared running set until the run finishes; reusing one
    id across tests makes the in-flight dedupe guard see a phantom
    'already running' from a previous test's straggler worker.
    """
    return {"id": job_id, "name": f"bg run {job_id}", "prompt": "hi",
            "schedule": {"kind": "cron", "expr": "0 9 * * *"}}


def _bound_session_key(key="agent:main:telegram:dm:123"):
    """Context manager binding the approval session key contextvar."""
    import contextlib

    from tools.approval import _approval_session_key

    @contextlib.contextmanager
    def _cm():
        token = _approval_session_key.set(key)
        try:
            yield
        finally:
            _approval_session_key.reset(token)

    return _cm()


class TestBackgroundDispatch:
    def test_dispatches_and_returns_handle_immediately(self):
        """With a routable session, run claims sync then dispatches async."""
        run_started = threading.Event()
        run_release = threading.Event()

        def slow_run_one_job(job, **kw):
            run_started.set()
            assert run_release.wait(timeout=5.0)
            return True

        with _bound_session_key():
            with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True) as m_claim, \
                 patch("cron.scheduler.run_one_job", side_effect=slow_run_one_job), \
                 patch("tools.cronjob_tools.get_job",
                       return_value={"last_status": "ok", "last_error": None}):
                res = _try_dispatch_background_run(_job('job-bg-01'))

        try:
            # Returned BEFORE the job finished — that's the whole point.
            assert res is not None
            assert res["claimed"] is True
            assert res["dispatched"] is True
            assert res["delegation_id"]
            m_claim.assert_called_once_with("job-bg-01")
            # The job actually starts on the daemon executor.
            assert run_started.wait(timeout=5.0), "job never started in background"
        finally:
            run_release.set()

    def test_completion_event_reaches_shared_queue(self):
        """The finished run pushes a type='async_delegation' event carrying
        the job outcome onto process_registry.completion_queue."""
        import time

        from tools.process_registry import process_registry

        # The runner executes on a daemon thread — the patches must stay
        # active until the completion event lands, so poll INSIDE the blocks.
        with _bound_session_key("agent:main:telegram:dm:777"):
            with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
                 patch("cron.scheduler.run_one_job", return_value=True), \
                 patch("tools.cronjob_tools.get_job",
                       return_value={"last_status": "ok", "last_error": None,
                                     "next_run_at": "2026-08-07T09:00:00"}):
                res = _try_dispatch_background_run(_job('job-bg-02'))
                assert res["dispatched"] is True

                found = None
                for _ in range(100):
                    try:
                        evt = process_registry.completion_queue.get_nowait()
                    except Exception:
                        time.sleep(0.05)
                        continue
                    if (evt.get("type") == "async_delegation"
                            and evt.get("delegation_id") == res["delegation_id"]):
                        found = evt
                        break
                    process_registry.completion_queue.put(evt)
                    time.sleep(0.05)
        assert found is not None, "completion event never reached the queue"
        assert found["session_key"] == "agent:main:telegram:dm:777"
        assert found["status"] == "completed"
        assert "bg run" in (found.get("summary") or "")
        assert "Next scheduled run" in found["summary"]

    def test_failed_run_reports_error_status_in_event(self):
        import time

        from tools.process_registry import process_registry

        with _bound_session_key("agent:main:telegram:dm:778"):
            with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
                 patch("cron.scheduler.run_one_job", return_value=True), \
                 patch("tools.cronjob_tools.get_job",
                       return_value={"last_status": "error",
                                     "last_error": "provider exploded"}):
                res = _try_dispatch_background_run(_job('job-bg-03'))
                assert res["dispatched"] is True

                found = None
                for _ in range(100):
                    try:
                        evt = process_registry.completion_queue.get_nowait()
                    except Exception:
                        time.sleep(0.05)
                        continue
                    if evt.get("delegation_id") == res["delegation_id"]:
                        found = evt
                        break
                    process_registry.completion_queue.put(evt)
                    time.sleep(0.05)
        assert found is not None
        assert found["status"] == "error"
        assert "provider exploded" in (found.get("error") or "")

    def test_claim_lost_reports_immediately_without_dispatch(self):
        """Paused/already-firing jobs report in the tool response, not as a
        delayed completion event."""
        with _bound_session_key():
            with patch("tools.cronjob_tools.claim_job_for_fire", return_value=False), \
                 patch("tools.cronjob_tools.get_job",
                       return_value={**_JOB, "enabled": False}), \
                 patch("tools.async_delegation.dispatch_async_delegation") as m_disp:
                res = _try_dispatch_background_run(_job('job-bg-04'))
        assert res["claimed"] is False
        assert "paused/disabled" in res["error"]
        m_disp.assert_not_called()


class TestSyncFallbacks:
    def test_no_session_key_falls_back_to_sync(self):
        """Direct Python callers (no agent session) keep the sync path."""
        res = _try_dispatch_background_run(_job('job-bg-05'))
        assert res is None

    def test_async_delivery_unsupported_falls_back_to_sync(self):
        """One-shot runtimes (hermes -z, cron child, Kanban) keep sync."""
        with _bound_session_key():
            with patch("gateway.session_context.async_delivery_supported",
                       return_value=False):
                res = _try_dispatch_background_run(_job('job-bg-06'))
        assert res is None

    def test_pool_at_capacity_runs_inline(self):
        """A rejected dispatch must not strand the already-taken claim."""
        with _bound_session_key():
            with patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
                 patch("tools.async_delegation.dispatch_async_delegation",
                       return_value={"status": "rejected", "error": "capacity"}), \
                 patch("cron.scheduler.run_one_job", return_value=True) as m_run, \
                 patch("tools.cronjob_tools.get_job",
                       return_value={"last_status": "ok", "last_error": None}):
                res = _try_dispatch_background_run(_job('job-bg-07'))
        assert res["dispatched"] is False
        assert res["success"] is True
        m_run.assert_called_once()   # ran inline on this thread


class TestInFlightDedupe:
    """Manual runs must not double-fire a job that is already mid-run
    (salvaged from #53395 by @izumi0uu): the fire claim's 300s TTL is
    routinely outlived by real jobs, so the claim alone can't prevent it."""

    def test_run_claimed_job_skips_when_already_running(self):
        """The authoritative guard: _run_claimed_job refuses to fire a job
        whose id is already registered in the scheduler running set."""
        from cron import scheduler as sched
        from tools.cronjob_tools import _run_claimed_job

        assert sched.try_register_running_job("job-bg-08")   # simulate ticker mid-run
        try:
            with patch("cron.scheduler.run_one_job") as m_run:
                res = _run_claimed_job(_job('job-bg-08'))
            assert res["success"] is False
            assert "already running" in res["error"]
            m_run.assert_not_called()
        finally:
            sched.release_running_job("job-bg-08")

    def test_run_claimed_job_registers_and_releases(self):
        """A normal run holds the registration for run_one_job's duration and
        releases it after — visible to get_running_job_ids mid-run."""
        from cron import scheduler as sched
        from tools.cronjob_tools import _run_claimed_job

        seen_during_run = {}

        def probe_run(job, **kw):
            seen_during_run["registered"] = "job-bg-09" in sched.get_running_job_ids()
            return True

        with patch("cron.scheduler.run_one_job", side_effect=probe_run), \
             patch("tools.cronjob_tools.get_job",
                   return_value={"last_status": "ok", "last_error": None}):
            res = _run_claimed_job(_job('job-bg-09'))

        assert res["success"] is True
        assert seen_during_run["registered"] is True
        assert "job-bg-09" not in sched.get_running_job_ids()   # released after

    def test_background_dispatch_reports_running_job_immediately(self):
        """The dispatch path pre-checks the running set so a mid-run job
        reports in the tool response, not as a delayed completion event."""
        from cron import scheduler as sched

        assert sched.try_register_running_job("job-bg-10")
        try:
            with _bound_session_key():
                with patch("tools.cronjob_tools.claim_job_for_fire") as m_claim, \
                     patch("tools.async_delegation.dispatch_async_delegation") as m_disp:
                    res = _try_dispatch_background_run(_job('job-bg-10'))
            assert res["claimed"] is False
            assert "already running" in res["error"]
            m_claim.assert_not_called()   # no claim consumed for a skipped run
            m_disp.assert_not_called()
        finally:
            sched.release_running_job("job-bg-10")

    def test_ticker_guard_uses_shared_helpers(self):
        """The ticker's _submit_with_guard and manual runs share ONE dedupe
        owner: registration through either side blocks the other."""
        from cron import scheduler as sched

        # Manual-run registration…
        assert sched.try_register_running_job("job-shared-1")
        try:
            # …is exactly what the ticker-side helper consults.
            assert not sched.try_register_running_job("job-shared-1")
            assert "job-shared-1" in sched.get_running_job_ids()
        finally:
            sched.release_running_job("job-shared-1")
        assert "job-shared-1" not in sched.get_running_job_ids()
        # Idempotent release: never raises on a non-member.
        sched.release_running_job("job-shared-1")


class TestCronjobRunToolIntegration:
    def test_run_action_returns_background_note(self):
        """cronjob(action='run') surfaces the handle + do-not-wait note."""
        with _bound_session_key():
            with patch("tools.cronjob_tools.resolve_job_ref", return_value=_job('job-bg-12')), \
                 patch("tools.cronjob_tools.claim_job_for_fire", return_value=True), \
                 patch("cron.scheduler.run_one_job", return_value=True), \
                 patch("tools.cronjob_tools.get_job",
                       return_value={"id": "job-bg-12", "name": "bg run",
                                     "last_status": "ok", "last_error": None}):
                out = json.loads(cronjob(action="run", job_id="job-bg-12"))

        assert out["success"] is True
        assert out["job"]["executed"] is True
        assert out["job"]["execution_mode"] == "background"
        assert out["job"]["delegation_id"]
        assert "background" in out["note"]

    def test_run_action_sync_path_unchanged_without_session(self):
        """No session context → the legacy synchronous behavior (executed +
        execution_success populated from the completed run)."""
        ran = {"job": "after-run", "last_status": "ok", "last_error": None}
        with patch("tools.cronjob_tools.resolve_job_ref", return_value=_job('job-bg-13')), \
             patch("tools.cronjob_tools.claim_job_for_fire", return_value=True) as m_claim, \
             patch("cron.scheduler.run_one_job", return_value=True) as m_run, \
             patch("tools.cronjob_tools.get_job", return_value=ran):
            out = json.loads(cronjob(action="run", job_id="job-bg-13"))

        assert out["success"] is True
        assert out["job"]["executed"] is True
        assert out["job"]["execution_success"] is True
        m_claim.assert_called_once_with("job-bg-13")
        m_run.assert_called_once()
