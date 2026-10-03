"""Test that action='run' always clears the in-memory firing lock (#107559).

Before this fix, background manual runs left a stale in-memory firing flag
after completion, permanently blocking subsequent `action=run` calls until
gateway restart.

Root cause: the `run` handler fires background jobs on a worker thread (via
`_try_dispatch_background_run` -> `_runner()` closure -> `_run_claimed_job`),
but the lock-release code in `_run_claimed_job` relied on a local `_registered`
variable that was False in the worker's stack, so `release_running_job()` was
never called.

Fix: unconditionally call `release_running_job()` in both the `finally` block
and the `except` block, removing the `_registered` guard entirely.
"""
import json
from unittest.mock import patch, MagicMock

from tools.cronjob_tools import cronjob, _run_claimed_job


_JOB = {
    "id": "job-107559",
    "name": "test lock release",
    "prompt": "hello",
    "schedule": {"kind": "cron", "expr": "0 9 * * *"},
}


def test_run_claimed_job_always_releases_lock_on_success():
    """Successful runs MUST release the in-memory firing lock."""
    released = []
    claimed_job = {**_JOB, "fire_claim": {"by": "manual-owner"}}
    
    with patch("cron.scheduler.try_register_running_job", return_value=True), \
         patch("cron.scheduler.run_one_job", return_value=True), \
         patch("cron.scheduler.release_running_job", side_effect=lambda job_id: released.append(job_id)), \
         patch("tools.cronjob_tools.get_job", return_value={"last_status": "ok", "last_error": None}):
        res = _run_claimed_job(claimed_job)
    
    assert res["claimed"] is True
    assert res["success"] is True
    assert released == ["job-107559"], "Lock must be released exactly once"


def test_run_claimed_job_always_releases_lock_on_exception():
    """Runs that raise an exception MUST release the in-memory firing lock."""
    released = []
    claimed_job = {**_JOB, "fire_claim": {"by": "manual-owner"}}
    
    with patch("cron.scheduler.try_register_running_job", return_value=True), \
         patch("cron.scheduler.run_one_job", side_effect=RuntimeError("boom")), \
         patch("cron.scheduler.release_running_job", side_effect=lambda job_id: released.append(job_id)), \
         patch("tools.cronjob_tools.mark_job_run"):
        res = _run_claimed_job(claimed_job)
    
    assert res["claimed"] is True
    assert res["success"] is False
    assert "boom" in res["error"]
    assert released == ["job-107559"], "Lock must be released even on exception"


def test_manual_run_twice_does_not_block_second_fire():
    """Two sequential manual runs of the same job must both succeed (#107559).
    
    Regression test: before the fix, the second `action=run` returned
    `execution_skipped: "Job is already being fired by the scheduler"`
    indefinitely because the first run left a stale in-memory flag.
    """
    refreshed = {"id": "job-107559", "last_status": "ok", "last_error": None}
    claimed_job = {**_JOB, "fire_claim": {"by": "manual-owner"}}
    
    # Track registration state across both fires
    registered_jobs = set()
    
    def mock_try_register(job_id):
        if job_id in registered_jobs:
            return False  # Already running
        registered_jobs.add(job_id)
        return True
    
    def mock_release(job_id):
        registered_jobs.discard(job_id)
    
    with patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)), \
         patch("tools.cronjob_tools.claim_job_for_fire", return_value=claimed_job), \
         patch("cron.scheduler.try_register_running_job", side_effect=mock_try_register), \
         patch("cron.scheduler.run_one_job", return_value=True), \
         patch("cron.scheduler.release_running_job", side_effect=mock_release), \
         patch("tools.cronjob_tools.get_job", return_value=refreshed), \
         patch("tools.cronjob_tools._notify_provider_jobs_changed_safe"):
        
        # First fire
        out1 = json.loads(cronjob(action="run", job_id="job-107559"))
        assert out1["success"] is True, f"First fire failed: {out1}"
        assert out1["job"]["executed"] is True
        assert out1["job"]["execution_success"] is True
        assert "execution_skipped" not in out1["job"]
        
        # Second fire (reproducer for #107559)
        out2 = json.loads(cronjob(action="run", job_id="job-107559"))
        assert out2["success"] is True, f"Second fire failed: {out2}"
        assert out2["job"]["executed"] is True, \
            f"Second fire was skipped: {out2.get('job', {}).get('execution_skipped')}"
        assert out2["job"]["execution_success"] is True
        assert "execution_skipped" not in out2["job"], \
            f"Second fire was blocked: {out2['job']['execution_skipped']}"
