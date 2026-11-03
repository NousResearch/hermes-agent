"""A manual cron run's completion notice reports the model and API calls its run session recorded.

The background ``cronjob(action='run')`` completion used to render ``Model: ?`` and ``API calls: 0``
whatever the job did: the runner hard-coded ``api_calls: 0`` and the dispatch only knew the job's
model pin (None for a job on the default model). The run's own session in state.db is the record.
"""
from unittest.mock import patch

import pytest

from cron import jobs
from tests.fakes.cron_owner import admit_fire, cron_owner
from tools.cronjob_tools import _try_dispatch_background_run
from tools.process_registry_notifications import format_process_notification


@pytest.fixture()
def owner(tmp_path, monkeypatch):
    authority = cron_owner(tmp_path / "home", monkeypatch)
    yield authority
    authority.db.close()


def _completion(job, run_one_job):
    """Dispatch ``job`` through the real background path and return its completion event."""
    from tools.approval_context import _approval_session_key
    from tools.process_registry import process_registry

    token = _approval_session_key.set("agent:main:telegram:dm:4242")
    others = []
    try:
        with patch("tools.cronjob_tools.claim_job_for_fire", side_effect=lambda jid, **kw: dict(job)), \
             patch("cron.scheduler.run_one_job", side_effect=run_one_job):
            res = _try_dispatch_background_run(dict(job))
            assert res["dispatched"] is True
            while True:  # a blocking get is the signal; Empty after the deadline fails the test
                evt = process_registry.completion_queue.get(timeout=10)
                if evt.get("delegation_id") == res["delegation_id"]:
                    return evt
                others.append(evt)
    finally:
        for evt in others:
            process_registry.completion_queue.put(evt)
        _approval_session_key.reset(token)


@pytest.mark.asyncio
async def test_notice_carries_the_run_sessions_model_and_api_calls(owner):
    job = jobs.create_job(prompt="report", schedule="every 1h", deliver="local")
    run_sid, fire = await admit_fire(owner, job["id"])
    owner.db._conn.execute("UPDATE sessions SET model = ?, api_call_count = ? WHERE id = ?",
                           ("vendor/run-model", 3, run_sid))
    owner.db._conn.commit()

    def run_one_job(claimed, **kw):
        claimed["execution_id"] = fire  # what the real run_one_job stamps on the claimed snapshot
        jobs.mark_job_run(claimed["id"], True)
        return True

    evt = _completion(job, run_one_job)
    assert (evt.get("model"), evt.get("api_calls")) == ("vendor/run-model", 3)
    text = format_process_notification(evt)
    assert "Role: cron_run   Model: vendor/run-model" in text
    assert "API calls: 3" in text


def test_notice_omits_fields_no_run_session_recorded(owner):
    job = jobs.create_job(prompt="report", schedule="every 1h", deliver="local")

    def run_one_job(claimed, **kw):  # no execution id, no run session (e.g. a refused fire)
        jobs.mark_job_run(claimed["id"], True)
        return True

    evt = _completion(job, run_one_job)
    assert "api_calls" not in evt and not evt.get("model")
    text = format_process_notification(evt)
    assert "Model: ?" not in text and "API calls" not in text
    assert "Role: cron_run" in text


def test_refused_fire_of_a_pinned_job_prints_no_model(owner):
    """The job's model pin is not what ran: a fire with no run session reports no model at all."""
    job = jobs.create_job(prompt="report", schedule="every 1h", deliver="local", model="vendor/pinned-model")

    def run_one_job(claimed, **kw):  # refused before any run session existed
        jobs.mark_job_run(claimed["id"], False, "refused")
        return False

    evt = _completion(job, run_one_job)
    assert not evt.get("model")
    assert "pinned-model" not in format_process_notification(evt)
