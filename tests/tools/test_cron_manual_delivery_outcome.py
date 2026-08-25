"""Manual completion must report the exact execution's delivery outcome (#94926)."""

import time

import pytest


@pytest.mark.parametrize(
    "response,deliver,delivery_error,outcome,wording",
    [
        ("visible", "telegram:123", None, "delivered", "delivery confirmed"),
        ("[SILENT]", "telegram:123", None, "suppressed", "delivery suppressed"),
        ("", "telegram:123", None, "suppressed", "delivery suppressed"),
        ("visible", "local", None, "suppressed", "locally only"),
        ("visible", "telegram:123", "fixture send failed", "failed", "delivery FAILED"),
    ],
)
def test_completion_uses_real_execution_outcome(
    monkeypatch, response, deliver, delivery_error, outcome, wording
):
    from cron import executions, jobs, scheduler
    from tools import cronjob_tools as tool

    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **kw: (True, response, response, None)
    )
    sends = []

    def deliver_result(job, content, **kwargs):
        sends.append(content)
        return delivery_error

    monkeypatch.setattr(scheduler, "_deliver_result", deliver_result)
    job = jobs.create_job(prompt="fixture", schedule="every 1h", deliver=deliver,
                          no_agent=True, script="fixture.py")
    result = tool._execute_job_now(job)
    record = executions.latest_execution(job["id"])
    assert record["delivery_outcome"] == outcome
    assert bool(sends) == (bool(response) and response != "[SILENT]")
    completion = tool._manual_run_completion(
        result, job["id"], "fixture", deliver, time.time()
    )
    assert wording in completion["summary"]
    assert result["delivery_outcome"] == record["delivery_outcome"]


@pytest.mark.parametrize("outcome", [None, "unrecognized", "suppressed", "queued"])
def test_completion_does_not_infer_delivery_from_a_later_job_record(monkeypatch, outcome):
    from tools import cronjob_tools as tool

    monkeypatch.setattr(tool, "get_job", lambda job_id: {
        "last_status": "ok", "last_delivery_error": None,
    })
    monkeypatch.setattr(tool, "_latest_job_output_excerpt", lambda job_id: None)
    result = {"success": True, "delivery_outcome": outcome}
    summary = tool._manual_run_completion(
        result, "fixture", "fixture", "telegram:123", time.time()
    )["summary"]
    assert "delivered there" not in summary
    assert "delivery confirmed" not in summary
    assert ("unverified" in summary or "suppressed" in summary)
