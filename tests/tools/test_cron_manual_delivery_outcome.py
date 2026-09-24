"""Manual completion must report the exact execution's delivery outcome (#94926)."""

import time

import pytest


@pytest.mark.parametrize(
    "response,deliver,delivery_error,outcome,wording,run_error,failure_deliver",
    [
        ("visible", "telegram:123", None, "delivered", "delivery confirmed", None, None),
        ("[SILENT]", "telegram:123", None, "suppressed", "delivery suppressed", None, None),
        ("", "telegram:123", None, "suppressed", "delivery suppressed", None, None),
        ("visible", "local", None, "suppressed", "locally only", None, None),
        ("visible", "telegram:123", "fixture send failed", "failed", "delivery FAILED", None, "discord:456"),
        ("visible", "local", None, "delivered", "delivery confirmed", "fixture failure", "telegram:456"),
        ("visible", "telegram:123", None, "suppressed", "locally only", "fixture failure", "local"),
    ],
)
def test_completion_uses_real_execution_outcome(
    monkeypatch, response, deliver, delivery_error, outcome, wording, run_error, failure_deliver
):
    from cron import executions, jobs, scheduler
    from tools import cronjob_tools as tool

    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **kw: (not run_error, response, response, run_error)
    )
    sends = []

    def deliver_result(job, content, **kwargs):
        sends.append(content)
        return delivery_error

    monkeypatch.setattr(scheduler, "_deliver_result", deliver_result)
    job = jobs.create_job(prompt="fixture", schedule="every 1h", deliver=deliver,
                          no_agent=True, script="fixture.py", failure_deliver=failure_deliver)
    result = tool._execute_job_now(job)
    record = executions.latest_execution(job["id"])
    assert record["delivery_outcome"] == outcome
    assert bool(sends) == (bool(response) and response != "[SILENT]")
    completion = tool._manual_run_completion(
        result, job["id"], "fixture", deliver, time.time()
    )
    assert wording in completion["summary"]
    assert result["delivery_outcome"] == record["delivery_outcome"]
    expected_target = (failure_deliver or deliver) if run_error else deliver
    assert f"Delivery target: {expected_target}" in completion["summary"]


@pytest.mark.parametrize("outcome,target", [
    (None, "telegram:123"), ("unrecognized", "telegram:123"),
    ("suppressed", "telegram:123"), ("queued", "telegram:123"), (None, "local"),
])
def test_completion_does_not_infer_delivery_from_a_later_job_record(monkeypatch, outcome, target):
    from tools import cronjob_tools as tool

    if target == "local":
        from cron import jobs, scheduler

        def fail_dispatch(job):
            raise RuntimeError("fixture worker dispatch failure")

        monkeypatch.setattr(scheduler, "_launch_external_cron_worker", fail_dispatch)
        job = jobs.create_job(prompt="fixture", schedule="every 1h", deliver="local")
        result = tool._execute_job_now(job)
        assert not result["success"]
        assert result["delivery_outcome"] is None
    else:
        result = {"success": True, "delivery_outcome": outcome}
    monkeypatch.setattr(tool, "get_job", lambda job_id: {
        "last_status": "ok", "last_delivery_error": None,
    })
    monkeypatch.setattr(tool, "_latest_job_output_excerpt", lambda job_id: None)
    summary = tool._manual_run_completion(
        result, "fixture", "fixture", target, time.time()
    )["summary"]
    assert "delivered there" not in summary
    assert "delivery confirmed" not in summary
    assert "saved locally" not in summary
    assert ("unverified" in summary or "suppressed" in summary)
