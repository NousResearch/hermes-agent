"""Synthetic behavioral evidence for cron manual-completion follow-ups.

No provider, platform transport, private archive, or external worker is used.
The job store, claims, scheduler bookkeeping and execution ledger are real.
"""

from types import SimpleNamespace

import pytest


@pytest.fixture
def manual_run(tmp_path, monkeypatch):
    from cron import delivery_queue, jobs, scheduler
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from tools import cronjob_tools

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)
    state = SimpleNamespace(response="Synthetic answer", delivery_error=None, sends=[])
    monkeypatch.setattr(
        scheduler, "run_job",
        lambda job, **kwargs: (
            True, "## Response\n\n" + state.response, state.response, None,
        ),
    )

    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})

    async def send(platform, config, chat_id, content, **kwargs):
        state.sends.append(content)
        if state.delivery_error:
            return {"success": False, "error": state.delivery_error}
        return {"success": True}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", send)
    enqueue = delivery_queue.enqueue

    def enqueue_and_drain(*args, **kwargs):
        queued = enqueue(*args, **kwargs)
        assert scheduler.drain_delivery_queue({}, None) == 1
        return queued

    monkeypatch.setattr(delivery_queue, "enqueue", enqueue_and_drain)
    with jobs.use_cron_store(tmp_path):
        state.job = jobs.create_job(
            prompt="Synthetic task", schedule="every 1h", deliver="telegram:fixture",
        )
        state.run = lambda: cronjob_tools._execute_job_now(state.job)
        yield state


@pytest.mark.parametrize("first_fails", [False, True])
def test_producer_uses_own_delivery_outcome_after_release(manual_run, monkeypatch, first_fails):
    from cron import executions, jobs, scheduler

    state = manual_run
    failure = "Synthetic transport refusal"
    state.delivery_error = failure if first_fails else None
    original_release = scheduler.release_running_job
    records = []
    later_results = []

    def release_then_run(job_id, **kwargs):
        original_release(job_id, **kwargs)
        monkeypatch.setattr(scheduler, "release_running_job", original_release)
        records.append(executions.latest_execution(job_id))
        state.delivery_error = None if first_fails else failure
        later_results.append(state.run())
        records.append(executions.latest_execution(job_id))

    monkeypatch.setattr(scheduler, "release_running_job", release_then_run)
    first = state.run()
    assert len(state.sends) == 2
    assert records[0]["id"] != records[1]["id"]
    assert records[0]["delivery_outcome"] == ("failed" if first_fails else "delivered")
    assert records[1]["delivery_outcome"] == ("delivered" if first_fails else "failed")
    assert jobs.get_job(state.job["id"])["last_status"] == ("ok" if first_fails else "delivery_failed")
    assert later_results[0]["success"] is first_fails
    assert first["success"] is not first_fails
    if first_fails:
        assert failure in first["error"]
    else:
        assert first["error"] is None
