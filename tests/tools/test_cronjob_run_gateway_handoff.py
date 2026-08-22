"""Regression tests for manual cron runs yielding continuable delivery to gateway."""

from __future__ import annotations


def _job(job_id: str = "manual-gateway-handoff") -> dict:
    return {
        "id": job_id,
        "name": "manual handoff",
        "prompt": "brief me",
        "deliver": "discord:123456789",
        "attach_to_session": True,
        "schedule": {"kind": "interval", "seconds": 3600},
    }


def test_execute_job_now_queues_for_gateway_before_taking_fire_claim(monkeypatch):
    from cron import scheduler as sched
    from tools import cronjob_tools

    monkeypatch.setattr(sched, "_should_yield_cron_to_live_gateway", lambda job: True)

    updates: list[tuple[str, dict]] = []
    claims: list[str] = []
    monkeypatch.setattr(
        cronjob_tools,
        "update_job",
        lambda job_id, update: (
            updates.append((job_id, update)) or {**_job(job_id), **update}
        ),
    )
    monkeypatch.setattr(
        cronjob_tools,
        "claim_job_for_fire",
        lambda job_id, **_: (
            claims.append(job_id) or {**_job(job_id), "fire_claim": {"by": "owner"}}
        ),
    )

    result = cronjob_tools._execute_job_now(_job())

    assert result["claimed"] is False
    assert result["success"] is True
    assert result["queued_for_gateway"] is True
    assert "live gateway" in result["error"]
    assert updates and updates[0][0] == "manual-gateway-handoff"
    assert "next_run_at" in updates[0][1]
    assert claims == []
