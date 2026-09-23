"""Manual runs of a satellite job that delivers through the primary gateway's profile route.

Regression for #120330. A multiplexed satellite profile with no ``platforms.<p>`` credential of
its own posts through the primary's bot via a root ``gateway.profile_routes`` entry, so only the
gateway process holding that bot can deliver its jobs. ``hermes -p <profile> cron run`` used to
run the whole agent turn in the CLI process anyway, fail delivery with ``platform 'telegram' not
configured/enabled`` and overwrite the job's ``last_status`` with ``delivery_failed``.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.cron import _run_outcome

ROUTED_TARGET = "telegram:-1004306455751:14"


@pytest.fixture
def keeper_job(tmp_path, monkeypatch):
    """A ``keeper`` satellite home whose primary routes one Telegram topic to it, plus one job there."""
    root = tmp_path / "root"
    keeper_home = root / "profiles" / "keeper"
    keeper_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({"gateway": {
        "multiplex_profiles": True,
        "profile_routes": [{"name": "ops", "platform": "telegram", "chat_id": "-1004306455751",
                            "thread_id": "14", "profile": "keeper"}],
    }}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(keeper_home))  # what `hermes -p keeper` sets
    from cron.jobs import create_job
    return create_job(prompt="infra report", schedule="0 9 * * *", name="infra", deliver=ROUTED_TARGET)


def _run(job_id, *, gateway_serves_profile):
    from tools.cronjob_tools import cronjob
    with patch("hermes_cli.cron._builtin_gateway_liveness", return_value=gateway_serves_profile), \
         patch("cron.scheduler.run_one_job", return_value=True) as m_run:
        out = json.loads(cronjob(action="run", job_id=job_id))
    return out, m_run


def test_routed_run_is_queued_for_the_gateway_that_serves_the_profile(keeper_job):
    from cron.jobs import get_job
    out, m_run = _run(keeper_job["id"], gateway_serves_profile=True)

    assert out["success"] is True
    m_run.assert_not_called()  # no agent turn in a process that cannot deliver its result
    stored = get_job(keeper_job["id"])
    assert stored["next_run_at"] == stored["manual_run_at"]  # due on the gateway's next tick
    assert stored.get("last_status") is None  # the manual run does not rewrite the job's status
    assert _run_outcome(out["job"]) == "It will run on the next scheduler tick."


def test_routed_run_without_a_serving_gateway_fails_before_the_turn(keeper_job):
    from cron.jobs import get_job
    out, m_run = _run(keeper_job["id"], gateway_serves_profile=False)

    assert out["success"] is False
    assert "profile route" in out["error"]
    m_run.assert_not_called()
    stored = get_job(keeper_job["id"])
    assert stored["next_run_at"] == keeper_job["next_run_at"]
    assert "manual_run_at" not in stored


@pytest.mark.parametrize("case", ["local delivery", "own credential", "inside the gateway", "paused"])
def test_run_keeps_the_in_process_path_when_not_routed_only(keeper_job, monkeypatch, case):
    """Everything but a runnable job reachable only through the primary route runs as before."""
    from cron.jobs import get_job, pause_job, update_job
    from tools import cronjob_tools
    if case == "local delivery":
        update_job(keeper_job["id"], {"deliver": "local"})
    elif case == "own credential":
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:keeper-own-bot")
    elif case == "inside the gateway":
        runner = SimpleNamespace(adapters={}, _gateway_loop=None)
        monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)
    else:
        pause_job(keeper_job["id"])

    with patch.object(cronjob_tools, "claim_job_for_fire",
                      wraps=cronjob_tools.claim_job_for_fire) as m_claim:
        _run(keeper_job["id"], gateway_serves_profile=True)

    m_claim.assert_called_once()  # the in-process claim -> run_one_job path, unchanged
    assert "manual_run_at" not in get_job(keeper_job["id"])
