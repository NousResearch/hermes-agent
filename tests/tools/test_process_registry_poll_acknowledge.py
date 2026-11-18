"""#94455: ``process(action='poll', acknowledge=true)`` on an exited process tells Hermes the agent
handled the result inline, so the gateway watcher must not inject a stale completion turn. A plain
poll stays read-only, so a status check still gets its autonomous turn (#10156)."""
import asyncio
import json

import pytest

import tools.process_registry as pr_module
from tests.waiting import wait_until
from tools.process_registry import ProcessRegistry


@pytest.fixture()
def registry(monkeypatch):
    monkeypatch.setattr("hermes_cli.process_identity._update_supervised_pgids", lambda verb, pgids: None)
    reg = ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", reg)
    return reg


def _watcher_turns(session_id: str) -> list:
    """Run the real gateway process watcher; return the synthetic completion turns it enqueues."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    turns = []

    async def _enqueue(synth_text, evt):
        turns.append(synth_text)
        return True

    async def _not_busy(*_a):
        return False

    runner._enqueue_process_completion_notification = _enqueue
    runner._launching_turn_active = _not_busy
    runner._load_background_notifications_mode = lambda *_a, **_k: "off"
    runner.adapters = {}
    watcher = {"session_id": session_id, "check_interval": 0.01, "platform": "telegram", "chat_id": "1",
               "session_key": "agent:main:telegram:dm:1", "notify_on_complete": True}
    asyncio.run(asyncio.wait_for(runner._run_process_watcher(watcher), 10))
    return turns


def _exited_process(registry, tmp_path):
    session = registry.spawn_local("echo DONE", cwd=str(tmp_path), session_key="agent:main:telegram:dm:1")
    session.notify_on_complete = True
    wait_until(lambda: session.exited and not registry._reader_finalizing(session), timeout=10,
               what="background process exit and final output drain")
    return session


@pytest.mark.parametrize("acknowledge, turns_expected", [(True, 0), (False, 1)])
def test_acknowledged_poll_suppresses_the_stale_completion_turn(registry, tmp_path, acknowledge, turns_expected):
    session = _exited_process(registry, tmp_path)
    args = {"action": "poll", "session_id": session.id, **({"acknowledge": True} if acknowledge else {})}
    seen = json.loads(pr_module._handle_process(args))
    assert seen["status"] == "exited" and "DONE" in seen["output_preview"]
    assert len(_watcher_turns(session.id)) == turns_expected
