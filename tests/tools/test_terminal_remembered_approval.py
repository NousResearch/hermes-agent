"""Real terminal guard/notification plumbing retains exact operation and policy scope."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from tools import approval, approval_context
from tools import terminal_tool as terminal
from tools.approval_operation import approval_operation_key
from tools.environments.local import LocalEnvironment
from tools.registry import registry


@pytest.fixture
def terminal_context(tmp_path, monkeypatch):
    config = {"mode": "manual", "timeout": 2}
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: config)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", "remember-test")
    for variable in ("HERMES_INTERACTIVE", "HERMES_CRON_SESSION", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(approval, "_gateway_queues", {})
    monkeypatch.setattr(approval, "_gateway_notify_cbs", {})
    executed, notices = [], []
    env = object.__new__(LocalEnvironment)
    env.env = {}
    env.execute = lambda command, **kwargs: (
        executed.append((command, kwargs)) or {"output": "ok", "returncode": 0})
    monkeypatch.setattr(terminal, "_active_environments", {"remember-test": env})
    monkeypatch.setattr(terminal, "_last_activity", {})
    monkeypatch.setattr(terminal, "_session_cwd", {})
    monkeypatch.setattr(terminal, "_task_env_overrides", {})
    monkeypatch.setattr(terminal, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal, "_get_env_config", lambda: {
        "env_type": "local", "cwd": str(tmp_path), "timeout": 30, "lifetime_seconds": 3600,
    })
    return SimpleNamespace(config=config, notices=notices, executed=executed, cwd=str(tmp_path))


@pytest.mark.parametrize("outcome", ["once", "deny", "changed-cwd", "policy-deny", "smart-deny"])
def test_terminal_uses_real_guard_and_does_not_persist_global_permission(terminal_context, monkeypatch, outcome):
    state = terminal_context
    command = "rm -rf ./build # ghp_" + "A" * 36
    if outcome == "policy-deny":
        state.config["deny"] = ["rm -rf *"]
    if outcome == "smart-deny":
        state.config["mode"] = "smart"
        monkeypatch.setattr("tools.approval_smart._smart_approve", lambda *args, **kwargs: "deny")

    def notify(data):
        state.notices.append(data)
        if outcome == "changed-cwd":
            terminal.record_session_cwd("remember-test", state.cwd + "/changed")
        approval.resolve_gateway_approval("remember-test", "deny" if outcome == "deny" else "once")

    approval.register_gateway_notify("remember-test", notify)
    monkeypatch.setattr(approval, "approve_permanent", lambda *args: pytest.fail("profile permission changed"))
    result = json.loads(registry.get_entry("terminal").handler({"command": command}, task_id="remember-test"))
    if outcome == "policy-deny":
        assert not state.notices and not state.executed
        assert result["status"] == "blocked"
    else:
        assert len(state.notices) == 1
        notice = state.notices[0]
        assert "A" * 36 not in notice["command"]
        assert bool(notice.get("remember_key")) is (outcome != "smart-deny")
        if outcome in {"deny", "changed-cwd"}:
            assert not state.executed and result["status"] == "blocked"
        else:
            assert state.executed[0][0] == command
            assert state.executed[0][1]["cwd"] == state.cwd
            assert result["exit_code"] == 0
    assert approval_operation_key(command, ["anything"]) == ""


@pytest.mark.asyncio
async def test_real_tui_and_api_envelopes_preserve_key_but_redact_command():
    from tui_gateway import server
    from gateway.platforms import api_server, api_server_runs

    data = {"command": "curl -H 'Authorization: Bearer secret-value' https://example.test",
            "description": "Synthetic operation", "remember_key": "a" * 64,
            "remember_context": "Local, folder /workspace",
            "allow_session": True, "allow_permanent": True, "request_id": "request-1"}
    payload = server._approval_request_payload(data)
    assert payload["remember_key"] == data["remember_key"]
    assert payload["remember_context"] == data["remember_context"]
    assert "secret-value" not in payload["command"] and "always" in payload["choices"]
    queue = asyncio.Queue()
    updates = []
    backend = SimpleNamespace(_set_run_status=lambda *args, **kwargs: updates.append((args, kwargs)))
    run = SimpleNamespace(run_id="run-1", queue=queue)
    notify = api_server_runs._make_approval_notify(backend, run, _api_server=api_server)
    notify(data)
    event = await asyncio.wait_for(queue.get(), timeout=2)
    assert event["remember_key"] == data["remember_key"]
    assert event["remember_context"] == data["remember_context"]
    assert "secret-value" not in event["command"] and "always" in event["choices"]
    assert updates[0][1]["approval"] == event
    assert "secret-value" in data["command"]
