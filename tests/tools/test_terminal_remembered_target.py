"""Remembered decisions identify the acquired SSH environment, not newer settings."""

import json
from pathlib import Path

import pytest

from gateway import hosted_room_messaging_approvals as approvals
from tools import approval, terminal_tool as terminal
from tools.environments.ssh import SSHEnvironment
from tools.registry import registry
from tests.tools.test_terminal_remembered_approval import terminal_context
from tests.gateway.test_hosted_room_approval_rules import pending, finish


def inert_ssh(state, host, executed):
    env = object.__new__(SSHEnvironment)
    env.host, env.user, env.port, env.key_path = host, "worker", 22, ""
    env.control_socket = Path(state.cwd) / "inert.sock"
    env.env = {}

    def execute(command, **kwargs):
        executed.append((env._build_ssh_command()[-1], command, kwargs["cwd"]))
        return {"output": "inert transport", "returncode": 0}

    env.execute = execute
    return env


def ssh_settings(state):
    return {"env_type": "ssh", "cwd": state.cwd, "timeout": 30, "lifetime_seconds": 3600,
            "ssh_host": "new.invalid", "ssh_user": "worker", "ssh_port": 22}


def test_cached_and_recreated_ssh_connections_cannot_share_a_remembered_grant(terminal_context, monkeypatch):
    state = terminal_context
    monkeypatch.setattr(terminal, "_get_env_config", lambda: ssh_settings(state))
    executed = []

    def notify(data):
        state.notices.append(data)
        approval.resolve_gateway_approval("remember-test", "once")

    approval.register_gateway_notify("remember-test", notify)
    terminal._active_environments["remember-test"] = inert_ssh(state, "old.invalid", executed)
    first = json.loads(registry.get_entry("terminal").handler({"command": "rm -rf ./build"}, task_id="remember-test"))
    terminal._active_environments.clear()
    monkeypatch.setattr(terminal, "_create_configured_env", lambda *args, **kwargs: inert_ssh(state, "new.invalid", executed))
    second = json.loads(registry.get_entry("terminal").handler({"command": "rm -rf ./build"}, task_id="remember-test"))
    assert first["exit_code"] == second["exit_code"] == 0
    assert [item[0] for item in executed] == ["worker@old.invalid", "worker@new.invalid"]
    assert "old.invalid" in state.notices[0]["remember_context"]
    assert "new.invalid" in state.notices[1]["remember_context"]
    assert state.notices[0]["remember_key"] != state.notices[1]["remember_key"]
    db = Path(state.cwd) / "home-side.db"
    original, attempt = pending(db, key=state.notices[0]["remember_key"], context=state.notices[0]["remember_context"])
    approvals.begin_approval_command(db, command_id="owner-grant", pending=original, choice="remember")
    approvals.apply_pending_decision(db, pending=original, choice="once", command_id="owner-grant", apply=lambda: {"resolved": 1})
    finish(db, attempt)
    later, _ = pending(db, key=state.notices[1]["remember_key"], context=state.notices[1]["remember_context"], suffix="2")
    assert not approvals.queue_remembered_approval(db, room_id="group-a", member_id="writer", action=later)


@pytest.mark.parametrize("change", ["host", "user", "port", "key_path", "cached-local"])
def test_changed_or_mismatched_execution_target_does_not_reuse_permission(terminal_context, monkeypatch, change):
    state = terminal_context
    monkeypatch.setattr(terminal, "_get_env_config", lambda: ssh_settings(state))
    executed = []
    env = inert_ssh(state, "old.invalid", executed)
    if change != "cached-local":
        terminal._active_environments["remember-test"] = env

    def notify(data):
        state.notices.append(data)
        if change != "cached-local":
            setattr(env, change, 2222 if change == "port" else "changed")
        approval.resolve_gateway_approval("remember-test", "once")

    approval.register_gateway_notify("remember-test", notify)
    result = json.loads(registry.get_entry("terminal").handler({"command": "rm -rf ./build"}, task_id="remember-test"))
    assert len(state.notices) == 1
    if change == "cached-local":
        assert "remember_key" not in state.notices[0]
        assert result["exit_code"] == 0
    else:
        assert result["status"] == "blocked" and not executed
