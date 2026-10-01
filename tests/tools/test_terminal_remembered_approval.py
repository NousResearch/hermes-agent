"""The terminal guard names the exact operation it asks about, and runs nothing else."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import approval, approval_context
from tools import terminal_tool as terminal
from tools.approval_operation import approval_operation_key
from tools.environments.local import LocalEnvironment
from tools.environments.ssh import SSHEnvironment
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


def run(command="rm -rf ./build"):
    return json.loads(registry.get_entry("terminal").handler({"command": command}, task_id="remember-test"))


@pytest.mark.parametrize("outcome", ["once", "deny", "changed-cwd", "policy-deny", "smart-deny"])
def test_terminal_names_the_operation_without_widening_permission(terminal_context, monkeypatch, outcome):
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
    result = run(command)
    if outcome == "policy-deny":
        assert not state.notices and not state.executed
        assert result["status"] == "blocked"
    else:
        notice, = state.notices
        assert "A" * 36 not in notice["command"]
        assert bool(notice.get("remember_key")) is (outcome != "smart-deny")
        if outcome != "smart-deny":
            assert notice["remember_context"] == f"Local, folder {state.cwd}"
        if outcome in {"deny", "changed-cwd"}:
            assert not state.executed and result["status"] == "blocked"
        else:
            assert state.executed[0][0] == command
            assert state.executed[0][1]["cwd"] == state.cwd
            assert result["exit_code"] == 0
    assert approval_operation_key(command, ["anything"]) == ""


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


def test_each_ssh_connection_is_its_own_operation(terminal_context, monkeypatch):
    state = terminal_context
    monkeypatch.setattr(terminal, "_get_env_config", lambda: ssh_settings(state))
    executed = []

    def notify(data):
        state.notices.append(data)
        approval.resolve_gateway_approval("remember-test", "once")

    approval.register_gateway_notify("remember-test", notify)
    terminal._active_environments["remember-test"] = inert_ssh(state, "old.invalid", executed)
    first = run()
    terminal._active_environments.clear()
    monkeypatch.setattr(terminal, "_create_configured_env", lambda *args, **kwargs: inert_ssh(state, "new.invalid", executed))
    second = run()
    assert first["exit_code"] == second["exit_code"] == 0
    assert [item[0] for item in executed] == ["worker@old.invalid", "worker@new.invalid"]
    assert "old.invalid" in state.notices[0]["remember_context"]
    assert "new.invalid" in state.notices[1]["remember_context"]
    assert state.notices[0]["remember_key"] != state.notices[1]["remember_key"]


@pytest.mark.parametrize("change", ["host", "user", "port", "key_path", "cached-local"])
def test_a_connection_that_changes_while_pending_runs_nothing(terminal_context, monkeypatch, change):
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
    result = run()
    assert len(state.notices) == 1
    if change == "cached-local":
        assert "remember_key" not in state.notices[0]
        assert result["exit_code"] == 0
    else:
        assert result["status"] == "blocked" and not executed


def test_background_commands_are_never_named_as_repeatable(terminal_context):
    state = terminal_context

    def notify(data):
        state.notices.append(data)
        approval.resolve_gateway_approval("remember-test", "deny")

    approval.register_gateway_notify("remember-test", notify)
    registry.get_entry("terminal").handler({"command": "rm -rf ./build", "background": True},
                                           task_id="remember-test")
    notice, = state.notices
    assert "remember_key" not in notice
