"""Malformed terminal calls must fail before provisioning any backend."""

import json

import pytest

from tools import terminal_tool as terminal_module


@pytest.mark.parametrize("command", ["echo before\x00echo after", "\x00", {"command": "echo"}, None])
def test_malformed_command_is_rejected_before_backend_resolution(monkeypatch, command):
    calls = []

    def unexpected_config():
        calls.append("config")
        raise AssertionError("Malformed input reached backend configuration")

    monkeypatch.setattr(terminal_module, "_get_env_config", unexpected_config)
    result = json.loads(terminal_module.registry.dispatch("terminal", {"command": command}))

    assert calls == []
    assert result["status"] == "error"
    assert result["exit_code"] != 0
    assert "Invalid command" in result["error"]


def test_shell_escape_remains_executable(monkeypatch):
    config = {
        "env_type": "local", "cwd": "/tmp", "timeout": 30, "lifetime_seconds": 300,
        "local_persistent": False, "docker_image": None,
    }
    monkeypatch.setattr(terminal_module, "_get_env_config", lambda: config)
    # Exercise planning without provisioning a real persistent shell.
    plan = terminal_module._plan_execution(
        r"printf '\0'", task_id="escaped-command", timeout=None,
        background=False, _host_local=True,
    )
    assert plan.config is config
