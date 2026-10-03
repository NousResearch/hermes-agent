"""Regression coverage for the Windows Bash /dev/tcp crash containment guard."""

import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest


def _run(command, monkeypatch):
    from tools.terminal_tool import terminal_tool

    config = {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    env = MagicMock()
    env.cwd = "/tmp"
    env.execute.return_value = {"output": "ok", "returncode": 0}
    with ExitStack() as stack:
        stack.enter_context(patch("tools.terminal_tool._get_env_config", return_value=config))
        stack.enter_context(patch("tools.terminal_tool._start_cleanup_thread"))
        stack.enter_context(patch("tools.terminal_tool._active_environments", {"default": env}))
        stack.enter_context(patch("tools.terminal_tool._last_activity", {"default": 0}))
        stack.enter_context(patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}))
        stack.enter_context(patch("tools.terminal_tool.gateway_lifecycle_block", return_value=None))
        stack.enter_context(patch("tools.terminal_tool.self_repo_block", return_value=None))
        result = json.loads(terminal_tool(command=command))
    return result, env


@pytest.mark.parametrize(
    "command",
    [
        "bash -c 'echo > /dev/tcp/192.0.2.1/22'",
        'plink host "timeout 5 bash -c \'echo > /dev/tcp/192.0.2.1/22\'"',
    ],
)
def test_blocks_executed_bash_dev_tcp_redirect_before_shell_spawn(command, monkeypatch):
    monkeypatch.setattr("tools.terminal_tool_guards.platform.system", lambda: "Windows")
    result, env = _run(command, monkeypatch)

    assert result["output"] == ""
    assert result["exit_code"] == 1
    assert result["status"] == "blocked"
    assert "Bash /dev/tcp redirect" in result["error"]
    assert "not run" in result["error"]
    env.execute.assert_not_called()


@pytest.mark.parametrize(
    "command",
    [
        "echo /dev/tcp/192.0.2.1/22",
        "printf '%s\\n' 'echo > /dev/tcp/192.0.2.1/22'",
        "bash -c 'echo connected'",
    ],
)
def test_allows_non_executed_or_non_redirect_dev_tcp_text(command, monkeypatch):
    monkeypatch.setattr("tools.terminal_tool_guards.platform.system", lambda: "Windows")
    result, env = _run(command, monkeypatch)

    assert result.get("status") != "blocked"
    env.execute.assert_called_once()


def test_posix_keeps_bash_dev_tcp_available(monkeypatch):
    monkeypatch.setattr("tools.terminal_tool_guards.platform.system", lambda: "Darwin")

    result, env = _run("bash -c 'echo > /dev/tcp/192.0.2.1/22'", monkeypatch)

    assert result.get("status") != "blocked"
    env.execute.assert_called_once()
