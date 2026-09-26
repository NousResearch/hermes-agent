"""``command_guard`` plugin hook: consulted by ``check_all_command_guards`` for every terminal
command, so a direct ``terminal_tool()`` caller (which ``pre_tool_call`` never sees) meets the same
veto, typed as ``status: blocked``, and no session setting (yolo) can bypass it."""

import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.fixture
def guard(monkeypatch):
    mgr = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    monkeypatch.setattr(mgr, "discover_and_load", lambda force=False: None)
    seen = []

    def _guard(command, env_type, session_key, **_):
        seen.append((command, env_type))
        return {"action": "block", "message": "no rm in this workspace"} if command.startswith("rm ") else None

    PluginContext(PluginManifest(name="envelope", source="user"), mgr).register_hook("command_guard", _guard)
    return seen


def _terminal(command):
    from tools.terminal_tool import terminal_tool

    env = MagicMock()
    env.execute.return_value = {"output": "ok", "returncode": 0}
    env.cwd = "/tmp"
    config = {"env_type": "local", "timeout": 180, "cwd": "/tmp", "host_cwd": None, "modal_mode": "auto",
              "docker_image": "", "singularity_image": "", "modal_image": "", "daytona_image": ""}
    with ExitStack() as stack:
        stack.enter_context(patch("tools.terminal_tool._get_env_config", return_value=config))
        stack.enter_context(patch("tools.terminal_tool._start_cleanup_thread"))
        stack.enter_context(patch("tools.terminal_tool._active_environments", {"default": env}))
        stack.enter_context(patch("tools.terminal_tool._last_activity", {"default": 0}))
        stack.enter_context(patch("tools.terminal_tool._session_cwd", {}))
        return json.loads(terminal_tool(command=command)), env


def test_direct_terminal_call_meets_the_guard_under_yolo(guard, monkeypatch):
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    result, env = _terminal("rm build.log")
    assert result["status"] == "blocked"
    assert "no rm in this workspace" in result["error"]
    env.execute.assert_not_called()

    # Positive control: the same path, a command the guard allows, runs.
    result, env = _terminal("echo ok")
    assert result.get("status") != "blocked"
    env.execute.assert_called_once()
    assert [c for c, _ in guard] == ["rm build.log", "echo ok"]
