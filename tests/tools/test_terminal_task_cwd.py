"""Regression tests for task/session cwd propagation in terminal_tool."""

import json
import os
import shutil

import tools.terminal_tool as terminal_tool


def _minimal_terminal_config(cwd="/default"):
    return {
        "env_type": "local",
        "cwd": cwd,
        "timeout": 60,
    }


def test_foreground_command_uses_registered_task_cwd_for_existing_environment(monkeypatch):
    """ACP can update task cwd after the local env exists; foreground must honor it."""
    calls = []

    class FakeEnv:
        env = {}

        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"output": "ok", "returncode": 0}

    task_id = "acp-session-1"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": "/workspace/acp"}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config())
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type: {"approved": True},
    )

    result = json.loads(terminal_tool.terminal_tool(command="pwd", task_id=task_id))

    assert result["exit_code"] == 0
    assert calls == [("pwd", {"timeout": 60, "cwd": "/workspace/acp"})]


def test_explicit_workdir_still_wins_over_registered_task_cwd(monkeypatch):
    calls = []

    class FakeEnv:
        env = {}

        def execute(self, command, **kwargs):
            calls.append(kwargs)
            return {"output": "ok", "returncode": 0}

    task_id = "acp-session-1"
    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {task_id: {"cwd": "/workspace/acp"}})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config())
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type: {"approved": True},
    )

    result = json.loads(
        terminal_tool.terminal_tool(
            command="pwd",
            task_id=task_id,
            workdir="/explicit/workdir",
        )
    )

    assert result["exit_code"] == 0
    assert calls == [{"timeout": 60, "cwd": "/explicit/workdir"}]


def test_get_env_config_uses_requested_workdir_when_process_cwd_deleted(monkeypatch, tmp_path):
    requested = tmp_path / "requested"
    requested.mkdir()

    def missing_cwd():
        raise FileNotFoundError("deleted cwd")

    monkeypatch.setattr(terminal_tool.os, "getcwd", missing_cwd)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    config = terminal_tool._get_env_config(requested_workdir=str(requested))

    assert config["cwd"] == str(requested)


def test_terminal_call_from_deleted_cwd_uses_explicit_workdir(monkeypatch, tmp_path):
    original_cwd = os.getcwd()
    deleted_cwd = tmp_path / "deleted-cwd"
    deleted_cwd.mkdir()
    requested_workdir = tmp_path / "requested-workdir"
    requested_workdir.mkdir()

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type: {"approved": True},
    )

    os.chdir(deleted_cwd)
    shutil.rmtree(deleted_cwd)
    try:
        result = json.loads(
            terminal_tool.terminal_tool(
                command="printf 'terminal-ok\\n'; pwd; uname -a",
                workdir=str(requested_workdir),
                timeout=10,
            )
        )
    finally:
        os.chdir(original_cwd)

    assert result["exit_code"] == 0
    assert "terminal-ok" in result["output"]
    assert str(requested_workdir) in result["output"].splitlines()


def test_terminal_call_from_deleted_cwd_falls_back_without_workdir(monkeypatch, tmp_path):
    original_cwd = os.getcwd()
    deleted_cwd = tmp_path / "deleted-cwd"
    deleted_cwd.mkdir()
    fallback_home = tmp_path / "home"
    fallback_home.mkdir()

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("HOME", str(fallback_home))
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type: {"approved": True},
    )

    os.chdir(deleted_cwd)
    shutil.rmtree(deleted_cwd)
    try:
        result = json.loads(
            terminal_tool.terminal_tool(
                command="printf 'fallback-ok\\n'; pwd",
                timeout=10,
            )
        )
    finally:
        os.chdir(original_cwd)

    assert result["exit_code"] == 0
    assert "fallback-ok" in result["output"]
    assert str(fallback_home) in result["output"].splitlines()
