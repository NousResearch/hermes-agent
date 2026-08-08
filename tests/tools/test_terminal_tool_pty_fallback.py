import json
from types import SimpleNamespace

import tools.terminal_tool as terminal_tool_module
from tools import process_registry as process_registry_module


def _base_config(tmp_path, env_type="local"):
    return {
        "env_type": env_type,
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "cwd": str(tmp_path),
        "timeout": 30,
    }


def _patch_common(monkeypatch, config, dummy_env):
    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool_module, "_check_all_guards", lambda *_a, **_kw: {"approved": True}
    )
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", dummy_env)
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)


def test_command_requires_pipe_stdin_detects_gh_with_token():
    assert terminal_tool_module._command_requires_pipe_stdin(
        "gh auth login --hostname github.com --git-protocol https --with-token"
    ) is True
    assert terminal_tool_module._command_requires_pipe_stdin(
        "gh auth login --web"
    ) is False


def test_terminal_background_keeps_pty_for_regular_interactive_commands(monkeypatch, tmp_path):
    config = _base_config(tmp_path)
    dummy_env = SimpleNamespace(env={})
    captured = {}

    def fake_spawn_local(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id="proc_test", pid=1234, notify_on_complete=False)

    _patch_common(monkeypatch, config, dummy_env)
    monkeypatch.setattr(process_registry_module.process_registry, "spawn_local", fake_spawn_local)

    try:
        result = json.loads(
            terminal_tool_module.terminal_tool(
                command="python3 -c \"print(input())\"",
                background=True,
                pty=True,
            )
        )
    finally:
        terminal_tool_module._active_environments.pop("default", None)
        terminal_tool_module._last_activity.pop("default", None)

    assert captured["use_pty"] is True
    assert "pty_note" not in result


def test_foreground_pty_still_runs_but_reports_the_dropped_pty(monkeypatch, tmp_path):
    """Foreground has no PTY path, so pty=true is dropped — say so.

    The command must still run (skills pass pty=true to one-shot commands like
    `codex exec` and consume the inline output), but the result has to carry the
    remedy instead of letting the agent rediscover it after a hang.
    """
    config = _base_config(tmp_path)
    executed = []
    dummy_env = SimpleNamespace(
        env={},
        cwd=str(tmp_path),
        execute=lambda command, **kwargs: (
            executed.append(command) or {"output": "done", "returncode": 0}
        ),
    )

    _patch_common(monkeypatch, config, dummy_env)

    try:
        result = json.loads(
            terminal_tool_module.terminal_tool(command="codex exec 'ship it'", pty=True)
        )
    finally:
        terminal_tool_module._active_environments.pop("default", None)
        terminal_tool_module._last_activity.pop("default", None)

    assert executed == ["codex exec 'ship it'"]
    assert result["output"] == "done"
    assert result["exit_code"] == 0
    assert "background=true" in result["pty_note"]


def test_foreground_pty_timeout_carries_the_pty_note(monkeypatch, tmp_path):
    """A terminal-hungry command on a pipe expires — the timeout must explain why."""
    config = _base_config(tmp_path)

    def _timeout(command, **kwargs):
        raise RuntimeError("command timeout exceeded")

    dummy_env = SimpleNamespace(env={}, cwd=str(tmp_path), execute=_timeout)
    _patch_common(monkeypatch, config, dummy_env)

    try:
        result = json.loads(
            terminal_tool_module.terminal_tool(command="htop", pty=True)
        )
    finally:
        terminal_tool_module._active_environments.pop("default", None)
        terminal_tool_module._last_activity.pop("default", None)

    assert result["exit_code"] == 124
    assert "background=true" in result["pty_note"]


def test_foreground_without_pty_has_no_pty_note(monkeypatch, tmp_path):
    config = _base_config(tmp_path)
    dummy_env = SimpleNamespace(
        env={},
        cwd=str(tmp_path),
        execute=lambda command, **kwargs: {"output": "hi", "returncode": 0},
    )

    _patch_common(monkeypatch, config, dummy_env)

    try:
        result = json.loads(terminal_tool_module.terminal_tool(command="echo hi"))
    finally:
        terminal_tool_module._active_environments.pop("default", None)
        terminal_tool_module._last_activity.pop("default", None)

    assert "pty_note" not in result


def test_pty_on_non_local_backend_reports_the_downgrade(monkeypatch, tmp_path):
    """spawn_via_env() has no PTY, so the request is dropped — but not silently."""
    config = _base_config(tmp_path, env_type="docker")
    dummy_env = SimpleNamespace(
        env={},
        cwd=str(tmp_path),
        execute=lambda command, **kwargs: {"output": "ok", "returncode": 0},
    )

    _patch_common(monkeypatch, config, dummy_env)

    try:
        result = json.loads(
            terminal_tool_module.terminal_tool(
                command="python3", pty=True, background=True
            )
        )
    finally:
        terminal_tool_module._active_environments.pop("default", None)
        terminal_tool_module._last_activity.pop("default", None)

    assert "docker" in result["pty_note"]
