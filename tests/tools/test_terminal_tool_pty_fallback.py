import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

import tools.terminal_tool as terminal_tool_module
from tools import process_registry as process_registry_module
from tools.environments.base import PtyUnavailableError
from tools.environments.local import LocalEnvironment


def _base_config(tmp_path):
    return {
        "env_type": "local",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "cwd": str(tmp_path),
        "timeout": 30,
    }


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

    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool_module, "_check_all_guards", lambda *_args, **_kwargs: {"approved": True})
    monkeypatch.setattr(process_registry_module.process_registry, "spawn_local", fake_spawn_local)
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", dummy_env)
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)

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


def test_terminal_foreground_forwards_pty_to_local_environment(monkeypatch, tmp_path):
    config = _base_config(tmp_path)
    captured = {}

    class FakeEnv:
        env = {}
        cwd = str(tmp_path)

        def execute(self, command, **kwargs):
            captured.update(kwargs)
            return {"output": "ok", "returncode": 0}

    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool_module,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", FakeEnv())
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)

    result = json.loads(
        terminal_tool_module.terminal_tool(command="tty", pty=True)
    )

    assert result["exit_code"] == 0
    assert captured["use_pty"] is True


def test_terminal_foreground_reports_when_pipe_stdin_disables_pty(monkeypatch, tmp_path):
    config = _base_config(tmp_path)
    captured = {}

    class FakeEnv:
        env = {}
        cwd = str(tmp_path)

        def execute(self, command, **kwargs):
            captured.update(kwargs)
            return {"output": "ok", "returncode": 0}

    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool_module,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", FakeEnv())
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)

    result = json.loads(
        terminal_tool_module.terminal_tool(
            command="gh auth login --with-token",
            pty=True,
        )
    )

    assert result["exit_code"] == 0
    assert captured["use_pty"] is False
    assert "expects piped stdin" in result["pty_note"]


def test_terminal_foreground_pty_unavailable_fails_without_retry(monkeypatch, tmp_path):
    config = _base_config(tmp_path)

    class FakeEnv:
        env = {}
        cwd = str(tmp_path)

        def execute(self, command, **kwargs):
            raise PtyUnavailableError("PTY dependency missing")

    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool_module,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setattr(
        terminal_tool_module.time,
        "sleep",
        lambda _seconds: pytest.fail("PTY availability errors must not retry"),
    )
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", FakeEnv())
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)

    result = json.loads(
        terminal_tool_module.terminal_tool(command="tty", pty=True)
    )

    assert result == {
        "output": "",
        "exit_code": -1,
        "error": "PTY dependency missing",
        "status": "unsupported",
    }


@pytest.mark.skipif(os.name == "nt", reason="POSIX tty assertion")
def test_local_foreground_pty_exposes_a_terminal(tmp_path):
    pytest.importorskip("ptyprocess")
    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    try:
        result = env.execute("tty", timeout=10, use_pty=True)
    finally:
        env.cleanup()

    assert result["returncode"] == 0
    assert result["output"].strip().startswith("/dev/")


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY integration")
def test_local_foreground_pty_preserves_exit_code_and_cwd(tmp_path):
    pytest.importorskip("ptyprocess")
    target = tmp_path / "target"
    target.mkdir()
    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    try:
        result = env.execute(
            f"cd {target}; printf reached; false",
            timeout=10,
            use_pty=True,
        )
    finally:
        env.cleanup()

    assert result["returncode"] == 1
    assert result["output"].strip() == "reached"
    assert env.cwd == str(target)
    assert "__HERMES_CWD_" not in result["output"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY integration")
def test_local_foreground_pty_timeout_returns_promptly(tmp_path):
    pytest.importorskip("ptyprocess")
    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    started = time.monotonic()
    try:
        result = env.execute("sleep 30", timeout=1, use_pty=True)
    finally:
        env.cleanup()

    assert result["returncode"] == 124
    assert "timed out after 1s" in result["output"]
    assert time.monotonic() - started < 5


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY integration")
def test_local_foreground_pty_interrupt_reaps_command(tmp_path):
    pytest.importorskip("ptyprocess")
    from tools.interrupt import set_interrupt

    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    result = {}

    def run_command():
        result.update(env.execute("sleep 30", timeout=10, use_pty=True))

    worker = threading.Thread(target=run_command)
    worker.start()
    try:
        time.sleep(0.5)
        set_interrupt(True, thread_id=worker.ident)
        worker.join(timeout=5)
    finally:
        set_interrupt(False, thread_id=worker.ident)
        env.cleanup()

    assert not worker.is_alive()
    assert result["returncode"] == 130
    assert "Command interrupted" in result["output"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY integration")
def test_local_foreground_pty_drains_output_larger_than_pipe_capacity(tmp_path):
    pytest.importorskip("ptyprocess")
    env = LocalEnvironment(cwd=str(tmp_path), timeout=10)
    try:
        result = env.execute(
            "printf '%70000s' x",
            timeout=10,
            use_pty=True,
        )
    finally:
        env.cleanup()

    assert result["returncode"] == 0
    assert len(result["output"].rstrip("\r\n")) == 70_000
    assert result["output"].rstrip().endswith("x")
