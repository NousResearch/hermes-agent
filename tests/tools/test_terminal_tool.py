"""Regression tests for terminal tool helpers."""

import json
from types import SimpleNamespace

import pytest
import tools.terminal_tool as terminal_tool


def setup_function():
    terminal_tool._cached_sudo_password = ""


def teardown_function():
    terminal_tool._cached_sudo_password = ""


@pytest.mark.parametrize(
    "workdir",
    [
        r"C:\hermes_workspace\project",
        r"D:\dev\agent_test",
        r"..\repo",
        "/tmp/project (old)",
        r"C:\hermes_workspace\O'Brien\repo",
    ],
)
def test_validate_workdir_accepts_platform_native_paths(workdir):
    assert terminal_tool._validate_workdir(workdir) is None


@pytest.mark.parametrize(
    "workdir",
    [
        r"C:\hermes_workspace\project; whoami",
        r"D:\dev\agent_test && whoami",
        "/tmp/project | cat",
        "/tmp/project`whoami`",
        "/tmp/project\nwhoami",
    ],
)
def test_validate_workdir_rejects_shell_metacharacters(workdir):
    error = terminal_tool._validate_workdir(workdir)

    assert error is not None
    assert "Blocked: workdir contains disallowed character" in error


def test_terminal_tool_passes_valid_workdir_to_environment(monkeypatch):
    captured = {}
    dummy_env = SimpleNamespace(
        env={},
        execute=lambda command, **kwargs: (
            captured.update({"command": command, "cwd": kwargs.get("cwd")})
            or {"output": "ok", "returncode": 0}
        ),
    )

    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {"env_type": "local", "cwd": ".", "timeout": 30},
    )
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setitem(terminal_tool._active_environments, "default", dummy_env)
    monkeypatch.setitem(terminal_tool._last_activity, "default", 0.0)

    try:
        result = json.loads(
            terminal_tool.terminal_tool(
                command="pwd",
                workdir=r"C:\hermes_workspace\project",
            )
        )
    finally:
        terminal_tool._active_environments.pop("default", None)
        terminal_tool._last_activity.pop("default", None)

    assert result["exit_code"] == 0
    assert captured["cwd"] == r"C:\hermes_workspace\project"


def test_terminal_tool_blocks_dangerous_workdir_before_execution(monkeypatch):
    dummy_env = SimpleNamespace(
        env={},
        execute=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dangerous workdir should be blocked before execute()")
        ),
    )

    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {"env_type": "local", "cwd": ".", "timeout": 30},
    )
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda *_args, **_kwargs: {"approved": True},
    )
    monkeypatch.setitem(terminal_tool._active_environments, "default", dummy_env)
    monkeypatch.setitem(terminal_tool._last_activity, "default", 0.0)

    try:
        result = json.loads(
            terminal_tool.terminal_tool(
                command="pwd",
                workdir=r"C:\hermes_workspace\project; whoami",
            )
        )
    finally:
        terminal_tool._active_environments.pop("default", None)
        terminal_tool._last_activity.pop("default", None)

    assert result["status"] == "blocked"
    assert "disallowed character" in result["error"]


def test_searching_for_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "rg --line-number --no-heading --with-filename 'sudo' . | head -n 20"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_printf_literal_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "printf '%s\\n' sudo"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_non_command_argument_named_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "grep -n sudo README.md"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_actual_sudo_command_uses_configured_password(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo apt install -y ripgrep")

    assert transformed == "sudo -S -p '' apt install -y ripgrep"
    assert sudo_stdin == "testpass\n"


def test_actual_sudo_after_leading_env_assignment_is_rewritten(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("DEBUG=1 sudo whoami")

    assert transformed == "DEBUG=1 sudo -S -p '' whoami"
    assert sudo_stdin == "testpass\n"


def test_explicit_empty_sudo_password_tries_empty_without_prompt(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("interactive sudo prompt should not run for explicit empty password")

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo true")

    assert transformed == "sudo -S -p '' true"
    assert sudo_stdin == "\n"


def test_cached_sudo_password_is_used_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool._cached_sudo_password = "cached-pass"

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("echo ok && sudo whoami")

    assert transformed == "echo ok && sudo -S -p '' whoami"
    assert sudo_stdin == "cached-pass\n"
