"""Regression tests for sudo detection and terminal tool metadata."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import tools.terminal_tool as terminal_tool


def setup_function():
    terminal_tool._reset_cached_sudo_passwords()


def teardown_function():
    terminal_tool._reset_cached_sudo_passwords()


def test_searching_for_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "rg --line-number --no-heading --with-filename 'sudo' . | head -n 20"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_terminal_schema_advertises_persistent_env_state():
    description = terminal_tool.TERMINAL_TOOL_DESCRIPTION

    assert "exported environment variables persist between calls" in description
    assert "activate a virtualenv" in description
    assert "once per session" in description


def test_terminal_schema_for_pwsh_is_foreground_only_and_internally_consistent():
    schema = terminal_tool._terminal_schema_for_shell("pwsh")
    properties = schema["parameters"]["properties"]

    assert "PowerShell commands" in schema["description"]
    assert "Background and PTY execution are not implemented" in schema["description"]
    assert "not supported when terminal.shell is pwsh" in properties["background"]["description"]
    assert "not supported when terminal.shell is pwsh" in properties["pty"]["description"]
    assert "use background=true" not in properties["timeout"]["description"]
    assert "unavailable when terminal.shell is pwsh" in properties["notify_on_complete"]["description"]
    assert "unavailable when terminal.shell is pwsh" in properties["watch_patterns"]["description"]


def test_terminal_schema_for_bash_keeps_background_and_pty_guidance():
    schema = terminal_tool._terminal_schema_for_shell("bash")
    properties = schema["parameters"]["properties"]

    assert "Execute shell commands on a Linux environment" in schema["description"]
    assert "returning a session_id" in properties["background"]["description"]
    assert "interactive CLI tools" in properties["pty"]["description"]
    assert "use background=true" in properties["timeout"]["description"]


@pytest.mark.windows_only
def test_registered_terminal_schema_uses_pwsh_at_process_start():
    env = os.environ.copy()
    env.update({"TERMINAL_SHELL": "pwsh", "TERMINAL_ENV": "local"})
    repo_root = Path(__file__).resolve().parents[2]
    probe = (
        "import json; "
        "from tools.terminal_tool import TERMINAL_SCHEMA; "
        "print('HERMES_SCHEMA=' + json.dumps(TERMINAL_SCHEMA))"
    )

    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    marker = next(
        line for line in completed.stdout.splitlines()
        if line.startswith("HERMES_SCHEMA=")
    )
    schema = json.loads(marker.removeprefix("HERMES_SCHEMA="))
    properties = schema["parameters"]["properties"]

    assert "PowerShell commands" in schema["description"]
    assert "not supported when terminal.shell is pwsh" in properties["background"]["description"]
    assert "not supported when terminal.shell is pwsh" in properties["pty"]["description"]


@pytest.mark.windows_only
@pytest.mark.parametrize(
    ("tool_argument", "sentinel", "mode_name"),
    [
        ("pty=True", "PTY_ONLY_EXECUTED", "PTY"),
        ("background=True", "BACKGROUND_ONLY_EXECUTED", "background"),
    ],
)
def test_pwsh_unsupported_mode_fails_closed_at_tool_entry(
    tool_argument,
    sentinel,
    mode_name,
):
    env = os.environ.copy()
    env.update({"TERMINAL_SHELL": "pwsh", "TERMINAL_ENV": "local"})
    repo_root = Path(__file__).resolve().parents[2]
    probe = (
        "import json; "
        "from tools.terminal_tool import terminal_tool; "
        f"result = terminal_tool(command=\"Write-Output '{sentinel}'\", {tool_argument}); "
        "print('HERMES_RESULT=' + result)"
    )

    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    marker = next(
        line for line in completed.stdout.splitlines()
        if line.startswith("HERMES_RESULT=")
    )
    result = json.loads(marker.removeprefix("HERMES_RESULT="))

    assert result.get("status") != "success"
    assert sentinel not in result.get("output", "")
    assert "PowerShell" in result["error"]
    assert mode_name in result["error"]


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


def test_explicit_empty_sudo_password_tries_empty_without_prompt(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("interactive sudo prompt should not run for explicit empty password")

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo true")

    assert transformed == "sudo -S -p '' true"
    assert sudo_stdin == "\n"


def test_validate_workdir_blocks_shell_metacharacters_in_windows_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project; rm -rf /")
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project$(whoami)")
    assert terminal_tool._validate_workdir("C:\\Users\\Alice\\project\nwhoami")


def test_validate_workdir_allows_unicode_filesystem_paths():
    assert terminal_tool._validate_workdir(
        "/Users/alice/Documents/Obs_Hermes_Data/项目-projects/客户拜访"
    ) is None
    assert terminal_tool._validate_workdir("/tmp/テスト") is None
    assert terminal_tool._validate_workdir("/home/jürgen/über projekt") is None


def test_validate_workdir_still_blocks_metachars_in_unicode_paths():
    # Widening to Unicode letters must not open the injection boundary:
    # shell metacharacters and control chars stay rejected even when mixed
    # with non-ASCII path segments.
    assert terminal_tool._validate_workdir("/tmp/テスト; rm -rf /")
    assert terminal_tool._validate_workdir("/tmp/项目$(whoami)")
    assert terminal_tool._validate_workdir("/tmp/über`id`")
    assert terminal_tool._validate_workdir("/tmp/テスト\nwhoami")
    assert terminal_tool._validate_workdir("/tmp/项目|cat /etc/passwd")
    assert terminal_tool._validate_workdir("/tmp/ü\x00ber")


def test_count_real_sudo_invocations_ignores_mentions(monkeypatch):
    assert terminal_tool._count_real_sudo_invocations("grep sudo README.md") == 0
    assert terminal_tool._count_real_sudo_invocations("sudo a; sudo b") == 2
