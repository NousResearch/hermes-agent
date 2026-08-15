"""Regression tests for the terminal tool (sudo handling, workdir validation, and timeout classification)."""

import json
import subprocess

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


def test_is_timeout_error_recognizes_builtin_timeout():
    assert terminal_tool._is_timeout_error(TimeoutError()) is True


def test_is_timeout_error_recognizes_subprocess_timeout_expired():
    exc = subprocess.TimeoutExpired("sleep 10", timeout=5)
    assert terminal_tool._is_timeout_error(exc) is True


def test_is_timeout_error_recognizes_timeout_messages():
    assert terminal_tool._is_timeout_error(RuntimeError("connection timeout")) is True
    assert terminal_tool._is_timeout_error(RuntimeError("request timed out")) is True


def test_is_timeout_error_does_not_flag_unrelated_errors():
    assert terminal_tool._is_timeout_error(RuntimeError("something went wrong")) is False


class _FakeTimeoutEnv:
    def __init__(self, exc):
        self.exc = exc
        self.cwd = None

    def execute(self, *args, **kwargs):
        raise self.exc


def _run_terminal_with_timeout_exc(monkeypatch, exc):
    terminal_tool._active_environments.clear()
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", "/tmp")
    monkeypatch.setattr(terminal_tool.time, "sleep", lambda _x: None)
    monkeypatch.setattr(
        terminal_tool,
        "_create_environment",
        lambda *args, **kwargs: _FakeTimeoutEnv(exc),
    )
    return json.loads(terminal_tool.terminal_tool("sleep 10", force=True, timeout=5))


def test_terminal_tool_timeout_error_returns_124_without_retry(monkeypatch):
    result = _run_terminal_with_timeout_exc(monkeypatch, TimeoutError())
    assert result["exit_code"] == 124
    assert "timed out after 5 seconds" in result["error"].lower()


def test_terminal_tool_subprocess_timeout_expired_returns_124_without_retry(monkeypatch):
    exc = subprocess.TimeoutExpired("sleep 10", timeout=5)
    result = _run_terminal_with_timeout_exc(monkeypatch, exc)
    assert result["exit_code"] == 124
    assert "timed out after 5 seconds" in result["error"].lower()
