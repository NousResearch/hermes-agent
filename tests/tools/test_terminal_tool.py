"""Regression tests for sudo detection and sudo password handling."""

import json

import pytest

import tools.terminal_tool as terminal_tool
from tools.environments.base import BaseEnvironment


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


def test_registered_empty_sudo_callback_preserves_skip(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.setattr(terminal_tool, "_sudo_nopasswd_works", lambda: False)
    terminal_tool.set_sudo_password_callback(lambda: "")
    try:
        transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo true")
    finally:
        terminal_tool.set_sudo_password_callback(None)

    assert transformed == "sudo true"
    assert sudo_stdin is None


def test_cancelled_sudo_prompt_stops_before_command_execution(monkeypatch, tmp_path):
    task_id = "sudo-cancel-test"

    class MinimalEnvironment(BaseEnvironment):
        def __init__(self):
            super().__init__(cwd=str(tmp_path), timeout=60)
            self.run_bash_calls = 0

        def _run_bash(self, *_args, **_kwargs):
            self.run_bash_calls += 1
            return object()

        def _wait_for_process(self, *_args, **_kwargs):
            return {"output": "executed", "returncode": 0}

        def cleanup(self):
            pass

    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(terminal_tool, "_sudo_nopasswd_works", lambda: False)
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda _task_id: task_id)
    monkeypatch.setattr(terminal_tool, "resolve_task_overrides", lambda _task_id: {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {"env_type": "local", "cwd": str(tmp_path), "timeout": 60},
    )
    environment = MinimalEnvironment()
    monkeypatch.setitem(terminal_tool._active_environments, task_id, environment)
    monkeypatch.setitem(terminal_tool._last_activity, task_id, 0.0)
    terminal_tool.set_sudo_password_callback(lambda: None)
    try:
        result = json.loads(terminal_tool.terminal_tool("sudo shutdown now", force=True))
    finally:
        terminal_tool.set_sudo_password_callback(None)

    assert result == {
        "output": "",
        "exit_code": 130,
        "error": "Command cancelled: sudo password prompt was dismissed.",
        "status": "cancelled",
    }
    assert environment.run_bash_calls == 0


@pytest.mark.parametrize("background", [False, True])
def test_nonlocal_sudo_dismissal_has_foreground_background_parity(
    monkeypatch, tmp_path, background
):
    import tools.process_registry as process_registry_module

    task_id = f"sudo-cancel-{'background' if background else 'foreground'}"
    calls = []

    class FakeEnvironment:
        cwd = str(tmp_path)

        def execute(self, *_args, **_kwargs):
            calls.append("foreground")
            raise terminal_tool.SudoPasswordPromptCancelled

    class FakeRegistry:
        def spawn_via_env(self, **_kwargs):
            calls.append("background")
            raise terminal_tool.SudoPasswordPromptCancelled

    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda _task_id: task_id)
    monkeypatch.setattr(terminal_tool, "resolve_task_overrides", lambda _task_id: {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "docker",
            "docker_image": "inert-test-image",
            "cwd": str(tmp_path),
            "timeout": 60,
        },
    )
    monkeypatch.setitem(terminal_tool._active_environments, task_id, FakeEnvironment())
    monkeypatch.setitem(terminal_tool._last_activity, task_id, 0.0)
    monkeypatch.setattr(process_registry_module, "process_registry", FakeRegistry())

    result = json.loads(
        terminal_tool.terminal_tool(
            "sudo true",
            background=background,
            task_id=task_id,
            force=True,
        )
    )

    assert result == {
        "output": "",
        "exit_code": 130,
        "error": "Command cancelled: sudo password prompt was dismissed.",
        "status": "cancelled",
    }
    assert calls == ["background" if background else "foreground"]


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
