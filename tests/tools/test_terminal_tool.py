"""Regression tests for sudo detection and sudo password handling."""

import tools.terminal_tool as terminal_tool


def setup_function():
    terminal_tool._cached_sudo_password = ""


def teardown_function():
    terminal_tool._cached_sudo_password = ""


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


def test_validate_workdir_allows_windows_drive_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project") is None
    assert terminal_tool._validate_workdir("C:/Users/Alice/project") is None


def test_validate_workdir_allows_windows_unc_paths():
    assert terminal_tool._validate_workdir(r"\\server\share\project") is None


def test_validate_workdir_blocks_shell_metacharacters_in_windows_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project; rm -rf /")
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project$(whoami)")
    assert terminal_tool._validate_workdir("C:\\Users\\Alice\\project\nwhoami")
