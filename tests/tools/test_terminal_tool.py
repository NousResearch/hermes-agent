"""Regression tests for sudo detection and sudo password handling."""

import tools.terminal_tool as terminal_tool


def setup_function():
    terminal_tool._cached_sudo_password = ""
    terminal_tool._cached_passwordless_sudo = None


def teardown_function():
    terminal_tool._cached_sudo_password = ""
    terminal_tool._cached_passwordless_sudo = None


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



def test_local_passwordless_sudo_uses_noninteractive_flag_without_prompt(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("TERMINAL_ENV", "local")

    monkeypatch.setattr(terminal_tool, "_can_use_passwordless_sudo", lambda: True)

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("interactive sudo prompt should not run when sudo -n already works")

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo systemctl status ssh")

    assert transformed == "sudo -n systemctl status ssh"
    assert sudo_stdin is None



def test_non_local_backends_skip_passwordless_probe_and_can_still_prompt(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("TERMINAL_ENV", "ssh")

    def _fail_probe():
        raise AssertionError("local passwordless sudo probe should not run for non-local backends")

    monkeypatch.setattr(terminal_tool, "_can_use_passwordless_sudo", _fail_probe)
    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", lambda timeout_seconds=45: "remote-pass")

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo whoami")

    assert transformed == "sudo -S -p '' whoami"
    assert sudo_stdin == "remote-pass\n"



def test_passwordless_rewrite_does_not_duplicate_existing_dash_n(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(terminal_tool, "_can_use_passwordless_sudo", lambda: True)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo -n true && sudo whoami")

    assert transformed == "sudo -n true && sudo -n whoami"
    assert sudo_stdin is None



def test_passwordless_rewrite_does_not_collapse_multiline_commands(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(terminal_tool, "_can_use_passwordless_sudo", lambda: True)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo\n-n true")

    assert transformed == "sudo -n\n-n true"
    assert sudo_stdin is None
