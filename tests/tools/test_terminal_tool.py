"""Regression tests for sudo detection and sudo password handling."""

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
    assert "do not re-source the same environment before every command" in description


def test_terminal_schema_discourages_single_command_wrappers():
    description = terminal_tool.TERMINAL_TOOL_DESCRIPTION

    assert "Use terminal directly for a single shell command" in description
    assert "may add whole-script approval" in description
    assert "without avoiding this tool's command-level checks" in description
    # Keep this standalone schema usable when execute_code is disabled.
    assert "execute_code" not in description


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
    terminal_tool._set_cached_sudo_password("cached-pass")

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("echo ok && sudo whoami")

    assert transformed == "echo ok && sudo -S -p '' whoami"
    assert sudo_stdin == "cached-pass\n"


def test_registered_sudo_callback_is_used_without_interactive_env(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.setattr(terminal_tool, "_sudo_nopasswd_works", lambda: False)

    calls = []

    def sudo_callback():
        calls.append("called")
        return "callback-pass"

    terminal_tool.set_sudo_password_callback(sudo_callback)
    try:
        transformed, sudo_stdin = terminal_tool._transform_sudo_command(
            "echo ok | sudo tee /tmp/hermes-test"
        )
    finally:
        terminal_tool.set_sudo_password_callback(None)

    assert calls == ["called"]
    assert transformed == "echo ok | sudo -S -p '' tee /tmp/hermes-test"
    assert sudo_stdin == "callback-pass\n"


def test_cached_sudo_password_isolated_by_session_key(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-a")
    terminal_tool._set_cached_sudo_password("alpha-pass")

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-b")
    assert terminal_tool._get_cached_sudo_password() == ""

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-a")
    assert terminal_tool._get_cached_sudo_password() == "alpha-pass"


def test_passwordless_sudo_skips_interactive_prompt_and_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError(
            "interactive sudo prompt should not run when sudo -n already works"
        )

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)
    monkeypatch.setattr(terminal_tool, "_sudo_nopasswd_works", lambda: True, raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo whoami")

    assert transformed == "sudo whoami"
    assert sudo_stdin is None


def test_passwordless_sudo_probe_rechecks_local_terminal(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    calls = []

    class Result:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return Result(0 if len(calls) == 1 else 1)

    monkeypatch.setattr(terminal_tool.subprocess, "run", fake_run)

    assert terminal_tool._sudo_nopasswd_works() is True
    assert terminal_tool._sudo_nopasswd_works() is False
    assert len(calls) == 2
    assert calls[0][0] == ["sudo", "-n", "true"]
    assert calls[1][0] == ["sudo", "-n", "true"]


def test_passwordless_sudo_probe_is_disabled_for_nonlocal_terminal_env(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    def _fail_run(*_args, **_kwargs):
        raise AssertionError("host sudo probe must not run for non-local terminal envs")

    monkeypatch.setattr(terminal_tool.subprocess, "run", _fail_run)

    assert terminal_tool._sudo_nopasswd_works() is False


def test_validate_workdir_allows_windows_drive_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project") is None
    assert terminal_tool._validate_workdir("C:/Users/Alice/project") is None


def test_validate_workdir_allows_windows_unc_paths():
    assert terminal_tool._validate_workdir(r"\\server\share\project") is None


def test_validate_workdir_blocks_shell_metacharacters_in_windows_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project; rm -rf /")
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project$(whoami)")
    assert terminal_tool._validate_workdir("C:\\Users\\Alice\\project\nwhoami")


def test_get_env_config_ignores_bad_docker_json_for_local_backend(monkeypatch):
    """Docker-only JSON env vars must not break the default local backend."""
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "None")
    monkeypatch.setenv("TERMINAL_DOCKER_ENV", "not-json")
    monkeypatch.setenv("TERMINAL_DOCKER_FORWARD_ENV", "not-json")
    monkeypatch.setenv("TERMINAL_DOCKER_EXTRA_ARGS", "not-json")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "local"
    assert config["docker_volumes"] == []
    assert config["docker_env"] == {}
    assert config["docker_forward_env"] == []
    assert config["docker_extra_args"] == []


def test_get_env_config_ignores_bad_docker_json_for_ssh_backend(monkeypatch):
    """Non-container remote backends should also ignore Docker-only JSON."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "None")
    monkeypatch.setenv("TERMINAL_DOCKER_ENV", "not-json")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "ssh"
    assert config["docker_volumes"] == []
    assert config["docker_env"] == {}


def test_get_env_config_preserves_ssh_tilde_cwd(monkeypatch):
    """SSH cwd '~' is expanded by the remote shell, not the Hermes host."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", "~")
    monkeypatch.setenv("HOME", "/opt/data")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "ssh"
    assert config["cwd"] == "~"


def test_get_env_config_preserves_ssh_tilde_child_cwd(monkeypatch):
    """SSH cwd '~/x' must not become the local/container HOME path."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", "~/project")
    monkeypatch.setenv("HOME", "/opt/data")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "ssh"
    assert config["cwd"] == "~/project"


def test_get_env_config_still_rejects_bad_docker_json_for_docker_backend(monkeypatch):
    """Selecting Docker should keep the existing actionable config error."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "None")

    try:
        terminal_tool._get_env_config()
    except ValueError as exc:
        assert "TERMINAL_DOCKER_VOLUMES" in str(exc)
    else:
        raise AssertionError("Docker backend must validate TERMINAL_DOCKER_VOLUMES")


def test_sudo_wrong_password_failure_detects_rejection_output():
    output = (
        "sudo: Authentication failed, try again.\n\n"
        "sudo: maximum 3 incorrect authentication attempts\n"
    )
    assert terminal_tool._sudo_wrong_password_failure(output) is True


def test_sudo_wrong_password_failure_ignores_tty_required_message():
    output = "sudo: a terminal is required to authenticate"
    assert terminal_tool._sudo_wrong_password_failure(output) is False


def test_invalidate_cached_sudo_on_auth_failure_clears_session_cache(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    terminal_tool._set_cached_sudo_password("wrong-pass")

    cleared = terminal_tool._invalidate_cached_sudo_on_auth_failure(
        "sudo apt install fprintd",
        "sudo: Authentication failed, try again.",
    )

    assert cleared is True
    assert terminal_tool._get_cached_sudo_password() == ""


def test_invalidate_cached_sudo_on_auth_failure_keeps_env_password(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "from-env")
    terminal_tool._set_cached_sudo_password("wrong-pass")

    cleared = terminal_tool._invalidate_cached_sudo_on_auth_failure(
        "sudo true",
        "sudo: Authentication failed, try again.",
    )

    assert cleared is False
    assert terminal_tool._get_cached_sudo_password() == "wrong-pass"


def test_transform_sudo_command_pipes_one_password_line_per_invocation(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command(
        "sudo true && sudo whoami"
    )

    assert transformed == "sudo -S -p '' true && sudo -S -p '' whoami"
    assert sudo_stdin == "testpass\ntestpass\n"


def test_count_real_sudo_invocations_ignores_mentions(monkeypatch):
    assert terminal_tool._count_real_sudo_invocations("grep sudo README.md") == 0
    assert terminal_tool._count_real_sudo_invocations("sudo a; sudo b") == 2
