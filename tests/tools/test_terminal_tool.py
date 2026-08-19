"""Regression tests for sudo detection and sudo password handling."""

import tools.terminal_tool as terminal_tool
from tools.registry import registry


def setup_function():
    terminal_tool._reset_cached_sudo_passwords()


def teardown_function():
    terminal_tool._reset_cached_sudo_passwords()


def test_terminal_tool_registers_at_import_time():
    """Importing terminal_tool must register the terminal tool definition."""
    assert registry.get_entry("terminal") is not None


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


def test_get_env_config_preserves_remote_ssh_mnt_cwd(monkeypatch):
    """A remote SSH path must never be rewritten as a host Windows path."""
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_CWD", "/mnt/c/remote/project")

    config = terminal_tool._get_env_config()

    assert config["cwd"] == "/mnt/c/remote/project"


def test_get_env_config_converts_wsl_cwd_for_windows_local(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_LOCAL_CWD", "/mnt/c/Users/Alice/project")
    monkeypatch.setattr(terminal_tool.platform, "system", lambda: "Windows")

    config = terminal_tool._get_env_config()

    assert config["cwd"] == r"C:\Users\Alice\project"


def test_get_env_config_preserves_mnt_cwd_for_non_windows_local(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_LOCAL_CWD", "/mnt/c/project")
    monkeypatch.setattr(terminal_tool.platform, "system", lambda: "Linux")

    config = terminal_tool._get_env_config()

    assert config["cwd"] == "/mnt/c/project"


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
