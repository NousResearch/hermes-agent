"""Regression tests for sudo detection and sudo password handling."""

import pytest

from tools import terminal_tool
from tools import terminal_tool_sudo


def setup_function():
    terminal_tool_sudo._reset_cached_sudo_passwords()


def teardown_function():
    terminal_tool_sudo._reset_cached_sudo_passwords()


def test_searching_for_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "rg --line-number --no-heading --with-filename 'sudo' . | head -n 20"
    transformed, sudo_stdin = terminal_tool_sudo._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None








def test_actual_sudo_command_uses_configured_password(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool_sudo._transform_sudo_command("sudo apt install -y ripgrep")

    assert transformed == "sudo -S -p '' apt install -y ripgrep"
    assert sudo_stdin == "testpass\n"


def test_explicit_empty_sudo_password_tries_empty_without_prompt(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("interactive sudo prompt should not run for explicit empty password")

    monkeypatch.setattr(terminal_tool_sudo, "_prompt_for_sudo_password", _fail_prompt)

    transformed, sudo_stdin = terminal_tool_sudo._transform_sudo_command("sudo true")

    assert transformed == "sudo -S -p '' true"
    assert sudo_stdin == "\n"


def test_headless_sudo_never_runs_backend_nopasswd_probe(monkeypatch):
    """No prompt can fire without a UI, so the backend round trip must not be paid."""
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool.set_sudo_password_callback(None)

    def _fail_probe():
        raise AssertionError("headless sudo must not probe the backend")

    assert terminal_tool_sudo._transform_sudo_command("sudo true", sudo_nopasswd_check=_fail_probe) == (
        "sudo true", None)


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


def test_literal_sudo_executables_receive_password_stdin(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    for prefix in ("", "VAR='a b' ", "env ", "'/usr/bin/env' -i -u UNUSED X=1 ",
                   "env --unset=UNUSED --chdir /tmp -- X=1 ", "env -uUNUSED -C/tmp "):
        for executable in ("sudo", "/usr/bin/sudo", "'/opt/my tools/sudo'", '"/usr/bin/sudo"'):
            command = prefix + executable + " -u root true"
            rewritten, stdin = terminal_tool_sudo._transform_sudo_command(command)
            assert rewritten == prefix + executable + " -S -p '' -u root true"
            assert stdin == "testpass\n"


def test_sudo_rewrite_preserves_env_operands_and_prose(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    commands = (
        "echo '/usr/bin/sudo true'", "env echo sudo true", "env -u sudo echo ok",
        "env --chdir sudo echo ok", "env --unset=sudo echo ok", "env -- sudo=1 echo ok",
        ">/tmp/sudo echo ok", "env 2>/tmp/sudo echo ok", "env > /tmp/sudo echo ok",
        "/tmp/{a,b}/sudo true", "env X=1 -u UNUSED sudo", "env - -u UNUSED sudo", "env echo /usr/bin/sudo", "/tmp/*/sudo true",
        "env -S 'sudo true'", "env --unknown sudo true", "env --help sudo",
        "bash -c 'sudo true'", "echo ok # prose; /usr/bin/sudo true",
        '"/usr/bin/sudo', "env -u sudo", "env X=sudo", '"X=1" /usr/bin/sudo true',
    )
    for command in commands:
        assert terminal_tool_sudo._transform_sudo_command(command) == (command, None)


def test_count_real_sudo_invocations_ignores_mentions(monkeypatch):
    assert terminal_tool_sudo._count_real_sudo_invocations("grep sudo README.md") == 0
    assert terminal_tool_sudo._count_real_sudo_invocations("sudo a; sudo b") == 2


def _fresh_cwd(monkeypatch, env_type, shared=None, backend=None):
    """Resolve cwd with no leftover per-backend env var and a clean auto-save latch."""
    monkeypatch.setenv("TERMINAL_ENV", env_type)
    monkeypatch.delenv(f"TERMINAL_{env_type.upper()}_CWD", raising=False)
    if shared is None:
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
    else:
        monkeypatch.setenv("TERMINAL_CWD", shared)
    if backend is not None:
        monkeypatch.setenv(f"TERMINAL_{env_type.upper()}_CWD", backend)
    terminal_tool._AUTO_SAVED_CWDS.discard(env_type)
    return terminal_tool._get_env_config()["cwd"]


def test_terminal_tool_registers_at_import_time():
    from tools.registry import registry

    assert registry.get_entry("terminal") is not None


def test_ssh_cwd_keeps_mnt_path(monkeypatch):
    cwd = _fresh_cwd(monkeypatch, "ssh", shared="/mnt/c/Users/Alice/project")
    assert cwd == "/mnt/c/Users/Alice/project"


def test_ssh_tilde_cwd_is_not_expanded(monkeypatch):
    cwd = _fresh_cwd(monkeypatch, "ssh", backend="~/project")
    assert cwd == "~/project"


@pytest.mark.platforms("linux")
def test_local_cwd_keeps_mnt_path_off_windows(monkeypatch):
    cwd = _fresh_cwd(monkeypatch, "local", shared="/mnt/c/project")
    assert cwd == "/mnt/c/project"


def test_local_mnt_path_converts_only_for_windows_local():
    path = "/mnt/c/Users/Alice/project"
    assert terminal_tool._convert_cross_backend_cwd("local", path, system="Windows") == r"C:\Users\Alice\project"
    assert terminal_tool._convert_cross_backend_cwd("ssh", path, system="Windows") == path
    assert terminal_tool._convert_cross_backend_cwd("local", "/mnt/c/project", system="Linux") == "/mnt/c/project"


def test_wsl_cwd_converts_windows_drive(monkeypatch):
    cwd = _fresh_cwd(monkeypatch, "wsl", shared=r"C:\Users\Alice\project")
    assert cwd == "/mnt/c/Users/Alice/project"


def test_wsl_cwd_discards_non_linux_shared_path(monkeypatch):
    cwd = _fresh_cwd(monkeypatch, "wsl", shared="/var/lib/hermes")
    assert cwd == ""


def test_wsl_keeps_tmp_and_opt_directories_not_lookalike_names(monkeypatch):
    assert _fresh_cwd(monkeypatch, "wsl", shared="/tmp") == "/tmp"
    assert _fresh_cwd(monkeypatch, "wsl", shared="/tmp/work") == "/tmp/work"
    assert _fresh_cwd(monkeypatch, "wsl", shared="/opt/app") == "/opt/app"
    assert _fresh_cwd(monkeypatch, "wsl", shared="/tmpfoo") == ""
    assert _fresh_cwd(monkeypatch, "wsl", shared="/optfoo") == ""


def test_explicit_backend_cwd_wins_over_shared(monkeypatch):
    cwd = _fresh_cwd(monkeypatch, "docker", shared="/tmp/shared", backend="/opt/app")
    assert cwd == "/opt/app"


def test_wsl_cwd_is_remembered_without_touching_environ(monkeypatch):
    monkeypatch.delenv("TERMINAL_WSL_CWD", raising=False)
    cwd = _fresh_cwd(monkeypatch, "wsl", shared="/home/agents/project")
    assert cwd == "/home/agents/project"
    assert "TERMINAL_WSL_CWD" not in __import__("os").environ
    from hermes_cli.config import read_raw_config

    assert read_raw_config().get("terminal", {}).get("wsl_cwd") == "/home/agents/project"


def test_other_backends_do_not_rewrite_config_on_cwd_resolve(monkeypatch):
    cwd = _fresh_cwd(monkeypatch, "docker", shared="/opt/app")
    assert cwd == "/opt/app"
    from hermes_cli.config import read_raw_config

    assert "docker_cwd" not in (read_raw_config().get("terminal") or {})
