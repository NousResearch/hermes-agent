"""Regression tests for sudo detection and sudo password handling."""

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


def test_non_interactive_sudo_probe_passes_through_unchanged(monkeypatch):
    """#94534: ``sudo -n`` means "fail immediately if a password would be
    required" and never reads a piped password — rewriting it to
    ``sudo -S -p '' -n`` makes the probe always fail with "a password is
    required" even when one is configured, so the model concludes sudo is
    broken. -n invocations must pass through with no prompt and no stdin."""
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    for command in (
        "sudo -n true",
        "sudo --non-interactive id",
        "sudo -nv true",
        "sudo -u janet-admin -n id",
        "sudo --user janet-admin --non-interactive id",
        "sudo --user=janet-admin -n id",
        "sudo -g wheel -n id",
    ):
        transformed, sudo_stdin = terminal_tool_sudo._transform_sudo_command(command)
        assert transformed == command
        assert sudo_stdin is None


def test_non_interactive_sudo_probe_never_prompts_interactively(monkeypatch):
    """Even with no configured password and an interactive UI available, a
    ``-n`` probe must not trigger the 45s sudo password prompt (#94534)."""
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("interactive sudo prompt must not run for a -n probe")

    monkeypatch.setattr(terminal_tool_sudo, "_prompt_for_sudo_password", _fail_prompt)

    transformed, sudo_stdin = terminal_tool_sudo._transform_sudo_command(
        "sudo -n true", sudo_nopasswd_check=lambda: False
    )

    assert transformed == "sudo -n true"
    assert sudo_stdin is None


def test_compound_command_rewrites_only_non_n_sudo_invocations(monkeypatch):
    """``sudo -n true && sudo apt update`` keeps the probe verbatim while the
    real invocation takes the normal password-pipe rewrite (#94534)."""
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool_sudo._transform_sudo_command(
        "sudo -n true && sudo apt update"
    )

    assert transformed == "sudo -n true && sudo -S -p '' apt update"
    assert sudo_stdin == "testpass\n"


def test_sudo_flag_like_argument_of_child_command_is_not_non_interactive(monkeypatch):
    """``-n`` belonging to the command sudo runs, not to sudo itself, must
    not disable the rewrite."""
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool_sudo._transform_sudo_command(
        "sudo rm -n /tmp/thing"
    )

    assert transformed == "sudo -S -p '' rm -n /tmp/thing"
    assert sudo_stdin == "testpass\n"


def test_non_interactive_sudo_probe_with_option_values_keeps_scanning(monkeypatch):
    """Option-order cases from #94534 review: an option whose value arrives
    as a separate token (``-u janet-admin``, ``--user janet-admin``) must
    not stop the flag scan — sudo still sees a later ``-n``, so the
    invocation passes through verbatim. Without a ``-n`` the same shapes
    take the normal rewrite, and the separate value is consumed verbatim
    even when it looks like a flag (getopt takes the next argv element as
    the option's argument)."""
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    for command in (
        "sudo -u janet-admin -n id",
        "sudo --user janet-admin --non-interactive id",
        "sudo -u 'janet admin' -n id",
    ):
        transformed, sudo_stdin = terminal_tool_sudo._transform_sudo_command(command)
        assert transformed == command
        assert sudo_stdin is None

    transformed, sudo_stdin = terminal_tool_sudo._transform_sudo_command(
        "sudo -u janet-admin id"
    )
    assert transformed == "sudo -S -p '' -u janet-admin id"
    assert sudo_stdin == "testpass\n"

    transformed, sudo_stdin = terminal_tool_sudo._transform_sudo_command(
        "sudo -u -n id"
    )
    assert transformed == "sudo -S -p '' -u -n id"
    assert sudo_stdin == "testpass\n"


def test_non_interactive_sudo_probe_covers_chdir_chroot_host_and_clusters(monkeypatch):
    """#94592 review: the remaining value-taking sudo options (``-D``/``--chdir``,
    ``--chroot``, ``--host``), the ``-A/-B/-H/-N`` argument-less clusters, and
    quoted flags must all let the scan reach a later ``-n``. Conversely ``-n``
    consumed as a value (``sudo -D -n id`` changes directory to ``-n``) is not
    sudo's non-interactive flag and keeps the password-pipe rewrite."""
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    for command in (
        "sudo -D / -n id",
        "sudo --chdir / --non-interactive id",
        "sudo --chroot / -n id",
        "sudo --host web01 -n id",
        "sudo --host=web01 -n id",
        "sudo -Hn id",
        "sudo -An id",
        "sudo -Bn id",
        "sudo -Nn id",
        'sudo "-n" id',
        "sudo '-n' id",
    ):
        transformed, sudo_stdin = terminal_tool_sudo._transform_sudo_command(command)
        assert transformed == command
        assert sudo_stdin is None

    for command, rewritten in (
        ("sudo -D -n id", "sudo -S -p '' -D -n id"),
        ("sudo --chdir -n id", "sudo -S -p '' --chdir -n id"),
        ("sudo -H id", "sudo -S -p '' -H id"),
    ):
        transformed, sudo_stdin = terminal_tool_sudo._transform_sudo_command(command)
        assert transformed == rewritten
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
