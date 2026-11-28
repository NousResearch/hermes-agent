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


def test_sudo_inside_a_heredoc_body_is_data(monkeypatch):
    """A heredoc body is written verbatim and never parsed as shell code.

    Counting a ``sudo`` line in there made the terminal tool demand a password for a script
    that only runs later on another machine (#158788).
    """
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("a heredoc body must not trigger the sudo prompt")

    monkeypatch.setattr(terminal_tool_sudo, "_prompt_for_sudo_password", _fail_prompt)

    commands = (
        "cat > x.sh <<'EOF'\nsudo -n tc qdisc del dev eth0 ingress\nEOF\nssh h 'bash x.sh'",
        "cat > x.sh <<EOF\nsudo apt install -y foo\nEOF",
        "cat > x.sh <<-EOF\n\t\tsudo whoami\n\tEOF\necho done",
        'cat <<"END"\nsudo rm -rf /\nEND',
        "cat <<\\EOF\nsudo x\nEOF",
        "cat <<A <<B\naaa\nA\nsudo in b\nB",
        "cat <<EOF\nsudo x",  # unterminated: the rest of the command is the body
    )
    for command in commands:
        assert terminal_tool_sudo._count_real_sudo_invocations(command) == 0, command
        assert terminal_tool_sudo._transform_sudo_command(command) == (command, None), command


def test_sudo_after_a_heredoc_body_is_still_counted(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    command = "cat > x.sh <<'EOF'\nsudo inner\nEOF\nsudo apt install -y foo"
    assert terminal_tool_sudo._count_real_sudo_invocations(command) == 1
    assert terminal_tool_sudo._transform_sudo_command(command) == (
        "cat > x.sh <<'EOF'\nsudo inner\nEOF\nsudo -S -p '' apt install -y foo",
        "testpass\n",
    )


def test_heredoc_lookalikes_are_not_bodies(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    for command in (
        "echo $((1<<3)); sudo ls",       # arithmetic shift, not a heredoc
        'echo \'<<EOF\'\nsudo ls',        # quoted, so no heredoc opens
        "echo x # <<EOF\nsudo ls",       # comment, so no heredoc opens
        'cat <<< "sudo x"; sudo ls',     # here-string, no body
    ):
        assert terminal_tool_sudo._count_real_sudo_invocations(command) == 1, command


def test_sudo_rewrite_keeps_heredoc_bytes_and_tiles_the_command():
    command = "cat > x.sh <<'EOF'\nsudo -n tc qdisc del dev eth0 ingress\n{ & } $(x) 'q'\nEOF\ncat x.sh"
    assert terminal_tool_sudo._rewrite_real_sudo_invocations(command) == (command, 0)
    tiled = "".join(command[start:end] for _kind, start, end, _at_start
                    in terminal_tool_sudo._scan_shell(command))
    assert tiled == command
