"""ssh's ControlMaster-fallback lines never lead an SSH command's output.

sshd allows MaxSessions (default 10) sessions per connection, and every command to one host rides
the client's ControlMaster. Past that limit the master refuses the session: ssh runs the command on
a direct connection (exit status unaffected) but first logs two lines on stderr, which the backend
merges into the command's output. Parallel ``vision_analyze`` reads then failed to decode the image
("non-image data"), and every other reply parsed from output (file reads, RPC request files, the
background poller's header) started with those lines.

The fake client below prints what OpenSSH prints in that case, then runs the command the way sshd
does (the words after the host, joined and handed to a shell), so the real ``SSHEnvironment``
execute path runs end to end without an sshd.
"""

import shlex

import pytest

from tools.environments import ssh as ssh_env
from tools.environments.ssh import SSHEnvironment
from tools.file_operations import ShellFileOperations


@pytest.fixture
def refused_session_env(tmp_path, monkeypatch):
    """An SSHEnvironment whose every command meets a ControlMaster that refused the session."""
    sock = shlex.quote(str(tmp_path / "0123456789abcdef.sock"))
    client = tmp_path / "ssh"
    client.write_text(
        "#!/bin/sh\n"
        "printf 'mux_client_request_session: session request failed: Session open refused by peer\\r\\n' >&2\n"
        f"printf 'ControlSocket %s already exists, disabling multiplexing\\r\\n' {sock} >&2\n"
        'exec sh -c "$*"\n')
    client.chmod(0o755)
    monkeypatch.setattr(ssh_env, "_ensure_ssh_available", lambda: None)
    monkeypatch.setattr(SSHEnvironment, "_build_ssh_command", lambda self, extra_args=None, send_env=(): [str(client)])
    monkeypatch.setattr(SSHEnvironment, "get_temp_dir", lambda self: str(tmp_path))
    work = tmp_path / "work"
    work.mkdir()
    env = SSHEnvironment(host="example.invalid", user="u", cwd=str(work), sync_files=False)
    yield env
    env.cleanup()


@pytest.mark.platforms("posix")
def test_command_output_starts_with_the_commands_own_text(refused_session_env):
    result = refused_session_env.execute("printf 'hello\\n'")
    assert result["returncode"] == 0
    assert result["output"] == "hello\n"


@pytest.mark.platforms("posix")
def test_file_read_returns_exactly_the_files_lines(refused_session_env, tmp_path):
    path = tmp_path / "work" / "notes.txt"
    path.write_text("first line\nsecond line\n")
    read = ShellFileOperations(refused_session_env).read_file(str(path))
    assert read.error is None
    assert "first line" in read.content and "second line" in read.content
    assert "multiplexing" not in read.content and "Session open refused" not in read.content


@pytest.mark.platforms("posix")
def test_the_same_text_from_the_command_itself_is_kept(refused_session_env):
    """Only ssh's own CRLF-terminated lines at the head go: a command that prints the same words
    (LF-terminated, or after its own output) keeps them."""
    own = ("ControlSocket /x.sock already exists, disabling multiplexing\n"
           "ok\n"
           "mux_client_request_session: session request failed: Session open refused by peer\r\n")
    result = refused_session_env.execute(f"printf '%s' {shlex.quote(own)}")
    assert result["output"] == own
