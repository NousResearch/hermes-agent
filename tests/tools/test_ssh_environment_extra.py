"""Additional unit tests for SSHEnvironment — covering connection, exec,
file-sync helpers, and cleanup paths not reached by the existing test files.

All tests are hermetic: no real SSH connections, no real subprocesses.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from tools.environments import ssh as ssh_env
from tools.environments.ssh import SSHEnvironment


# ---------------------------------------------------------------------------
# Shared fixture: fully-stubbed SSHEnvironment ready for method-level tests
# ---------------------------------------------------------------------------


@pytest.fixture
def env(monkeypatch):
    """Return an SSHEnvironment whose __init__ side-effects are all mocked.

    The environment is constructed with host='h', user='u', port=22,
    key_path='', cwd='~', timeout=60.  _detect_remote_home returns
    '/home/u' so _remote_home is predictable in assertions.
    """
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr(SSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(SSHEnvironment, "_detect_remote_home", lambda self: "/home/u")
    monkeypatch.setattr(SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(SSHEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(
        ssh_env, "FileSyncManager",
        lambda **kw: type("M", (), {"sync": lambda self, **k: None,
                                     "sync_back": lambda self: None})(),
    )
    return SSHEnvironment(host="h", user="u")


# ---------------------------------------------------------------------------
# _establish_connection
# ---------------------------------------------------------------------------


class TestEstablishConnection:
    """Tests for the SSH handshake subprocess."""

    @pytest.fixture(autouse=True)
    def _stub_init(self, monkeypatch):
        """Stub everything in __init__ EXCEPT _establish_connection itself."""
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
        monkeypatch.setattr(SSHEnvironment, "_detect_remote_home", lambda self: "/home/u")
        monkeypatch.setattr(SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
        monkeypatch.setattr(SSHEnvironment, "init_session", lambda self: None)
        monkeypatch.setattr(
            ssh_env, "FileSyncManager",
            lambda **kw: type("M", (), {"sync": lambda self, **k: None,
                                         "sync_back": lambda self: None})(),
        )

    def test_success(self, monkeypatch):
        """A zero-return-code subprocess means connection established, no raise."""
        monkeypatch.setattr(
            ssh_env.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="SSH connection established\n", stderr=""),
        )
        # Should not raise
        SSHEnvironment(host="h", user="u")

    def test_non_zero_returncode_raises(self, monkeypatch):
        """A non-zero return code should raise RuntimeError with ssh stderr."""
        monkeypatch.setattr(
            ssh_env.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 255, stdout="", stderr="Connection refused"),
        )
        with pytest.raises(RuntimeError, match="SSH connection failed"):
            SSHEnvironment(host="h", user="u")

    def test_stderr_included_in_error_message(self, monkeypatch):
        """The error message should include the ssh stderr text."""
        monkeypatch.setattr(
            ssh_env.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Host key verification failed"),
        )
        with pytest.raises(RuntimeError, match="Host key verification failed"):
            SSHEnvironment(host="h", user="u")

    def test_timeout_raises_runtime_error(self, monkeypatch):
        """A subprocess.TimeoutExpired should be converted to RuntimeError."""
        def _raise_timeout(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 15)

        monkeypatch.setattr(ssh_env.subprocess, "run", _raise_timeout)
        with pytest.raises(RuntimeError, match="timed out"):
            SSHEnvironment(host="h", user="u")


# ---------------------------------------------------------------------------
# _detect_remote_home
# ---------------------------------------------------------------------------


class TestDetectRemoteHome:
    """Tests for the remote $HOME detection subprocess."""

    @pytest.fixture(autouse=True)
    def _stub_init(self, monkeypatch):
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
        monkeypatch.setattr(SSHEnvironment, "_establish_connection", lambda self: None)
        monkeypatch.setattr(SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
        monkeypatch.setattr(SSHEnvironment, "init_session", lambda self: None)
        monkeypatch.setattr(
            ssh_env, "FileSyncManager",
            lambda **kw: type("M", (), {"sync": lambda self, **k: None,
                                         "sync_back": lambda self: None})(),
        )

    def test_uses_stdout_when_successful(self, monkeypatch):
        """When subprocess succeeds, _remote_home should equal the stdout value."""
        calls = []

        def _fake_run(cmd, **kw):
            calls.append(cmd)
            if calls[-1][-1] == "echo $HOME":
                return subprocess.CompletedProcess(cmd, 0, stdout="/home/remoteuser\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(ssh_env.subprocess, "run", _fake_run)
        env = SSHEnvironment(host="h", user="remoteuser")
        assert env._remote_home == "/home/remoteuser"

    def test_fallback_root_user_when_subprocess_fails(self, monkeypatch):
        """When the subprocess raises, root user should fall back to /root."""
        def _raise(cmd, **kw):
            raise OSError("ssh not found")

        monkeypatch.setattr(ssh_env.subprocess, "run", _raise)
        env = SSHEnvironment(host="h", user="root")
        assert env._remote_home == "/root"

    def test_fallback_nonroot_user_when_subprocess_fails(self, monkeypatch):
        """When the subprocess raises, non-root user should fall back to /home/<user>."""
        def _raise(cmd, **kw):
            raise OSError("ssh not found")

        monkeypatch.setattr(ssh_env.subprocess, "run", _raise)
        env = SSHEnvironment(host="h", user="alice")
        assert env._remote_home == "/home/alice"

    def test_fallback_when_returncode_nonzero(self, monkeypatch):
        """When the subprocess returns non-zero with empty stdout, fall back."""
        monkeypatch.setattr(
            ssh_env.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error"),
        )
        env = SSHEnvironment(host="h", user="bob")
        assert env._remote_home == "/home/bob"


# ---------------------------------------------------------------------------
# _ensure_remote_dirs
# ---------------------------------------------------------------------------


class TestEnsureRemoteDirs:
    """Verify that _ensure_remote_dirs creates the expected .hermes tree."""

    def test_creates_expected_dirs(self, monkeypatch):
        """_ensure_remote_dirs should issue a single SSH call with all four dirs."""
        # Build env without stubbing _ensure_remote_dirs so we call the real one.
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _: "/usr/bin/ssh")
        monkeypatch.setattr(SSHEnvironment, "_establish_connection", lambda self: None)
        monkeypatch.setattr(SSHEnvironment, "_detect_remote_home", lambda self: "/home/u")
        monkeypatch.setattr(SSHEnvironment, "init_session", lambda self: None)
        monkeypatch.setattr(
            ssh_env, "FileSyncManager",
            lambda **kw: type("M", (), {"sync": lambda self, **k: None,
                                         "sync_back": lambda self: None})(),
        )

        run_calls = []

        def _capture_run(cmd, **kw):
            run_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(ssh_env.subprocess, "run", _capture_run)
        # _ensure_remote_dirs is called during __init__ here
        env = SSHEnvironment(host="h", user="u")

        assert len(run_calls) >= 1
        # The mkdir call should mention all four expected dirs
        all_cmds = " ".join(" ".join(c) for c in run_calls)
        assert "/home/u/.hermes" in all_cmds
        assert "/home/u/.hermes/skills" in all_cmds
        assert "/home/u/.hermes/credentials" in all_cmds
        assert "/home/u/.hermes/cache" in all_cmds


# ---------------------------------------------------------------------------
# _scp_upload
# ---------------------------------------------------------------------------


class TestScpUpload:
    """Unit tests for single-file SCP upload."""

    def test_success(self, env, tmp_path, monkeypatch):
        """A successful scp run should not raise."""
        src = tmp_path / "file.txt"
        src.write_text("hello")

        monkeypatch.setattr(
            ssh_env.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0),
        )
        # Should not raise
        env._scp_upload(str(src), "/home/u/.hermes/skills/file.txt")

    def test_failure_raises_runtime_error(self, env, tmp_path, monkeypatch):
        """A non-zero scp returncode should raise RuntimeError with scp stderr."""
        src = tmp_path / "file.txt"
        src.write_text("hello")

        run_calls = []

        def _mock_run(cmd, **kw):
            run_calls.append(cmd)
            # First call is the mkdir over SSH — succeed.
            # Second call is the actual scp — fail.
            if "scp" in cmd[0]:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no such file")
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(ssh_env.subprocess, "run", _mock_run)
        with pytest.raises(RuntimeError, match="scp failed"):
            env._scp_upload(str(src), "/home/u/.hermes/skills/file.txt")

    def test_scp_command_includes_control_socket(self, env, tmp_path, monkeypatch):
        """The scp command must reuse the ControlMaster socket for performance."""
        src = tmp_path / "f.txt"
        src.write_text("x")

        captured = []

        def _capture(cmd, **kw):
            captured.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(ssh_env.subprocess, "run", _capture)
        env._scp_upload(str(src), "/home/u/.hermes/skills/f.txt")

        # The scp command is the second call (mkdir first)
        scp_cmd = next(c for c in captured if c[0] == "scp")
        assert f"ControlPath={env.control_socket}" in " ".join(scp_cmd)

    def test_scp_uses_custom_port(self, monkeypatch, tmp_path):
        """scp should use -P for non-default ports."""
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _: "/usr/bin/ssh")
        monkeypatch.setattr(SSHEnvironment, "_establish_connection", lambda self: None)
        monkeypatch.setattr(SSHEnvironment, "_detect_remote_home", lambda self: "/home/u")
        monkeypatch.setattr(SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
        monkeypatch.setattr(SSHEnvironment, "init_session", lambda self: None)
        monkeypatch.setattr(
            ssh_env, "FileSyncManager",
            lambda **kw: type("M", (), {"sync": lambda self, **k: None,
                                         "sync_back": lambda self: None})(),
        )
        env = SSHEnvironment(host="h", user="u", port=2222)

        src = tmp_path / "g.txt"
        src.write_text("g")
        captured = []

        def _capture(cmd, **kw):
            captured.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(ssh_env.subprocess, "run", _capture)
        env._scp_upload(str(src), "/home/u/.hermes/skills/g.txt")

        scp_cmd = next(c for c in captured if c[0] == "scp")
        assert "-P" in scp_cmd and "2222" in scp_cmd


# ---------------------------------------------------------------------------
# _ssh_delete
# ---------------------------------------------------------------------------


class TestSshDelete:
    """Unit tests for batch remote file deletion."""

    def test_success(self, env, monkeypatch):
        """A zero return code should not raise."""
        monkeypatch.setattr(
            ssh_env.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0),
        )
        env._ssh_delete(["/home/u/.hermes/skills/old.py"])

    def test_failure_raises(self, env, monkeypatch):
        """A non-zero return code should raise RuntimeError."""
        monkeypatch.setattr(
            ssh_env.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Permission denied"),
        )
        with pytest.raises(RuntimeError, match="remote rm failed"):
            env._ssh_delete(["/home/u/.hermes/skills/gone.py"])

    def test_rm_command_contains_all_paths(self, env, monkeypatch):
        """All paths should appear in the single SSH call."""
        captured = []

        def _capture(cmd, **kw):
            captured.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(ssh_env.subprocess, "run", _capture)
        paths = [
            "/home/u/.hermes/skills/a.py",
            "/home/u/.hermes/cache/b.txt",
        ]
        env._ssh_delete(paths)

        assert len(captured) == 1
        cmd_str = " ".join(captured[0])
        assert "rm -f" in cmd_str
        for p in paths:
            assert p in cmd_str


# ---------------------------------------------------------------------------
# _ssh_bulk_download
# ---------------------------------------------------------------------------


class TestSshBulkDownload:
    """Unit tests for remote .hermes/ tar download."""

    def test_success_writes_to_dest(self, env, tmp_path, monkeypatch):
        """A zero return code should write the tar to dest without raising."""
        dest = tmp_path / "archive.tar"

        monkeypatch.setattr(
            ssh_env.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0),
        )
        env._ssh_bulk_download(dest)
        # The file is opened in wb mode even if empty — it should exist
        assert dest.exists()

    def test_failure_raises_runtime_error(self, env, tmp_path, monkeypatch):
        """A non-zero return code should raise RuntimeError."""
        dest = tmp_path / "bad.tar"

        def _fail(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 1, stderr=b"disk full")

        monkeypatch.setattr(ssh_env.subprocess, "run", _fail)
        with pytest.raises(RuntimeError, match="SSH bulk download failed"):
            env._ssh_bulk_download(dest)

    def test_ssh_command_targets_remote_hermes_dir(self, env, tmp_path, monkeypatch):
        """The tar command should archive the correct remote .hermes path."""
        dest = tmp_path / "out.tar"
        captured = []

        def _capture(cmd, **kw):
            captured.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(ssh_env.subprocess, "run", _capture)
        env._ssh_bulk_download(dest)

        assert len(captured) == 1
        cmd_str = " ".join(captured[0])
        # Remote base without leading slash (tar -C / rel_base pattern)
        assert "home/u/.hermes" in cmd_str
        assert "tar cf -" in cmd_str


# ---------------------------------------------------------------------------
# _run_bash
# ---------------------------------------------------------------------------


class TestRunBash:
    """Unit tests for _run_bash — verify SSH command shape."""

    def test_non_login_uses_bash_c(self, env, monkeypatch):
        """Non-login _run_bash should use 'bash -c' on the remote."""
        popped = []

        def _popen(cmd, stdin_data=None):
            popped.append(cmd)
            return MagicMock(stdout=iter([]), returncode=0)

        monkeypatch.setattr(ssh_env, "_popen_bash", _popen)
        env._run_bash("echo hi", login=False)

        cmd_str = " ".join(popped[0])
        assert "bash -c" in cmd_str
        assert "bash -l" not in cmd_str

    def test_login_uses_bash_l_c(self, env, monkeypatch):
        """Login _run_bash should use 'bash -l -c' on the remote."""
        popped = []

        def _popen(cmd, stdin_data=None):
            popped.append(cmd)
            return MagicMock(stdout=iter([]), returncode=0)

        monkeypatch.setattr(ssh_env, "_popen_bash", _popen)
        env._run_bash("echo hi", login=True)

        cmd_str = " ".join(popped[0])
        assert "bash -l -c" in cmd_str

    def test_command_is_quoted_in_argv(self, env, monkeypatch):
        """The command string should be shlex-quoted in the SSH argv."""
        import shlex
        popped = []

        def _popen(cmd, stdin_data=None):
            popped.append(cmd)
            return MagicMock(stdout=iter([]), returncode=0)

        monkeypatch.setattr(ssh_env, "_popen_bash", _popen)
        env._run_bash("echo 'hello world'", login=False)

        quoted = shlex.quote("echo 'hello world'")
        assert quoted in popped[0]


# ---------------------------------------------------------------------------
# _before_execute
# ---------------------------------------------------------------------------


class TestBeforeExecute:
    """_before_execute must delegate to the FileSyncManager."""

    def test_calls_sync_manager(self, env):
        sync_calls = []
        env._sync_manager.sync = lambda **kw: sync_calls.append(kw)
        env._before_execute()
        assert len(sync_calls) == 1


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    """Tests for cleanup() — sync_back and socket removal."""

    def test_calls_sync_back(self, env, monkeypatch):
        """cleanup() should call sync_back on the FileSyncManager."""
        synced_back = []
        env._sync_manager.sync_back = lambda: synced_back.append(True)

        # Stub subprocess.run so the SSH exit call doesn't fail
        monkeypatch.setattr(
            ssh_env.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0),
        )
        env.cleanup()
        assert synced_back == [True]

    def test_removes_control_socket_file(self, env, monkeypatch, tmp_path):
        """cleanup() should delete the control socket file if it exists."""
        # Write a real file at the socket path so unlink has something to do
        fake_socket = tmp_path / "fake.sock"
        fake_socket.touch()
        env.control_socket = fake_socket

        env._sync_manager.sync_back = lambda: None
        monkeypatch.setattr(
            ssh_env.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0),
        )
        env.cleanup()
        assert not fake_socket.exists()

    def test_cleanup_tolerates_missing_socket(self, env, monkeypatch, tmp_path):
        """cleanup() should not raise when the control socket does not exist."""
        # Point to a non-existent path
        env.control_socket = tmp_path / "nonexistent.sock"
        env._sync_manager.sync_back = lambda: None
        monkeypatch.setattr(
            ssh_env.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0),
        )
        # Should not raise
        env.cleanup()

    def test_cleanup_tolerates_subprocess_error(self, env, monkeypatch, tmp_path):
        """cleanup() should not raise when the SSH exit command fails."""
        fake_socket = tmp_path / "fake.sock"
        fake_socket.touch()
        env.control_socket = fake_socket

        env._sync_manager.sync_back = lambda: None

        def _raise(cmd, **kw):
            raise subprocess.SubprocessError("connection lost")

        monkeypatch.setattr(ssh_env.subprocess, "run", _raise)
        # Should not raise
        env.cleanup()
