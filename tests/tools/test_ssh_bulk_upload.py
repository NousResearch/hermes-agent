"""Tests for SSH bulk upload via tar pipe."""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from tools.environments import ssh as ssh_env
from tools.environments.file_sync import quoted_mkdir_command, unique_parent_dirs
from tools.environments.ssh import SSHEnvironment


def _mock_proc(*, returncode=0, poll_return=0, communicate_return=(b"", b""),
               stderr_read=b""):
    """Create a MagicMock mimicking subprocess.Popen for tar/ssh pipes."""
    m = MagicMock()
    m.stdout = MagicMock()
    m.returncode = returncode
    m.poll.return_value = poll_return
    m.communicate.return_value = communicate_return
    m.stderr = MagicMock()
    m.stderr.read.return_value = stderr_read
    return m


@pytest.fixture
def mock_env(monkeypatch):
    """Create an SSHEnvironment with mocked connection/sync."""
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/testuser")
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(
        ssh_env, "FileSyncManager",
        lambda **kw: type("M", (), {"sync": lambda self, **k: None})(),
    )
    return SSHEnvironment(host="example.com", user="testuser")


class TestSSHBulkUpload:
    """Unit tests for _ssh_bulk_upload — tar pipe mechanics."""

    def test_empty_files_is_noop(self, mock_env):
        """Empty file list should not spawn any subprocesses."""
        with patch.object(subprocess, "run") as mock_run, \
             patch.object(subprocess, "Popen") as mock_popen:
            mock_env._ssh_bulk_upload([])
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_mkdir_batched_into_single_call(self, mock_env, tmp_path):
        """All parent directories should be created in one SSH call."""
        # Create test files
        f1 = tmp_path / "a.txt"
        f1.write_text("aaa")
        f2 = tmp_path / "b.txt"
        f2.write_text("bbb")

        files = [
            (str(f1), "/home/testuser/.hermes/skills/a.txt"),
            (str(f2), "/home/testuser/.hermes/credentials/b.txt"),
        ]

        # Mock subprocess.run for mkdir and Popen for tar pipe
        mock_run = MagicMock(return_value=subprocess.CompletedProcess([], 0))

        def make_proc(cmd, **kwargs):
            m = MagicMock()
            m.stdout = MagicMock()
            m.returncode = 0
            m.poll.return_value = 0
            m.communicate.return_value = (b"", b"")
            m.stderr = MagicMock()
            m.stderr.read.return_value = b""
            return m

        with patch.object(subprocess, "run", mock_run), \
             patch.object(subprocess, "Popen", side_effect=make_proc):
            mock_env._ssh_bulk_upload(files)

        # Exactly one subprocess.run call for mkdir
        assert mock_run.call_count == 1
        mkdir_cmd = mock_run.call_args[0][0]
        # Should contain mkdir -p with both parent dirs
        mkdir_str = " ".join(mkdir_cmd)
        assert "mkdir -p" in mkdir_str
        assert "/home/testuser/.hermes/skills" in mkdir_str
        assert "/home/testuser/.hermes/credentials" in mkdir_str

    def test_staging_symlinks_mirror_remote_layout(self, mock_env, tmp_path):
        """Staged file in staging dir should mirror the remote path structure.

        On platforms where symlinks are available (Linux/macOS) the staged
        entry is a symlink; on Windows it may be a regular copy.  Either way
        the file must exist at the expected path and contain the right data.
        """
        f1 = tmp_path / "local_a.txt"
        f1.write_text("content a")

        files = [
            (str(f1), "/home/testuser/.hermes/skills/my_skill.md"),
        ]

        staging_paths = []

        def capture_tar_cmd(cmd, **kwargs):
            if cmd[0] == "tar":
                # Capture the staging dir from -C argument
                c_idx = cmd.index("-C")
                staging_dir = cmd[c_idx + 1]
                # Check the staged entry exists at the base-relative path
                expected = os.path.join(staging_dir, "skills/my_skill.md")
                staging_paths.append(expected)
                # File must exist (either as symlink or copy)
                assert os.path.exists(expected), f"Expected staged file at {expected}"
                # Content must match the source
                with open(expected, "r") as fh:
                    assert fh.read() == "content a"

            mock = MagicMock()
            mock.stdout = MagicMock()
            mock.returncode = 0
            mock.poll.return_value = 0
            mock.communicate.return_value = (b"", b"")
            mock.stderr = MagicMock()
            mock.stderr.read.return_value = b""
            return mock

        with patch.object(subprocess, "run",
                          return_value=subprocess.CompletedProcess([], 0)), \
             patch.object(subprocess, "Popen", side_effect=capture_tar_cmd):
            mock_env._ssh_bulk_upload(files)

        assert len(staging_paths) == 1, "tar command should have been called"


    def test_bulk_upload_never_stages_remote_home_prefix(self, mock_env, tmp_path):
        """Regression: do not archive /home/<user> path components."""
        f1 = tmp_path / "nested.txt"
        f1.write_text("nested")
        files = [(str(f1), "/home/testuser/.hermes/cache/nested.txt")]

        def capture_tar_cmd(cmd, **kwargs):
            if cmd[0] == "tar":
                c_idx = cmd.index("-C")
                staging_dir = cmd[c_idx + 1]
                assert not os.path.exists(os.path.join(staging_dir, "home"))
                expected = os.path.join(staging_dir, "cache/nested.txt")
                assert os.path.islink(expected)

            mock = MagicMock()
            mock.stdout = MagicMock()
            mock.returncode = 0
            mock.poll.return_value = 0
            mock.communicate.return_value = (b"", b"")
            mock.stderr = MagicMock()
            mock.stderr.read.return_value = b""
            return mock

        with patch.object(subprocess, "run",
                          return_value=subprocess.CompletedProcess([], 0)), \
             patch.object(subprocess, "Popen", side_effect=capture_tar_cmd):
            mock_env._ssh_bulk_upload(files)


    def test_timeout_kills_both_processes(self, mock_env, tmp_path):
        """TimeoutExpired during communicate should kill both processes."""
        f1 = tmp_path / "t.txt"
        f1.write_text("t")
        files = [(str(f1), "/home/testuser/.hermes/skills/t.txt")]

        mock_tar = MagicMock()
        mock_tar.stdout = MagicMock()
        mock_tar.returncode = None
        mock_tar.poll.return_value = None

        mock_ssh = MagicMock()
        mock_ssh.communicate.side_effect = subprocess.TimeoutExpired("ssh", 120)
        mock_ssh.returncode = None

        def make_proc(cmd, **kwargs):
            if cmd[0] == "tar":
                return mock_tar
            return mock_ssh

        with patch.object(subprocess, "run",
                          return_value=subprocess.CompletedProcess([], 0)), \
             patch.object(subprocess, "Popen", side_effect=make_proc):
            with pytest.raises(RuntimeError, match="SSH bulk upload timed out"):
                mock_env._ssh_bulk_upload(files)

        mock_tar.kill.assert_called_once()
        mock_ssh.kill.assert_called_once()


class TestSSHBulkUploadWiring:
    """Verify bulk_upload_fn is wired into FileSyncManager."""

    def test_filesyncmanager_receives_bulk_upload_fn(self, monkeypatch):
        """SSHEnvironment should pass _ssh_bulk_upload to FileSyncManager."""
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/root")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)

        captured_kwargs = {}

        class FakeSyncManager:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

            def sync(self, **kw):
                pass

        monkeypatch.setattr(ssh_env, "FileSyncManager", FakeSyncManager)

        env = SSHEnvironment(host="h", user="u")

        assert "bulk_upload_fn" in captured_kwargs
        assert captured_kwargs["bulk_upload_fn"] is not None
        # Should be the bound method
        assert callable(captured_kwargs["bulk_upload_fn"])


class TestSharedHelpers:
    """Direct unit tests for file_sync.py helpers."""

    def test_quoted_mkdir_command_basic(self):
        result = quoted_mkdir_command(["/a", "/b/c"])
        assert result == "mkdir -p /a /b/c"


    def test_unique_parent_dirs_empty(self):
        assert unique_parent_dirs([]) == []


class TestSSHBulkUploadEdgeCases:
    """Edge cases for _ssh_bulk_upload."""

    def test_ssh_popen_failure_kills_tar(self, mock_env, tmp_path):
        """If SSH Popen raises, tar process must be killed and cleaned up."""
        f1 = tmp_path / "e.txt"
        f1.write_text("e")
        files = [(str(f1), "/home/testuser/.hermes/skills/e.txt")]

        mock_tar = _mock_proc()

        call_count = 0

        def failing_ssh_popen(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_tar  # tar Popen succeeds
            raise OSError("SSH binary not found")

        with patch.object(subprocess, "run",
                          return_value=subprocess.CompletedProcess([], 0)), \
             patch.object(subprocess, "Popen", side_effect=failing_ssh_popen):
            with pytest.raises(OSError, match="SSH binary not found"):
                mock_env._ssh_bulk_upload(files)

        mock_tar.kill.assert_called_once()
        mock_tar.wait.assert_called_once()
