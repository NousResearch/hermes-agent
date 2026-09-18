"""Regression tests for BSD/macOS remotes in the SSH file-sync path.

Each test here pins a bug that made `terminal.backend: ssh` unusable against a
macOS host (the common case for a Mac-to-Mac fleet):

* the extractor was sent GNU-only ``--no-overwrite-dir``, which BSD tar
  rejects -- killing the local tar with SIGPIPE and reporting the failure as
  if the *local* side were at fault;
* ``_ssh_bulk_download`` tarred the whole remote ``~/.hermes`` (gigabytes,
  mostly the remote's own install) even though sync-back only ever applies
  files it previously pushed.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from tools.environments import ssh as ssh_env
from tools.environments.ssh import SSHEnvironment


@pytest.fixture
def mock_env(monkeypatch):
    """An SSHEnvironment with connection/sync stubbed out."""
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


class TestUploadTarFlagNegotiation:
    """The extract command must not hard-code the GNU-only flag."""

    def _capture_extract_cmd(self, mock_env, tmp_path):
        """Run an upload and return the remote extract command."""
        f = tmp_path / "a.txt"
        f.write_text("x")
        seen = {}

        def fake_popen(cmd, **kwargs):
            # The tar side is a list starting with "tar"; the ssh side carries
            # the remote command as its final argument.
            if cmd[0] != "tar":
                seen["cmd"] = cmd[-1]
            proc = MagicMock()
            proc.returncode = 0
            proc.stdout = MagicMock()
            proc.stderr = MagicMock()
            proc.stderr.read.return_value = b""
            proc.communicate.return_value = (b"", b"")
            proc.poll.return_value = 0
            return proc

        with patch.object(subprocess, "run", return_value=MagicMock(returncode=0, stderr="")), \
             patch.object(subprocess, "Popen", side_effect=fake_popen):
            mock_env._ssh_bulk_upload([(str(f), "/home/testuser/.hermes/a.txt")])
        return seen["cmd"]

    def test_offers_both_gnu_and_bsd_forms(self, mock_env, tmp_path):
        """GNU gets --no-overwrite-dir; BSD gets a plain extract."""
        cmd = self._capture_extract_cmd(mock_env, tmp_path)
        assert "--no-overwrite-dir" in cmd, "GNU remotes still need the flag"
        assert "else exec tar xf -" in cmd, "BSD remotes need a fallback"

    def test_flag_is_guarded_not_unconditional(self, mock_env, tmp_path):
        """The pre-fix bug: the flag was passed to every remote unconditionally."""
        cmd = self._capture_extract_cmd(mock_env, tmp_path)
        head = cmd.split("--no-overwrite-dir", 1)[0]
        assert "if tar" in head, "the flag must sit behind a capability test"

    def test_negotiation_is_a_single_round_trip(self, mock_env, tmp_path):
        """No extra probe connection -- the remote shell decides in-band."""
        f = tmp_path / "a.txt"
        f.write_text("x")
        with patch.object(subprocess, "run", return_value=MagicMock(returncode=0, stderr="")) as run, \
             patch.object(subprocess, "Popen") as popen:
            proc = popen.return_value
            proc.returncode = 0
            proc.stderr.read.return_value = b""
            proc.communicate.return_value = (b"", b"")
            proc.poll.return_value = 0
            mock_env._ssh_bulk_upload([(str(f), "/home/testuser/.hermes/a.txt")])
            # Exactly one subprocess.run: the batched mkdir.
            assert run.call_count == 1


class TestBulkDownloadScope:
    """_ssh_bulk_download must not tar the entire remote home."""

    def _capture_remote_cmd(self, mock_env, tmp_path, paths):
        """Run a download and return the remote command string."""
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        with patch.object(subprocess, "run", side_effect=fake_run):
            mock_env._ssh_bulk_download(tmp_path / "out.tar", paths)
        return " ".join(seen["cmd"])

    def test_targeted_paths_use_stdin_list(self, mock_env, tmp_path):
        """Given explicit paths, tar reads them from stdin (-T -), not the whole tree."""
        cmd = self._capture_remote_cmd(
            mock_env, tmp_path, ["/home/testuser/.hermes/skills/a/SKILL.md"]
        )
        assert "-T -" in cmd
        # The remote's own multi-GB install must never be walked.
        assert "hermes-agent" not in cmd

    def test_no_paths_falls_back_to_full_tree(self, mock_env, tmp_path):
        """Without a path list the old whole-directory behaviour is preserved."""
        cmd = self._capture_remote_cmd(mock_env, tmp_path, None)
        assert "-T -" not in cmd
