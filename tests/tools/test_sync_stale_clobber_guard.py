"""Regression tests: stale sync snapshots must never revert newer local files.

2026-09-19 incident — kanban workers' skill writes (and the skill ledger) kept silently
reverting to older byte-for-byte images: the SSH backend's file sync extracted tar snapshots
straight back over the live tree, and a push stalled behind the busy multiplexed channel
delivered bytes it had read minutes earlier. Guards under test (tools/environments/):

1. ssh.SSHEnvironment._sync_target_is_local() disables file sync entirely when the SSH target
   IS the agent's own hermes home — a self-sync can only rewrite live files with older bytes.
2. ssh.SSHEnvironment._keep_newer_extract_flag() adds GNU tar's --keep-newer-files to the bulk
   extract so an older archive member never replaces a newer destination, and every skip is
   surfaced as a WARNING with the path.
3. file_sync.FileSyncManager._apply_staged_file() refuses to overwrite a host file whose mtime
   is newer than the staged copy (the pull-side half of the same class).
4. The push archive carries files only (explicit ``--null -T`` member list) and the extract
   passes --keep-newer-files WITHOUT --no-overwrite-dir — GNU tar rejects that pair (exit 2),
   which would fail every upload. A file-only archive also keeps the dir-metadata protection
   that --no-overwrite-dir existed for.
"""

import hashlib
import logging
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_constants import get_hermes_home
from tools.environments import ssh as ssh_env
from tools.environments.file_sync import FileSyncManager
from tools.environments.ssh import SSHEnvironment


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def quiet_env(monkeypatch):
    """An SSHEnvironment whose construction never touches real ssh."""
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/testuser")
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_sync_target_is_local", lambda self: False)
    monkeypatch.setattr(
        ssh_env, "FileSyncManager",
        lambda **kw: type("M", (), {"sync": lambda self, **k: None})(),
    )
    return SSHEnvironment(host="example.com", user="testuser")


# ── 1. Same-tree probe ───────────────────────────────────────────────────────


class TestSameTreeProbe:
    def _build(self, monkeypatch, run_ssh):
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: str(get_hermes_home().parent))
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_run_ssh", run_ssh)
        made = []
        monkeypatch.setattr(ssh_env, "FileSyncManager",
                            lambda **kw: made.append(kw) or type("M", (), {"sync": lambda self, **k: None})())
        return SSHEnvironment(host="localhost", user="cameron"), made

    def test_probe_reads_back_nonce_and_disables_sync(self, monkeypatch):
        """A remote that can read the fresh nonce shares our tree -> no sync manager at all."""
        def fake_run_ssh(self, remote_cmd, timeout):
            # Emulate a target that reads the same tree: it can cat the file we just wrote.
            probe = next(iter(get_hermes_home().glob(".sync-probe-*")), None)
            return subprocess.CompletedProcess([], 0, stdout=probe.read_text() if probe else "", stderr="")

        env, made = self._build(monkeypatch, fake_run_ssh)

        assert env._sync_manager is None
        assert made == []  # FileSyncManager never constructed
        env.cleanup()  # guarded path: no manager, no sync_back
        assert list(get_hermes_home().glob(".sync-probe-*")) == []  # probe file cleaned up

    def test_probe_mismatch_keeps_sync_enabled(self, monkeypatch):
        def fake_run_ssh(self, remote_cmd, timeout):
            return subprocess.CompletedProcess([], 1, stdout="", stderr="No such file")

        env, made = self._build(monkeypatch, fake_run_ssh)

        assert env._sync_manager is not None
        assert len(made) == 1

    def test_probe_failure_keeps_sync_enabled(self, monkeypatch):
        """Any probe error must fall back to the previous behaviour, never to a silent disable."""
        def boom(self, remote_cmd, timeout):
            raise OSError("no route to host")

        env, made = self._build(monkeypatch, boom)

        assert env._sync_manager is not None
        assert len(made) == 1
        assert list(get_hermes_home().glob(".sync-probe-*")) == []


# ── 2. Push-side guard (--keep-newer-files) ──────────────────────────────────


class TestKeepNewerExtractFlag:
    def test_gnu_tar_gets_the_flag(self, quiet_env, monkeypatch):
        monkeypatch.setattr(quiet_env, "_run_ssh", lambda cmd, timeout: subprocess.CompletedProcess(
            [], 0, stdout="tar (GNU tar) 1.35\n", stderr=""))
        assert quiet_env._keep_newer_extract_flag() == " --keep-newer-files"

    def test_non_gnu_tar_does_not(self, quiet_env, monkeypatch):
        monkeypatch.setattr(quiet_env, "_run_ssh", lambda cmd, timeout: subprocess.CompletedProcess(
            [], 0, stdout="bsdtar 3.5.1 - libarchive 3.5.1\n", stderr=""))
        assert quiet_env._keep_newer_extract_flag() == ""

    def test_probe_error_means_no_flag(self, quiet_env, monkeypatch):
        def boom(cmd, timeout):
            raise OSError("ssh gone")

        monkeypatch.setattr(quiet_env, "_run_ssh", boom)
        assert quiet_env._keep_newer_extract_flag() == ""

    def test_skips_surface_as_warning_with_paths(self, caplog):
        stderr = (
            "tar: Current \u2018./skills/note-taking/obsidian-sync-management/SKILL.md\u2019 is newer or same age\n"
            'tar: Current "./skills/research/rss-feeds/SKILL.md" is newer or same age\n'
        ).encode()
        with caplog.at_level(logging.WARNING, logger="tools.environments.ssh"):
            SSHEnvironment._warn_extract_skipped_newer(stderr)
        assert "stale-push protection" in caplog.text
        assert "obsidian-sync-management/SKILL.md" in caplog.text
        assert "rss-feeds/SKILL.md" in caplog.text

    def test_unrelated_stderr_is_quiet(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.environments.ssh"):
            SSHEnvironment._warn_extract_skipped_newer(b"tar: removing leading `/'\n")
        assert caplog.text == ""


# ── 2b. Extract command composition ──────────────────────────────────────────


class TestExtractCommandComposition:
    """The extract command must never combine --keep-newer-files with --no-overwrite-dir.

    GNU tar treats both as members of one mutually-exclusive overwrite-policy group: passing
    the pair makes the remote extract exit 2 ("`--no-overwrite-dir' cannot be used with
    `--keep-newer-files'") and the whole upload fails. The GNU path therefore passes
    keep-newer alone; the archive is file-only so no directory metadata can be stamped.
    """

    def _capture(self, monkeypatch, gnu: bool):
        monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/testuser")
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_ensure_remote_dirs", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)
        monkeypatch.setattr(ssh_env.SSHEnvironment, "_sync_target_is_local", lambda self: False)
        monkeypatch.setattr(
            ssh_env, "FileSyncManager",
            lambda **kw: type("M", (), {"sync": lambda self, **k: None})(),
        )
        env = SSHEnvironment(host="example.com", user="testuser")
        env._remote_gnu_tar = gnu  # skip the live ``tar --version`` probe
        captured: dict = {}

        def fake_popen(cmd, **kwargs):
            m = MagicMock()
            m.stdout = MagicMock()
            m.returncode = 0
            m.poll.return_value = 0
            m.communicate.return_value = (b"", b"")
            m.stderr = MagicMock()
            m.stderr.read.return_value = b""
            if cmd[0] == "tar":
                captured["tar_cmd"] = list(cmd)
                listing = cmd[cmd.index("-T") + 1]
                with open(listing, "rb") as fh:
                    captured["listing"] = fh.read()
            else:
                captured["ssh_cmd"] = list(cmd)
            return m

        monkeypatch.setattr(ssh_env.subprocess, "run",
                            lambda *a, **k: subprocess.CompletedProcess([], 0))
        monkeypatch.setattr(ssh_env.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(env, "_build_ssh_command", lambda: ["ssh", "target"])
        return env, captured

    def _run(self, monkeypatch, gnu):
        env, captured = self._capture(monkeypatch, gnu)
        host = "/tmp/regression-src.md"
        with open(host, "w", encoding="utf-8") as fh:
            fh.write("payload")
        env._ssh_bulk_upload([(host, "/home/testuser/.hermes/skills/example/SKILL.md")])
        return captured

    def test_gnu_extract_uses_keep_newer_alone(self, monkeypatch):
        captured = self._run(monkeypatch, gnu=True)
        extract = captured["ssh_cmd"][-1]
        assert extract.startswith("tar xf -")
        assert "--keep-newer-files" in extract
        assert "--no-overwrite-dir" not in extract
        assert extract.endswith("-C /home/testuser/.hermes")

    def test_non_gnu_extract_keeps_no_overwrite_dir(self, monkeypatch):
        captured = self._run(monkeypatch, gnu=False)
        extract = captured["ssh_cmd"][-1]
        assert "--no-overwrite-dir" in extract
        assert "--keep-newer-files" not in extract

    def test_archive_is_file_only(self, monkeypatch):
        """No directory members in the stream: tar reads an explicit NUL-separated name list."""
        captured = self._run(monkeypatch, gnu=True)
        tar_cmd = captured["tar_cmd"]
        assert "--null" in tar_cmd and "-T" in tar_cmd
        assert tar_cmd[-1] != "."  # the old recursive ``tar ... .`` form is gone
        names = captured["listing"].split(b"\0")
        assert b"./skills/example/SKILL.md" in names

    def test_empty_upload_is_still_a_noop(self, monkeypatch):
        env, captured = self._capture(monkeypatch, gnu=True)
        env._ssh_bulk_upload([])
        assert captured == {}


# ── 3. Pull-side guard (sync_back) ───────────────────────────────────────────


class TestSyncBackNewerHostGuard:
    REMOTE = "/root/.hermes/skills/example/SKILL.md"

    def _manager(self):
        return FileSyncManager(get_files_fn=lambda: [], upload_fn=lambda *a: None,
                               delete_fn=lambda *a: None)

    def test_refuses_to_overwrite_newer_host_file(self, tmp_path, caplog):
        staged, host = tmp_path / "staged.md", tmp_path / "host.md"
        staged.write_bytes(b"old remote bytes")
        os.utime(staged, (1_000, 1_000))
        host.write_bytes(b"newer local bytes written after the snapshot")
        os.utime(host, (2_000, 2_000))

        with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
            applied = self._manager()._apply_staged_file(
                str(staged), self.REMOTE, [(str(host), self.REMOTE)], set())

        assert applied == 0
        assert host.read_bytes() == b"newer local bytes written after the snapshot"
        assert "refusing to overwrite newer local file" in caplog.text
        assert _sha(b"newer local bytes written after the snapshot")[:12] in caplog.text
        assert _sha(b"old remote bytes")[:12] in caplog.text

    def test_unknown_staged_mtime_falls_back_to_previous_behaviour(self, tmp_path, caplog):
        """A zero/unreadable staged mtime cannot prove staleness — historical apply wins."""
        staged, host = tmp_path / "staged.md", tmp_path / "host.md"
        staged.write_bytes(b"remote bytes")
        os.utime(staged, (0, 0))
        host.write_bytes(b"local bytes")
        os.utime(host, (2_000, 2_000))

        with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
            applied = self._manager()._apply_staged_file(
                str(staged), self.REMOTE, [(str(host), self.REMOTE)], set())

        assert applied == 1
        assert host.read_bytes() == b"remote bytes"
        assert "refusing" not in caplog.text

    def test_still_applies_when_staged_copy_is_newer(self, tmp_path):
        staged, host = tmp_path / "staged.md", tmp_path / "host.md"
        host.write_bytes(b"older local bytes")
        os.utime(host, (1_000, 1_000))
        staged.write_bytes(b"newer remote bytes")
        os.utime(staged, (2_000, 2_000))

        applied = self._manager()._apply_staged_file(
            str(staged), self.REMOTE, [(str(host), self.REMOTE)], set())

        assert applied == 1
        assert host.read_bytes() == b"newer remote bytes"

    def test_identical_content_is_not_flagged_as_clobber(self, tmp_path, caplog):
        payload = b"same bytes"
        staged, host = tmp_path / "staged.md", tmp_path / "host.md"
        staged.write_bytes(payload)
        os.utime(staged, (1_000, 1_000))
        host.write_bytes(payload)
        os.utime(host, (2_000, 2_000))

        with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
            self._manager()._apply_staged_file(
                str(staged), self.REMOTE, [(str(host), self.REMOTE)], set())

        # Identical bytes are not a conflict (unchanged pre-existing behaviour); the guard only
        # fires when the contents actually differ.
        assert "refusing" not in caplog.text
        assert host.read_bytes() == payload
