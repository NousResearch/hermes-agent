"""Tests for LocalEnvironment recovery when ``self.cwd`` is deleted.

When a tool call inside the persistent terminal session ``rm -rf``'s its own
working directory, the next ``subprocess.Popen(..., cwd=self.cwd)`` would
otherwise raise ``FileNotFoundError`` before bash starts, wedging every
subsequent terminal/file-tool call until the gateway restarts.

Regression coverage for https://github.com/NousResearch/hermes-agent/issues/17558.
"""

import os
import shutil
import tempfile
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.environments.local import (
    LocalEnvironment,
    _is_usable_cwd,
    _resolve_safe_cwd,
)


class TestResolveSafeCwd:
    """Pure-function unit tests for the recovery helper."""

    def test_returns_cwd_when_directory_exists(self, tmp_path):
        path = str(tmp_path)
        assert _resolve_safe_cwd(path) == path

    def test_walks_up_to_first_existing_ancestor(self, tmp_path):
        nested = tmp_path / "child" / "grandchild"
        nested.mkdir(parents=True)
        deleted = str(nested)
        shutil.rmtree(tmp_path / "child")

        # The deepest existing ancestor on the path is tmp_path itself.
        assert _resolve_safe_cwd(deleted) == str(tmp_path)

    def test_falls_back_when_path_is_empty(self):
        assert _resolve_safe_cwd("") == tempfile.gettempdir()

    def test_returns_tempdir_when_nothing_on_path_exists(self, monkeypatch):
        monkeypatch.setattr(os.path, "isdir", lambda p: False)
        assert _resolve_safe_cwd("/no/such/dir") == tempfile.gettempdir()

    def test_returns_root_when_only_root_exists(self, monkeypatch):
        """If every ancestor except the filesystem root is gone, the root
        itself is still a valid recovery target — don't skip it just because
        ``os.path.dirname('/') == '/'`` is the loop's exit condition."""
        sep = os.path.sep
        monkeypatch.setattr(os.path, "isdir", lambda p: p == sep)
        assert _resolve_safe_cwd("/no/such/deep/dir") == sep

    def test_walks_up_past_inaccessible_dir(self, monkeypatch, tmp_path):
        """A directory that exists but is not searchable (``os.X_OK`` fails)
        must be treated like a missing one — ``os.path.isdir`` alone returns
        True for it, so recovery would otherwise hand ``Popen`` a cwd it
        can't ``chdir`` into and raise ``PermissionError`` (#65583)."""
        locked = tmp_path / "locked"
        locked.mkdir()
        blocked = str(locked)

        # Simulate ``/root``-style access: the dir exists (isdir True) but the
        # runtime user can't enter it (X_OK False).  tmp_path itself stays
        # accessible so it becomes the recovery target.
        monkeypatch.setattr(
            os, "access", lambda p, mode: p != blocked
        )

        assert _resolve_safe_cwd(blocked) == str(tmp_path)


class TestIsUsableCwd:
    """``_is_usable_cwd`` must require BOTH existence and search access."""

    def test_existing_accessible_dir_is_usable(self, tmp_path):
        assert _is_usable_cwd(str(tmp_path)) is True

    def test_empty_is_not_usable(self):
        assert _is_usable_cwd("") is False

    def test_missing_is_not_usable(self, tmp_path):
        assert _is_usable_cwd(str(tmp_path / "nope")) is False

    def test_inaccessible_dir_is_not_usable(self, monkeypatch, tmp_path):
        blocked = str(tmp_path)
        monkeypatch.setattr(os, "access", lambda p, mode: False)
        # isdir is still True, but the missing X_OK makes it unusable.
        assert os.path.isdir(blocked) is True
        assert _is_usable_cwd(blocked) is False


def _fake_interrupt():
    return threading.Event()


def _make_fake_popen(captured: dict, fds: list):
    """Build a fake ``Popen`` whose ``stdout`` exposes a real OS file
    descriptor so ``BaseEnvironment._wait_for_process`` can call
    ``select.select([fd], ...)`` and ``os.read(fd, ...)`` against it without
    tripping ``TypeError: fileno() returned a non-integer`` from a MagicMock
    ``fileno()`` (or worse, accidentally reading from the test runner's own
    stdout).

    The pipe's write end is closed immediately so the drain loop sees EOF on
    the first iteration.  Every fd handed out is appended to ``fds`` so the
    caller can clean up after the test.
    """
    def fake_popen(cmd, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = kwargs.get("env", {})
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        stdout = os.fdopen(read_fd, "rb", buffering=0)
        fds.append(stdout)
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 0
        proc.stdout = stdout
        proc.stdin = MagicMock()
        return proc
    return fake_popen


def _close_fds(fds):
    for f in fds:
        try:
            f.close()
        except Exception:
            pass


class TestRunBashCwdRecovery:
    """End-to-end recovery: deleted ``self.cwd`` must not crash Popen."""

    def test_recovers_when_cwd_deleted_after_init(self, tmp_path, caplog):
        """Reproduces the wedge from #17558: cwd was valid when the
        snapshot was taken, but a subsequent command deleted it before the
        next ``Popen``."""
        wedged = tmp_path / "wedge-repro"
        wedged.mkdir()

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=str(wedged), timeout=10)

        # The previous tool call deleted the working directory.
        shutil.rmtree(wedged)
        assert env.cwd == str(wedged) and not os.path.isdir(env.cwd)

        captured = {}
        fds: list = []
        try:
            with patch("tools.environments.local._find_bash", return_value="/bin/bash"), \
                 patch("subprocess.Popen", side_effect=_make_fake_popen(captured, fds)), \
                 patch("tools.terminal_tool._interrupt_event", _fake_interrupt()), \
                 caplog.at_level("WARNING", logger="tools.environments.local"):
                env.execute("echo hello")
        finally:
            _close_fds(fds)

        # Popen must have been handed a real, existing directory.
        assert captured["cwd"] == str(tmp_path)
        assert os.path.isdir(captured["cwd"])

        # ``self.cwd`` is updated so the next call doesn't re-warn.
        assert env.cwd == str(tmp_path)

        # The warning surfaces the wedge so it isn't silently masked.
        assert any("unusable" in rec.message for rec in caplog.records)

    @pytest.mark.skipif(
        os.name == "nt", reason="POSIX directory-permission semantics only"
    )
    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root bypasses directory search permissions",
    )
    def test_recovers_when_cwd_is_inaccessible(self, tmp_path, caplog):
        """Reproduces #65583: the configured cwd exists but the runtime user
        can't enter it (the ``/root`` case under a non-root systemd/cron
        runtime).  ``os.path.isdir`` returns True, so without the ``os.X_OK``
        guard ``Popen`` would raise ``PermissionError`` and wedge every
        terminal/file-tool call while the job still reports success."""
        locked = tmp_path / "locked"
        locked.mkdir()

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=str(locked), timeout=10)

        # Remove search/execute permission so the process can't chdir into it.
        os.chmod(locked, 0o000)
        try:
            assert os.path.isdir(str(locked)) is True
            assert os.access(str(locked), os.X_OK) is False

            captured = {}
            fds: list = []
            try:
                with patch("tools.environments.local._find_bash", return_value="/bin/bash"), \
                     patch("subprocess.Popen", side_effect=_make_fake_popen(captured, fds)), \
                     patch("tools.terminal_tool._interrupt_event", _fake_interrupt()), \
                     caplog.at_level("WARNING", logger="tools.environments.local"):
                    env.execute("echo hello")
            finally:
                _close_fds(fds)
        finally:
            # Restore so tmp_path cleanup can remove the tree.
            os.chmod(locked, 0o755)

        # Popen must have been handed an accessible directory, never the
        # locked one that would raise PermissionError.
        assert captured["cwd"] != str(locked)
        assert os.path.isdir(captured["cwd"])
        assert os.access(captured["cwd"], os.X_OK)

        assert env.cwd == captured["cwd"]
        assert any("unusable" in rec.message for rec in caplog.records)

    def test_no_warning_when_cwd_still_exists(self, tmp_path, caplog):
        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=str(tmp_path), timeout=10)

        captured = {}
        fds: list = []
        try:
            with patch("tools.environments.local._find_bash", return_value="/bin/bash"), \
                 patch("subprocess.Popen", side_effect=_make_fake_popen(captured, fds)), \
                 patch("tools.terminal_tool._interrupt_event", _fake_interrupt()), \
                 caplog.at_level("WARNING", logger="tools.environments.local"):
                env.execute("echo hello")
        finally:
            _close_fds(fds)

        assert captured["cwd"] == str(tmp_path)
        assert env.cwd == str(tmp_path)
        assert not any("missing on disk" in rec.message for rec in caplog.records)


class TestUpdateCwdRejectsMissingPaths:
    """``_update_cwd`` must not propagate a deleted path back into ``self.cwd``."""

    def test_skips_assignment_when_marker_path_missing(self, tmp_path):
        original = tmp_path / "starting"
        original.mkdir()

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=str(original), timeout=10)

        # Simulate the stale-marker case: the prior command emitted a cwd
        # marker for a directory that has since been deleted.
        deleted = tmp_path / "wedge-repro"
        marker = env._cwd_marker

        env._update_cwd(
            {"output": f"x\n{marker}{deleted}{marker}\n", "returncode": 0}
        )

        assert env.cwd == str(original)

    def test_accepts_assignment_when_marker_path_exists(self, tmp_path):
        original = tmp_path / "starting"
        original.mkdir()
        new_dir = tmp_path / "next"
        new_dir.mkdir()

        with patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None):
            env = LocalEnvironment(cwd=str(original), timeout=10)
        marker = env._cwd_marker

        env._update_cwd(
            {"output": f"x\n{marker}{new_dir}{marker}\n", "returncode": 0}
        )

        assert env.cwd == str(new_dir)
