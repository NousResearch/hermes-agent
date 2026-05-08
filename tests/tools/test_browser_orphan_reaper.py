"""Tests for _reap_orphaned_browser_sessions() — kills orphaned agent-browser
daemons whose Python parent exited without cleaning up."""

import os
import signal
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def fake_tmpdir(tmp_path):
    """Patch _socket_safe_tmpdir to return a temp dir we control."""
    with patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)):
        yield tmp_path


@pytest.fixture(autouse=True)
def _isolate_sessions():
    """Ensure _active_sessions is empty for each test."""
    import tools.browser_tool as bt
    orig = bt._active_sessions.copy()
    bt._active_sessions.clear()
    yield
    bt._active_sessions.clear()
    bt._active_sessions.update(orig)


def _make_socket_dir(tmpdir, session_name, pid=None, owner_pid=None):
    """Create a fake agent-browser socket directory with optional PID files.

    Args:
        tmpdir: base temp directory
        session_name: name like "h_abc1234567" or "cdp_abc1234567"
        pid: daemon PID to write to <session>.pid (None = no file)
        owner_pid: owning hermes PID to write to <session>.owner_pid
                   (None = no file; tests the legacy path)
    """
    d = tmpdir / f"agent-browser-{session_name}"
    d.mkdir()
    if pid is not None:
        (d / f"{session_name}.pid").write_text(str(pid))
    if owner_pid is not None:
        (d / f"{session_name}.owner_pid").write_text(str(owner_pid))
    return d


class TestReapOrphanedBrowserSessions:
    """Tests for the orphan reaper function."""

    def test_no_socket_dirs_is_noop(self, fake_tmpdir):
        """No socket dirs => nothing happens, no errors."""
        from tools.browser_tool import _reap_orphaned_browser_sessions
        _reap_orphaned_browser_sessions()  # should not raise

    def test_stale_dir_without_pid_file_is_removed(self, fake_tmpdir):
        """Socket dir with no PID file is cleaned up."""
        from tools.browser_tool import _reap_orphaned_browser_sessions
        d = _make_socket_dir(fake_tmpdir, "h_abc1234567")
        assert d.exists()
        _reap_orphaned_browser_sessions()
        assert not d.exists()

    def test_stale_dir_with_dead_pid_is_removed(self, fake_tmpdir):
        """Socket dir whose daemon PID is dead gets cleaned up."""
        from tools.browser_tool import _reap_orphaned_browser_sessions
        d = _make_socket_dir(fake_tmpdir, "h_dead123456", pid=999999999)
        assert d.exists()
        _reap_orphaned_browser_sessions()
        assert not d.exists()

    def test_orphaned_alive_daemon_is_killed(self, fake_tmpdir):
        """Alive daemon not tracked by _active_sessions gets SIGTERM (legacy path).

        No owner_pid file => falls back to tracked_names check.
        """
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "h_orphan12345", pid=12345)

        kill_calls = []
        original_kill = os.kill

        def mock_kill(pid, sig):
            kill_calls.append((pid, sig))
            if sig == 0:
                return  # pretend process exists
            # Don't actually kill anything

        with patch("os.kill", side_effect=mock_kill):
            _reap_orphaned_browser_sessions()

        # Should have checked existence (sig 0) then killed (SIGTERM)
        assert (12345, 0) in kill_calls
        assert (12345, signal.SIGTERM) in kill_calls

    def test_tracked_session_is_not_reaped(self, fake_tmpdir):
        """Sessions tracked in _active_sessions are left alone (legacy path)."""
        import tools.browser_tool as bt
        from tools.browser_tool import _reap_orphaned_browser_sessions

        session_name = "h_tracked1234"
        d = _make_socket_dir(fake_tmpdir, session_name, pid=12345)

        # Register the session as actively tracked
        bt._active_sessions["some_task"] = {"session_name": session_name}

        kill_calls = []

        def mock_kill(pid, sig):
            kill_calls.append((pid, sig))

        with patch("os.kill", side_effect=mock_kill):
            _reap_orphaned_browser_sessions()

        # Should NOT have tried to kill anything
        assert len(kill_calls) == 0
        # Dir should still exist
        assert d.exists()

    def test_permission_error_on_kill_check_skips(self, fake_tmpdir):
        """If we can't check the PID (PermissionError), skip it."""
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "h_perm1234567", pid=12345)

        def mock_kill(pid, sig):
            if sig == 0:
                raise PermissionError("not our process")

        with patch("os.kill", side_effect=mock_kill):
            _reap_orphaned_browser_sessions()

        # Dir should still exist (we didn't touch someone else's process)
        assert d.exists()

    def test_cdp_sessions_are_also_reaped(self, fake_tmpdir):
        """CDP sessions (cdp_ prefix) are also scanned."""
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "cdp_abc1234567")
        assert d.exists()
        _reap_orphaned_browser_sessions()
        # No PID file → cleaned up
        assert not d.exists()

    def test_non_hermes_dirs_are_ignored(self, fake_tmpdir):
        """Socket dirs that don't match our naming pattern are left alone."""
        from tools.browser_tool import _reap_orphaned_browser_sessions

        # Create a dir that doesn't match h_* or cdp_* pattern
        d = fake_tmpdir / "agent-browser-other_session"
        d.mkdir()
        (d / "other_session.pid").write_text("12345")

        _reap_orphaned_browser_sessions()

        # Should NOT be touched
        assert d.exists()

    def test_corrupt_pid_file_is_cleaned(self, fake_tmpdir):
        """PID file with non-integer content is cleaned up."""
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "h_corrupt1234")
        (d / "h_corrupt1234.pid").write_text("not-a-number")

        _reap_orphaned_browser_sessions()
        assert not d.exists()


class TestOwnerPidCrossProcess:
    """Tests for owner_pid-based cross-process safe reaping.

    The owner_pid file records which hermes process owns a daemon so that
    concurrent hermes processes don't reap each other's active browser
    sessions.  Added to fix orphan accumulation from crashed processes.
    """

    def test_alive_owner_is_not_reaped_even_when_untracked(self, fake_tmpdir):
        """Daemon with alive owner_pid is NOT reaped, even if not in our _active_sessions.

        This is the core cross-process safety check: Process B scanning while
        Process A is using a browser must not kill A's daemon.
        """
        from tools.browser_tool import _reap_orphaned_browser_sessions

        # Use our own PID as the "owner" — guaranteed alive
        d = _make_socket_dir(
            fake_tmpdir, "h_alive_owner", pid=12345, owner_pid=os.getpid()
        )

        kill_calls = []

        def mock_kill(pid, sig):
            kill_calls.append((pid, sig))
            if pid == os.getpid() and sig == 0:
                return  # real existence check: owner alive
            if sig == 0:
                return  # pretend daemon exists too
            # Don't actually kill anything

        with patch("os.kill", side_effect=mock_kill):
            _reap_orphaned_browser_sessions()

        # We should have checked the owner (sig 0) but never tried to kill
        # the daemon.
        assert (12345, signal.SIGTERM) not in kill_calls
        # Dir should still exist
        assert d.exists()

    def test_dead_owner_triggers_reap(self, fake_tmpdir):
        """Daemon whose owner_pid is dead gets reaped."""
        from tools.browser_tool import _reap_orphaned_browser_sessions

        # PID 999999999 almost certainly doesn't exist
        d = _make_socket_dir(
            fake_tmpdir, "h_dead_owner1", pid=12345, owner_pid=999999999
        )

        kill_calls = []

        def mock_kill(pid, sig):
            kill_calls.append((pid, sig))
            if pid == 999999999 and sig == 0:
                raise ProcessLookupError  # owner dead
            if pid == 12345 and sig == 0:
                return  # daemon still alive
            # SIGTERM to daemon — noop in test

        with patch("os.kill", side_effect=mock_kill):
            _reap_orphaned_browser_sessions()

        # Owner checked (returned dead), daemon checked (alive), daemon killed
        assert (999999999, 0) in kill_calls
        assert (12345, 0) in kill_calls
        assert (12345, signal.SIGTERM) in kill_calls
        # Dir cleaned up
        assert not d.exists()

    def test_corrupt_owner_pid_falls_back_to_legacy(self, fake_tmpdir):
        """Corrupt owner_pid file → fall back to tracked_names check."""
        import tools.browser_tool as bt
        from tools.browser_tool import _reap_orphaned_browser_sessions

        session_name = "h_corrupt_own"
        d = _make_socket_dir(fake_tmpdir, session_name, pid=12345)
        # Write garbage to owner_pid file
        (d / f"{session_name}.owner_pid").write_text("not-a-pid")

        # Register session so legacy fallback leaves it alone
        bt._active_sessions["task"] = {"session_name": session_name}

        kill_calls = []

        def mock_kill(pid, sig):
            kill_calls.append((pid, sig))

        with patch("os.kill", side_effect=mock_kill):
            _reap_orphaned_browser_sessions()

        # Legacy path took over → tracked → not reaped
        assert (12345, signal.SIGTERM) not in kill_calls
        assert d.exists()

    def test_owner_pid_permission_error_treated_as_alive(self, fake_tmpdir):
        """If os.kill(owner, 0) raises PermissionError, treat owner as alive.

        PermissionError means the PID exists but is owned by a different user —
        we must not assume the owner is dead (could kill someone else's daemon).
        """
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(
            fake_tmpdir, "h_perm_owner1", pid=12345, owner_pid=22222
        )

        kill_calls = []

        def mock_kill(pid, sig):
            kill_calls.append((pid, sig))
            if pid == 22222 and sig == 0:
                raise PermissionError("not our user")

        with patch("os.kill", side_effect=mock_kill):
            _reap_orphaned_browser_sessions()

        # Must NOT have tried to kill the daemon
        assert (12345, signal.SIGTERM) not in kill_calls
        assert d.exists()

    def test_write_owner_pid_creates_file_with_current_pid(
        self, fake_tmpdir, monkeypatch
    ):
        """_write_owner_pid(dir, session) writes <session>.owner_pid with os.getpid()."""
        import tools.browser_tool as bt

        session_name = "h_ownertest01"
        socket_dir = fake_tmpdir / f"agent-browser-{session_name}"
        socket_dir.mkdir()

        bt._write_owner_pid(str(socket_dir), session_name)

        owner_pid_file = socket_dir / f"{session_name}.owner_pid"
        assert owner_pid_file.exists()
        assert owner_pid_file.read_text().strip() == str(os.getpid())

    def test_write_owner_pid_is_idempotent(self, fake_tmpdir):
        """Calling _write_owner_pid twice leaves a single owner_pid file."""
        import tools.browser_tool as bt

        session_name = "h_idempot1234"
        socket_dir = fake_tmpdir / f"agent-browser-{session_name}"
        socket_dir.mkdir()

        bt._write_owner_pid(str(socket_dir), session_name)
        bt._write_owner_pid(str(socket_dir), session_name)

        files = list(socket_dir.glob("*.owner_pid"))
        assert len(files) == 1
        assert files[0].read_text().strip() == str(os.getpid())

    def test_write_owner_pid_swallows_oserror(self, fake_tmpdir, monkeypatch):
        """OSError (e.g. permission denied) doesn't propagate — the reaper
        falls back to the legacy tracked_names heuristic in that case.
        """
        import tools.browser_tool as bt

        def raise_oserror(*a, **kw):
            raise OSError("permission denied")

        monkeypatch.setattr("builtins.open", raise_oserror)

        # Must not raise
        bt._write_owner_pid(str(fake_tmpdir), "h_readonly123")

    def test_run_browser_command_calls_write_owner_pid(
        self, fake_tmpdir, monkeypatch
    ):
        """_run_browser_command wires _write_owner_pid after mkdir."""
        import tools.browser_tool as bt

        session_name = "h_wiringtest1"

        # Short-circuit Popen so we exit after the owner_pid write
        class _FakePopen:
            def __init__(self, *a, **kw):
                raise RuntimeError("short-circuit after owner_pid")

        monkeypatch.setattr(bt.subprocess, "Popen", _FakePopen)
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "/bin/true")
        monkeypatch.setattr(
            bt, "_requires_real_termux_browser_install", lambda *a: False
        )
        monkeypatch.setattr(bt, "_chromium_installed", lambda: True)
        monkeypatch.setattr(
            bt, "_get_session_info",
            lambda task_id: {"session_name": session_name},
        )

        calls = []
        orig_write = bt._write_owner_pid

        def _spy(*a, **kw):
            calls.append(a)
            orig_write(*a, **kw)

        monkeypatch.setattr(bt, "_write_owner_pid", _spy)

        with patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(fake_tmpdir)):
            try:
                bt._run_browser_command(task_id="test_task", command="goto", args=[])
            except Exception:
                pass

        assert calls, "_run_browser_command must call _write_owner_pid"
        # First positional arg is the socket_dir, second is the session_name
        socket_dir_arg, session_name_arg = calls[0][0], calls[0][1]
        assert session_name_arg == session_name
        assert session_name in socket_dir_arg


class TestEmergencyCleanupRunsReaper:
    """Verify atexit-registered cleanup sweeps orphans even without an active session."""

    def test_emergency_cleanup_calls_reaper(self, fake_tmpdir, monkeypatch):
        """_emergency_cleanup_all_sessions must call _reap_orphaned_browser_sessions."""
        import tools.browser_tool as bt

        # Reset the _cleanup_done flag so the cleanup actually runs
        monkeypatch.setattr(bt, "_cleanup_done", False)

        reaper_called = []
        orig_reaper = bt._reap_orphaned_browser_sessions

        def _spy_reaper():
            reaper_called.append(True)
            orig_reaper()

        monkeypatch.setattr(bt, "_reap_orphaned_browser_sessions", _spy_reaper)

        # No active sessions — reaper should still run
        bt._emergency_cleanup_all_sessions()

        assert reaper_called, (
            "Reaper must run on exit even with no active sessions"
        )
