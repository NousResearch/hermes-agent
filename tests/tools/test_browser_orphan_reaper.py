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


def _make_socket_dir(tmpdir, session_name, pid=None):
    """Create a fake agent-browser socket directory with optional PID file."""
    d = tmpdir / f"agent-browser-{session_name}"
    d.mkdir()
    if pid is not None:
        (d / f"{session_name}.pid").write_text(str(pid))
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
        """Alive daemon not tracked by _active_sessions gets SIGTERM."""
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
        """Sessions tracked in _active_sessions are left alone."""
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
