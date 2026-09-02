"""Tests for _reap_orphaned_browser_sessions() — kills orphaned agent-browser
daemons whose Python parent exited without cleaning up."""

import os
import time
from unittest.mock import patch

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
        with patch(
            "tools.browser_tool._socket_dir_idle_seconds",
            return_value=10_000,
        ):
            _reap_orphaned_browser_sessions()
        assert not d.exists()

    def test_fresh_dir_without_pid_file_survives_creator_race(self, fake_tmpdir):
        """A concurrent reaper must not delete a session still starting."""
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "h_starting1234")
        with patch(
            "tools.browser_tool._socket_dir_idle_seconds",
            return_value=0.0,
        ):
            _reap_orphaned_browser_sessions()

        assert d.exists()


    def test_alive_legacy_daemon_is_reaped(self, fake_tmpdir):
        """Alive, untracked, legacy (no owner_pid) daemon is reaped.

        Post-#21561 the liveness probe goes through
        ``gateway.status._pid_exists`` (which wraps ``psutil.pid_exists``
        because ``os.kill(pid, 0)`` is a footgun on Windows — bpo-14484).
        With no owner_pid file and no tracked-name entry, the reaper
        terminates the daemon (and its process tree) and removes its socket
        dir regardless of whether termination succeeded (best-effort
        semantics).
        """
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "h_perm1234567", pid=12345)

        terminate_calls = []

        def mock_terminate(pid, expected_start=None):
            terminate_calls.append(pid)

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("gateway.status.get_process_start_time", return_value=777), \
             patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid", side_effect=mock_terminate):
            _reap_orphaned_browser_sessions()

        assert 12345 in terminate_calls
        assert not d.exists()

    def test_unfingerprintable_daemon_is_refused(self, fake_tmpdir):
        """No start-time fingerprint -> the kill is refused (fail closed).

        The reaper reads the PID from a world-writable temp dir; a PID whose
        identity cannot be pinned could be recycled between the verify and the
        tree-kill, so it must be left alone (and the socket dir kept for a
        later sweep).
        """
        from tools.browser_tool import _reap_orphaned_browser_sessions

        _make_socket_dir(fake_tmpdir, "h_perm7654321", pid=12345)
        terminate_calls = []

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("gateway.status.get_process_start_time", return_value=None), \
             patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=lambda pid, expected_start=None: terminate_calls.append(pid)):
            _reap_orphaned_browser_sessions()

        assert terminate_calls == []


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

        def mock_terminate(pid):
            kill_calls.append(pid)

        # Owner alive → reaper skips without ever probing the daemon.
        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid", side_effect=mock_terminate):
            _reap_orphaned_browser_sessions()

        assert 12345 not in kill_calls
        assert d.exists()


    def test_owner_pid_permission_error_treated_as_alive(self, fake_tmpdir):
        """Owner PID owned by another user → treat as alive.

        Post-#21561 this is handled inside ``gateway.status._pid_exists``
        (via psutil's ``OpenProcess`` returning ``ERROR_ACCESS_DENIED`` on
        Windows, or via the POSIX fallback's ``except PermissionError``
        branch). Exposed to callers as ``alive=True``.
        """
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(
            fake_tmpdir, "h_perm_owner1", pid=12345, owner_pid=22222
        )

        kill_calls = []

        def mock_terminate(pid):
            kill_calls.append(pid)

        # Owner 22222 reported alive (PermissionError collapses to True
        # inside _pid_exists). Daemon never probed, never terminated.
        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid", side_effect=mock_terminate):
            _reap_orphaned_browser_sessions()

        assert 12345 not in kill_calls
        assert d.exists()


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


class TestReaperIdentityGuard:
    """Tests for _verify_reapable_browser_daemon — the #14073 fix.

    The reaper reads daemon PIDs from world-writable, predictably-named temp
    dirs.  Before tree-killing a live PID it must confirm the process really is
    *this* session's agent-browser daemon, defeating planted pid files and
    recycled PIDs that would otherwise become an arbitrary same-user DoS.
    """

    class _FakeProc:
        def __init__(self, name="agent-browser", cmdline=None, environ=None,
                     raise_environ=False):
            self._name = name
            self._cmdline = cmdline if cmdline is not None else []
            self._environ = environ or {}
            self._raise_environ = raise_environ

        def name(self):
            return self._name

        def cmdline(self):
            return self._cmdline

        def environ(self):
            if self._raise_environ:
                import psutil
                raise psutil.AccessDenied()
            return self._environ

    def _run(self, fake_proc, socket_dir, session_name="h_sess123456",
             daemon_pid=12345, no_such=False, access_denied=False):
        import psutil
        from tools.browser_tool import _verify_reapable_browser_daemon

        def _factory(pid):
            if no_such:
                raise psutil.NoSuchProcess(pid)
            if access_denied:
                raise psutil.AccessDenied(pid)
            return fake_proc

        with patch("psutil.Process", side_effect=_factory):
            return _verify_reapable_browser_daemon(
                daemon_pid, socket_dir, session_name)

    def test_real_daemon_bound_via_cmdline_is_reapable(self):
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(
            name="agent-browser",
            cmdline=["agent-browser", "open", "--session", "h_sess123456",
                     "--socket-dir", socket_dir],
        )
        assert self._run(proc, socket_dir) is True

    def test_daemon_bound_via_environ_is_reapable(self):
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(
            name="agent-browser-linux-x64",
            cmdline=["agent-browser-linux-x64", "daemon"],  # no dir in cmd
            environ={"AGENT_BROWSER_SOCKET_DIR": socket_dir},
        )
        assert self._run(proc, socket_dir) is True


    def test_recycled_pid_browser_not_bound_to_our_dir_is_refused(self):
        """An agent-browser process for a DIFFERENT session must not be reaped.

        Models PID reuse / a concurrent unrelated daemon: it looks like
        agent-browser but is bound to another socket dir.
        """
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(
            name="agent-browser",
            cmdline=["agent-browser", "open", "--session", "h_OTHER999",
                     "--socket-dir", "/tmp/agent-browser-h_OTHER999"],
            environ={"AGENT_BROWSER_SOCKET_DIR":
                     "/tmp/agent-browser-h_OTHER999"},
        )
        assert self._run(proc, socket_dir) is False


    def test_planted_pid_survives_full_reaper_path(self, fake_tmpdir):
        """End-to-end through the reaper: a planted non-browser PID is spared.

        No owner_pid (legacy path), not tracked, PID 'alive' — but the live
        process is `sleep`, not agent-browser, so it must be left alone and the
        socket dir retained.
        """
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "h_planted9999", pid=12345)

        terminate_calls = []
        proc = self._FakeProc(name="sleep", cmdline=["/bin/sleep", "600"])

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("psutil.Process", return_value=proc), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=lambda pid: terminate_calls.append(pid)):
            _reap_orphaned_browser_sessions()

        assert terminate_calls == [], "planted non-browser PID must not be killed"
        assert d.exists(), "socket dir retained for a later sweep"


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


def _age_socket_dir(d, seconds):
    """Backdate every mtime under ``d`` so it looks idle for ``seconds``."""
    old = time.time() - seconds
    for p in d.iterdir():
        os.utime(p, (old, old))
    os.utime(d, (old, old))


class TestSocketDirIdleSeconds:
    """Unit tests for the idle-age signal backing the leak escape hatch."""

    def test_missing_dir_returns_none(self, tmp_path):
        from tools.browser_tool import _socket_dir_idle_seconds
        assert _socket_dir_idle_seconds(str(tmp_path / "nope")) is None

    def test_fresh_dir_is_near_zero(self, tmp_path):
        from tools.browser_tool import _socket_dir_idle_seconds
        d = tmp_path / "agent-browser-h_fresh"
        d.mkdir()
        assert _socket_dir_idle_seconds(str(d)) < 5

    def test_entry_mtime_beats_stale_dir_mtime(self, tmp_path):
        """Rewriting an existing file must count as activity.

        Command names repeat (``_stdout_click`` is rewritten on every click),
        and overwriting an existing file does NOT bump the *directory* mtime.
        Reading only the directory mtime would therefore report a busy session
        as idle and reap it.  The reaper must scan entries too.
        """
        from tools.browser_tool import _socket_dir_idle_seconds
        d = tmp_path / "agent-browser-h_reuse"
        d.mkdir()
        f = d / "_stdout_click"
        f.write_text("x")
        _age_socket_dir(d, 7200)
        assert _socket_dir_idle_seconds(str(d)) > 7000

        f.write_text("y")  # rewrite in place — dir mtime stays stale
        assert time.time() - os.path.getmtime(d) > 7000, "precondition"
        assert _socket_dir_idle_seconds(str(d)) < 5


class TestLeakedDaemonWithLiveOwner:
    """Idle-age escape hatch for untracked daemons whose owner is still alive.

    ``owner_alive is True`` alone made a leaked daemon immortal: in-memory
    tracking is lost on any exception path between spawn and registration,
    yet the owner PID stays up, so the reaper skipped it forever.  Observed in
    the wild — five agent-browser daemons accumulated over 10 days inside one
    long-lived hermes process, pinning ~5 CPU cores and driving load to 100+.

    The daemon-side ``AGENT_BROWSER_IDLE_TIMEOUT_MS`` is not a backstop here:
    it does not fire when the daemon itself is wedged (e.g. Chrome's framework
    was replaced underneath it by an auto-update).
    """

    def test_fresh_untracked_daemon_with_live_owner_is_spared(self, fake_tmpdir):
        """Within the grace window, cross-process safety still wins."""
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(
            fake_tmpdir, "h_fresh_owner", pid=12345, owner_pid=os.getpid()
        )
        kill_calls = []

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=kill_calls.append):
            _reap_orphaned_browser_sessions()

        assert 12345 not in kill_calls
        assert d.exists()

    def test_idle_untracked_daemon_with_live_owner_is_reaped(self, fake_tmpdir):
        """Past the grace window, an untracked daemon is treated as leaked."""
        from tools.browser_tool import (
            BROWSER_ORPHAN_GRACE_SECONDS,
            _reap_orphaned_browser_sessions,
        )

        d = _make_socket_dir(
            fake_tmpdir, "h_leaked_owner", pid=12345, owner_pid=os.getpid()
        )
        _age_socket_dir(d, BROWSER_ORPHAN_GRACE_SECONDS + 600)
        kill_calls = []

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("gateway.status.get_process_start_time", return_value=777), \
             patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=lambda pid, expected_start=None: kill_calls.append(pid)):
            _reap_orphaned_browser_sessions()

        assert 12345 in kill_calls
        assert not d.exists()

    def test_tracked_daemon_with_live_owner_is_spared_at_any_age(self, fake_tmpdir):
        """A session this process still tracks is never reaped, however old.

        Idle age is a fallback for *lost* bookkeeping, not an override of
        bookkeeping that is present and says the session is live.
        """
        import tools.browser_tool as bt
        from tools.browser_tool import (
            BROWSER_ORPHAN_GRACE_SECONDS,
            _reap_orphaned_browser_sessions,
        )

        d = _make_socket_dir(
            fake_tmpdir, "h_tracked_old", pid=12345, owner_pid=os.getpid()
        )
        _age_socket_dir(d, BROWSER_ORPHAN_GRACE_SECONDS * 10)
        bt._active_sessions["task-1"] = {"session_name": "h_tracked_old"}
        kill_calls = []

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=kill_calls.append):
            _reap_orphaned_browser_sessions()

        assert 12345 not in kill_calls
        assert d.exists()

    def test_unknown_idle_age_fails_safe(self, fake_tmpdir):
        """Unreadable mtime => treat as too young to reap, never guess."""
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(
            fake_tmpdir, "h_unknown_age", pid=12345, owner_pid=os.getpid()
        )
        kill_calls = []

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.browser_tool._socket_dir_idle_seconds", return_value=None), \
             patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=kill_calls.append):
            _reap_orphaned_browser_sessions()

        assert 12345 not in kill_calls
        assert d.exists()

    def test_identity_guard_still_gates_the_new_path(self, fake_tmpdir):
        """The escape hatch must not bypass _verify_reapable_browser_daemon.

        That guard is the anti-spoof / anti-PID-recycle defense (issue #14073);
        an idle daemon is still only reapable if it verifies.
        """
        from tools.browser_tool import (
            BROWSER_ORPHAN_GRACE_SECONDS,
            _reap_orphaned_browser_sessions,
        )

        d = _make_socket_dir(
            fake_tmpdir, "h_unverified", pid=12345, owner_pid=os.getpid()
        )
        _age_socket_dir(d, BROWSER_ORPHAN_GRACE_SECONDS + 600)
        kill_calls = []

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=False), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=kill_calls.append):
            _reap_orphaned_browser_sessions()

        assert 12345 not in kill_calls
        assert d.exists()


class TestPeriodicOrphanReap:
    """The reaper must run repeatedly, not only at cleanup-thread startup.

    A startup-only reap can never recover from a leak that appears *after*
    boot — which is exactly what happens in a hermes process that stays up
    for days.
    """

    def test_reaper_runs_on_every_interval_not_just_startup(self):
        import tools.browser_tool as bt

        cycles_to_run = 21
        reap_calls = []
        remaining = {"n": cycles_to_run}

        def fake_cleanup():
            remaining["n"] -= 1
            if remaining["n"] <= 0:
                bt._cleanup_running = False

        orig_running = bt._cleanup_running
        bt._cleanup_running = True
        try:
            with patch("tools.browser_tool._reap_orphaned_browser_sessions",
                       side_effect=lambda: reap_calls.append(1)), \
                 patch("tools.browser_tool._cleanup_inactive_browser_sessions",
                       side_effect=fake_cleanup), \
                 patch("tools.browser_tool.time.sleep"):
                bt._browser_cleanup_thread_worker()
        finally:
            bt._cleanup_running = orig_running

        every = max(1, round(bt.BROWSER_ORPHAN_REAP_INTERVAL / 30))
        expected = len([c for c in range(cycles_to_run) if c % every == 0])
        assert len(reap_calls) == expected
        assert len(reap_calls) > 1, "startup-only reap would give exactly 1"
