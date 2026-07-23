"""Tests for _reap_orphaned_browser_sessions() — kills orphaned agent-browser
daemons whose Python parent exited without cleaning up."""

import os
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
        """Alive daemon not tracked by _active_sessions is terminated (legacy path).

        No owner_pid file => falls back to tracked_names check.
        """
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "h_orphan12345", pid=12345)

        kill_calls = []

        def mock_terminate(pid):
            kill_calls.append(pid)

        # Post-#21561 the liveness probe goes through
        # ``gateway.status._pid_exists`` (which wraps ``psutil.pid_exists``
        # so it's safe on Windows — ``os.kill(pid, 0)`` is bpo-14484).
        # The identity guard (#14073) is mocked True here — its own behavior
        # is covered by TestReaperIdentityGuard below.
        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid", side_effect=mock_terminate):
            _reap_orphaned_browser_sessions()

        assert 12345 in kill_calls

    def test_tracked_session_is_not_reaped(self, fake_tmpdir):
        """Sessions tracked in _active_sessions are left alone (legacy path)."""
        import tools.browser_tool as bt
        from tools.browser_tool import _reap_orphaned_browser_sessions

        session_name = "h_tracked1234"
        d = _make_socket_dir(fake_tmpdir, session_name, pid=12345)

        # Register the session as actively tracked
        bt._active_sessions["some_task"] = {"session_name": session_name}

        kill_calls = []

        def mock_terminate(pid):
            kill_calls.append(pid)

        with patch("tools.process_registry.ProcessRegistry._terminate_host_pid", side_effect=mock_terminate):
            _reap_orphaned_browser_sessions()

        # Should NOT have tried to terminate anything
        assert len(kill_calls) == 0
        # Dir should still exist
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

        def mock_terminate(pid):
            terminate_calls.append(pid)

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid", side_effect=mock_terminate):
            _reap_orphaned_browser_sessions()

        assert 12345 in terminate_calls
        assert not d.exists()

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


class TestAppDirSessionsAreLeftAlone:
    """Regression: ``_reap_orphaned_browser_sessions`` never touches
    ``~/.agent-browser/`` (agent-browser's "app dir" used by direct CLI
    invocations outside Hermes).

    A live daemon there has no ``owner_pid`` file Hermes can prove ownership
    through, and the environmental identity guard only proves the PID is an
    agent-browser daemon for the session name — not that it is orphaned or
    Hermes-owned.  A user running ``agent-browser open`` in a separate
    terminal must survive any Hermes reaper pass.

    Reviewed by teknium1 in PR #65701 (Blocking): the app-dir sweep this
    PR originally added could terminate live direct-CLI daemons.  The sweep
    was removed; this regression test pins the removal.
    """

    def _make_app_dir_session(self, app_dir, session_name, pid):
        """Create a flat ``<session>.pid`` file in ``~/.agent-browser``."""
        app_dir.mkdir(parents=True, exist_ok=True)
        (app_dir / f"{session_name}.pid").write_text(str(pid))
        return app_dir / f"{session_name}.pid"

    def test_live_app_dir_daemon_survives_reaper(self, fake_tmpdir, tmp_path):
        """A live direct-CLI (app-dir) daemon is never terminated.

        Scenario:
          - ``~/.agent-browser/`` has a ``live_session.pid`` file pointing
            at a process that WOULD pass the identity+binding guard.
          - ``_reap_orphaned_browser_sessions`` is called.
          - The reaper only globs ``agent-browser-h_*``/``-cdp_*``/``-hermes_*``
            socket dirs in the system tmpdir.  The app dir is outside that
            glob, so the live CLI daemon is never even visited by the
            reaper, let alone terminated.

        This is the keystone regression from teknium1's review: prior to
        PR #65701's revision, ``_reap_app_dir_sessions`` would have
        terminated this PID.  It must now survive.
        """
        from tools.browser_tool import _reap_orphaned_browser_sessions

        # Stage a live daemon in ``~/.agent-browser/``.  We don't need to
        # patch ``os.path.expanduser`` because the reaper no longer even
        # reads the app dir, but we DO isolate by writing into the test's
        # tmp_path under a ``.agent-browser`` subdir — the regression is
        # structural: the reaper code path no longer calls any app-dir
        # helper, so the file we write here is never inspected.
        app_dir = tmp_path / ".agent-browser"
        self._make_app_dir_session(app_dir, "live_session", pid=4242)

        terminate_calls = []

        def _record_terminate(pid):
            terminate_calls.append(pid)

        # If for any reason the reaper DID start scanning the app dir, the
        # identity+binding guard must still refuse to terminate — we mock
        # ``_verify_reapable_browser_daemon`` to (correctly) declare the
        # session reapable, and assert that even so, no terminate fires
        # because the reaper's glob never finds the file.
        with patch("tools.browser_tool._verify_reapable_browser_daemon",
                   return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=_record_terminate), \
             patch("gateway.status._pid_exists", return_value=True):
            _reap_orphaned_browser_sessions()

        # Sentinel: the live CLI daemon must never be terminated.
        assert 4242 not in terminate_calls, (
            "Reaper reached into ~/.agent-browser/ and terminated a live "
            "direct-CLI daemon — app-dir sessions must be left alone (teknium1 "
            "PR #65701 Blocking review)."
        )
        assert terminate_calls == [], (
            f"Reaper terminated unexpected PIDs: {terminate_calls}"
        )
        # And the .pid file we staged must be untouched.
        assert (app_dir / "live_session.pid").read_text() == "4242"


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

    def test_dead_owner_triggers_reap(self, fake_tmpdir):
        """Daemon whose owner_pid is dead gets reaped."""
        from tools.browser_tool import _reap_orphaned_browser_sessions

        # PID 999999999 almost certainly doesn't exist
        d = _make_socket_dir(
            fake_tmpdir, "h_dead_owner1", pid=12345, owner_pid=999999999
        )

        kill_calls = []

        def mock_terminate(pid):
            kill_calls.append(pid)

        # Owner 999999999 dead, daemon 12345 alive.
        pid_alive = {999999999: False, 12345: True}
        with patch("gateway.status._pid_exists",
                   side_effect=lambda pid: pid_alive.get(int(pid), False)), \
             patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid", side_effect=mock_terminate):
            _reap_orphaned_browser_sessions()

        assert 12345 in kill_calls
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

        def mock_terminate(pid):
            kill_calls.append(pid)

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid", side_effect=mock_terminate):
            _reap_orphaned_browser_sessions()

        # Legacy path took over → tracked → not reaped
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

    def test_planted_pid_for_non_browser_process_is_refused(self):
        """A planted .pid pointing at e.g. `sleep 600` must NOT be reaped."""
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(name="sleep", cmdline=["/bin/sleep", "600"])
        assert self._run(proc, socket_dir) is False

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

    def test_browser_name_but_environ_denied_and_no_cmdline_bind_refused(self):
        """Looks like browser, cmdline doesn't bind, environ() denied -> refuse."""
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(
            name="agent-browser",
            cmdline=["agent-browser", "daemon"],  # no dir
            raise_environ=True,
        )
        assert self._run(proc, socket_dir) is False

    def test_vanished_process_is_not_reapable(self):
        socket_dir = "/tmp/agent-browser-h_sess123456"
        assert self._run(None, socket_dir, no_such=True) is False

    def test_access_denied_on_identity_read_refuses(self):
        socket_dir = "/tmp/agent-browser-h_sess123456"
        assert self._run(None, socket_dir, access_denied=True) is False

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


class TestDaemonStartFailureDetection:
    """Tests for ``_is_daemon_start_failure`` and the retry-on-failure path."""

    def test_detects_daemon_failed_to_start(self):
        from tools.browser_tool import _is_daemon_start_failure
        result = {"success": False, "error": "✗ Daemon failed to start (port: 127.0.0.1:50838)"}
        assert _is_daemon_start_failure(result) is True

    def test_ignores_other_errors(self):
        from tools.browser_tool import _is_daemon_start_failure
        assert _is_daemon_start_failure({"error": "Some other error"}) is False
        assert _is_daemon_start_failure({"success": True, "error": None}) is False
        assert _is_daemon_start_failure({}) is False
        assert _is_daemon_start_failure(None) is False

    def test_is_daemon_start_failure_false_for_none_result(self):
        from tools.browser_tool import _is_daemon_start_failure
        assert _is_daemon_start_failure(None) is False


class TestDaemonRetryOnStartFailure:
    """Verify ``_run_browser_command`` retries once after reaping on daemon
    start failure, and does NOT retry again on the second failure."""

    @pytest.fixture(autouse=True)
    def _isolate(self):
        import tools.browser_tool as bt
        orig = bt._active_sessions.copy()
        orig_activity = bt._session_last_activity.copy()
        bt._active_sessions.clear()
        bt._session_last_activity.clear()
        yield
        bt._active_sessions.clear()
        bt._active_sessions.update(orig)
        bt._session_last_activity.clear()
        bt._session_last_activity.update(orig_activity)

    @pytest.fixture(autouse=True)
    def _stub_browser_infra(self, monkeypatch, tmp_path):
        """Stub all non-target functions so ``_run_browser_command`` reaches
        the daemon detection block without touching real subprocess / disk /
        network paths.

        Returns a dict ``{"call_count": {"n": 0}}`` that tests mutate to
        control which invocation of Popen should simulate failure or success.
        """
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "/bin/true")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install",
                            lambda *a: False)
        # On non-Linux hosts ``_needs_chromium_sandbox_bypass`` reads
        # ``/proc/sys/kernel/unprivileged_userns_clone`` via ``open()``, which
        # raises on Windows.  Stub it to False so we don't trigger that path.
        monkeypatch.setattr(bt, "_needs_chromium_sandbox_bypass", lambda: False)
        monkeypatch.setattr(bt, "_chromium_installed", lambda: True)
        monkeypatch.setattr(bt, "_is_local_mode", lambda: True)
        monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
        monkeypatch.setattr(
            bt, "_get_session_info",
            lambda task_id: {"session_name": "h_test12345", "cdp_url": None},
        )
        # ``write_owner_pid`` writes a sidecar file used for cross-process
        # orphan detection — safe to short-circuit here.
        monkeypatch.setattr(bt, "_write_owner_pid", lambda *a, **kw: None)

    @staticmethod
    def _make_fake_popen(tmp_path, session_name, command,
                          call_count, stdout_per_call, stderr_per_call):
        """Return a fake ``subprocess.Popen`` substitute.

        On call N (1-indexed) it writes ``stdout_per_call[N]`` and
        ``stderr_per_call[N]`` into the temp files that
        ``_run_browser_command`` creates with ``os.open`` before invoking
        Popen.  This mirrors what the real agent-browser binary does with its
        stdout/stderr redirection, and avoids monkeypatching ``builtins.open``
        globally.
        """
        import os

        socket_dir = os.path.join(str(tmp_path),
                                  f"agent-browser-{session_name}")
        stdout_path = os.path.join(socket_dir, f"_stdout_{command}")
        stderr_path = os.path.join(socket_dir, f"_stderr_{command}")

        def _fake_popen(cmd, *args, **kw):
            # 1-indexed call counter
            call_count["n"] += 1
            n = call_count["n"]
            stdout_text = stdout_per_call.get(n, "")
            stderr_text = stderr_per_call.get(n, "")
            # Write the simulated daemon output to the temp files.  They were
            # already opened (and FDs closed before we're invoked) by
            # ``_run_browser_command``.  'w' truncates so the files stay clean.
            with open(stdout_path, "w", encoding="utf-8") as f:
                f.write(stdout_text)
            with open(stderr_path, "w", encoding="utf-8") as f:
                f.write(stderr_text)

            class _FakeProc:
                # ``returncode`` is read after ``wait()`` returns.  Non-zero rc
                # triggers the error path that looks at the stderr content.
                returncode = 1

                def wait(self, timeout=None):
                    return self.returncode

            return _FakeProc()

        return _fake_popen

    def test_retry_calls_reaper_and_resets_session(self, monkeypatch, tmp_path):
        """On first daemon failure, reaper runs, session resets, and command
        is retried exactly once."""
        import tools.browser_tool as bt

        call_count = {"n": 0}
        reap_called = []
        reset_called = []

        def _fake_reaper():
            reap_called.append(True)

        def _fake_reset(task_id):
            reset_called.append(task_id)

        _fake_popen = self._make_fake_popen(
            tmp_path=tmp_path,
            session_name="h_test12345",
            command="open",
            call_count=call_count,
            # Call 1: empty stdout, stderr says "Daemon failed to start".
            # Call 2: success JSON, no stderr.
            stdout_per_call={1: "", 2: '{"success": true, "data": {}}'},
            stderr_per_call={
                1: "Daemon failed to start (port: 127.0.0.1:50838)",
            },
        )

        monkeypatch.setattr(bt, "_reap_orphaned_browser_sessions", _fake_reaper)
        monkeypatch.setattr(bt, "_reset_session_for_retry", _fake_reset)
        monkeypatch.setattr(bt.subprocess, "Popen", _fake_popen)

        result = bt._run_browser_command(
            task_id="test", command="open", args=["https://example.com"]
        )

        assert reap_called, "reaper must be called on daemon start failure"
        assert reset_called, "session reset must be called before retry"
        assert call_count["n"] == 2, "command must be retried exactly once"
        assert result.get("success") is True

    def test_no_retry_on_second_daemon_failure(self, monkeypatch, tmp_path):
        """If the retry also fails with daemon start error, return the error
        without retrying again."""
        import tools.browser_tool as bt

        call_count = {"n": 0}

        _fake_popen = self._make_fake_popen(
            tmp_path=tmp_path,
            session_name="h_test12345",
            command="open",
            call_count=call_count,
            # Both calls emit the daemon-failure stderr / empty stdout.
            stdout_per_call={1: "", 2: ""},
            stderr_per_call={
                1: "Daemon failed to start (port: 127.0.0.1:50838)",
                2: "Daemon failed to start (port: 127.0.0.1:50838)",
            },
        )

        monkeypatch.setattr(bt, "_reap_orphaned_browser_sessions", lambda: None)
        monkeypatch.setattr(bt, "_reset_session_for_retry", lambda tid: None)
        monkeypatch.setattr(bt.subprocess, "Popen", _fake_popen)

        result = bt._run_browser_command(
            task_id="test", command="open", args=["https://example.com"]
        )

        assert call_count["n"] == 2, "must try twice (original + 1 retry), no more"
        assert result.get("success") is False
        assert "Daemon failed to start" in result.get("error", "")
