"""Tests for _reap_orphaned_browser_sessions() — kills orphaned agent-browser
daemons whose Python parent exited without cleaning up."""

import os
import threading
import time
from pathlib import Path
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

        def mock_terminate(pid, *, expected_start=None):
            assert expected_start == 777
            kill_calls.append(pid)

        # Identity verification returns the kernel start fingerprint, and the
        # post-kill liveness check confirms the daemon is gone.
        with patch("gateway.status._pid_exists", side_effect=[True, False]), \
             patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=777), \
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
        terminates the verified daemon tree. It removes the socket directory
        only after the daemon is confirmed gone and no process references the
        managed profile.
        """
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "h_perm1234567", pid=12345)

        terminate_calls = []

        def mock_terminate(pid, *, expected_start=None):
            assert expected_start == 777
            terminate_calls.append(pid)

        with patch("gateway.status._pid_exists", side_effect=[True, False]), \
             patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=777), \
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

        def mock_terminate(pid, *, expected_start=None):
            assert expected_start == 777
            kill_calls.append(pid)

        # Owner dead, daemon alive, then daemon gone after guarded termination.
        with patch("gateway.status._pid_exists",
                   side_effect=[False, True, False]), \
             patch("tools.browser_tool._verify_reapable_browser_daemon", return_value=777), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid", side_effect=mock_terminate):
            _reap_orphaned_browser_sessions()

        assert 12345 in kill_calls
        assert not d.exists()

    def test_corrupt_owner_pid_is_preserved_fail_closed(self, fake_tmpdir):
        """Present but corrupt owner metadata is never treated as unowned legacy."""
        from tools.browser_tool import _reap_orphaned_browser_sessions

        session_name = "h_corrupt_own"
        d = _make_socket_dir(fake_tmpdir, session_name, pid=12345)
        (d / f"{session_name}.owner_pid").write_text("not-a-pid")
        kill_calls = []

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=lambda *args, **kwargs: kill_calls.append((args, kwargs))):
            _reap_orphaned_browser_sessions()

        assert kill_calls == []
        assert d.exists()

    def test_partial_owner_pid_fails_closed_even_when_untracked(self, fake_tmpdir):
        """A concurrent reader seeing partial metadata must not treat it as legacy."""
        from tools.browser_tool import _reap_orphaned_browser_sessions

        session_name = "h_partial_owner"
        d = _make_socket_dir(fake_tmpdir, session_name, pid=12345)
        (d / f"{session_name}.owner_pid").write_text("")
        terminate_calls = []

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.browser_tool._verify_reapable_browser_daemon",
                   return_value=777), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=lambda *args, **kwargs: terminate_calls.append((args, kwargs))):
            _reap_orphaned_browser_sessions()

        assert terminate_calls == []
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

    def test_write_owner_pid_publishes_by_atomic_replace(self, fake_tmpdir):
        """Readers see the old complete value until the new value is published."""
        import tools.browser_tool as bt

        session_name = "h_atomic_owner"
        socket_dir = fake_tmpdir / f"agent-browser-{session_name}"
        socket_dir.mkdir()
        owner_file = socket_dir / f"{session_name}.owner_pid"
        owner_file.write_text("11111")
        values_seen_at_replace = []
        real_replace = os.replace

        def observed_replace(src, dst):
            assert Path(src).parent == socket_dir
            assert Path(dst).parent == socket_dir
            values_seen_at_replace.append(owner_file.read_text())
            real_replace(src, dst)

        with patch("os.replace", side_effect=observed_replace):
            bt._write_owner_pid(str(socket_dir), session_name)

        assert values_seen_at_replace == ["11111"]
        assert owner_file.read_text() == str(os.getpid())
        assert list(socket_dir.glob(f".{session_name}.owner_pid.*")) == []

    def test_concurrent_owner_pid_reader_never_observes_partial_content(
        self, fake_tmpdir
    ):
        import tools.browser_tool as bt

        session_name = "h_concurrent_owner"
        socket_dir = fake_tmpdir / f"agent-browser-{session_name}"
        socket_dir.mkdir()
        owner_file = socket_dir / f"{session_name}.owner_pid"
        owner_file.write_text("11111")
        stop = threading.Event()
        reader_ready = threading.Event()
        first_read = threading.Event()
        writes_active = threading.Event()
        overlapping_read = threading.Event()
        invalid_values = []
        read_count = []

        def reader():
            reader_ready.set()
            while not stop.is_set():
                try:
                    value = owner_file.read_text().strip()
                except OSError as exc:
                    invalid_values.append(type(exc).__name__)
                    continue
                read_count.append(value)
                first_read.set()
                if writes_active.is_set():
                    overlapping_read.set()
                if not value.isdigit():
                    invalid_values.append(value)

        thread = threading.Thread(target=reader)
        thread.start()
        assert reader_ready.wait(timeout=1)
        assert first_read.wait(timeout=1)
        try:
            writes_active.set()
            for _ in range(100):
                bt._write_owner_pid(str(socket_dir), session_name)
            assert overlapping_read.wait(timeout=1)
        finally:
            stop.set()
            thread.join(timeout=2)

        assert not thread.is_alive()
        assert read_count
        assert invalid_values == []

    def test_write_owner_pid_swallows_oserror(self, fake_tmpdir, monkeypatch):
        """OSError (e.g. permission denied) doesn't propagate — the reaper
        falls back to the legacy tracked_names heuristic in that case.
        """
        import tools.browser_tool as bt

        def raise_oserror(*a, **kw):
            raise OSError("permission denied")

        monkeypatch.setattr("builtins.open", raise_oserror)

        # Must not raise or leave a partially published temporary file.
        bt._write_owner_pid(str(fake_tmpdir), "h_readonly123")
        assert list(fake_tmpdir.glob(".h_readonly123.owner_pid.*")) == []

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

        with patch("gateway.status.get_process_start_time", return_value=777), \
             patch("psutil.Process", side_effect=_factory):
            return _verify_reapable_browser_daemon(
                daemon_pid, socket_dir, session_name)

    def test_real_daemon_bound_via_cmdline_is_reapable(self):
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(
            name="agent-browser",
            cmdline=["agent-browser", "open", "--session", "h_sess123456",
                     "--socket-dir", socket_dir],
        )
        assert self._run(proc, socket_dir) == 777

    def test_daemon_bound_via_environ_is_reapable(self):
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(
            name="agent-browser-linux-x64",
            cmdline=["agent-browser-linux-x64", "daemon"],  # no dir in cmd
            environ={"AGENT_BROWSER_SOCKET_DIR": socket_dir},
        )
        assert self._run(proc, socket_dir) == 777

    def test_planted_pid_for_non_browser_process_is_refused(self):
        """A planted .pid pointing at e.g. `sleep 600` must NOT be reaped."""
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(name="sleep", cmdline=["/bin/sleep", "600"])
        assert self._run(proc, socket_dir) is None

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
        assert self._run(proc, socket_dir) is None

    def test_browser_name_but_environ_denied_and_no_cmdline_bind_refused(self):
        """Looks like browser, cmdline doesn't bind, environ() denied -> refuse."""
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(
            name="agent-browser",
            cmdline=["agent-browser", "daemon"],  # no dir
            raise_environ=True,
        )
        assert self._run(proc, socket_dir) is None

    def test_vanished_process_is_not_reapable(self):
        socket_dir = "/tmp/agent-browser-h_sess123456"
        assert self._run(None, socket_dir, no_such=True) is None

    def test_access_denied_on_identity_read_refuses(self):
        socket_dir = "/tmp/agent-browser-h_sess123456"
        assert self._run(None, socket_dir, access_denied=True) is None

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

    def test_verified_daemon_kill_uses_kernel_start_guard(self, fake_tmpdir):
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "h_guardedkill", pid=12345)
        terminate_calls = []
        # First liveness check sees the daemon; the post-kill check sees it gone.
        with patch("gateway.status._pid_exists", side_effect=[True, False]), \
             patch("tools.browser_tool._verify_reapable_browser_daemon",
                   return_value=24680), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=lambda *args, **kwargs: terminate_calls.append((args, kwargs))):
            _reap_orphaned_browser_sessions()

        assert terminate_calls == [((12345,), {"expected_start": 24680})]
        assert not d.exists()

    def test_surviving_daemon_preserves_socket_metadata(self, fake_tmpdir):
        from tools.browser_tool import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "h_daemonsurvives", pid=12345)
        terminate_calls = []
        with patch("gateway.status._pid_exists", side_effect=[True, True]), \
             patch("tools.browser_tool._verify_reapable_browser_daemon",
                   return_value=24680), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=lambda *args, **kwargs: terminate_calls.append((args, kwargs))):
            _reap_orphaned_browser_sessions()

        assert terminate_calls == [((12345,), {"expected_start": 24680})]
        assert d.exists()


class TestOrphanedChromiumReaper:
    """Security boundaries for Hermes-owned managed Chromium profiles."""

    class _FakeProc:
        def __init__(self, pid, cmdline, *, age=600, parent=None,
                     name="chrome", username="alex"):
            self.pid = pid
            self.info = {
                "pid": pid,
                "name": name,
                "cmdline": cmdline,
                "create_time": time.time() - age,
            }
            self._parent = parent
            self._username = username

        def parent(self):
            return self._parent

        def name(self):
            return self.info["name"]

        def cmdline(self):
            return self.info["cmdline"]

        def create_time(self):
            return self.info["create_time"]

        def username(self):
            return self._username

    def _session(self, tmpdir, session="h_owned"):
        socket_dir = tmpdir / f"agent-browser-{session}"
        socket_dir.mkdir()
        (socket_dir / f"{session}.owner_pid").write_text("999999")
        profile = socket_dir / f"agent-browser-chrome-{session}"
        profile.mkdir()
        return session, socket_dir, profile

    def _chrome(self, profile, *, pid=321, age=600, parent=None,
                username="alex", extra=None):
        cmdline = ["/usr/bin/chromium", f"--user-data-dir={profile}", "--headless=new"]
        if extra:
            cmdline.extend(extra)
        return self._FakeProc(pid, cmdline, age=age, parent=parent,
                              username=username)

    def _process_patches(self, proc, terminated, *, current_user="alex",
                         survives=False, kernel_start=777):
        import psutil

        current = self._FakeProc(os.getpid(), ["python"], username=current_user)

        def lookup(pid):
            if pid == os.getpid():
                return current
            if pid == proc.pid:
                if terminated and not survives:
                    raise psutil.NoSuchProcess(pid)
                return proc
            raise psutil.NoSuchProcess(pid)

        def terminate(pid, *, expected_start=None):
            assert pid == proc.pid
            assert expected_start == kernel_start
            terminated.append(pid)

        def iter_processes(*_args, **_kwargs):
            if terminated and not survives:
                return []
            return [proc]

        return (
            patch("psutil.process_iter", side_effect=iter_processes),
            patch("psutil.Process", side_effect=lookup),
            patch("gateway.status.get_process_start_time", return_value=kernel_start),
            patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                  side_effect=terminate),
        )

    def test_exact_dead_owner_session_is_reaped_with_start_guard(self, fake_tmpdir):
        from tools.browser_tool import _reap_session_chromium

        session, socket_dir, profile = self._session(fake_tmpdir)
        proc = self._chrome(profile)
        terminated = []
        p1, p2, p3, p4 = self._process_patches(proc, terminated)
        with p1, p2, p3, p4:
            assert _reap_session_chromium(
                str(socket_dir), session, min_age_seconds=120
            ) == 1
        assert terminated == [proc.pid]
        assert not profile.exists()

    def test_young_process_is_preserved(self, fake_tmpdir):
        from tools.browser_tool import _reap_session_chromium

        session, socket_dir, profile = self._session(fake_tmpdir)
        proc = self._chrome(profile, age=30)
        terminated = []
        p1, p2, p3, p4 = self._process_patches(proc, terminated)
        with p1, p2, p3, p4:
            assert _reap_session_chromium(
                str(socket_dir), session, min_age_seconds=120
            ) == 0
        assert terminated == []
        assert profile.exists()

    def test_live_agent_browser_parent_is_preserved(self, fake_tmpdir):
        from tools.browser_tool import _reap_session_chromium

        session, socket_dir, profile = self._session(fake_tmpdir)
        parent = self._FakeProc(111, ["agent-browser", "daemon"],
                                name="agent-browser")
        proc = self._chrome(profile, parent=parent)
        terminated = []
        p1, p2, p3, p4 = self._process_patches(proc, terminated)
        with p1, p2, p3, p4:
            _reap_session_chromium(str(socket_dir), session,
                                    min_age_seconds=0)
        assert terminated == []
        assert profile.exists()

    def test_cross_platform_username_mismatch_is_preserved(self, fake_tmpdir):
        from tools.browser_tool import _reap_session_chromium

        session, socket_dir, profile = self._session(fake_tmpdir)
        proc = self._chrome(profile, username="other-user")
        terminated = []
        p1, p2, p3, p4 = self._process_patches(proc, terminated)
        with p1, p2, p3, p4:
            _reap_session_chromium(str(socket_dir), session,
                                    min_age_seconds=0)
        assert terminated == []
        assert profile.exists()

    def test_duplicate_user_data_dir_is_rejected(self, fake_tmpdir):
        from tools.browser_tool import _reap_session_chromium

        session, socket_dir, profile = self._session(fake_tmpdir)
        proc = self._chrome(profile, extra=["--user-data-dir=/tmp/other"])
        terminated = []
        p1, p2, p3, p4 = self._process_patches(proc, terminated)
        with p1, p2, p3, p4:
            _reap_session_chromium(str(socket_dir), session,
                                    min_age_seconds=0)
        assert terminated == []
        assert profile.exists()

    def test_surviving_same_process_preserves_profile(self, fake_tmpdir):
        from tools.browser_tool import _reap_session_chromium

        session, socket_dir, profile = self._session(fake_tmpdir)
        proc = self._chrome(profile)
        terminated = []
        p1, p2, p3, p4 = self._process_patches(
            proc, terminated, survives=True
        )
        with p1, p2, p3, p4:
            _reap_session_chromium(str(socket_dir), session,
                                    min_age_seconds=0)
        assert terminated == [proc.pid]
        assert profile.exists()

    def test_symlinked_profile_is_rejected(self, fake_tmpdir):
        from tools.browser_tool import _managed_chrome_profile

        session = "h_symlink"
        socket_dir = fake_tmpdir / f"agent-browser-{session}"
        socket_dir.mkdir()
        target = fake_tmpdir / "target"
        target.mkdir()
        (socket_dir / f"agent-browser-chrome-{session}").symlink_to(
            target, target_is_directory=True
        )
        assert _managed_chrome_profile(str(socket_dir), session) is None
        assert target.exists()

    def test_unreadable_final_process_scan_preserves_socket(self, fake_tmpdir):
        import tools.browser_tool as bt

        session, socket_dir, profile = self._session(fake_tmpdir)

        class _UnreadableProc:
            info = {"name": "chromium", "cmdline": None}

        with patch("psutil.process_iter", return_value=[_UnreadableProc()]):
            assert bt._remove_browser_socket_dir_if_safe(
                str(socket_dir), session
            ) is False

        assert socket_dir.exists()
        assert profile.exists()

    def test_unreadable_unrelated_process_does_not_block_cleanup(self, fake_tmpdir):
        import tools.browser_tool as bt

        session, socket_dir, _ = self._session(fake_tmpdir)

        class _UnrelatedProc:
            info = {"name": "sleep", "cmdline": None}

        with patch("psutil.process_iter", return_value=[_UnrelatedProc()]):
            assert bt._remove_browser_socket_dir_if_safe(
                str(socket_dir), session
            ) is True

        assert not socket_dir.exists()

    def test_global_scan_requires_dead_owner_metadata(self, fake_tmpdir):
        import tools.browser_tool as bt

        session, socket_dir, _ = self._session(fake_tmpdir)
        calls = []
        with patch("gateway.status._pid_exists", return_value=False), \
             patch.object(bt, "_reap_session_chromium",
                          side_effect=lambda *a, **k: calls.append((a, k)) or 0):
            bt._reap_orphaned_browser_chromes(min_age_seconds=120)
        assert calls == [((str(socket_dir), session), {"min_age_seconds": 120})]

    def test_global_scan_preserves_live_owner(self, fake_tmpdir):
        import tools.browser_tool as bt

        self._session(fake_tmpdir)
        with patch("gateway.status._pid_exists", return_value=True), \
             patch.object(bt, "_reap_session_chromium") as reap:
            bt._reap_orphaned_browser_chromes(min_age_seconds=120)
        reap.assert_not_called()

    def test_dead_daemon_cleanup_preserves_socket_for_bound_young_chromium(
        self, fake_tmpdir
    ):
        """The later daemon sweep may not erase a profile Chrome preserved."""
        import tools.browser_tool as bt

        session, socket_dir, profile = self._session(fake_tmpdir)
        daemon_pid = 4242
        (socket_dir / f"{session}.pid").write_text(str(daemon_pid))
        proc = self._chrome(profile, age=30)
        terminated = []
        liveness_checks = []

        def pid_exists(pid):
            liveness_checks.append(pid)
            return False  # both the Hermes owner and daemon are dead

        p1, p2, p3, p4 = self._process_patches(proc, terminated)
        with p1, p2, p3, p4, \
             patch("gateway.status._pid_exists", side_effect=pid_exists):
            bt._reap_orphaned_browser_chromes(min_age_seconds=120)
            bt._reap_orphaned_browser_sessions()

        assert terminated == []
        assert liveness_checks == [999999, 999999, daemon_pid]
        assert socket_dir.exists()
        assert profile.exists()

    def test_cleanup_thread_scans_chrome_before_daemon_with_timeout(self, monkeypatch):
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_cleanup_running", False)
        calls = []
        monkeypatch.setattr(
            bt, "_reap_orphaned_browser_chromes",
            lambda *, min_age_seconds: calls.append(("chrome", min_age_seconds)),
        )
        monkeypatch.setattr(
            bt, "_reap_orphaned_browser_sessions",
            lambda: calls.append(("daemon", None)),
        )
        bt._browser_cleanup_thread_worker()
        assert calls == [
            ("chrome", bt.BROWSER_SESSION_INACTIVITY_TIMEOUT),
            ("daemon", None),
        ]


class TestEmergencyCleanupRunsReaper:
    """Verify atexit-registered cleanup sweeps orphans even without an active session."""

    def test_emergency_cleanup_calls_reaper(self, fake_tmpdir, monkeypatch):
        """_emergency_cleanup_all_sessions must call _reap_orphaned_browser_sessions."""
        import tools.browser_tool as bt

        # Reset the _cleanup_done flag so the cleanup actually runs
        monkeypatch.setattr(bt, "_cleanup_done", False)

        daemon_reaper_called = []
        chrome_reaper_called = []
        monkeypatch.setattr(
            bt,
            "_reap_orphaned_browser_sessions",
            lambda: daemon_reaper_called.append(True),
        )
        monkeypatch.setattr(
            bt,
            "_reap_orphaned_browser_chromes",
            lambda *, min_age_seconds: chrome_reaper_called.append(
                min_age_seconds
            ),
        )

        # No active sessions — both isolated reaper hooks should still run.
        bt._emergency_cleanup_all_sessions()

        assert daemon_reaper_called
        assert chrome_reaper_called == [bt.BROWSER_SESSION_INACTIVITY_TIMEOUT]
