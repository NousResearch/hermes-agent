"""Tests for _reap_orphaned_browser_sessions() — kills orphaned agent-browser
daemons whose Python parent exited without cleaning up."""

import os
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from tools import browser_tool_lifecycle as bt_lifecycle


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


def _killable(*pids):
    """``_pid_exists`` / ``_terminate_host_pid`` fakes where each PID stays alive until terminated."""
    alive = set(pids)
    calls = []

    def terminate(pid, expected_start=None):
        calls.append(pid)
        alive.discard(pid)

    return (lambda pid: pid in alive), terminate, calls


class TestReapOrphanedBrowserSessions:
    """Tests for the orphan reaper function."""


    def test_stale_dir_without_pid_file_is_removed(self, fake_tmpdir):
        """Socket dir with no PID file is cleaned up."""
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions
        d = _make_socket_dir(fake_tmpdir, "h_abc1234567")
        assert d.exists()
        with patch(
            "tools.browser_tool_lifecycle._socket_dir_idle_seconds",
            return_value=10_000,
        ):
            _reap_orphaned_browser_sessions()
        assert not d.exists()

    def test_fresh_dir_without_pid_file_survives_creator_race(self, fake_tmpdir):
        """A concurrent reaper must not delete a session still starting."""
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "h_starting1234")
        with patch(
            "tools.browser_tool_lifecycle._socket_dir_idle_seconds",
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
        dir once the daemon is read back as gone.
        """
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "h_perm1234567", pid=12345)
        pid_exists, terminate, terminate_calls = _killable(12345)

        with patch("gateway.status._pid_exists", side_effect=pid_exists), \
             patch("gateway.status.get_process_start_time", return_value=777), \
             patch("tools.browser_tool_lifecycle._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid", side_effect=terminate):
            _reap_orphaned_browser_sessions()

        assert 12345 in terminate_calls
        assert not d.exists()

    def test_real_profile_attach_daemon_is_reaped_when_owner_is_dead(self, fake_tmpdir):
        """#100855: the shared ``hermes-real-profile`` attach daemon is not ``<prefix>_<hex>``
        named, so the reaper's glob never saw it and a wedged daemon + headless Chrome outlived
        gateway restarts. Same owner-liveness rule as every other lane: dead owner => reaped."""
        import tools.browser_tool as bt
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, bt._REAL_PROFILE_SESSION, pid=4242, owner_pid=99999)
        pid_exists, terminate, terminate_calls = _killable(4242)  # daemon alive, owning hermes gone

        with patch("gateway.status._pid_exists", side_effect=pid_exists), \
             patch("gateway.status.get_process_start_time", return_value=777), \
             patch("tools.browser_tool_lifecycle._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid", side_effect=terminate):
            _reap_orphaned_browser_sessions()

        assert terminate_calls == [4242]
        assert not d.exists()

    def test_unfingerprintable_daemon_is_refused(self, fake_tmpdir):
        """No start-time fingerprint -> the kill is refused (fail closed).

        The reaper reads the PID from a world-writable temp dir; a PID whose
        identity cannot be pinned could be recycled between the verify and the
        tree-kill, so it must be left alone (and the socket dir kept for a
        later sweep).
        """
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

        _make_socket_dir(fake_tmpdir, "h_perm7654321", pid=12345)
        terminate_calls = []

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("gateway.status.get_process_start_time", return_value=None), \
             patch("tools.browser_tool_lifecycle._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=lambda pid, expected_start=None: terminate_calls.append(pid)):
            _reap_orphaned_browser_sessions()

        assert terminate_calls == []


    @pytest.mark.parametrize("content", [b"not-a-number", b"", b"\xff\xfe\x00", b"0", b"-1", b"2147483648", b"99999999999"],
                             ids=["non_integer", "empty", "undecodable", "zero", "negative", "pid_t_max_plus_one", "overflow"])
    @pytest.mark.parametrize("idle_s", [0.0, 10_000], ids=["fresh", "stale"])
    def test_corrupt_pid_file_is_retained(self, fake_tmpdir, content, idle_s):
        """Malformed, non-positive, or out-of-range PID data cannot prove the daemon is gone: the dir and
        PID file stay and nothing is signalled, even long past the grace window."""
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "h_corrupt1234")
        pid_file = d / "h_corrupt1234.pid"
        pid_file.write_bytes(content)
        terminate_calls = []

        with patch("tools.browser_tool_lifecycle._socket_dir_idle_seconds", return_value=idle_s), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=lambda pid, expected_start=None: terminate_calls.append(pid)):
            _reap_orphaned_browser_sessions()

        assert terminate_calls == []
        assert pid_file.read_bytes() == content

    def test_reaper_continues_after_one_directory_error(self, fake_tmpdir):
        """An uninspectable directory must stay fail-closed without starving later dirs."""
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

        _make_socket_dir(fake_tmpdir, "h_first1234")
        _make_socket_dir(fake_tmpdir, "h_second1234")
        calls = []

        def reap_one(socket_dir, session_name, tracked_names):
            calls.append(session_name)
            if len(calls) == 1:
                raise RuntimeError("unreadable process metadata")
            return False

        with patch("tools.browser_tool_lifecycle._best_effort", return_value=None), \
             patch("tools.browser_tool_lifecycle._reap_socket_dir", side_effect=reap_one):
            _reap_orphaned_browser_sessions()

        assert len(calls) == 2
        assert set(calls) == {"h_first1234", "h_second1234"}

    def test_dead_daemon_pid_dir_is_removed(self, fake_tmpdir):
        """A valid PID confirmed dead releases its dir without a kill."""
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

        d = _make_socket_dir(fake_tmpdir, "h_dead1234567", pid=12345)
        terminate_calls = []

        with patch("gateway.status._pid_exists", return_value=False), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=lambda pid, expected_start=None: terminate_calls.append(pid)):
            _reap_orphaned_browser_sessions()

        assert terminate_calls == []
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
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

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







class TestReaperIdentityGuard:
    """Tests for _verify_reapable_browser_daemon — the #14073 fix.

    The reaper reads daemon PIDs from world-writable, predictably-named temp
    dirs.  Before tree-killing a live PID it must confirm the process really is
    *this* session's agent-browser daemon, defeating planted pid files and
    recycled PIDs that would otherwise become an arbitrary same-user DoS.
    """

    @pytest.fixture(autouse=True)
    def _selected_binary(self, tmp_path):
        """PM's selected daemon binary (a real file, never executed) and the image a fake
        process runs by default — the identity the guard must verify."""
        self.selected = tmp_path / "pm" / "bin" / "agent-browser-linux-x64"
        self.selected.parent.mkdir(parents=True)
        self.selected.write_bytes(b"not a real binary")
        self.imposter = tmp_path / "python3"
        self.imposter.write_bytes(b"some other image")
        with patch("pm.installed_package",
                   side_effect=lambda name: SimpleNamespace(binary=self.selected) if name == "agent-browser" else None), \
             patch("pm.ensure", side_effect=AssertionError("identity check must never install")), \
             patch("tools.browser_tool_install._is_termux_environment", return_value=False):
            yield

    class _FakeProc:
        def __init__(self, name="agent-browser", cmdline=None, environ=None,
                     raise_environ=False, exe=None, raise_exe=False):
            self._name = name
            self._cmdline = cmdline if cmdline is not None else []
            self._environ = environ or {}
            self._raise_environ = raise_environ
            self._exe = exe
            self._raise_exe = raise_exe
            self._proc = SimpleNamespace(exe=self._native_exe)

        def name(self):
            return self._name

        def exe(self):
            if self._raise_exe:
                import psutil
                raise psutil.AccessDenied()
            return self._exe

        def _native_exe(self):
            return self.exe()

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
        from tools.browser_tool_lifecycle import _verify_reapable_browser_daemon

        if fake_proc._exe is None:
            fake_proc._exe = str(self.selected)  # runs PM's selected binary unless a test says otherwise

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

    def test_chrome_argument_with_our_socket_dir_is_refused(self):
        """Socket-dir arguments must not double as executable identity.

        Ordinary Chrome can have a user-data-dir under a path whose predictable
        basename contains ``agent-browser-<session>``.  That proves path binding,
        but it must never establish that the process is agent-browser.
        """
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(
            name="chrome",
            cmdline=["/usr/bin/chrome", f"--user-data-dir={socket_dir}"],
            exe=str(self.imposter),
        )
        assert self._run(proc, socket_dir) is False

    _AGENT_BROWSER_NAMES = pytest.mark.parametrize("name,argv0", [
        ("agent-browser", "agent-browser"),
        ("agent-browser-l", "/opt/npm/node_modules/agent-browser/bin/agent-browser-linux-x64"),
        ("agent-browser-linux-musl-arm64", "agent-browser-linux-musl-arm64"),
        ("agent-browser-darwin-arm64", "/usr/local/lib/node_modules/agent-browser/bin/agent-browser-darwin-arm64"),
        ("agent-browser-win32-x64.exe", r"C:\npm\node_modules\agent-browser\bin\agent-browser-win32-x64.exe"),
    ])

    @_AGENT_BROWSER_NAMES
    def test_process_running_the_selected_binary_is_reapable(self, name, argv0):
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(name=name, cmdline=[argv0, "daemon"],
                              environ={"AGENT_BROWSER_SOCKET_DIR": socket_dir})
        assert self._run(proc, socket_dir) is True

    @_AGENT_BROWSER_NAMES
    def test_agent_browser_name_or_argv0_on_another_image_is_refused(self, name, argv0):
        """Process name and argv[0] are caller-chosen: a Python (or any) process can claim
        ``agent-browser`` while its real image is something else entirely."""
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(name=name, cmdline=[argv0, "daemon"], exe=str(self.imposter),
                              environ={"AGENT_BROWSER_SOCKET_DIR": socket_dir})
        assert self._run(proc, socket_dir) is False

    def test_selected_binary_reached_through_a_symlink_is_reapable(self, tmp_path):
        link = tmp_path / "agent-browser"
        link.symlink_to(self.selected)
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(cmdline=["agent-browser", "daemon"], exe=str(link),
                              environ={"AGENT_BROWSER_SOCKET_DIR": socket_dir})
        assert self._run(proc, socket_dir) is True

    def test_external_shim_that_is_not_the_daemon_image_is_refused(self, tmp_path):
        """A PATH install selects a launcher shim; the daemon runs a different native binary
        that cannot be proven to be the selected one — fail closed."""
        shim = tmp_path / "npm" / "bin" / "agent-browser"
        shim.parent.mkdir(parents=True)
        shim.write_text("#!/usr/bin/env node\n")
        shim.chmod(0o755)
        native = tmp_path / "npm" / "lib" / "agent-browser-linux-x64"
        native.parent.mkdir(parents=True)
        native.write_bytes(b"native")
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(name="agent-browser-linux-x64", cmdline=[str(native), "daemon"], exe=str(native),
                              environ={"AGENT_BROWSER_SOCKET_DIR": socket_dir})
        with patch("pm.installed_package", return_value=None), \
             patch("tools.browser_tool_install.shutil.which", return_value=str(shim)):
            assert self._run(proc, socket_dir) is False

    def test_no_selected_binary_is_refused_without_installing(self):
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(cmdline=["agent-browser", f"--socket-dir={socket_dir}"])
        ensure = Mock()
        with patch("pm.installed_package", return_value=None), \
             patch("tools.browser_tool_install.shutil.which", return_value=None), \
             patch("pm.ensure", ensure):
            assert self._run(proc, socket_dir) is False
        ensure.assert_not_called()

    def test_unreadable_process_image_is_refused(self):
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(cmdline=["agent-browser", f"--socket-dir={socket_dir}"], raise_exe=True)
        assert self._run(proc, socket_dir) is False

    def test_psutil_argv0_fallback_is_not_treated_as_process_image(self):
        """psutil's public exe() guesses from an executable absolute argv[0] when the native
        image query is denied. That caller-controlled guess cannot authorize daemon cleanup."""
        import psutil
        from tools.browser_tool_lifecycle import _verify_reapable_browser_daemon

        socket_dir = "/tmp/agent-browser-h_sess123456"
        self.selected.chmod(0o755)  # psutil's fallback checks only isfile + X_OK; it never executes it
        pid = 12345

        def denied_native_exe():
            raise psutil.AccessDenied(pid=pid)

        native_proc = SimpleNamespace(
            exe=denied_native_exe,
            name=lambda: "agent-browser",
            cmdline=lambda: [str(self.selected), "--socket-dir", socket_dir],
            environ=lambda: {"AGENT_BROWSER_SOCKET_DIR": socket_dir},
        )
        wrapped_proc = psutil.Process(os.getpid())
        object.__setattr__(wrapped_proc, "_proc", native_proc)
        object.__setattr__(wrapped_proc, "_exe", None)

        # Exercise the installed psutil public wrapper: it incorrectly reports argv[0] as exe.
        assert wrapped_proc.exe() == str(self.selected)
        with patch("psutil.Process", return_value=wrapped_proc):
            assert _verify_reapable_browser_daemon(pid, socket_dir, "h_sess123456") is False

    def test_pm_binary_lookup_error_keeps_daemon_state(self, fake_tmpdir):
        """A broken PM lookup fails closed without propagating or deleting session state."""
        session_name = "h_lookup1234"
        d = _make_socket_dir(fake_tmpdir, session_name, pid=12345)
        pid_file = d / f"{session_name}.pid"
        socket_dir = str(d)
        proc = self._FakeProc(
            cmdline=["agent-browser", "daemon"],
            environ={"AGENT_BROWSER_SOCKET_DIR": socket_dir},
            exe=str(self.selected),
        )

        with patch("gateway.status._pid_exists", return_value=True), patch(
            "gateway.status.get_process_start_time", return_value=777
        ):
            with patch("psutil.Process", return_value=proc), patch(
                "tools.browser_tool_install._find_agent_browser",
                side_effect=RuntimeError("PM metadata is unreadable"),
            ), patch(
                "tools.process_registry.ProcessRegistry._terminate_host_pid"
            ) as terminate:
                stopped = bt_lifecycle._stop_session_daemon(socket_dir, session_name)

        assert stopped is False
        assert d.exists()
        assert pid_file.read_text() == "12345"
        terminate.assert_not_called()

    def test_process_registry_error_keeps_daemon_state(self, fake_tmpdir):
        """Unexpected termination API errors are unconfirmed and preserve the PID/socket state."""
        session_name = "h_registry1234"
        d = _make_socket_dir(fake_tmpdir, session_name, pid=12345)
        pid_file = d / f"{session_name}.pid"
        socket_dir = str(d)
        proc = self._FakeProc(
            cmdline=["agent-browser", "daemon"],
            environ={"AGENT_BROWSER_SOCKET_DIR": socket_dir},
            exe=str(self.selected),
        )

        with patch("gateway.status._pid_exists", return_value=True), patch(
            "gateway.status.get_process_start_time", return_value=777
        ):
            with patch("psutil.Process", return_value=proc), patch(
                "tools.process_registry.ProcessRegistry._terminate_host_pid",
                side_effect=RuntimeError("termination API failed"),
            ) as terminate:
                stopped = bt_lifecycle._stop_session_daemon(socket_dir, session_name)

        assert stopped is False
        assert d.exists()
        assert pid_file.read_text() == "12345"
        terminate.assert_called_once_with(12345, 777)

    def test_unexpected_verification_error_keeps_daemon_state(self, fake_tmpdir):
        """Unexpected stop-verification errors must not escape or erase the retry state."""
        session_name = "h_verifyerr1234"
        d = _make_socket_dir(fake_tmpdir, session_name, pid=12345)
        pid_file = d / f"{session_name}.pid"
        socket_dir = str(d)

        with patch("gateway.status._pid_exists", return_value=True), patch(
            "tools.browser_tool_lifecycle._verified_daemon_start",
            side_effect=RuntimeError("verification failed"),
        ):
            stopped = bt_lifecycle._stop_session_daemon(socket_dir, session_name)

        assert stopped is False
        assert d.exists()
        assert pid_file.read_text() == "12345"

    def test_agent_browser_only_in_later_argv_is_refused(self):
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(name="python3", exe=str(self.imposter),
                              cmdline=["python3", "/tmp/agent-browser", f"--socket-dir={socket_dir}"])
        assert self._run(proc, socket_dir) is False

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

    def test_recycled_pid_carrying_only_socket_dir_basename_is_refused(self):
        """The socket-dir BASENAME anywhere in argv is not a binding (#116884).

        `agent-browser-<session>` is predictable, so a recycled PID whose argv merely
        mentions it (a grep, a shell) must not pass the binding gate; only the full
        normalized path as an argv token (or the environ match) binds.
        """
        socket_dir = "/tmp/agent-browser-h_sess123456"
        proc = self._FakeProc(
            name="bash",
            cmdline=["grep", "agent-browser-h_sess123456", "/var/log/syslog"],
            environ={},
        )
        assert self._run(proc, socket_dir) is False
        # Control: the full path as a `--flag=value` token still binds.
        bound = self._FakeProc(
            name="agent-browser",
            cmdline=["agent-browser", "daemon", f"--socket-dir={socket_dir}/"],
        )
        assert self._run(bound, socket_dir) is True


    def test_planted_pid_survives_full_reaper_path(self, fake_tmpdir):
        """End-to-end through the reaper: a planted non-browser PID is spared.

        No owner_pid (legacy path), not tracked, PID 'alive' — but the live
        process is `sleep`, not agent-browser, so it must be left alone and the
        socket dir retained.
        """
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

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




def _age_socket_dir(d, seconds):
    """Backdate every mtime under ``d`` so it looks idle for ``seconds``."""
    old = time.time() - seconds
    for p in d.iterdir():
        os.utime(p, (old, old))
    os.utime(d, (old, old))


class TestSocketDirIdleSeconds:
    """Unit tests for the idle-age signal backing the leak escape hatch."""

    def test_missing_dir_returns_none(self, tmp_path):
        from tools.browser_tool_lifecycle import _socket_dir_idle_seconds
        assert _socket_dir_idle_seconds(str(tmp_path / "nope")) is None


    def test_entry_mtime_beats_stale_dir_mtime(self, tmp_path):
        """Rewriting an existing file must count as activity.

        Command names repeat (``_stdout_click`` is rewritten on every click),
        and overwriting an existing file does NOT bump the *directory* mtime.
        Reading only the directory mtime would therefore report a busy session
        as idle and reap it.  The reaper must scan entries too.
        """
        from tools.browser_tool_lifecycle import _socket_dir_idle_seconds
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
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

        d = _make_socket_dir(
            fake_tmpdir, "h_fresh_owner", pid=12345, owner_pid=os.getpid()
        )
        kill_calls = []

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.browser_tool_lifecycle._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=kill_calls.append):
            _reap_orphaned_browser_sessions()

        assert 12345 not in kill_calls
        assert d.exists()

    def test_idle_untracked_daemon_with_live_owner_is_reaped(self, fake_tmpdir):
        """Past the grace window, an untracked daemon is treated as leaked."""
        from tools.browser_tool import BROWSER_ORPHAN_GRACE_SECONDS
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

        d = _make_socket_dir(
            fake_tmpdir, "h_leaked_owner", pid=12345, owner_pid=os.getpid()
        )
        _age_socket_dir(d, BROWSER_ORPHAN_GRACE_SECONDS + 600)
        pid_exists, terminate, kill_calls = _killable(12345, os.getpid())

        with patch("gateway.status._pid_exists", side_effect=pid_exists), \
             patch("gateway.status.get_process_start_time", return_value=777), \
             patch("tools.browser_tool_lifecycle._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid", side_effect=terminate):
            _reap_orphaned_browser_sessions()

        assert 12345 in kill_calls
        assert not d.exists()

    def test_tracked_daemon_with_live_owner_is_spared_at_any_age(self, fake_tmpdir):
        """A session this process still tracks is never reaped, however old.

        Idle age is a fallback for *lost* bookkeeping, not an override of
        bookkeeping that is present and says the session is live.
        """
        import tools.browser_tool as bt
        from tools.browser_tool import BROWSER_ORPHAN_GRACE_SECONDS
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

        d = _make_socket_dir(
            fake_tmpdir, "h_tracked_old", pid=12345, owner_pid=os.getpid()
        )
        _age_socket_dir(d, BROWSER_ORPHAN_GRACE_SECONDS * 10)
        bt._active_sessions["task-1"] = {"session_name": "h_tracked_old"}
        kill_calls = []

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.browser_tool_lifecycle._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=kill_calls.append):
            _reap_orphaned_browser_sessions()

        assert 12345 not in kill_calls
        assert d.exists()

    def test_unknown_idle_age_fails_safe(self, fake_tmpdir):
        """Unreadable mtime => treat as too young to reap, never guess."""
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

        d = _make_socket_dir(
            fake_tmpdir, "h_unknown_age", pid=12345, owner_pid=os.getpid()
        )
        kill_calls = []

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.browser_tool_lifecycle._socket_dir_idle_seconds", return_value=None), \
             patch("tools.browser_tool_lifecycle._verify_reapable_browser_daemon", return_value=True), \
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
        from tools.browser_tool import BROWSER_ORPHAN_GRACE_SECONDS
        from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

        d = _make_socket_dir(
            fake_tmpdir, "h_unverified", pid=12345, owner_pid=os.getpid()
        )
        _age_socket_dir(d, BROWSER_ORPHAN_GRACE_SECONDS + 600)
        kill_calls = []

        with patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.browser_tool_lifecycle._verify_reapable_browser_daemon", return_value=False), \
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
            with patch("tools.browser_tool_lifecycle._reap_orphaned_browser_sessions",
                       side_effect=lambda: reap_calls.append(1)), \
                 patch("tools.browser_tool_lifecycle._cleanup_inactive_browser_sessions",
                       side_effect=fake_cleanup), \
                 patch("tools.browser_tool.time.sleep"):
                bt_lifecycle._browser_cleanup_thread_worker()
        finally:
            bt._cleanup_running = orig_running

        assert len(reap_calls) > 1, "startup-only reap would give exactly 1"
