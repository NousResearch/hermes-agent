"""Browser daemon teardown binds the verified identity through to the kill and drops
the socket dir (the orphan reaper's retry record) only once the daemon is read back as gone."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.browser_tool as bt
from tools import browser_tool_lifecycle as lifecycle
from tools import browser_tool_session as bt_session
from tools.process_registry import ProcessRegistry

SESSION = "h_reaptest123"
PID = 12345
DISCARD_TASK = "task-timeout"


def _reap(socket_dir):
    return lifecycle._reap_socket_dir(str(socket_dir), SESSION, set())


def _release(socket_dir):
    lifecycle._release_session_resources("task-reap", {"session_name": SESSION, "bb_session_id": None})


def _discard(socket_dir):
    session_info = {"session_name": SESSION, "bb_session_id": None, "cdp_url": None}
    bt._active_sessions[DISCARD_TASK] = session_info
    bt_session._discard_timed_out_browser_session(DISCARD_TASK, session_info, str(socket_dir))
    assert DISCARD_TASK not in bt._active_sessions  # evicted whether or not the stop is confirmed


ENTRYPOINTS = pytest.mark.parametrize(
    "teardown", [_reap, _release, _discard], ids=["orphan_reaper", "session_release", "timeout_discard"])


@pytest.fixture
def socket_dir(tmp_path):
    d = tmp_path / f"agent-browser-{SESSION}"
    d.mkdir()
    (d / f"{SESSION}.pid").write_text(str(PID))
    with patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
         patch("tools.browser_tool_cdp._stop_cdp_supervisor", lambda _tid: None):
        yield d
    bt._active_sessions.pop(DISCARD_TASK, None)
    bt._session_last_activity.pop(DISCARD_TASK, None)


class _FakeDaemon:
    """One PID whose start time can change (recycle) and that dies only if termination works."""

    def __init__(self, start=777, dies_on_terminate=True):
        self.alive = True
        self.start = start
        self.dies_on_terminate = dies_on_terminate
        self.terminate_calls = []

    def pid_exists(self, pid):
        return pid == PID and self.alive

    def start_time(self, pid):
        return self.start if self.pid_exists(pid) else None

    def terminate(self, pid, expected_start=None):
        self.terminate_calls.append((pid, expected_start))
        if self.dies_on_terminate:
            self.alive = False

    def patches(self, verify=lambda *_a: True):
        return (
            patch("gateway.status._pid_exists", side_effect=self.pid_exists),
            patch("gateway.status.get_process_start_time", side_effect=self.start_time),
            patch("tools.browser_tool_lifecycle._verify_reapable_browser_daemon", side_effect=verify),
            patch("tools.process_registry.ProcessRegistry._terminate_host_pid", side_effect=self.terminate),
            patch.object(ProcessRegistry, "_KILL_SETTLE_SECONDS", 0.0),
        )


def _run(daemon, teardown, socket_dir, **kw):
    p1, p2, p3, p4, p5 = daemon.patches(**kw)
    with p1, p2, p3, p4, p5:
        return teardown(socket_dir)


class TestIdentityBinding:
    @ENTRYPOINTS
    def test_pid_recycled_right_after_verification_is_never_signalled(self, socket_dir, teardown):
        daemon = _FakeDaemon(start=777)

        def verify_then_recycle(*_a):
            daemon.start = 888  # the verified daemon exits; its PID now names a stranger
            return True

        _run(daemon, teardown, socket_dir, verify=verify_then_recycle)

        assert daemon.terminate_calls == []
        assert (socket_dir / f"{SESSION}.pid").exists()

    @ENTRYPOINTS
    def test_kill_carries_the_fingerprint_that_was_verified(self, socket_dir, teardown):
        daemon = _FakeDaemon(start=777)

        _run(daemon, teardown, socket_dir)

        assert daemon.terminate_calls == [(PID, 777)]


class TestUnconfirmedTermination:
    @ENTRYPOINTS
    def test_noop_termination_keeps_the_retry_record(self, socket_dir, teardown):
        daemon = _FakeDaemon(dies_on_terminate=False)

        result = _run(daemon, teardown, socket_dir)

        assert daemon.terminate_calls == [(PID, 777)]
        assert (socket_dir / f"{SESSION}.pid").read_text() == str(PID)
        if teardown is _reap:
            assert result is False  # not counted as reaped

    @ENTRYPOINTS
    def test_pid_reused_after_termination_keeps_the_retry_record(self, socket_dir, teardown):
        """The daemon dies and a stranger immediately takes its PID: a live PID with a new start
        time is not proof that the PID is gone, so the stop stays unconfirmed."""
        daemon = _FakeDaemon(start=777)

        def terminate_then_recycle(pid, expected_start=None):
            daemon.terminate_calls.append((pid, expected_start))
            daemon.start = 888

        daemon.terminate = terminate_then_recycle
        result = _run(daemon, teardown, socket_dir)

        assert daemon.terminate_calls == [(PID, 777)]
        assert (socket_dir / f"{SESSION}.pid").read_text() == str(PID)
        if teardown is _reap:
            assert result is False

    @pytest.mark.parametrize("current_start", [888, None], ids=["recycled", "unreadable"])
    def test_live_pid_is_running_whatever_its_start_time(self, current_start):
        daemon = _FakeDaemon(start=current_start)
        with patch("gateway.status._pid_exists", side_effect=daemon.pid_exists), \
             patch("gateway.status.get_process_start_time", side_effect=lambda _pid: current_start):
            assert lifecycle._daemon_still_running(PID) is True

    def test_unreadable_start_time_after_kill_counts_as_alive(self, socket_dir):
        daemon = _FakeDaemon(dies_on_terminate=False)

        def terminate_then_hide_identity(pid, expected_start=None):
            daemon.terminate_calls.append((pid, expected_start))
            daemon.start_time = lambda _pid: None

        with patch("gateway.status._pid_exists", side_effect=daemon.pid_exists), \
             patch("gateway.status.get_process_start_time", side_effect=lambda pid: daemon.start_time(pid)), \
             patch("tools.browser_tool_lifecycle._verify_reapable_browser_daemon", return_value=True), \
             patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
                   side_effect=terminate_then_hide_identity), \
             patch.object(ProcessRegistry, "_KILL_SETTLE_SECONDS", 0.0):
            assert _reap(socket_dir) is False

        assert socket_dir.exists()


class TestConfirmedTermination:
    @ENTRYPOINTS
    def test_confirmed_stop_removes_the_socket_dir(self, socket_dir, teardown):
        daemon = _FakeDaemon()

        result = _run(daemon, teardown, socket_dir)

        assert not daemon.alive
        assert not socket_dir.exists()
        if teardown is _reap:
            assert result is True

    def test_daemon_already_closed_politely_is_released_without_a_kill(self, socket_dir):
        """A responsive local session exits on ``close``; release just removes its dir."""
        daemon = _FakeDaemon()
        daemon.alive = False

        _run(daemon, _release, socket_dir)

        assert daemon.terminate_calls == []
        assert not socket_dir.exists()


class TestPidFileRecord:
    @pytest.mark.parametrize("teardown", [_release, _discard], ids=["session_release", "timeout_discard"])
    def test_corrupt_pid_file_keeps_the_dir_and_signals_nothing(self, socket_dir, teardown):
        (socket_dir / f"{SESSION}.pid").write_text("not-a-pid")
        daemon = _FakeDaemon()

        _run(daemon, teardown, socket_dir)

        assert daemon.terminate_calls == []
        assert (socket_dir / f"{SESSION}.pid").read_text() == "not-a-pid"

    @pytest.mark.parametrize("teardown", [_release, _discard], ids=["session_release", "timeout_discard"])
    def test_absent_pid_file_keeps_the_dir_without_a_kill(self, socket_dir, teardown):
        """An absent PID is not evidence that an in-flight daemon has exited."""
        (socket_dir / f"{SESSION}.pid").unlink()
        daemon = _FakeDaemon()

        _run(daemon, teardown, socket_dir)

        assert daemon.terminate_calls == []
        assert socket_dir.exists()
        assert not (socket_dir / f"{SESSION}.pid").exists()


def _select_daemon_binary(binary):
    """PM selects ``binary`` as the agent-browser daemon; installing is never allowed."""
    return (
        patch("pm.installed_package",
              side_effect=lambda name: SimpleNamespace(binary=Path(binary)) if name == "agent-browser" else None),
        patch("pm.ensure", side_effect=AssertionError("teardown must never install agent-browser")),
        patch("tools.browser_tool_install._is_termux_environment", return_value=False),
    )


@pytest.mark.platforms("posix")
class TestRealChildProcess:
    """End to end against a harmless Python child posing as the session's daemon (no browser)."""

    @pytest.fixture(autouse=True)
    def _python_is_the_selected_daemon(self):
        # The child's real image is this interpreter, so selecting it makes the child a genuine daemon.
        p1, p2, p3 = _select_daemon_binary(sys.executable)
        with p1, p2, p3:
            yield

    @staticmethod
    def _spawn(socket_dir, code):
        # argv[0] claims agent-browser; identity comes from the real image (the interpreter).
        child = subprocess.Popen(
            ["agent-browser", "-c", code, str(socket_dir)], executable=sys.executable,
            stdout=subprocess.PIPE, text=True)
        assert child.stdout.readline().strip() == "ready"
        (socket_dir / f"{SESSION}.pid").write_text(str(child.pid))
        return child

    def test_reaper_stops_a_verified_daemon_and_removes_its_dir(self, socket_dir):
        child = self._spawn(socket_dir, "import time; print('ready', flush=True); time.sleep(60)")
        try:
            assert _reap(socket_dir) is True
            assert child.wait(timeout=10) is not None
            assert not socket_dir.exists()
        finally:
            child.kill()
            child.wait()

    def test_reaper_keeps_the_dir_when_the_daemon_survives(self, socket_dir):
        child = self._spawn(
            socket_dir,
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(60)")
        try:
            # Zero grace: SIGTERM only, no SIGKILL escalation — the child ignores it and lives.
            with patch.object(ProcessRegistry, "_daemon_term_grace_seconds", return_value=0.0), \
                 patch.object(ProcessRegistry, "_KILL_SETTLE_SECONDS", 0.2):
                assert _reap(socket_dir) is False
            assert child.poll() is None
            assert (socket_dir / f"{SESSION}.pid").read_text() == str(child.pid)
        finally:
            child.kill()
            child.wait()

    def test_timeout_discard_keeps_the_dir_when_the_daemon_survives(self, socket_dir):
        child = self._spawn(
            socket_dir,
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(60)")
        try:
            with patch.object(ProcessRegistry, "_daemon_term_grace_seconds", return_value=0.0), \
                 patch.object(ProcessRegistry, "_KILL_SETTLE_SECONDS", 0.2):
                _discard(socket_dir)
            assert child.poll() is None
            assert (socket_dir / f"{SESSION}.pid").read_text() == str(child.pid)
        finally:
            child.kill()
            child.wait()

    @ENTRYPOINTS
    def test_child_claiming_agent_browser_on_another_image_is_never_signalled(self, socket_dir, tmp_path, teardown):
        """argv[0]=agent-browser and an exact socket-dir binding, but the real image is not the
        selected daemon binary: nothing is signalled and the retry record stays."""
        selected = tmp_path / "agent-browser-linux-x64"
        selected.write_bytes(b"selected daemon image, never executed")
        child = self._spawn(socket_dir, "import time; print('ready', flush=True); time.sleep(60)")
        try:
            p1, p2, p3 = _select_daemon_binary(selected)
            with p1, p2, p3:
                result = teardown(socket_dir)
            assert child.poll() is None
            assert (socket_dir / f"{SESSION}.pid").read_text() == str(child.pid)
            if teardown is _reap:
                assert result is False
        finally:
            child.kill()
            child.wait()
