"""Tests for process wait timeout-result clarity (not-an-error semantics)."""

import time

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession

@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return ProcessRegistry()

def _spawn_sleeper(registry, notify=False):
    session = registry.spawn_local("sleep 30", cwd="/tmp", task_id="t-waitclar")
    session.notify_on_complete = notify
    return session.id

class TestWaitTimeoutClarity:
    def test_wait_timeout_marks_process_running(self, registry):
        sid = _spawn_sleeper(registry)
        try:
            r = registry.wait(sid, timeout=1)
            assert r["status"] == "timeout"
            assert r["process_running"] is True
        finally:
            registry.kill_process(sid)

    def test_clamped_wait_keeps_clamp_note_and_running_semantics(self, registry, monkeypatch):
        monkeypatch.setenv("TERMINAL_TIMEOUT", "1")
        sid = _spawn_sleeper(registry)
        try:
            r = registry.wait(sid, timeout=600)
            assert r["status"] == "timeout"
            assert r["process_running"] is True
        finally:
            registry.kill_process(sid)

    def test_exited_process_unaffected(self, registry):
        session = registry.spawn_local("true", cwd="/tmp", task_id="t-waitclar")
        r = registry.wait(session.id, timeout=10)
        assert r["status"] == "exited"
        assert "process_running" not in r

class TestWaitYieldRelease:
    """A mid-turn steer/redirect (request_yield on the tool-worker tid) releases a
    process wait instead of parking the user's message behind it (kimi-code#3697 class)."""

    def test_yield_releases_wait_and_keeps_process_running(self, registry):
        import threading
        import time

        from tools.interrupt import request_yield

        sid = _spawn_sleeper(registry)
        try:
            result = {}

            def waiter():
                result["r"] = registry.wait(sid, timeout=15)

            t = threading.Thread(target=waiter)
            t.start()
            time.sleep(0.3)
            request_yield(t.ident)
            t.join(5)
            assert not t.is_alive(), "wait did not release within 5s of the yield request"
            r = result["r"]
            assert r["status"] == "interrupted"
            assert r["process_running"] is True
            # The process was not killed and the yield bit was consumed.
            assert registry.poll(sid)["status"] == "running"
            from tools.interrupt import is_thread_yield_requested
            assert not is_thread_yield_requested(t.ident)
        finally:
            registry.kill_process(sid)


class TestPtyWaitReconcile:
    """wait()/poll() must reconcile a PTY session whose reader never observes
    the exit instead of spinning to the deadline (#137437: on Windows ConPTY
    the reader's ``pty.isalive()`` stayed satisfied long after the child was
    gone, so five consecutive waits each burned the full 420s tool ceiling)."""

    class _StuckPty:
        """A PTY whose liveness probe stays satisfied after the child died."""

        exitstatus = None

        def __init__(self):
            self.closed = False

        def isalive(self):
            return True

        def close(self):
            self.closed = True

    def _pty_session(self, registry, sid="proc_pty_stuck"):
        session = ProcessSession(
            id=sid, command="ssh host 'timeout 1800 true'", task_id="t-ptyrec",
            started_at=time.time())
        session.pid = 4194304
        session.host_start_time = 111
        session._pty = self._StuckPty()
        registry._running[session.id] = session
        return session

    def test_wait_returns_exited_when_os_reports_pid_gone(self, registry, monkeypatch):
        monkeypatch.setattr(
            ProcessRegistry, "_is_host_pid_alive", staticmethod(lambda pid: False))
        session = self._pty_session(registry)
        r = registry.wait(session.id, timeout=5)
        assert r["status"] == "exited"
        # The reader never reaped an exit status: unknown, not success.
        assert r["exit_code"] == -1
        assert session._completion_event.is_set()
        assert session._pty.closed

    def test_wait_exits_when_pid_was_recycled_onto_a_stranger(self, registry, monkeypatch):
        monkeypatch.setattr(
            ProcessRegistry, "_is_host_pid_alive", staticmethod(lambda pid: True))
        monkeypatch.setattr(
            ProcessRegistry, "_safe_host_start_time", staticmethod(lambda pid: 222))
        session = self._pty_session(registry)
        r = registry.wait(session.id, timeout=5)
        assert r["status"] == "exited"

    def test_wait_keeps_waiting_while_pid_is_alive_and_ours(self, registry, monkeypatch):
        monkeypatch.setattr(
            ProcessRegistry, "_is_host_pid_alive", staticmethod(lambda pid: True))
        monkeypatch.setattr(
            ProcessRegistry, "_safe_host_start_time", staticmethod(lambda pid: 111))
        session = self._pty_session(registry)
        r = registry.wait(session.id, timeout=1)
        assert r["status"] == "timeout"
        assert r["process_running"] is True
        assert session.exited is False

    def test_pty_exitstatus_is_used_when_present(self, registry, monkeypatch):
        monkeypatch.setattr(
            ProcessRegistry, "_is_host_pid_alive", staticmethod(lambda pid: False))
        session = self._pty_session(registry)
        session._pty.exitstatus = 3
        r = registry.poll(session.id)
        assert r["status"] == "exited"
        assert r["exit_code"] == 3
