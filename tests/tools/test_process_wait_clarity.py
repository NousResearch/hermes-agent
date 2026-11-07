"""Tests for process wait timeout-result clarity (not-an-error semantics)."""

import pytest

from tools.process_registry import ProcessRegistry

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

    def test_redirect_during_an_earlier_tool_releases_a_later_wait(self, registry):
        """The redirect's yield fans out to the tids running at that moment; a tool that ignores
        it (execute_code) exits and drops the bit, so the batch's next tool (a process wait) used
        to park the user's message for its whole window (codex#55644 class). The pending steer
        now carries the yield to later workers until it is drained."""
        import threading
        import time

        from agent.interrupt_control import InterruptControlMixin
        from agent.tool_executor import _registered_tool_worker
        from tests.waiting import wait_until

        class _Agent(InterruptControlMixin):
            _executing_tools = True
            _interrupt_requested = False
            _pending_steer = None
            _pending_redirect = None
            api_mode = "chat_completions"

            def __init__(self):
                self._pending_steer_lock = threading.Lock()
                self._pending_redirect_lock = threading.Lock()
                self._tool_worker_threads = set()
                self._tool_worker_threads_lock = threading.Lock()
                self._execution_thread_id = None

        agent = _Agent()
        sid = _spawn_sleeper(registry)

        def run_worker(body):
            out = {}

            def target():
                with _registered_tool_worker(agent):
                    out["r"] = body()

            t = threading.Thread(target=target)
            t.start()
            return t, out

        try:
            redirected = threading.Event()
            earlier, _ = run_worker(lambda: redirected.wait(5))  # a tool that never consumes yields
            wait_until(lambda: agent._tool_worker_threads, timeout=5)
            assert agent.redirect("stop waiting, answer me") is True
            redirected.set()
            earlier.join(5)

            t0 = time.monotonic()
            later, out = run_worker(lambda: registry.wait(sid, timeout=15))
            later.join(5)
            assert not later.is_alive(), "later wait parked the redirect"
            assert out["r"]["status"] == "interrupted" and time.monotonic() - t0 < 5

            # Once the steer is drained, the next batch's waits are not released by it.
            assert agent._drain_pending_steer() == "stop waiting, answer me"
            after, out = run_worker(lambda: registry.wait(sid, timeout=1))
            after.join(5)
            assert out["r"]["status"] == "timeout"
        finally:
            registry.kill_process(sid)
