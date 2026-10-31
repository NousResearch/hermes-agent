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


class TestWaitUntil:
    """``wait(until=...)`` ends as soon as the awaited text is printed; a later wait only sees new output."""

    def test_returns_on_match_and_consumes_it(self, registry, tmp_path):
        import time

        session = registry.spawn_local(
            "echo 'listening on :1'; sleep 1; echo 'listening on :2'; sleep 30", cwd=str(tmp_path), task_id="t-until")
        try:
            start = time.monotonic()
            first = registry.wait(session.id, timeout=20, until="listening on")
            second = registry.wait(session.id, timeout=20, until="listening on")
            assert (first["status"], second["status"]) == ("matched", "matched")
            assert first["process_running"] is True
            assert time.monotonic() - start < 10  # never held the 20 s window
            assert registry.wait(session.id, timeout=1, until="listening on")["status"] == "timeout"
        finally:
            registry.kill_process(session.id)

    def test_exit_before_match_returns_exit(self, registry, tmp_path):
        session = registry.spawn_local("echo bye", cwd=str(tmp_path), task_id="t-until")
        assert registry.wait(session.id, timeout=10, until="never printed")["status"] == "exited"
