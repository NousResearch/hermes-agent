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
            assert "not an error" in r["timeout_note"]
            assert "Uptime" in r["timeout_note"]
        finally:
            registry.kill_process(sid)

    def test_wait_timeout_suggests_notify_when_unset(self, registry):
        sid = _spawn_sleeper(registry, notify=False)
        try:
            r = registry.wait(sid, timeout=1)
            assert "notify_on_complete=true" in r["timeout_note"]
        finally:
            registry.kill_process(sid)

    def test_wait_timeout_defers_to_notify_when_set(self, registry):
        sid = _spawn_sleeper(registry, notify=True)
        try:
            r = registry.wait(sid, timeout=1)
            assert "you will be notified on exit" in r["timeout_note"]
        finally:
            registry.kill_process(sid)

    def test_clamped_wait_keeps_clamp_note_and_running_semantics(self, registry, monkeypatch):
        monkeypatch.setenv("TERMINAL_TIMEOUT", "1")
        sid = _spawn_sleeper(registry)
        try:
            r = registry.wait(sid, timeout=600)
            assert r["status"] == "timeout"
            assert "clamped" in r["timeout_note"]
            assert "not an error" in r["timeout_note"]
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
            assert "still running" in r["note"]
            # The process was not killed and the yield bit was consumed.
            assert registry.poll(sid)["status"] == "running"
            from tools.interrupt import is_thread_yield_requested
            assert not is_thread_yield_requested(t.ident)
        finally:
            registry.kill_process(sid)

    def test_wait_without_yield_still_times_out(self, registry):
        sid = _spawn_sleeper(registry)
        try:
            r = registry.wait(sid, timeout=1)
            assert r["status"] == "timeout"
        finally:
            registry.kill_process(sid)
