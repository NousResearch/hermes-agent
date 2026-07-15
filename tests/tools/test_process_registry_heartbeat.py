"""Tests for the ProcessRegistry activity-heartbeat sweep.

A silent background process (no stdout for minutes) produces no reader-thread
ticks, so liveness has to come from a time-based sweep instead. These tests
drive ``_run_heartbeat_once`` synchronously rather than sleeping for a real
``HEARTBEAT_INTERVAL_S`` tick.
"""

import time

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture()
def registry():
    return ProcessRegistry()


def _running_session(sid, session_key, exited=False) -> ProcessSession:
    return ProcessSession(
        id=sid,
        command="sleep 999",
        task_id="t1",
        session_key=session_key,
        started_at=time.time(),
        exited=exited,
    )


class TestHeartbeatSweep:
    def test_sweep_invokes_callback_for_each_alive_session_key(self, registry):
        registry._running["p1"] = _running_session("p1", "key-a")
        registry._running["p2"] = _running_session("p2", "key-b")

        seen = []
        registry.set_heartbeat_callback(seen.append)
        registry._run_heartbeat_once()

        assert sorted(seen) == ["key-a", "key-b"]

    def test_sweep_skips_exited_sessions(self, registry):
        registry._running["p1"] = _running_session("p1", "key-a")
        registry._running["p2"] = _running_session("p2", "key-exited", exited=True)

        seen = []
        registry.set_heartbeat_callback(seen.append)
        registry._run_heartbeat_once()

        assert seen == ["key-a"]

    def test_sweep_dedupes_shared_session_key(self, registry):
        # Two processes spawned by the same gateway session share a key —
        # the callback (a single DB UPDATE) only needs to fire once.
        registry._running["p1"] = _running_session("p1", "key-a")
        registry._running["p2"] = _running_session("p2", "key-a")

        seen = []
        registry.set_heartbeat_callback(seen.append)
        registry._run_heartbeat_once()

        assert seen == ["key-a"]

    def test_sweep_skips_sessions_without_session_key(self, registry):
        registry._running["p1"] = _running_session("p1", session_key="")

        seen = []
        registry.set_heartbeat_callback(seen.append)
        registry._run_heartbeat_once()

        assert seen == []

    def test_sweep_is_noop_without_registered_callback(self, registry):
        registry._running["p1"] = _running_session("p1", "key-a")
        registry._run_heartbeat_once()  # must not raise

    def test_sweep_swallows_callback_exceptions(self, registry):
        registry._running["p1"] = _running_session("p1", "key-a")
        registry._running["p2"] = _running_session("p2", "key-b")

        seen = []

        def flaky_cb(key):
            if key == "key-a":
                raise RuntimeError("boom")
            seen.append(key)

        registry.set_heartbeat_callback(flaky_cb)
        registry._run_heartbeat_once()  # must not raise despite key-a failing

        assert seen == ["key-b"]

    def test_set_heartbeat_callback_starts_thread_once(self, registry):
        registry.set_heartbeat_callback(lambda key: None)
        thread1 = registry._heartbeat_thread
        assert thread1 is not None
        assert thread1.daemon is True
        assert thread1.is_alive()

        registry.set_heartbeat_callback(lambda key: None)
        assert registry._heartbeat_thread is thread1
