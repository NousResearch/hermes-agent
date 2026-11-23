"""Idle slash-worker eviction: the reaper tick closes a worker that has not run a command for
``_SLASH_WORKER_IDLE_S`` WITHOUT ending its session.

Every session teardown funnel gates on a dead transport, so a dashboard session on a live
WebSocket never finalizes and its worker (a full MCP fleet) accumulates for the gateway's
lifetime. The worker is only a cache — slash.exec respawns it on demand and already handles a
dead worker — so the reaper evicts the worker alone: no session end, no state.db write, no
``session.reclaimed`` announcement.
"""

import io
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from tui_gateway import server

NOW = 1_700_000_000.0
IDLE_S = 900.0


class _FakeProc:
    pid = 424242

    def __init__(self):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()


def _make_worker(monkeypatch, *, last_used):
    """A real ``_SlashWorker`` shell (no subprocess): its close() lands on the patched
    ``_stop_worker_group``, so ``len(stopped)`` counts real closes exactly."""
    stopped = []
    monkeypatch.setattr(server, "_stop_worker_group", lambda proc: stopped.append(proc))
    worker = object.__new__(server._SlashWorker)
    worker._lock = threading.Lock()
    worker._closed = False
    worker._seq = 0
    worker.stderr_tail = []
    worker.stdout_queue = queue.Queue()
    worker.last_used = last_used
    worker.proc = _FakeProc()
    return worker, stopped


def _make_session(worker, *, transport):
    """A session dict plus the list recording any agent.close() calls (the eviction pass must
    never make one; suite teardown may, after the assertions have run)."""
    agent_close_calls = []
    session = {
        "session_key": "sess-key",
        "transport": transport,
        "agent": SimpleNamespace(close=lambda: agent_close_calls.append(1)),
        "slash_worker": worker,
    }
    return session, agent_close_calls


@pytest.fixture
def registry(monkeypatch):
    sessions = {}
    monkeypatch.setattr(server, "_sessions", sessions)
    monkeypatch.setattr(server, "_SLASH_WORKER_IDLE_S", IDLE_S)
    monkeypatch.setattr(server, "_session_pending_kind", lambda sid: None)
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda sid, session: False)
    reclaimed = []
    monkeypatch.setattr(server, "_announce_session_reclaimed",
                        lambda session, end_reason: reclaimed.append(end_reason))
    return sessions, reclaimed


def test_idle_worker_on_live_transport_is_evicted_session_untouched(monkeypatch, registry):
    sessions, reclaimed = registry
    worker, stopped = _make_worker(monkeypatch, last_used=NOW - IDLE_S - 60)
    session, agent_close_calls = _make_session(worker, transport=object())  # live: not the sentinel
    sessions["sid-1"] = session

    evicted = server._evict_idle_slash_workers(NOW)

    assert evicted == 1
    assert len(stopped) == 1  # close() ran exactly once
    assert worker._closed is True
    assert session["slash_worker"] is None
    assert sessions["sid-1"] is session  # the session itself was NOT ended
    assert agent_close_calls == []  # the agent was not touched
    assert reclaimed == []  # no session.reclaimed announced


def test_worker_with_a_command_in_flight_is_skipped(monkeypatch, registry):
    sessions, _reclaimed = registry
    worker, stopped = _make_worker(monkeypatch, last_used=NOW - IDLE_S - 60)
    session, _agent_close_calls = _make_session(worker, transport=object())
    sessions["sid-1"] = session

    worker._lock.acquire()  # run() holds the worker lock for the whole command
    try:
        evicted = server._evict_idle_slash_workers(NOW)
    finally:
        worker._lock.release()

    assert evicted == 0
    assert stopped == []
    assert worker._closed is False
    assert session["slash_worker"] is worker


def test_young_worker_is_untouched(monkeypatch, registry):
    sessions, _reclaimed = registry
    worker, stopped = _make_worker(monkeypatch, last_used=NOW - 60)
    session, _agent_close_calls = _make_session(worker, transport=object())
    sessions["sid-1"] = session

    assert server._evict_idle_slash_workers(NOW) == 0
    assert stopped == []
    assert worker._closed is False
    assert session["slash_worker"] is worker


def test_session_mid_turn_is_exempt_even_with_an_idle_worker(monkeypatch, registry):
    sessions, _reclaimed = registry
    worker, stopped = _make_worker(monkeypatch, last_used=NOW - IDLE_S - 60)
    session, _agent_close_calls = _make_session(worker, transport=object())
    session["running"] = True
    sessions["sid-1"] = session

    assert server._evict_idle_slash_workers(NOW) == 0
    assert stopped == []
    assert session["slash_worker"] is worker


def test_detached_session_worker_closed_by_finalize_is_not_double_closed(monkeypatch, registry):
    sessions, _reclaimed = registry
    worker, stopped = _make_worker(monkeypatch, last_used=NOW - IDLE_S - 60)
    session, _agent_close_calls = _make_session(worker, transport=server._detached_ws_transport)
    sessions["sid-1"] = session
    worker.close()  # _finalize_session's chokepoint got there first, as it owns detached sessions
    assert len(stopped) == 1

    assert server._evict_idle_slash_workers(NOW) == 0
    assert len(stopped) == 1  # the eviction pass did not close it a second time
    assert session["slash_worker"] is worker  # the slot is finalize/teardown's to clear, not the pass's


def test_zero_tunable_disables_eviction(monkeypatch, registry):
    sessions, _reclaimed = registry
    monkeypatch.setattr(server, "_SLASH_WORKER_IDLE_S", 0.0)
    worker, stopped = _make_worker(monkeypatch, last_used=NOW - 10 * IDLE_S)
    session, _agent_close_calls = _make_session(worker, transport=object())
    sessions["sid-1"] = session

    assert server._evict_idle_slash_workers(NOW) == 0
    assert stopped == []
    assert session["slash_worker"] is worker


def test_reaper_tick_runs_the_eviction_pass(monkeypatch, registry):
    sessions, reclaimed = registry
    worker, stopped = _make_worker(monkeypatch, last_used=NOW - IDLE_S - 60)
    session, agent_close_calls = _make_session(worker, transport=object())
    session["last_active"] = time.time()
    session["created_at"] = time.time()
    sessions["sid-1"] = session
    # Neutralize the tick's other piggybacks; the eviction pass itself stays real.
    monkeypatch.setattr(server, "_flush_dirty_sessions", lambda: 0)
    monkeypatch.setattr(server, "_repair_missing_ws_orphan_reaps", lambda: None)
    monkeypatch.setattr(server, "_max_live_sessions", lambda: 0)
    monkeypatch.setattr(server, "_reclaim_orphaned_leases", lambda: None)

    server._reap_idle_sessions()

    assert len(stopped) == 1
    assert session["slash_worker"] is None
    assert sessions["sid-1"] is session
    assert agent_close_calls == []
    assert reclaimed == []
