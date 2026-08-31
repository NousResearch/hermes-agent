"""A killed ``hermes serve`` must not lose in-memory session transcripts.

Regression for #94724 (item 2, @ruangraung): a serve terminated mid-update
lost every un-flushed in-memory session — the next RPC failed with
"session-scoped RPC rejected: not in memory (detached/reaped runtime)" and no
store held the transcript. #95576 made serves survive *future* updates; this
covers the kill path itself:

* SIGTERM/SIGINT first flush in-memory sessions to state.db (bounded,
  best-effort, chained to the previously installed handler so uvicorn's
  graceful shutdown still runs).
* The idle-reaper tick piggybacks a periodic incremental flush so even a
  SIGKILL loses at most one flush interval.
"""

from __future__ import annotations

import os
import signal
import time

import pytest

from tui_gateway import server


class _FlushAgent:
    """Minimal agent exposing the real ``_persist_session`` flush contract."""

    def __init__(self, messages=None):
        self.session_id = "flush-agent"
        self.flush_calls: list[list] = []
        self._session_messages = (
            messages
            if messages is not None
            else [{"role": "user", "content": "unflushed turn"}]
        )

    def _persist_session(self, messages, conversation_history=None):
        self.flush_calls.append(list(messages))


@pytest.fixture
def registered_session():
    """Register a fake in-memory session; always deregister on exit."""
    registered: list[str] = []

    def _register(sid: str, agent, **extra):
        session = {"agent": agent, "session_key": sid, "running": False}
        session.update(extra)
        with server._sessions_lock:
            server._sessions[sid] = session
        registered.append(sid)
        return session

    yield _register

    with server._sessions_lock:
        for sid in registered:
            server._sessions.pop(sid, None)


def _restore_signal_state(prev_handlers):
    for signum, handler in prev_handlers.items():
        signal.signal(signum, handler)
    server._exit_flush_prev_handlers.clear()
    server._exit_flush_handlers_installed = False


def test_sigterm_flushes_populated_session_into_state_db(
    registered_session, tmp_path, monkeypatch
):
    """A populated in-memory session survives a SIGTERM into state.db."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "sess-sigterm-flush"
    db.create_session(sid, source="tui")

    class _DbAgent(_FlushAgent):
        def _persist_session(self, messages, conversation_history=None):
            super()._persist_session(messages, conversation_history)
            for msg in messages:
                if msg.get("_db_persisted"):
                    continue
                db.append_message(sid, msg["role"], msg["content"])
                msg["_db_persisted"] = True

    agent = _DbAgent(messages=[{"role": "user", "content": "survive the kill"}])
    registered_session(sid, agent)

    chained = {"called": False}

    def _prev_handler(signum, frame):
        chained["called"] = True

    prev = {signal.SIGTERM: signal.signal(signal.SIGTERM, _prev_handler)}
    try:
        assert server.install_exit_flush_signal_handlers() is True
        os.kill(os.getpid(), signal.SIGTERM)
        # The handler runs synchronously on the main thread at the next
        # bytecode boundary; poll briefly for robustness.
        deadline = time.monotonic() + 5.0
        while not chained["called"] and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        _restore_signal_state(prev)

    assert chained["called"], "previous SIGTERM handler must still be chained"
    assert agent.flush_calls, "SIGTERM must flush in-memory sessions"
    rows = db.get_messages(sid)
    assert any("survive the kill" in str(r.get("content", "")) for r in rows)


def test_exit_flush_is_bounded(registered_session):
    """A hung persist must never block exit longer than the budget."""

    class _HangingAgent(_FlushAgent):
        def _persist_session(self, messages, conversation_history=None):
            time.sleep(5.0)

    registered_session("sess-hang", _HangingAgent())

    start = time.monotonic()
    server._flush_sessions_before_exit(budget_s=0.3)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"exit flush blocked {elapsed:.1f}s past its budget"


def test_shutdown_sessions_flushes_before_teardown(monkeypatch):
    """The atexit path persists transcripts BEFORE slow per-session teardown."""
    order: list[str] = []

    monkeypatch.setattr(
        server, "_release_gateway_wake_owner", lambda: None, raising=False
    )
    monkeypatch.setattr(
        server,
        "_flush_sessions_before_exit",
        lambda budget_s=None: order.append("flush") or 0,
    )
    monkeypatch.setattr(
        server,
        "_close_session_by_id",
        lambda sid, **kw: order.append(f"close:{sid}"),
    )
    with server._sessions_lock:
        server._sessions["sess-order"] = {"agent": None, "session_key": "sess-order"}
    try:
        server._shutdown_sessions()
    finally:
        with server._sessions_lock:
            server._sessions.pop("sess-order", None)

    assert order and order[0] == "flush"
    assert "close:sess-order" in order


def test_periodic_flush_respects_interval_with_fake_clock(
    registered_session, monkeypatch
):
    monkeypatch.setattr(server, "_INCREMENTAL_FLUSH_INTERVAL_S", 300.0)
    agent = _FlushAgent()
    registered_session("sess-interval", agent)

    assert server._flush_dirty_sessions(now=1_000.0) == 1
    assert len(agent.flush_calls) == 1

    # Within the interval: no re-flush.
    assert server._flush_dirty_sessions(now=1_000.0 + 299.0) == 0
    assert len(agent.flush_calls) == 1

    # Past the interval: flushes again — SIGKILL loses at most one interval.
    assert server._flush_dirty_sessions(now=1_000.0 + 301.0) == 1
    assert len(agent.flush_calls) == 2


def test_periodic_flush_skips_running_sessions(registered_session, monkeypatch):
    """Mid-turn sessions are the turn thread's to persist — never race them."""
    monkeypatch.setattr(server, "_INCREMENTAL_FLUSH_INTERVAL_S", 300.0)
    agent = _FlushAgent()
    registered_session("sess-running", agent, running=True)

    assert server._flush_dirty_sessions(now=1_000.0) == 0
    assert agent.flush_calls == []


def test_idle_reaper_scan_piggybacks_incremental_flush(monkeypatch):
    """The existing reaper tick drives the flush — no new timer subsystem."""
    called = {"flush": 0}
    monkeypatch.setattr(
        server,
        "_flush_dirty_sessions",
        lambda now=None: called.__setitem__("flush", called["flush"] + 1) or 0,
    )
    monkeypatch.setattr(server, "_enforce_session_cap", lambda: None)
    monkeypatch.setattr(server, "_reclaim_orphaned_leases", lambda: None)
    server._reap_idle_sessions()
    assert called["flush"] == 1
