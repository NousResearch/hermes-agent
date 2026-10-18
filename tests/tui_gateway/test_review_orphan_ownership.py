"""Reclamation waits for review receipts; the isolated owner receives teardown."""
import threading
from types import SimpleNamespace

import pytest

from tui_gateway import server


@pytest.mark.parametrize("isolated", [False, True])
def test_orphan_and_lru_keep_review_but_timeout_is_bounded(monkeypatch, isolated):
    timers, retired, controls = [], [], []
    class Timer:
        def __init__(self, delay, callback):
            self.callback = callback
            timers.append(self)
        def start(self):
            pass
        def cancel(self):
            pass
    sid = "hidden-room-session"
    run = SimpleNamespace(worker_done=threading.Event())
    agent = SimpleNamespace(_background_review_workers=[run], _background_review_lock=threading.Lock(),
                            _review_shutdown_timeout_s=120)
    session = dict(agent=None if isolated else agent, session_key=sid, _sid=sid,
                   history_lock=threading.Lock(), running=False, transport=server._detached_ws_transport)
    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server.threading, "Timer", Timer)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda *a: False)
    monkeypatch.setattr(server, "_session_pending_kind", lambda *a: None)
    monkeypatch.setattr(server, "write_json", lambda *a: True)
    monkeypatch.setattr(server, "_finalize_session", lambda *a, **kw: retired.append(sid))
    monkeypatch.setattr(server, "_announce_session_reclaimed", lambda *a: None)
    supervisor = SimpleNamespace(is_running=lambda: True, control=lambda *a, **kw: controls.append((a, kw)))
    monkeypatch.setattr(server, "_compute_host_supervisor", supervisor)
    if isolated:
        session["_compute_host_active"] = True
        def emit(revision, pending):
            server._relay_compute_host_rpc({"method": "event", "params": {
                "type": "review.status", "session_id": sid,
                "payload": {"pending": pending, "revision": revision, "shutdown_timeout_s": 120}}})
        emit(2, True)
        emit(1, False)  # stale terminal delivery must not unpin newer work
    server._schedule_ws_orphan_reap(sid)
    timers[-1].callback()
    assert sid in server._sessions, "foreground completion discarded a still-owned review"
    assert not server._session_is_lru_evictable(sid, session)
    # No sleeps or global clock mocking: move only this grace's start into the past.
    session["_review_reclaim_started"] -= 121
    timers[-1].callback()
    assert sid not in server._sessions and retired == [sid]
    if isolated:
        assert controls == [((sid,), {"route_name": "session.close", "wait": False})]
    else:
        assert not controls
