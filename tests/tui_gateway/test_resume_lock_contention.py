"""A busy chat's history_lock must not stall other chats' resumes through the global resume lock."""

import threading
import time
from types import SimpleNamespace

import pytest

from tests.waiting import wait_until
from tui_gateway import server


def _live_session():
    return dict(transport=server._detached_ws_transport, running=True, history={}, history_lock=threading.Lock(),
                agent=SimpleNamespace(get_activity_summary=lambda: {"seconds_since_activity": 0}))


def _ctx():
    return SimpleNamespace(rid=1, owns_db=False, db=None, cols=80, omit_messages=True,
                           defer_history=False, target="stored", profile=None, profile_home=None,
                           profile_resume_cwd=None, found={}, inline_images=True)


def _resume(path, sid, session):
    if path == "reuse":
        return server._resume_reuse_live(_ctx(), sid, session)
    return server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "session.activate",
                                  "params": {"session_id": sid, "omit_messages": True}})


@pytest.mark.parametrize("path", ["reuse", "activate"])
def test_resume_of_idle_chat_is_not_stalled_by_busy_chat_history_lock(monkeypatch, path):
    busy, idle = _live_session(), _live_session()
    monkeypatch.setattr(server, "_sessions", {"busy": busy, "idle": idle})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    results = {}
    busy["history_lock"].acquire()  # the long holder: a turn or compression that keeps its lock
    try:
        busy_thread = threading.Thread(
            target=lambda: results.update(busy=_resume(path, "busy", busy)), daemon=True)
        busy_thread.start()
        # The busy resume has reached its wait: it holds the resume lock (baseline) or has already attached.
        wait_until(lambda: server._session_resume_lock.locked() or busy["transport"] is server._stdio_transport,
                   what="busy resume reached its history_lock wait")
        idle_thread = threading.Thread(
            target=lambda: results.update(idle=_resume(path, "idle", idle)), daemon=True)
        idle_thread.start()
        idle_thread.join(timeout=3)
        assert not idle_thread.is_alive(), "resume of an idle chat stalled behind another chat's history_lock"
        assert "result" in results["idle"]
    finally:
        busy["history_lock"].release()
    busy_thread.join(timeout=3)
    assert not busy_thread.is_alive()
    assert "result" in results["busy"]


def test_orphan_reaper_activity_check_does_not_wait_on_history_lock(monkeypatch):
    session = dict(_live_session(), _compute_host_turn_id="turn", _compute_host_activity_ns=time.perf_counter_ns())
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 600)
    result = {}
    session["history_lock"].acquire()  # the reaper runs under the global resume lock; a turn holds history_lock
    try:
        checker = threading.Thread(
            target=lambda: result.update(fresh=server._ws_orphan_turn_activity_is_fresh(session)), daemon=True)
        checker.start()
        checker.join(timeout=3)
        assert not checker.is_alive(), "orphan reaper activity check waits on history_lock under the global lock"
    finally:
        session["history_lock"].release()
    assert result["fresh"] is True
