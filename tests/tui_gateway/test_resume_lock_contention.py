"""A busy chat's history_lock must not stall other chats' resumes through the global resume lock."""

import threading
import time
from types import SimpleNamespace

import pytest

from tests.waiting import wait_until
from tui_gateway import server


def _live_session(key):
    return dict(
        transport=server._detached_ws_transport,
        running=True,
        history={},
        history_lock=threading.Lock(),
        session_key=key,
        agent=SimpleNamespace(
            get_activity_summary=lambda: {"seconds_since_activity": 0}
        ),
    )


def _ctx():
    return SimpleNamespace(
        rid=1,
        owns_db=False,
        db=None,
        cols=80,
        omit_messages=True,
        defer_history=False,
        target="stored",
        profile=None,
        profile_home=None,
        profile_resume_cwd=None,
        found={},
        inline_images=True,
    )


def _resume(path, sid, session):
    if path == "reuse":
        return server._resume_reuse_live(_ctx(), sid, session)
    method = "session.resume" if path == "resume" else "session.activate"
    return server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": {"session_id": sid, "omit_messages": True},
        }
    )


class _FakeTimer:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _SignallingLock:
    """Wraps a held lock; ``acquire()`` sets ``entered`` before it blocks, so a test sees the attempt."""

    def __init__(self, lock, entered):
        self._lock, self._entered = lock, entered

    def acquire(self, blocking=True, timeout=-1):
        self._entered.set()
        return self._lock.acquire(blocking, timeout)

    def release(self):
        self._lock.release()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self._lock.release()


@pytest.mark.parametrize("path", ["reuse", "activate", "resume"])
def test_resume_of_idle_chat_is_not_stalled_by_busy_chat_history_lock(
    monkeypatch, path
):
    # "resume" is session.resume with no state.db row: the live session is found by key (unpersisted path).
    # A bound transport makes that path rebind. At baseline the rebind took history_lock under the resume
    # lock; the fix attaches without history_lock, so the busy "resume" may finish without ever reaching it.
    monkeypatch.setattr(server, "current_transport", lambda: object())
    holder, busy_entered = threading.Lock(), threading.Event()
    busy, idle = (
        dict(_live_session("busy"), history_lock=_SignallingLock(holder, busy_entered)),
        _live_session("idle"),
    )
    monkeypatch.setattr(server, "_sessions", {"busy": busy, "idle": idle})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    results = {}
    holder.acquire()  # the long holder: a turn or compression that keeps its lock
    try:
        busy_thread = threading.Thread(
            target=lambda: results.update(busy=_resume(path, "busy", busy)), daemon=True
        )
        busy_thread.start()
        # Start the idle resume only once the busy one has provably reached contention: it attempted the
        # history_lock we hold, or it holds the resume lock, or it already finished (nothing left to stall).
        wait_until(
            lambda: (
                busy_entered.is_set()
                or server._session_resume_lock.locked()
                or not busy_thread.is_alive()
            ),
            what="busy resume reached history_lock, held the resume lock, or finished",
        )
        idle_thread = threading.Thread(
            target=lambda: results.update(idle=_resume(path, "idle", idle)), daemon=True
        )
        idle_thread.start()
        idle_thread.join(timeout=3)
        assert not idle_thread.is_alive(), (
            "resume of an idle chat stalled behind another chat's history_lock"
        )
        assert "result" in results["idle"]
    finally:
        busy["history_lock"].release()
    busy_thread.join(timeout=3)
    assert not busy_thread.is_alive()
    assert "result" in results["busy"]


@pytest.mark.parametrize("path", ["reuse", "resume"])
def test_resume_waits_for_held_resume_lock_before_rebinding_or_cancelling_reap(
    monkeypatch, path
):
    transport = object()
    monkeypatch.setattr(server, "current_transport", lambda: transport)
    session, timer = _live_session("sid"), _FakeTimer()
    monkeypatch.setattr(server, "_sessions", {"sid": session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {"sid": timer})
    results = {}
    server._session_resume_lock.acquire()  # grace expiry holds the resume lock: a resume must not interleave
    try:
        resumer = threading.Thread(
            target=lambda: results.update(r=_resume(path, "sid", session)), daemon=True
        )
        resumer.start()
        resumer.join(timeout=0.5)
        assert resumer.is_alive(), "resume ran past the held _session_resume_lock"
        assert not timer.cancelled, (
            "resume cancelled the orphan reap while grace expiry held the lock"
        )
        assert server._pending_ws_reaps.get("sid") is timer
    finally:
        server._session_resume_lock.release()
    resumer.join(timeout=3)
    assert not resumer.is_alive()
    assert "result" in results["r"]
    assert timer.cancelled
    assert "sid" not in server._pending_ws_reaps
    assert session["transport"] is transport


def test_orphan_reaper_activity_check_does_not_wait_on_history_lock(monkeypatch):
    session = dict(
        _live_session("k"),
        _compute_host_turn_id="turn",
        _compute_host_activity_ns=time.perf_counter_ns(),
    )
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 600)
    result = {}
    session[
        "history_lock"
    ].acquire()  # the reaper runs under the global resume lock; a turn holds history_lock
    try:
        checker = threading.Thread(
            target=lambda: result.update(
                fresh=server._ws_orphan_turn_activity_is_fresh(session)
            ),
            daemon=True,
        )
        checker.start()
        checker.join(timeout=3)
        assert not checker.is_alive(), (
            "orphan reaper activity check waits on history_lock under the global lock"
        )
    finally:
        session["history_lock"].release()
    assert result["fresh"] is True
