"""Reconnect ownership must not revive or reap a different detachment."""

import queue
import threading
import types

import pytest

from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport


@pytest.fixture
def detached_session(monkeypatch):
    sid = "continuity-live"
    session = {
        "agent": types.SimpleNamespace(),
        "session_key": "continuity-stored",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "profile_home": None,
        "transport": server._detached_ws_transport,
    }
    monkeypatch.setitem(server._sessions, sid, session)
    yield sid, session
    server._cancel_ws_orphan_reap(sid)


@pytest.fixture
def timers(monkeypatch):
    scheduled = []

    class Timer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.cancelled = False
            scheduled.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(server.threading, "Timer", Timer)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20.0)
    return scheduled


def _request(method, params, transport):
    token = bind_transport(transport)
    try:
        return server.handle_request(
            {"id": "continuity", "method": method, "params": params}
        )
    finally:
        reset_transport(token)


@pytest.mark.parametrize("phase", ["callback", "continuation"])
@pytest.mark.parametrize("deferral", ["delegation", "fresh_activity"])
def test_stale_reap_cannot_replace_new_detachment_timer(
    monkeypatch, detached_session, timers, phase, deferral
):
    sid, session = detached_session
    monkeypatch.setattr(
        server, "_session_has_active_delegations",
        lambda *_a, **_k: deferral == "delegation",
    )
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 600.0)
    session["agent"].get_activity_summary = lambda: {"seconds_since_activity": 1.0}
    original_schedule = server._schedule_ws_orphan_reap
    original_schedule(sid)
    timer_a = timers[-1]
    detachment_b = {}

    def reconnect_and_redetach():
        session["transport"] = object()
        server._cancel_ws_orphan_reap(sid)
        session["transport"] = server._detached_ws_transport
        original_schedule(sid)
        detachment_b["timer"] = timers[-1]

    if phase == "callback":
        reconnect_and_redetach()
        assert timer_a.cancelled
    else:
        def interleave(interleaved_sid, **stale_kwargs):
            assert interleaved_sid == sid
            reconnect_and_redetach()
            original_schedule(sid, **stale_kwargs)

        # Let A claim its callback, then reconnect after it releases the
        # lifecycle lock but before its deferred scheduling continuation runs.
        monkeypatch.setattr(server, "_schedule_ws_orphan_reap", interleave)

    # Timer.cancel cannot withdraw an already-dispatched callback.
    timer_a.callback()
    timer_b = detachment_b["timer"]
    assert server._pending_ws_reaps[sid] is timer_b
    assert not timer_b.cancelled
    assert timers == [timer_a, timer_b]
    assert not session.get("_client_gone_interrupt_requested")


def test_reap_clears_own_registration_after_session_removed(detached_session, timers):
    sid, session = detached_session
    server._schedule_ws_orphan_reap(sid)
    timer = timers[-1]
    assert server._pop_session_by_id(sid) is session
    timer.callback()
    assert sid not in server._pending_ws_reaps


@pytest.mark.parametrize("interrupt_claimed", [True, False])
def test_activate_respects_disconnect_interrupt_claim(
    monkeypatch, detached_session, timers, interrupt_claimed
):
    sid, session = detached_session
    timer = _arm_reap(monkeypatch, sid, session, timers, interrupt_claimed)
    _assert_reconnect("session.activate", sid, sid, session, timer, interrupt_claimed)


def _arm_reap(monkeypatch, sid, session, timers, interrupt_claimed):
    server._schedule_ws_orphan_reap(sid)
    timer = timers[-1]
    if interrupt_claimed:
        # Run the real claimant, but leave the interrupted turn unsettled.
        monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 0)
        monkeypatch.setattr(server, "_session_has_active_delegations", lambda *_a: False)
        monkeypatch.setattr(server, "_interrupt_session_turn", lambda *_a, **_k: False)
        timer.callback()
        timer = timers[-1]
        assert session["_client_gone_interrupt_requested"]
    return timer


def _assert_reconnect(method, target, sid, session, timer, interrupt_claimed):
    transport = object()
    response = _request(
        method, {"session_id": target, "omit_messages": True}, transport
    )
    if interrupt_claimed:
        assert response.get("error", {}).get("code") == 4009, response
        assert session["transport"] is server._detached_ws_transport
        assert transport not in session.get("viewers", {})
        assert server._pending_ws_reaps[sid] is timer
        assert not timer.cancelled
    else:
        assert response["result"]["session_id"] == sid
        assert session["transport"] is transport
        assert transport in session["viewers"]
        assert sid not in server._pending_ws_reaps
        assert timer.cancelled


@pytest.mark.parametrize("persisted", [True, False], ids=["persisted", "unpersisted"])
@pytest.mark.parametrize("interrupt_claimed", [True, False])
def test_resume_respects_disconnect_interrupt_claim(
    monkeypatch, tmp_path, detached_session, timers, persisted, interrupt_claimed
):
    from hermes_state import SessionDB

    sid, session = detached_session
    key = session["session_key"]
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    try:
        if persisted:
            db.create_session(key, source="desktop", cwd=str(tmp_path))
        timer = _arm_reap(monkeypatch, sid, session, timers, interrupt_claimed)
        _assert_reconnect("session.resume", key, sid, session, timer, interrupt_claimed)
        # Reconnecting a lazy session must not persist an empty draft.
        assert bool(db.get_session(key)) == persisted
    finally:
        db.close()


def test_resume_race_winner_keeps_committed_interrupt_reap(
    monkeypatch, detached_session, timers
):
    sid, session = detached_session
    timer = _arm_reap(monkeypatch, sid, session, timers, True)
    released = []
    lease = types.SimpleNamespace(release=lambda: released.append(True))
    winner = server._claim_or_reuse_live(
        "losing-resume", session["session_key"], {"profile_home": None}, lease
    )
    assert winner == (sid, session)
    assert released == [True]
    assert server._pending_ws_reaps.get(sid) is timer
    assert not timer.cancelled


@pytest.mark.parametrize("interrupt_claimed", [True, False])
@pytest.mark.parametrize("admission_refused", [True, False])
def test_prompt_reconnect_claim_precedes_admission_and_queue(
    monkeypatch, detached_session, timers, interrupt_claimed, admission_refused
):
    sid, session = detached_session
    timer = _arm_reap(monkeypatch, sid, session, timers, interrupt_claimed)
    admissions = []

    def admission(claimed_sid, claimed_session):
        admissions.append((claimed_sid, claimed_session))
        return "at capacity" if admission_refused else None

    monkeypatch.setattr(server, "_ensure_active_session_slot", admission)
    transport = object()
    response = _request(
        "prompt.submit", {"session_id": sid, "text": "next", "queued": True}, transport
    )
    if interrupt_claimed:
        assert response.get("error", {}).get("code") == 4009, response
        assert admissions == []
        assert session["transport"] is server._detached_ws_transport
        assert transport not in session.get("viewers", {})
        assert server._pending_ws_reaps[sid] is timer
        assert not timer.cancelled
        assert not session.get("queued_prompt")
    else:
        assert admissions == [(sid, session)]
        if admission_refused:
            assert response["error"]["code"] == 4090
            assert not session.get("queued_prompt")
        else:
            assert response["result"]["status"] == "queued"
            assert session["queued_prompt"]["text"] == "next"
            assert session["queued_prompt"]["transport"] is transport
        assert session["transport"] is transport
        assert transport in session.get("viewers", {})
        assert sid not in server._pending_ws_reaps
        assert timer.cancelled
    assert session["history"] == []


def test_prompt_does_not_rebind_a_record_removed_during_request(
    monkeypatch, detached_session
):
    sid, session = detached_session
    original_config = server._load_dashboard_process_isolation_config

    def remove_before_claim():
        server._pop_session_by_id(sid)
        return original_config()

    admissions = []
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", remove_before_claim)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_a: admissions.append(True))
    response = _request(
        "prompt.submit", {"session_id": sid, "text": "next", "queued": True}, object()
    )
    assert response.get("error", {}).get("code") == 4001, response
    assert session["transport"] is server._detached_ws_transport
    assert admissions == []
    assert not session.get("queued_prompt")


@pytest.mark.parametrize("method", ["session.activate", "session.resume", "prompt.submit"])
def test_reconnect_waits_for_in_progress_orphan_claim(
    monkeypatch, tmp_path, detached_session, timers, method
):
    from hermes_state import SessionDB

    sid, session = detached_session
    progress = queue.Queue()
    release_reap = threading.Event()
    errors = []
    responses = []
    admissions = []

    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                progress.put("waiting-for-claim")
                self.lock.acquire()
            return self

        def __exit__(self, *_args):
            self.lock.release()

    def pause_reap(*_args):
        progress.put("reaper")
        assert release_reap.wait(timeout=5.0)
        return False

    def run(fn):
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)

    def reconnect():
        responses.append(_request(
            method,
            {
                "session_id": session["session_key"] if method == "session.resume" else sid,
                "omit_messages": True,
                "text": "next",
                "queued": True,
            },
            object(),
        ))
        progress.put("reconnected")

    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_session_resume_lock", ObservedLock())
    monkeypatch.setattr(server, "_session_has_active_delegations", pause_reap)
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 0)
    monkeypatch.setattr(server, "_interrupt_session_turn", lambda *_a, **_k: False)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_a: admissions.append(True))
    server._schedule_ws_orphan_reap(sid)
    reap_thread = threading.Thread(target=run, args=(timers[-1].callback,))
    reconnect_thread = threading.Thread(target=run, args=(reconnect,))
    try:
        reap_thread.start()
        assert progress.get(timeout=5.0) == "reaper"
        reconnect_thread.start()
        # No negative sleeps: the request either contends on the claim lock
        # or completes incorrectly while the reaper is paused before marking.
        first_progress = progress.get(timeout=5.0)
        release_reap.set()
        reap_thread.join(timeout=5.0)
        reconnect_thread.join(timeout=5.0)
        assert not errors
        assert not reap_thread.is_alive()
        assert not reconnect_thread.is_alive()
        assert first_progress == "waiting-for-claim"
        assert responses[0]["error"]["code"] == 4009
        assert session["transport"] is server._detached_ws_transport
        assert not session.get("queued_prompt")
        assert admissions == []
        assert server._pending_ws_reaps[sid] is timers[-1]
    finally:
        release_reap.set()
        reap_thread.join(timeout=5.0)
        if reconnect_thread.ident is not None:
            reconnect_thread.join(timeout=5.0)
        db.close()
