"""Timer identity, not a sid alone, owns one native WS detachment."""
import threading
from types import SimpleNamespace

import pytest

from tui_gateway import server


@pytest.fixture
def orphan(monkeypatch):
    timers, torn_down, interrupts = [], [], []
    class Timer:
        def __init__(self, delay, callback):
            self.delay, self.callback, self.cancelled = delay, callback, False
            timers.append(self)
        def start(self):
            pass
        def cancel(self):
            self.cancelled = True
    session = {
        "agent": SimpleNamespace(session_id="stored"), "session_key": "stored", "history": [],
        "history_lock": threading.Lock(), "history_version": 0, "running": True,
        "attached_images": [], "cols": 80, "slash_worker": None,
        "transport": server._detached_ws_transport, "viewers": {},
        "_turn_attempt": (1, "task", 1), "_turn_attempt_live": (1, "task", 1),
    }
    monkeypatch.setattr(server, "_sessions", {"sid": session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server.threading, "Timer", Timer)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 10)
    monkeypatch.setattr(server, "_WS_ORPHAN_INTERRUPT_REAP_MAX_POLLS", 2)
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda *args: False)
    monkeypatch.setattr(server, "_ws_orphan_turn_activity_is_fresh", lambda session: False)
    monkeypatch.setattr(server, "_teardown_popped_session", lambda session, **kwargs:
                        torn_down.append(session) if session is not None else None)
    monkeypatch.setattr(server, "_interrupt_session_turn", lambda *args, **kwargs: interrupts.append(kwargs))
    return SimpleNamespace(session=session, timers=timers, torn_down=torn_down, interrupts=interrupts)


@pytest.mark.parametrize("race", ["cancelled_callback", "continuation", "interrupt_error", "abandoned", "fresh_detachment"])
def test_orphan_callback_cannot_steal_a_new_detachment(orphan, monkeypatch, race):
    session = orphan.session
    server._schedule_ws_orphan_reap("sid")
    old = orphan.timers[-1]
    if race == "cancelled_callback":
        server._cancel_ws_orphan_reap("sid")
        server._schedule_ws_orphan_reap("sid")
        current = orphan.timers[-1]
        old.callback()  # Timer.cancel cannot retract an already dispatched callback.
        assert server._pending_ws_reaps["sid"] is current
        assert orphan.interrupts == []
        assert "_client_gone_interrupt_polls" not in session
    elif race in {"continuation", "interrupt_error"}:
        replacement = []
        def interrupt(*args, **kwargs):
            assert server._pending_ws_reaps.get("sid") is old
            assert kwargs["attempt"] == session["_turn_attempt"]
            # A newer detachment can win while interrupt I/O runs without session locks.
            server._cancel_ws_orphan_reap("sid")
            server._schedule_ws_orphan_reap("sid")
            replacement.append(orphan.timers[-1])
            if race == "interrupt_error":
                raise RuntimeError("old interrupt failed")
        monkeypatch.setattr(server, "_interrupt_session_turn", interrupt)
        old.callback()
        assert len(replacement) == 1
        assert server._pending_ws_reaps["sid"] is replacement[0]
        assert len(orphan.timers) == 2
        assert session["_client_gone_interrupt_requested"] is True
    elif race == "abandoned":
        session.update(transport=object(), _client_gone_interrupt_requested=True, _client_gone_interrupt_polls=2)
        old.callback()
        assert "sid" not in server._pending_ws_reaps
        assert "_client_gone_interrupt_requested" not in session
        assert "_client_gone_interrupt_polls" not in session
    else:
        transport = object()
        session.update(transport=transport, _client_gone_interrupt_requested=True, _client_gone_interrupt_polls=2)
        original_schedule = server._schedule_ws_orphan_reap
        def schedule(sid, **kwargs):
            # Another native rebind cannot interleave with initial timer registration.
            assert server._session_resume_lock.locked()
            original_schedule(sid, **kwargs)
        monkeypatch.setattr(server, "_schedule_ws_orphan_reap", schedule)
        assert server._close_sessions_for_transport(transport) == (0, 1)
        monkeypatch.setattr(server, "_schedule_ws_orphan_reap", original_schedule)
        orphan.timers[-1].callback()
        assert session["_client_gone_interrupt_polls"] == 1
        assert len(orphan.interrupts) == 1
    assert server._sessions["sid"] is session
    assert orphan.torn_down == []


@pytest.mark.parametrize("entry", ["activate", "submit", "lazy_resume", "warm_resume"])
@pytest.mark.parametrize("state", ["settling", "removed", "reattach"])
def test_native_reattach_paths_share_orphan_ownership(orphan, monkeypatch, entry, state):
    session, transport = orphan.session, object()
    monkeypatch.setattr(server, "current_transport", lambda: transport)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *args: None)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *args: False)
    monkeypatch.setattr(server, "_child_run_active", lambda *args: False)
    monkeypatch.setattr(server, "_resolve_model", lambda: "offline-test")
    monkeypatch.setattr(server, "_attach_todo_state", lambda payload, session: payload)
    # Avoid DB reads; transport rebinding itself must still use the real lifecycle helper.
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    server._schedule_ws_orphan_reap("sid")
    timer = orphan.timers[-1]
    if state == "settling":
        session["_client_gone_interrupt_requested"] = True
    elif state == "removed":
        server._sessions.pop("sid")
    monkeypatch.setattr(server, "_sess_nowait", lambda *args: (session, None))
    monkeypatch.setattr(server, "_sess", lambda *args: (session, None))
    ctx = SimpleNamespace(
        rid=1, owns_db=False, profile=None, cols=80, omit_messages=True, target="stored",
        defer_history=False, messages=lambda history: [])
    if entry == "activate":
        result = server._methods["session.activate"](1, {"session_id": "sid", "omit_messages": True})
    elif entry == "submit":
        # Stop at the normal busy response, after rebind and before any model/build.
        monkeypatch.setattr(server, "_handle_busy_submit", lambda *args, **kwargs: server._ok(1, {"queued": True}))
        result = server._methods["prompt.submit"](1, {"session_id": "sid", "text": "Offline input"})
    elif entry == "lazy_resume":
        result = server._resume_live_unpersisted(ctx, "sid", session)
    else:
        result = server._resume_reuse_live(ctx, "sid", session)
    if state == "reattach":
        assert "result" in result, result
        assert session["transport"] is transport
        assert "sid" not in server._pending_ws_reaps
        assert timer.cancelled
    else:
        assert result["error"]["code"] == (4009 if state == "settling" else 4007)
        assert session["transport"] is server._detached_ws_transport
        assert server._pending_ws_reaps["sid"] is timer
        assert not timer.cancelled
