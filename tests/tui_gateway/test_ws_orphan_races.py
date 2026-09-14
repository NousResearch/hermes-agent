"""Orphan callbacks own only their detachment, never a later reconnect."""

from contextlib import nullcontext
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tui_gateway import server


@pytest.mark.parametrize("phase", ["before_callback", "before_continuation", "before_initial_timer", "cold_resume_claim"])
def test_obsolete_orphan_cannot_replace_new_detachment(monkeypatch, phase):
    timers = []

    class Timer:
        def __init__(self, delay, callback):
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            pass  # A dispatched callback can still execute after cancel().

    sid = "generation-race"
    session = dict(transport=server._detached_ws_transport, running=True,
                   agent=SimpleNamespace(get_activity_summary=lambda: {"seconds_since_activity": 0}))
    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server.threading, "Timer", Timer)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 600)
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda *a: False)
    if phase == "before_initial_timer":
        transport = object()
        session["transport"] = transport
        newest = None

        class DisconnectLock:
            def __enter__(self):
                pass

            def __exit__(self, *args):
                # A new client reconnects and drops as the old disconnect
                # releases its claim, before any out-of-lock scheduling.
                nonlocal newest
                server._cancel_ws_orphan_reap(sid)
                server._schedule_ws_orphan_reap(sid)
                newest = timers[-1]

        monkeypatch.setattr(server, "_session_resume_lock", DisconnectLock())
        assert server._close_sessions_for_transport(transport) == (0, 1)
        assert server._pending_ws_reaps[sid] is newest
        return
    server._schedule_ws_orphan_reap(sid)
    old = timers[-1]
    if phase == "cold_resume_claim":
        # A cold resume missed the live lookup before a concurrent resume won.
        # Its claim discovers that winner while orphan interrupt I/O is in flight.
        session["session_key"] = sid
        monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 0)
        replies = []

        def resume_during_interrupt(*a, **kw):
            ctx = server._Resume(1, {}, sid)
            replies.append(ctx.claim("unused", {}))

        monkeypatch.setattr(server, "_interrupt_session_turn", resume_during_interrupt)
        old.callback()
        assert replies[0]["error"]["code"] == 4009
        assert session["transport"] is server._detached_ws_transport
        assert session["_client_gone_interrupt_requested"]
        assert len(timers) == 2
        assert server._pending_ws_reaps[sid] is timers[-1]
        return

    def redetach():
        server._cancel_ws_orphan_reap(sid)
        session["transport"] = server._detached_ws_transport
        server._schedule_ws_orphan_reap(sid)
        return timers[-1]

    if phase == "before_callback":
        newest = redetach()
    else:
        # Interrupt I/O runs outside the resume lock. A reconnect/redetach
        # can win before the old callback registers its next poll.
        monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 0)
        def interrupt(*a, **kw):
            nonlocal newest
            session.pop("_client_gone_interrupt_requested", None)
            newest = redetach()
        monkeypatch.setattr(server, "_interrupt_session_turn", interrupt)
        newest = None
    old.callback()
    assert server._pending_ws_reaps[sid] is newest
    assert timers == [old, newest]


@pytest.mark.parametrize("transition", ["retire", "redetach"])
def test_orphan_interrupt_claim_clears_when_session_leaves_detached_state(monkeypatch, transition):
    timers = []

    class Timer:
        def __init__(self, _delay, callback):
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            pass

    sid = "writer-rebound"
    session = dict(transport=server._detached_ws_transport, running=True)
    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server.threading, "Timer", Timer)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 0)
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda *a: False)
    monkeypatch.setattr(server, "_interrupt_session_turn", lambda *a, **kw: False)

    server._schedule_ws_orphan_reap(sid)
    timers[0].callback()
    assert session["_client_gone_interrupt_requested"]

    session["transport"] = object()
    if transition == "redetach":
        # The bypass writer disconnects before the old settlement can retire.
        assert server._close_sessions_for_transport(session["transport"]) == (0, 1)
        timers[2].callback()
        assert session["_client_gone_interrupt_requested"]
        assert session["_client_gone_interrupt_polls"] == 1
        replacement = server._pending_ws_reaps[sid]
        timers[1].callback()
        assert server._pending_ws_reaps[sid] is replacement
        assert session["_client_gone_interrupt_requested"]
        return
    timers[1].callback()

    assert "_client_gone_interrupt_requested" not in session
    assert "_client_gone_interrupt_polls" not in session
    assert server._reattach_refusal(1, sid, session) is None
    assert sid not in server._pending_ws_reaps


@pytest.mark.parametrize("path", ["unpersisted", "reuse", "eager", "activate", "prompt"])
@pytest.mark.parametrize("claim", ["already_claimed", "wins_lock", "retired"])
def test_reconnect_cannot_cross_orphan_interrupt_claim(monkeypatch, path, claim):
    sid = "interrupt-race"
    session = dict(transport=server._detached_ws_transport, running=True,
                   history_lock=threading.Lock(), history=[], session_key="stored",
                   agent=SimpleNamespace(model="test"), queued_prompt=None)
    session["_client_gone_interrupt_requested"] = claim == "already_claimed"
    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {sid: Mock()})
    transport = object()
    monkeypatch.setattr(server, "current_transport", lambda: transport)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test")
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *a: None)
    monkeypatch.setattr(server, "_legacy_group_fence_error", lambda *a: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a: False)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(server, "_handle_busy_submit", lambda *a, **kw: {"result": {"queued": True}})
    monkeypatch.setattr(server, "_sess", lambda *a: (session, None))

    class ResumeLock:
        held = False

        def __enter__(self):
            assert not self.held, "resume path recursively acquired a non-reentrant lock"
            self.held = True
            if claim == "wins_lock":
                session["_client_gone_interrupt_requested"] = True
            elif claim == "retired":
                server._sessions.pop(sid, None)

        def __exit__(self, *args):
            self.held = False

    monkeypatch.setattr(server, "_session_resume_lock", ResumeLock())
    ctx = SimpleNamespace(rid=1, owns_db=False, db=None, cols=80, omit_messages=True,
                          defer_history=False, target="stored", profile=None,
                          profile_home=None, profile_resume_cwd=None, found={},
                          messages=lambda history: [], mint=lambda: ("unused", "tui", "."),
                          restore=lambda: ([], [], []), display_prefix=lambda: [])
    if path == "eager":
        monkeypatch.setattr(server, "_profile_build_scope", lambda *a: nullcontext())
        monkeypatch.setattr(server, "_make_agent_in_context", lambda *a, **kw: Mock())
        monkeypatch.setattr(server, "_find_live_session_by_key", lambda *a: (sid, session))
        response = server._resume_eager(ctx)
    elif path == "unpersisted":
        response = server._resume_live_unpersisted(ctx, sid, session)
    elif path == "reuse":
        response = server._resume_reuse_live(ctx, sid, session)
    else:
        name = {"activate": "session.activate", "prompt": "prompt.submit"}[path]
        response = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": name,
                                          "params": {"session_id": sid, "text": "continue", "omit_messages": True}})
    assert response.get("error", {}).get("code") == (4007 if claim == "retired" else 4009)
    assert session["transport"] is server._detached_ws_transport
    assert sid in server._pending_ws_reaps
    assert session["queued_prompt"] is None
