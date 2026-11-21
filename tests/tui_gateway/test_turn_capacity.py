"""Desktop capacity is held only between turn admission and settlement."""

import threading

import pytest

from hermes_cli import active_sessions
from tui_gateway import server


def test_tui_idle_owners_do_not_block_telegram_and_cannot_run_over_cap(monkeypatch):
    cfg = {"max_concurrent_sessions": 1}
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(server, "_canonical_ledger_refusal", lambda session: None)
    sessions = [dict(session_key=f"desktop-{i}", source="desktop", history_lock=threading.RLock())
                for i in range(2)]
    try:
        for i, session in enumerate(sessions):
            assert server._ensure_active_session_slot(f"ui-{i}", session) is None
        assert server._activate_active_session_slot(sessions[0]) is None
        refusal = server._activate_active_session_slot(sessions[1])
        assert getattr(refusal, "reason", None) == active_sessions.MAX_CONCURRENT_SESSIONS
        server._release_hosted_room_turn_slot(sessions[0])
        assert "active_session_lease" in sessions[0]
        telegram, refusal = active_sessions.try_acquire_active_session(
            session_id="telegram-turn", surface="gateway:telegram", config=cfg)
        assert refusal is None and telegram is not None
        try:
            refusal = server._activate_active_session_slot(sessions[0])
            assert getattr(refusal, "reason", None) == active_sessions.MAX_CONCURRENT_SESSIONS
        finally:
            telegram.release()
        assert server._activate_active_session_slot(sessions[1]) is None
    finally:
        for session in sessions:
            server._release_active_session_slot(session)


@pytest.mark.parametrize("settlement", ["notification", "missing_agent", "compute_done", "cancel_before_ready"])
def test_terminal_paths_free_capacity_without_releasing_ownership(monkeypatch, settlement):
    cfg = {"max_concurrent_sessions": 1}
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(server, "_canonical_ledger_refusal", lambda session: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_compute_host_session_info", lambda session: {})
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *a, **k: None)
    session = dict(session_key="terminal-owner", source="desktop", history_lock=threading.RLock(),
                   running=True, inflight_turn=None, history=[], agent=None, attached_images=[])
    try:
        assert server._ensure_active_session_slot("terminal-ui", session) is None
        assert server._activate_active_session_slot(session) is None
        owner = session["active_session_lease"]
        if settlement == "notification":
            server._notif_release_turn(session)
        elif settlement == "missing_agent":
            assert server._admit_prompt_turn("terminal-ui", session, "synthetic", None, None, None, None) is None
        elif settlement == "cancel_before_ready":
            session["_turn_cancel_requested"] = True
            monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *a: None)
            server._run_after_agent_ready("rid", "terminal-ui", session, "synthetic", None, None, None)
        else:
            server._on_compute_host_turn_done(
                "rid", "terminal-ui", session, {"type": "turn.end", "session_info_emitted": True})
        assert session["running"] is False
        assert session["active_session_lease"] is owner
        telegram, refusal = active_sessions.try_acquire_active_session(
            session_id="after-terminal", surface="gateway:telegram", config=cfg)
        assert refusal is None and telegram is not None
        telegram.release()
    finally:
        server._release_active_session_slot(session)
