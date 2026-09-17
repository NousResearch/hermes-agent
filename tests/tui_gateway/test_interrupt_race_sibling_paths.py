"""Sibling wedge sites: every caller whose release is fenced on the id _run_prompt_submit pops."""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server as srv


def _session(agent):
    return {
        "agent": agent,
        "session_key": "agent-session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "inflight_turn": None,
        "transport": None,
    }


def _cancel_during_admission(session):
    """A concurrent session.interrupt lands between the claim and the admission recheck."""

    def _slot(*_a, **_k):
        with session["history_lock"]:
            session["_turn_cancel_requested"] = True
        return None

    return _slot


def test_notification_dispatch_wedges_the_session(monkeypatch):
    session = _session(types.SimpleNamespace(clear_interrupt=lambda: None))
    monkeypatch.setattr(srv, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(srv, "_ensure_session_db_row", lambda *_a: True)
    monkeypatch.setattr(
        srv, "_ensure_active_session_slot", _cancel_during_admission(session)
    )
    monkeypatch.setattr(srv, "_notif_log_failure", lambda *_a, **_k: None)

    turn_id = srv._notif_claim_turn(session)
    assert turn_id
    with pytest.raises(RuntimeError):
        srv._notif_submit(
            "rid", "sid", session, "notification text", "notif dispatch",
            expected_turn_id=turn_id,
        )

    assert "_active_turn_id" not in session
    assert session["running"] is False, (
        "notification turn wedged the session busy: _notif_release_turn is fenced on a "
        "turn id that _run_prompt_submit already popped"
    )


def test_auto_continue_recovery_wedges_the_session(monkeypatch, tmp_path):
    from agent.turn_authorization import TurnAuthorization

    session = _session(types.SimpleNamespace(clear_interrupt=lambda: None))
    recovery = TurnAuthorization.blocked()
    monkeypatch.setattr(srv, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(srv, "_ensure_session_db_row", lambda *_a: True)
    monkeypatch.setattr(
        srv, "_ensure_active_session_slot", _cancel_during_admission(session)
    )
    # Mirror _maybe_schedule_auto_continue's kickoff claim exactly.
    with session["history_lock"]:
        session["running"] = True
        auto_turn_id = srv._activate_turn_identity(session)
        session["_turn_cancel_requested"] = False
        session["_active_turn_route"] = "inline"
        session["_active_turn_authorization"] = recovery

    started = srv._run_prompt_submit(
        "rid", "sid", session, "resume me",
        display_kind="auto_continue",
        turn_authorization=recovery,
        expected_turn_id=auto_turn_id,
    )

    assert started is False
    assert "_active_turn_id" not in session
    assert session["running"] is False, (
        "auto-continue recovery wedged the session busy: it never inspects the "
        "False return, and _notif_release_turn would be fenced out anyway"
    )
