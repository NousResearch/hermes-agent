"""Concurrent, real-code reproduction of the admission-race wedge."""

from __future__ import annotations

import threading
import time
import types

import agent.interrupt_compat as interrupt_compat
from agent.turn_authorization import TurnAuthorization
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


def test_real_interrupt_racing_real_queue_drain_wedges_the_session(monkeypatch):
    """session.interrupt + queue drain, both real, leave running=True with no worker.

    Only two BLOCKING points are injected, at places the production code already
    documents as slow and lock-free: active-session lease acquisition, and the
    provider hard interrupt ("some providers can't apply interrupt() until a
    blocking call returns").
    """
    lease_entered = threading.Event()
    lease_release = threading.Event()
    hard_interrupt_entered = threading.Event()
    hard_interrupt_release = threading.Event()

    session = _session(types.SimpleNamespace(clear_interrupt=lambda: None))
    session["queued_prompt"] = {"text": "queued work", "transport": None}
    srv._sessions["sid"] = session

    def slow_lease(*_a, **_k):
        lease_entered.set()
        assert lease_release.wait(5.0)
        return None  # lease granted

    def slow_hard_interrupt(*_a, **_k):
        hard_interrupt_entered.set()
        assert hard_interrupt_release.wait(5.0)

    monkeypatch.setattr(srv, "_ensure_active_session_slot", slow_lease)
    monkeypatch.setattr(
        interrupt_compat, "request_hard_interrupt", slow_hard_interrupt
    )
    monkeypatch.setattr(srv, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(srv, "_ensure_session_db_row", lambda *_a: True)
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_a: False)
    monkeypatch.setattr(srv, "_tts_stream_stop", lambda: None)
    monkeypatch.setattr(srv, "_retire_turn_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(srv, "_clear_pending", lambda *_a: None)

    drain_result = []
    interrupt_result = []
    drainer = threading.Thread(
        target=lambda: drain_result.append(
            srv._drain_queued_prompt("rid", "sid", session)
        )
    )
    drainer.start()
    try:
        assert lease_entered.wait(5.0), "drain never reached lease acquisition"
        assert session["running"] is True
        claimed_turn = session["_active_turn_id"]

        interrupter = threading.Thread(
            target=lambda: interrupt_result.append(
                srv._methods["session.interrupt"]("r", {"session_id": "sid"})
            )
        )
        interrupter.start()
        # The interrupt has claimed the turn and asked the provider to stop.
        assert hard_interrupt_entered.wait(5.0), "interrupt never claimed the turn"
        # Now the drain's lease returns and it observes the cancel request.
        lease_release.set()
        drainer.join(timeout=5.0)
        assert not drainer.is_alive()
        # Only now does the interrupt reach its identity-fenced release.
        hard_interrupt_release.set()
        interrupter.join(timeout=5.0)
        assert not interrupter.is_alive()
    finally:
        lease_release.set()
        hard_interrupt_release.set()
        srv._sessions.pop("sid", None)
        drainer.join(timeout=5.0)

    assert interrupt_result and interrupt_result[0]["result"]["status"] == "interrupted"
    assert claimed_turn
    assert "_active_turn_id" not in session, "identity was cleared by the drain"
    assert "_run_thread" not in session, "no worker thread exists"
    assert session["running"] is False, (
        "WEDGED: running=True with no worker and no active turn id. "
        "Every later release is identity-fenced on a turn id that no longer exists, "
        "so this session is permanently busy. "
        f"queued={session.get('queued_prompt')!r}"
    )


def test_wedged_session_refuses_all_further_work(monkeypatch):
    """Follow-up proof: once wedged, new prompts only queue and the drain refuses."""
    session = _session(types.SimpleNamespace(clear_interrupt=lambda: None))
    # Exactly the state the race above produces.
    session["running"] = True
    monkeypatch.setattr(srv, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_a: False)
    monkeypatch.setattr(srv, "_load_busy_input_mode", lambda: "queue")
    srv._sessions["sid"] = session
    try:
        response = srv._methods["prompt.submit"](
            "r", {"session_id": "sid", "text": "please answer"}
        )
    finally:
        srv._sessions.pop("sid", None)

    assert response["result"]["status"] == "queued"
    assert srv._drain_queued_prompt("rid", "sid", session) is False
    assert session["queued_prompt"]["text"] == "please answer"
