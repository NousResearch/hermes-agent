"""Probe: interrupt racing dispatch wedges the session busy forever."""

from __future__ import annotations

import threading
import types

from agent.turn_authorization import TurnAuthorization
from tui_gateway import server as srv


def _session(agent):
    return {
        "agent": agent,
        "session_key": "k",
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 80,
        "inflight_turn": None,
        "transport": None,
    }


def test_probe_interrupt_racing_dispatch_wedges_session_busy(monkeypatch):
    import agent.interrupt_compat as interrupt_compat

    agent = types.SimpleNamespace(session_id="a", clear_interrupt=lambda: None)
    session = _session(agent)
    authorization = TurnAuthorization.from_raw(None)
    # Turn A was admitted by prompt.submit (_lock_in_submit_turn) and its
    # _run_after_agent_ready thread is about to call _run_prompt_submit.
    session.update(
        running=True,
        _active_turn_id="A",
        _active_turn_authorization=authorization,
        _active_turn_route="inline",
    )
    srv._start_inflight_turn(session, "hello")

    monkeypatch.setattr(srv, "_emit", lambda *a, **kw: None)
    monkeypatch.setattr(srv, "_ensure_session_db_row", lambda *a, **kw: True)
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *a, **kw: False)
    monkeypatch.setattr(srv, "_session_active_turn_uses_compute_host", lambda *a, **kw: False)
    monkeypatch.setattr(srv, "_clear_pending", lambda *a, **kw: None)

    hard_interrupt_reached = threading.Event()
    let_interrupt_finish = threading.Event()

    def blocking_hard_interrupt(*_a, **_kw):
        # The claim is installed and _turn_cancel_requested is already True here.
        hard_interrupt_reached.set()
        assert let_interrupt_finish.wait(10)

    monkeypatch.setattr(
        interrupt_compat, "request_hard_interrupt", blocking_hard_interrupt)

    interrupt_thread = threading.Thread(
        target=lambda: srv._interrupt_session_turn("sid", session),
        name="interrupt")

    def slot_gate(*_a, **_kw):
        # _run_prompt_submit already passed its early gate (no cancel yet) and
        # released history_lock; the interrupt now lands in that exact window.
        interrupt_thread.start()
        assert hard_interrupt_reached.wait(10)
        return None

    monkeypatch.setattr(srv, "_ensure_active_session_slot", slot_gate)

    started = srv._run_prompt_submit(
        "rid", "sid", session, "hello",
        turn_authorization=authorization, expected_turn_id="A")

    print("submit started        =", started)
    print("after submit: running =", session.get("running"))
    print("after submit: id      =", session.get("_active_turn_id"))

    let_interrupt_finish.set()
    interrupt_thread.join(10)

    print("FINAL running         =", session.get("running"))
    print("FINAL _active_turn_id =", session.get("_active_turn_id"))
    print("FINAL _run_thread     =", session.get("_run_thread"))
    print("FINAL inflight_turn   =", (session.get("inflight_turn") or {}).get("user"))
    print("FINAL cancel_requested=", session.get("_turn_cancel_requested"))
    print("FINAL interrupt_claim =", session.get("_turn_interrupt_claim"))

    assert started is False
    # The refused dispatch must leave the session idle: every downstream release is
    # fenced on the turn id it popped, so a lingering running=True would be
    # unrecoverable (no worker exists to clear it).
    assert session.get("running") is False, (
        "session wedged BUSY: running=True with no turn identity, no run thread "
        "and no interrupt claim — every later prompt.submit queues forever and "
        "_drain_queued_prompt refuses because running is True"
    )
    assert session.get("_active_turn_id") is None
    assert session.get("_run_thread") is None

    # user-visible consequence: the next prompt RUNS instead of queueing forever.
    next_admission = srv._lock_in_submit_turn(
        "rid2", "sid", session, "next prompt", {}, False, None, None, None,
        TurnAuthorization.from_raw(None), False)
    assert next_admission[0] is not srv._SUBMIT_TURN_BECAME_BUSY, (
        "session still busy: the next prompt.submit was forced down the queue path"
    )
    assert next_admission[0] is None
