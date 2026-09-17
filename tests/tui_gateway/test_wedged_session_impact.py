"""A dispatch refused mid-admission must leave the session USABLE.

Regression for the wedge found in review: `_run_prompt_submit` popped the turn
identity while leaving `running=True`. Because every downstream release is
compare-and-swapped on that now-deleted id, nothing could ever clear the flag —
no worker existed to run a finalizer. The session stayed busy forever: later
prompts only queued, and `_drain_queued_prompt` refused a session it believed was
running. These tests pin the user-visible consequence (the next prompt RUNS)
rather than the internal flag alone.
"""
import threading
import types

import tui_gateway.server as srv


def _session(agent):
    return {
        "agent": agent,
        "session_key": "k",
        "history": [],
        "history_lock": threading.RLock(),
        "running": False,
        "cols": 80,
    }


def _refused_mid_admission(monkeypatch):
    """Drive the exact race: a cancel lands while dispatch is inside admission."""
    session = _session(types.SimpleNamespace(clear_interrupt=lambda: None))
    monkeypatch.setattr(srv, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(srv, "_ensure_session_db_row", lambda *_a: True)
    monkeypatch.setattr(srv, "_session_uses_compute_host", lambda *_a: False)

    def cancel_during_admission(*_a, **_k):
        with session["history_lock"]:
            session["_turn_cancel_requested"] = True
        return None

    monkeypatch.setattr(srv, "_ensure_active_session_slot", cancel_during_admission)
    turn_id = srv._notif_claim_turn(session)
    started = srv._run_prompt_submit(
        "rid", "sid", session, "notification text", expected_turn_id=turn_id
    )
    assert started is False
    # The refusal owns the release: identity and running go together.
    assert session["running"] is False
    assert session.get("_active_turn_id") is None
    assert session.get("_run_thread") is None
    return session


def test_refused_dispatch_leaves_session_idle(monkeypatch):
    _refused_mid_admission(monkeypatch)


def test_next_user_prompt_runs_instead_of_queueing_forever(monkeypatch):
    session = _refused_mid_admission(monkeypatch)
    session.pop("_turn_cancel_requested", None)
    monkeypatch.setattr(srv, "_load_busy_input_mode", lambda: "queue")
    srv._sessions["sid"] = session
    try:
        first = srv._methods["prompt.submit"]("r1", {"session_id": "sid", "text": "one"})
    finally:
        srv._sessions.pop("sid", None)

    # Previously this returned "queued" forever with no worker to drain it.
    assert first["result"]["status"] != "queued", (
        "session still busy after a refused dispatch: the user's next prompt was "
        "forced down the queue path with no worker able to drain it"
    )
