"""A turn that reaches admission after ``session.close`` must not leave an active-session lease.

``session.close`` waits a bounded grace for the turn thread and then finalizes, which releases the
lease. A turn thread that outlives the grace (typically the first prompt, still waiting for the
deferred agent build) used to claim a fresh lease when it finally reached admission, see the
session was closing, and return with the lease still registered. Reopening the chat in another
runtime was then refused as "open in another Hermes window/terminal" until the idle reaper's orphan
sweep, about five minutes later.
"""

from __future__ import annotations

import threading
import types

import pytest

from hermes_cli.active_sessions import active_session_registry_snapshot
from tui_gateway import server

SID = "closing-sid"


def _session(**extra) -> dict:
    return {
        "agent": None, "session_key": "closing-session-key", "history": [],
        "history_lock": threading.Lock(), "history_version": 0, "running": False,
        "attached_images": [], "image_counter": 0, "cols": 80, "slash_worker": None,
        "show_reasoning": False, "tool_progress_mode": "all", **extra}


def _built_agent(turns_run: list) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        session_id="closing-session-key", clear_interrupt=lambda: None,
        run_conversation=lambda *a, **k: turns_run.append(a) or {"final_response": "done"})


def _rpc(method: str, **params) -> dict:
    return server.handle_request({"id": method, "method": method, "params": {"session_id": SID, **params}})


def _registry_session_ids() -> list:
    return [entry.get("session_id") for entry in active_session_registry_snapshot()]


def _reopen_refusal():
    """The refusal that a new runtime for the same chat gets, or None. A successful claim is released again."""
    reopened = _session()
    refusal = server._ensure_active_session_slot("reopened-sid", reopened)
    server._release_active_session_slot(reopened)
    return refusal


@pytest.fixture
def turn_env(monkeypatch):
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda session: True)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
    # Production gives the turn thread 5 s before finalizing; the outcome is the same when it gives up at once.
    monkeypatch.setattr(server, "_TURN_SETTLE_BEFORE_CLOSE_SECONDS", 0.0)
    sessions: list[dict] = []
    yield sessions
    for session in sessions:
        server._release_active_session_slot(session)
    server._sessions.pop(SID, None)


def _join_turn(session: dict) -> None:
    run_thread = session["_run_thread"]
    run_thread.join(timeout=10)
    assert not run_thread.is_alive()


def test_close_while_first_prompt_waits_for_build_leaves_no_lease(turn_env):
    build_done = threading.Event()
    session = _session(agent_ready=build_done)
    turn_env.append(session)
    server._sessions[SID] = session
    turns_run: list = []

    submitted = _rpc("prompt.submit", text="hello")
    assert submitted["result"]["status"] == "streaming", submitted
    assert _registry_session_ids() == ["closing-session-key"]

    assert _rpc("session.close")["result"]["closed"] is True
    assert _registry_session_ids() == []

    session["agent"] = _built_agent(turns_run)
    build_done.set()
    _join_turn(session)

    assert {
        "leases": _registry_session_ids(), "reopen_refusal": _reopen_refusal(),
        "running": session["running"], "turns_run": turns_run,
    } == {"leases": [], "reopen_refusal": None, "running": False, "turns_run": []}


def test_close_landing_while_turn_claims_lease_leaves_no_lease(turn_env, monkeypatch):
    """The close finalizes between the turn's closing check and its lease claim."""
    ready = threading.Event()
    ready.set()
    turns_run: list = []
    session = _session(agent=_built_agent(turns_run), agent_ready=ready)
    turn_env.append(session)
    server._sessions[SID] = session
    claim = server._ensure_active_session_slot
    submitting_thread = threading.current_thread()
    at_claim: dict = {}

    def _claim_after_close(sid, claiming):
        if threading.current_thread() is not submitting_thread and not claiming.get("_closing"):
            at_claim["closed"] = _rpc("session.close")["result"]["closed"]
            at_claim["leases"] = _registry_session_ids()
        return claim(sid, claiming)

    monkeypatch.setattr(server, "_ensure_active_session_slot", _claim_after_close)

    submitted = _rpc("prompt.submit", text="hello")
    assert submitted["result"]["status"] == "streaming", submitted
    _join_turn(session)

    monkeypatch.setattr(server, "_ensure_active_session_slot", claim)
    assert at_claim == {"closed": True, "leases": []}
    assert {
        "leases": _registry_session_ids(), "reopen_refusal": _reopen_refusal(),
        "running": session["running"], "turns_run": turns_run,
    } == {"leases": [], "reopen_refusal": None, "running": False, "turns_run": []}


def test_closing_refusal_keeps_a_lease_the_session_already_held(turn_env):
    """Only a lease this admission claimed is released: one held from before the close is the close's
    own to hand off (an isolated compute-host turn defers it in _settle_isolated_turn_before_close)."""
    session = _session()
    turn_env.append(session)
    assert server._ensure_active_session_slot(SID, session) is None
    held = session["active_session_lease"]
    session["_closing"] = True

    assert server._admit_prompt_turn(SID, session, "hello", None, None, None, None) is None

    assert session.get("active_session_lease") is held
    assert _registry_session_ids() == ["closing-session-key"]
