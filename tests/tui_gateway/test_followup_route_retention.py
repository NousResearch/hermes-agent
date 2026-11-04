"""Keep a routed client alive while a claimed follow-up prepares its worker (#89978)."""

import contextlib
import threading

import pytest

from tests.tui_gateway.test_turn_state_truthful import _session
from tui_gateway import server


@pytest.fixture
def followup(monkeypatch):
    session = _session()
    events = []
    monkeypatch.setattr(server, "_sessions", {"route": session})
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_: None)
    monkeypatch.setattr(server, "_record_turn_marker", lambda *a, **k: None)
    monkeypatch.setattr(server, "_reopen_routed_session_row", lambda *a: None)
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: events.append((event, payload)))
    for name in ["_release_turn_scopes", "_post_turn_housekeeping", "_emit_settled_session_info"]:
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    monkeypatch.setattr(server, "_prepare_turn_input", lambda *a: None)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _: True)
    monkeypatch.setattr(server, "_routing_provenance_db", lambda _: contextlib.nullcontext(None))
    return session, events


def test_followup_retains_route_before_database_preparation_and_worker_publication(monkeypatch, followup):
    session, events = followup
    workers = []

    def before_database(_session):
        # This preparation can exceed Desktop's 500 ms settled-lease debounce.
        assert events == [("session.info", {"running": True})]
        return True

    def publish(run, **_kwargs):
        assert events == [("session.info", {"running": True})]
        workers.append(run)
        return object()

    monkeypatch.setattr(server, "_ensure_session_db_row", before_database)
    monkeypatch.setattr(server, "_start_session_work", publish)
    server._dispatch_followup_turn("rid", "route", session, "continue", "goal continuation")
    assert len(workers) == 1
    assert session["running"] is True
    assert not any(event == "message.start" for event, _ in events)
    workers[0]()
    assert [event for event, _ in events].count("message.start") == 1
    assert session["running"] is False


def test_refused_followup_cleanup_preserves_newer_turn_claim(monkeypatch, followup):
    session, events = followup
    refused = threading.Event()
    resume_cleanup = threading.Event()
    submit = server._run_prompt_submit

    def refuse_then_wait(*args, **kwargs):
        accepted = submit(*args, **kwargs)
        assert accepted is False and session["running"] is False
        refused.set()
        assert resume_cleanup.wait(timeout=5)
        return accepted

    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_: "temporarily not owned")
    monkeypatch.setattr(server, "_run_prompt_submit", refuse_then_wait)
    worker = threading.Thread(target=server._dispatch_followup_turn,
                              args=("rid", "route", session, "continue", "goal continuation"))
    worker.start()
    try:
        assert refused.wait(timeout=5)
        assert server._notif_claim_turn(session)
        assert session["_run_turn"] == 2
    finally:
        resume_cleanup.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert session["running"] is True
    assert [(event, payload) for event, payload in events if event == "session.info"] == [
        ("session.info", {"running": True})]


@pytest.mark.parametrize("refusal", ["closing", "replaced", "retiring", "ownership", "preparation_error"])
def test_refused_followup_settles_retention_without_starting_a_turn(monkeypatch, followup, refusal):
    session, events = followup
    def fail(_session):
        raise RuntimeError("database preparation failed")

    session["_closing"] = refusal == "closing"
    server._sessions["route"] = {} if refusal == "replaced" else session
    refusal_hooks = {
        "ownership": ("_ensure_active_session_slot", lambda *_: "not owned"),
        "preparation_error": ("_ensure_session_db_row", fail),
    }
    if hook := refusal_hooks.get(refusal):
        monkeypatch.setattr(server, *hook)
    monkeypatch.setattr(server, "_start_session_work", lambda *a, **k: None)
    server._dispatch_followup_turn("rid", "route", session, "continue", "goal continuation")
    assert session["running"] is False
    assert [(event, payload) for event, payload in events if event == "session.info"] == [
        ("session.info", {"running": True}), ("session.info", {"running": False})]
    assert not any(event == "message.start" for event, _ in events)
