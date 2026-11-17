"""A caller that pauses a turn to continue it itself says so: ``session.interrupt {resume: true}``.

A host plugin stops a running turn to apply a grant (a sandbox restart, a network policy reload) and then
continues the same reply with a hidden ``prompt.submit``. The turn ends like any interrupt (same stored rows,
same ``interrupted`` status); its ``message.complete`` carries ``paused: true``, so a client can show the reply
as continuing instead of stopped, and hold the user's next message until the continuation ran. The user's own
Stop (a plain interrupt) for the same turn wins.
"""

from __future__ import annotations

import contextlib
import threading
import types
from types import SimpleNamespace

import pytest

from tui_gateway import server
from tui_gateway.contracts.events import MessageCompletePayload
from tui_gateway.contracts.sessions import SessionInterruptParams


def record(sid: str, **state) -> dict:
    session = {
        "agent": None,
        "agent_ready": threading.Event(),
        "agent_error": None,
        "history": [],
        "history_lock": threading.RLock(),
        "running": False,
        "session_key": sid,
        **state,
    }
    server._sessions[sid] = session
    return session


@pytest.fixture
def sid(request):
    sid = f"pause-{request.node.name}"
    yield sid
    server._sessions.pop(sid, None)


def interrupt(sid: str, **params) -> dict:
    """Through the RPC admission, so the contract's params check runs as it does for a client."""
    return server.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "session.interrupt", "params": {"session_id": sid, **params}})


def running(sid: str) -> dict:
    return record(sid, agent_build_started=True, running=True, inflight_turn={"user": "read the plan"},
                  _run_thread=types.SimpleNamespace(is_alive=lambda: True))


def _turn(result):
    return SimpleNamespace(
        result=result, agent=SimpleNamespace(_session_title_hint=""), terminal_callback=None,
        receipt_committed=True, receipt_attempted=False, marker_key="", error_retained=False,
        error_detail="", prompt_text="read the plan",
    )


@pytest.fixture
def complete(monkeypatch):
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "render_message", lambda _text, _cols: None)
    monkeypatch.setattr(server, "_clear_inflight_turn", lambda _session: None)

    def run(session, result):
        session.setdefault("pending_title", None)
        session["history_lock"] = contextlib.nullcontext()
        payload, _raw, status = server._complete_turn_payload(session, _turn(result), None, 80)
        return payload, status

    return run


def test_the_contract_takes_resume_and_reports_paused():
    assert SessionInterruptParams.model_validate({"session_id": "s", "resume": True}).resume is True
    assert SessionInterruptParams.model_validate({"session_id": "s"}).resume is None
    assert MessageCompletePayload.model_validate({"status": "interrupted", "paused": True}).paused is True


def test_a_resume_interrupt_stops_the_turn_and_marks_it_paused(sid, complete):
    session = running(sid)

    response = interrupt(sid, resume=True)

    assert response.get("result", {}).get("status") == "interrupted", response
    assert session["_turn_cancel_requested"] is True
    payload, status = complete(session, {"final_response": "", "interrupted": True})
    assert (status, payload.get("paused")) == ("interrupted", True)
    # Consumed with the turn: the next turn's completion does not inherit it.
    payload, _ = complete(session, {"final_response": "", "interrupted": True})
    assert "paused" not in payload


def test_the_users_stop_wins_over_a_pause(sid, complete):
    session = running(sid)

    interrupt(sid, resume=True)
    interrupt(sid)
    interrupt(sid, resume=True)

    payload, status = complete(session, {"final_response": "", "interrupted": True})
    assert status == "interrupted"
    assert "paused" not in payload


def test_a_plain_interrupt_is_not_a_pause(sid, complete):
    session = running(sid)

    interrupt(sid)

    payload, _ = complete(session, {"final_response": "", "interrupted": True})
    assert "paused" not in payload


def test_a_turn_that_finished_before_the_pause_landed_is_not_paused(sid, complete):
    session = running(sid)

    interrupt(sid, resume=True)

    payload, status = complete(session, {"final_response": "done"})
    assert status == "complete"
    assert "paused" not in payload
    # Consumed with that turn all the same: a later completion of the record cannot report it.
    payload, _ = complete(session, {"final_response": "", "interrupted": True})
    assert "paused" not in payload


def test_a_pause_left_by_an_earlier_turn_does_not_reach_the_next_one(sid, complete):
    """The mark belongs to the turn it was asked for: a later turn interrupted some other way is not paused."""
    session = running(sid)
    interrupt(sid, resume=True)
    session["inflight_turn"] = {"user": "the next prompt"}  # that turn ended without a completion; a new one runs

    payload, status = complete(session, {"final_response": "", "interrupted": True})
    assert status == "interrupted"
    assert "paused" not in payload


def test_a_resume_interrupt_on_an_idle_session_marks_nothing(sid, complete):
    session = record(sid)

    interrupt(sid, resume=True)

    assert not session.get("inflight_turn")
    payload, _ = complete(session, {"final_response": "", "interrupted": True})
    assert "paused" not in payload


def correct(sid: str, verb: str) -> dict:
    return server.handle_request(
        {"jsonrpc": "2.0", "id": 2, "method": f"session.{verb}", "params": {"session_id": sid, "text": "and the appendix"}})


@pytest.mark.parametrize("verb", ["steer", "redirect"])
def test_a_correction_the_paused_turn_takes_keeps_the_pause(sid, complete, verb):
    """An accepted steer or redirect replaces the turn's inflight record with a copy (``_record_inflight_correction``);
    it is the same turn, so its completion still says ``paused``."""
    session = running(sid)
    session["agent"] = SimpleNamespace(steer=lambda text: True, redirect=lambda text: True,
                                       _supports_active_turn_redirect=True)
    before = session["inflight_turn"]

    interrupt(sid, resume=True)
    response = correct(sid, verb)

    assert response.get("result", {}).get("status") in ("queued", "redirected"), response
    assert session["inflight_turn"] is not before and session["inflight_turn"]["corrections"] == ["and the appendix"]
    payload, status = complete(session, {"final_response": "", "interrupted": True})
    assert (status, payload.get("paused")) == ("interrupted", True)


def test_the_users_stop_after_a_correction_still_wins(sid, complete):
    session = running(sid)
    session["agent"] = SimpleNamespace(steer=lambda text: True)

    interrupt(sid, resume=True)
    correct(sid, "steer")
    interrupt(sid)
    interrupt(sid, resume=True)

    payload, status = complete(session, {"final_response": "", "interrupted": True})
    assert status == "interrupted"
    assert "paused" not in payload
