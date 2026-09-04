"""TUI JSON-RPC contracts for Hermes-native non-blocking user input."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


QUESTIONS = [
    {
        "id": "name",
        "text": "What name should I use?",
        "options": [],
        "allow_free_text": True,
        "default": "",
    }
]


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    database.create_session("tui-session", source="desktop")
    database.create_session("other-session", source="desktop")
    try:
        yield database
    finally:
        database.close()


def _install_session(monkeypatch, server, db, *, sid="tui-sid", session_id="tui-session"):
    delivered = []

    class Agent:
        def __init__(self):
            self.session_id = session_id
            self._session_db = db
            self._current_turn_id = "turn-tui-1"
            self._model_request_active = None
            self._executing_tools = True

        def steer(self, text):
            delivered.append(text)
            return True

        def redirect(self, text):
            delivered.append(text)
            return True

    session = {
        "agent": Agent(),
        "session_key": sid,
        "history_lock": __import__("threading").RLock(),
        "agent_ready": __import__("threading").Event(),
        "running": True,
    }
    session["agent_ready"].set()
    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server, "_session_db", lambda _session: db)
    return session, delivered


def test_user_input_respond_is_session_scoped_and_emits_answer_event(monkeypatch, db):
    from tui_gateway import server

    db.create_pending_user_input(
        request_id="uir_tui_1",
        session_id="tui-session",
        questions=[{"id": "version", "text": "Version?", "options": ["stable", "beta"], "allow_free_text": False}],
        expires_at=time.time() + 60,
        turn_id="turn-tui-1",
    )
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))
    session, delivered = _install_session(monkeypatch, server, db)

    response = server.handle_request({
        "jsonrpc": "2.0",
        "id": "rpc-1",
        "method": "user_input.respond",
        "params": {"session_id": "tui-sid", "request_id": "uir_tui_1", "answers": {"version": "stable"}},
    })

    assert response["result"]["status"] == "answered"
    assert response["result"]["accepted"] is True
    assert delivered == [json.dumps({"version": "stable"}, ensure_ascii=False)]
    answer_events = [item for item in emitted if item[0] == "user_input.answer"]
    assert answer_events
    assert answer_events[-1][1] == "tui-sid"
    assert answer_events[-1][2]["request_id"] == "uir_tui_1"
    assert answer_events[-1][2]["status"] == "answered"

    # A different live session cannot resolve the same durable request.
    monkeypatch.setitem(server._sessions, "other-sid", {
        **session,
        "agent": SimpleNamespace(session_id="other-session", _session_db=db),
    })
    denied = server.handle_request({
        "jsonrpc": "2.0",
        "id": "rpc-2",
        "method": "user_input.respond",
        "params": {"session_id": "other-sid", "request_id": "uir_tui_1", "answers": {"version": "beta"}},
    })
    assert denied["error"]["code"] == 4001


def test_user_input_pending_replays_only_the_requesting_session(monkeypatch, db):
    from tui_gateway import server

    db.create_pending_user_input(
        request_id="uir_tui_2",
        session_id="tui-session",
        questions=QUESTIONS,
        context="reconnect",
        expires_at=time.time() + 60,
    )
    db.create_pending_user_input(
        request_id="uir_other",
        session_id="other-session",
        questions=QUESTIONS,
        expires_at=time.time() + 60,
    )
    _install_session(monkeypatch, server, db)

    response = server.handle_request({
        "jsonrpc": "2.0",
        "id": "rpc-3",
        "method": "user_input.pending",
        "params": {"session_id": "tui-sid"},
    })

    requests = response["result"]["requests"]
    assert [item["request_id"] for item in requests] == ["uir_tui_2"]
    assert requests[0]["session_id"] == "tui-session"


def test_user_input_request_is_emitted_on_the_session_event_rail(monkeypatch, db):
    from tools import user_input_tool
    from tui_gateway import server

    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))
    callback = server._agent_cbs("tui-sid")["user_input_callback"]
    monkeypatch.setattr(user_input_tool, "_shared_session_db", db)

    result = json.loads(
        user_input_tool.request_user_input(
            questions=QUESTIONS,
            context="event test",
            timeout_s=100,
            session_id="tui-session",
            turn_id="turn-tui-1",
            callback=callback,
            session_db=db,
            now=100.0,
        )
    )

    assert result["status"] == "pending"
    request_events = [item for item in emitted if item[0] == "user_input.request"]
    assert request_events
    event = request_events[-1]
    assert event[1] == "tui-sid"
    assert event[2]["request_id"] == result["request_id"]
    assert event[2]["status"] == "pending"
    assert event[2]["questions"] == QUESTIONS
    assert event[2]["expires_at"] == pytest.approx(200.0)
