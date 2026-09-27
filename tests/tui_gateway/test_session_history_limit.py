"""``session.history`` may bound its reply to the newest messages (#frame-cap reports).

A client that renders a 100-message window has to receive, and its transport has to carry, the whole
transcript — one long conversation then produces a reply above the client's frame cap and the
conversation stops opening for that client. ``limit`` bounds the reply to the newest display messages;
``count`` keeps its existing meaning (the whole lineage), so no existing caller sees a change.
"""

from __future__ import annotations

import threading

import pytest

from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session
from tui_gateway import server

MESSAGES = 40


@pytest.fixture()
def db(tmp_path, monkeypatch):
    store = open_persistence(profile_scope(home=tmp_path))
    seed_session(store, "stored-session", "desktop")
    seed_messages(store, "stored-session", *({"role": "user", "content": f"m{index}"} for index in range(MESSAGES)))
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_db", store)
    server._sessions["runtime-session"] = {
        "session_key": "stored-session",
        "history": [],
        "history_lock": threading.Lock(),
        "running": False,
        "agent": None,
    }
    try:
        yield store
    finally:
        server._sessions.pop("runtime-session", None)
        store.close()


@pytest.fixture()
def session(db):
    return "runtime-session"


def _history(session_id: str, **params) -> dict:
    response = server.handle_request(
        {"id": "1", "method": "session.history", "params": {"session_id": session_id, **params}}
    )
    assert isinstance(response, dict)
    assert "error" not in response, response
    return response["result"]


def test_limit_returns_the_newest_window_and_keeps_the_lineage_count(session):
    result = _history(session, limit=10)
    assert [message["text"] for message in result["messages"]] == [f"m{i}" for i in range(MESSAGES - 10, MESSAGES)]
    assert result["count"] == MESSAGES


def test_omitted_limit_returns_the_whole_transcript(session):
    result = _history(session)
    assert len(result["messages"]) == MESSAGES
    assert result["count"] == MESSAGES


def test_a_limit_past_the_lineage_returns_everything(session):
    result = _history(session, limit=MESSAGES + 100)
    assert len(result["messages"]) == MESSAGES
    assert result["count"] == MESSAGES


@pytest.mark.parametrize("limit", [0, -1])
def test_a_non_positive_limit_means_no_window(session, limit):
    result = _history(session, limit=limit)
    assert len(result["messages"]) == MESSAGES


def test_an_unknown_parameter_is_still_refused(session):
    """The refusal a mismatched client sees; `limit` must not have widened the contract."""
    response = server.handle_request(
        {"id": "2", "method": "session.history", "params": {"session_id": session, "window": 10}}
    )
    assert response["error"]["code"] == 4000
    assert "window" in response["error"]["message"]


def test_a_tool_row_at_the_window_edge_keeps_the_args_of_its_call(db, session):
    """The window cuts after the tool row is paired with the assistant row that issued its call."""
    call = {"id": "call-1", "type": "function",
            "function": {"name": "terminal", "arguments": '{"command": "ls"}'}}
    seed_messages(db, "stored-session",
                  {"role": "assistant", "content": "", "tool_calls": [call]},
                  {"role": "tool", "content": "file.txt", "tool_name": "terminal", "tool_call_id": "call-1"},
                  {"role": "assistant", "content": "done"})

    edge = _history(session, limit=2)["messages"]

    assert [message["role"] for message in edge] == ["tool", "assistant"]
    assert edge[0]["args"] == {"command": "ls"}
