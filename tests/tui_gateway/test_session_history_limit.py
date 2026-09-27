"""``session.history`` may bound its reply to the newest messages (#frame-cap reports).

A client that renders a 100-message window has to receive, and its transport has to carry, the whole
transcript — one long conversation then produces a reply above the client's frame cap and the
conversation stops opening for that client. ``limit`` bounds the reply; the count must stay the whole
lineage so a windowed client still has an authority to page "older" from.
"""

from __future__ import annotations

import threading

import pytest

from hermes_state import SessionDB
import tui_gateway.server as server

MESSAGES = 40


@pytest.fixture()
def session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("stored-session", source="desktop")
    for index in range(MESSAGES):
        db.append_message("stored-session", "user", f"m{index}")
    previous_db = server._db
    setattr(server, "_db", db)
    server._sessions["runtime-session"] = {
        "session_key": "stored-session",
        "history": [],
        "history_lock": threading.Lock(),
        "running": False,
        "agent": None,
    }
    try:
        yield "runtime-session"
    finally:
        server._sessions.pop("runtime-session", None)
        setattr(server, "_db", previous_db)
        db.close()


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
    # The count is the authority a windowed client pages "older" from, so the window never shrinks it.
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
