"""Tests for /drop handling in tui_gateway.

The TUI routes ``/drop`` through ``command.dispatch`` (it's in
``_PENDING_INPUT_COMMANDS``).  The server calls ``drop_messages`` on the
live agent to prune trailing raw messages from ``_session_messages`` and
deactivate the corresponding rows in state.db, then reloads the active
transcript into ``session["history"]``.

``/drop``    — removes from the last tool result to the tail (auto mode).
``/drop N``  — removes exactly N raw messages from the tail.
``/drop abc``— returns an error without touching history.
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# Fixtures (mirrors test_undo_command.py)
# ---------------------------------------------------------------------------


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def server(hermes_home):
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()
        mod._db = None


@pytest.fixture()
def db(hermes_home):
    return SessionDB(db_path=hermes_home / "state.db")


def _make_session(server, db, *, with_tool_result: bool = True):
    """Return (sid, session_key, session_dict, agent_mock).

    Builds a session with:
      user1, assistant1, user2, assistant(tool_call), tool_result, user3

    when ``with_tool_result=True``.  When False, only user+assistant pairs.
    """
    sid = "sid-drop"
    session_key = "tui-drop-1"
    db.create_session(session_key, source="tui")

    messages = []
    if with_tool_result:
        db.append_message(session_key, "user", "question 1")
        db.append_message(session_key, "assistant", "answer 1")
        db.append_message(session_key, "user", "question 2")
        # Simulate assistant tool call + result
        import json
        db.append_message(
            session_key, "assistant",
            json.dumps({"tool_calls": [{"id": "tc1", "type": "function",
                                        "function": {"name": "terminal",
                                                     "arguments": "{}"}}]})
        )
        db.append_message(session_key, "tool", "some tool output")
        db.append_message(session_key, "user", "question 3")
    else:
        db.append_message(session_key, "user", "question 1")
        db.append_message(session_key, "assistant", "answer 1")
        db.append_message(session_key, "user", "question 2")
        db.append_message(session_key, "assistant", "answer 2")

    history = db.get_messages_as_conversation(session_key)

    # Build a fake agent whose _session_messages IS the history list.
    agent = MagicMock()
    agent._session_messages = list(history)
    agent._session_db = db
    agent.session_id = session_key
    agent._last_flushed_db_idx = len(agent._session_messages)
    agent._empty_content_retries = 2
    agent._content_filter_refusal_count = 3

    s = {
        "session_key": session_key,
        "history": agent._session_messages,  # same list object
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "agent": agent,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid] = s
    server._db = db
    return sid, session_key, s, agent


def _call(server, method, **params):
    return server._methods[method](1, params)


# ---------------------------------------------------------------------------
# /drop auto mode (N=None → from last tool result)
# ---------------------------------------------------------------------------


def test_drop_auto_returns_exec_type(server, db):
    sid, _, _, _ = _make_session(server, db)
    resp = _call(server, "command.dispatch", session_id=sid, name="drop", arg="")
    assert "result" in resp, resp
    assert resp["result"]["type"] == "exec"
    assert "Dropped" in resp["result"]["output"]


def test_drop_auto_removes_from_last_tool_result(server, db):
    """Auto /drop should remove from the last tool result to the tail."""
    sid, _, s, agent = _make_session(server, db)
    before = len(agent._session_messages)
    resp = _call(server, "command.dispatch", session_id=sid, name="drop", arg="")
    assert "result" in resp, resp
    # tool_result + user3 were removed (at minimum 1 message dropped)
    assert len(agent._session_messages) < before


def test_drop_auto_no_tool_result_returns_error(server, db):
    """When there's no tool result, /drop should return an error."""
    sid, _, _, _ = _make_session(server, db, with_tool_result=False)
    resp = _call(server, "command.dispatch", session_id=sid, name="drop", arg="")
    # drop_messages returns error when no tool result exists
    assert "error" in resp


# ---------------------------------------------------------------------------
# /drop N (explicit message count)
# ---------------------------------------------------------------------------


def test_drop_n_removes_exactly_n_messages(server, db):
    sid, _, s, agent = _make_session(server, db)
    before = len(agent._session_messages)
    resp = _call(server, "command.dispatch", session_id=sid, name="drop", arg="2")
    assert "result" in resp, resp
    assert resp["result"]["type"] == "exec"
    assert len(agent._session_messages) == before - 2


def test_drop_n_1_removes_last_message(server, db):
    sid, _, s, agent = _make_session(server, db)
    before = len(agent._session_messages)
    resp = _call(server, "command.dispatch", session_id=sid, name="drop", arg="1")
    assert "result" in resp, resp
    assert len(agent._session_messages) == before - 1


def test_drop_n_output_mentions_count(server, db):
    sid, _, _, _ = _make_session(server, db)
    resp = _call(server, "command.dispatch", session_id=sid, name="drop", arg="1")
    assert "result" in resp, resp
    assert "1" in resp["result"]["output"]


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_drop_invalid_count_returns_error(server, db):
    sid, _, _, _ = _make_session(server, db)
    resp = _call(server, "command.dispatch", session_id=sid, name="drop", arg="abc")
    assert "error" in resp
    assert "invalid count" in resp["error"]["message"].lower()


def test_drop_refuses_when_session_busy(server, db):
    sid, _, s, _ = _make_session(server, db)
    s["running"] = True
    resp = _call(server, "command.dispatch", session_id=sid, name="drop", arg="")
    assert "error" in resp
    assert "busy" in resp["error"]["message"].lower()


def test_drop_errors_when_no_active_session(server):
    resp = _call(server, "command.dispatch", session_id="no-such-sid",
                 name="drop", arg="")
    assert "error" in resp
    assert "no active session" in resp["error"]["message"].lower()


# ---------------------------------------------------------------------------
# History reload and agent surgery
# ---------------------------------------------------------------------------


def test_drop_bumps_history_version(server, db):
    sid, _, s, _ = _make_session(server, db)
    before_version = s["history_version"]
    _call(server, "command.dispatch", session_id=sid, name="drop", arg="1")
    assert s["history_version"] > before_version


def test_drop_calls_invalidate_system_prompt(server, db):
    sid, _, _, agent = _make_session(server, db)
    agent._invalidate_system_prompt = MagicMock()
    _call(server, "command.dispatch", session_id=sid, name="drop", arg="1")
    agent._invalidate_system_prompt.assert_called_once()


def test_drop_resets_refusal_counters(server, db):
    """drop_messages resets _content_filter_refusal_count + other counters."""
    sid, _, _, agent = _make_session(server, db)
    # Confirm they are non-zero before the drop
    assert agent._content_filter_refusal_count == 3
    _call(server, "command.dispatch", session_id=sid, name="drop", arg="1")
    # After drop, the counter should be zeroed
    assert agent._content_filter_refusal_count == 0


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_drop_in_pending_input_commands(server):
    """/drop must be in _PENDING_INPUT_COMMANDS so slash.exec routes it
    to command.dispatch (not the slash-worker subprocess)."""
    assert "drop" in server._PENDING_INPUT_COMMANDS
