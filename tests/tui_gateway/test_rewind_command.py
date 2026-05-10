"""Tests for /rewind handling in tui_gateway.

The TUI routes ``/rewind`` through ``command.dispatch`` (it's in
``_PENDING_INPUT_COMMANDS`` because the CLI handler queues input the
slash-worker subprocess can't read). The server handles it directly,
mutates SessionDB to soft-delete rows, refreshes the in-memory session
history, fires the memory-provider hook with ``rewound=True``, and
returns ``{"type": "prefill", "message": <text>, "notice": ...}`` so
the Ink client drops the message into the composer for editing.
See issue #21910.
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


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
        mod._methods.clear()
        importlib.reload(mod)


@pytest.fixture()
def db(hermes_home):
    return SessionDB(db_path=hermes_home / "state.db")


@pytest.fixture()
def session_with_history(server, db):
    """Build a session with 3 user turns + assistant replies persisted in DB."""
    sid = "sid-rewind"
    session_key = "tui-rewind-1"
    db.create_session(session_key, source="tui")
    for i in range(1, 4):
        db.append_message(session_key, "user", f"question {i}")
        db.append_message(session_key, "assistant", f"answer {i}")
    history = db.get_messages_as_conversation(session_key)
    agent = MagicMock()
    agent._memory_manager = MagicMock()
    agent._last_flushed_db_idx = len(history)
    s = {
        "session_key": session_key,
        "history": list(history),
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "agent": agent,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid] = s
    # Wire the DB cache so _get_db() returns our fixture.
    server._db = db
    return sid, session_key, s, agent


def _call(server, method, **params):
    return server._methods[method](1, params)


def test_rewind_returns_prefill_with_target_text(server, session_with_history):
    sid, session_key, s, agent = session_with_history
    resp = _call(server, "command.dispatch", session_id=sid, name="rewind", arg="")
    result = resp["result"]
    assert result["type"] == "prefill"
    # v1 auto-picks the most recent user turn — "question 3"
    assert result["message"] == "question 3"
    assert "Rewound" in result["notice"]


def test_rewind_truncates_in_memory_history(server, session_with_history, db):
    sid, session_key, s, agent = session_with_history
    _call(server, "command.dispatch", session_id=sid, name="rewind", arg="")
    # After rewinding to "question 3", active history should be 4 rows:
    # user q1, asst a1, user q2, asst a2
    assert len(s["history"]) == 4
    roles = [m["role"] for m in s["history"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    # version bumped
    assert s["history_version"] == 1


def test_rewind_soft_deletes_rows_in_db(server, session_with_history, db):
    sid, session_key, _, _ = session_with_history
    _call(server, "command.dispatch", session_id=sid, name="rewind", arg="")
    # All rows still present
    all_rows = db.get_messages(session_key, include_inactive=True)
    assert len(all_rows) == 6
    # 2 inactive (the "question 3" row + its trailing siblings — here just
    # "question 3" + "answer 3", since target was the q3 user row).
    active = [r for r in all_rows if r["active"] == 1]
    assert len(active) == 4
    # rewind_count bumped
    sess = db.get_session(session_key)
    assert sess["rewind_count"] == 1


def test_rewind_notifies_memory_provider(server, session_with_history):
    sid, session_key, _, agent = session_with_history
    _call(server, "command.dispatch", session_id=sid, name="rewind", arg="")
    agent._memory_manager.on_session_switch.assert_called_once()
    args, kwargs = agent._memory_manager.on_session_switch.call_args
    assert args[0] == session_key
    assert kwargs["rewound"] is True
    assert kwargs["reset"] is False


def test_rewind_refuses_when_session_busy(server, session_with_history):
    sid, _, s, _ = session_with_history
    s["running"] = True
    resp = _call(server, "command.dispatch", session_id=sid, name="rewind", arg="")
    assert "error" in resp
    assert "busy" in resp["error"]["message"].lower()


def test_rewind_errors_when_no_active_session(server):
    resp = _call(server, "command.dispatch", session_id="no-such-sid", name="rewind", arg="")
    assert "error" in resp
    assert "no active session" in resp["error"]["message"].lower()


def test_rewind_in_pending_input_commands(server):
    """Registry sanity: /rewind must be in _PENDING_INPUT_COMMANDS so
    slash.exec rejects it and the TUI falls through to command.dispatch."""
    assert "rewind" in server._PENDING_INPUT_COMMANDS
