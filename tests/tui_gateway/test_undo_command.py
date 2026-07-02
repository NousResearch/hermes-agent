"""Tests for /undo handling in tui_gateway.

The TUI routes ``/undo`` through ``command.dispatch`` (it's in
``_PENDING_INPUT_COMMANDS`` because the CLI handler queues input the
slash-worker subprocess can't read). The server handles it directly,
mutates SessionDB to soft-delete rows, refreshes the in-memory session
history, fires the memory-provider hook with ``rewound=True``, and
returns ``{"type": "prefill", "message": <text>, "notice": ...}`` so
the Ink client drops the message into the composer for editing.

``/undo N`` backs up N user turns at once (default 1). See issue #21910.
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
    # Mocks are scoped to the initial import only (see
    # tests/tui_gateway/test_protocol.py for the rationale).
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")

    methods = dict(mod._methods)
    yield mod
    # Restore in place instead of clear+reload: importlib.reload
    # re-registers atexit hooks (duplicate ThreadPoolExecutor shutdowns
    # race the stderr buffer at interpreter exit — same class as PR #34217)
    # and re-captures module-level paths like _hermes_home against this
    # test's soon-deleted tmpdir, breaking later files in the same process.
    mod._methods.clear()
    mod._methods.update(methods)
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()
    mod._db = None


@pytest.fixture()
def db(hermes_home):
    return SessionDB(db_path=hermes_home / "state.db")


@pytest.fixture()
def session_with_history(server, db):
    """Build a session with 3 user turns + assistant replies persisted in DB."""
    sid = "sid-undo"
    session_key = "tui-undo-1"
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


def test_undo_returns_prefill_with_target_text(server, session_with_history):
    sid, session_key, s, agent = session_with_history
    resp = _call(server, "command.dispatch", session_id=sid, name="undo", arg="")
    result = resp["result"]
    assert result["type"] == "prefill"
    # Default /undo backs up one user turn — "question 3"
    assert result["message"] == "question 3"
    assert "Undid" in result["notice"]


def test_undo_truncates_in_memory_history(server, session_with_history, db):
    sid, session_key, s, agent = session_with_history
    _call(server, "command.dispatch", session_id=sid, name="undo", arg="")
    # After undoing to "question 3", active history should be 4 rows:
    # user q1, asst a1, user q2, asst a2
    assert len(s["history"]) == 4
    roles = [m["role"] for m in s["history"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    # version bumped
    assert s["history_version"] == 1


def test_undo_n_backs_up_multiple_turns(server, session_with_history, db):
    """/undo 2 backs up two user turns to "question 2"."""
    sid, session_key, s, agent = session_with_history
    resp = _call(server, "command.dispatch", session_id=sid, name="undo", arg="2")
    result = resp["result"]
    assert result["type"] == "prefill"
    assert result["message"] == "question 2"
    assert "2 turns" in result["notice"]
    # Active history truncated to user q1 + asst a1
    assert len(s["history"]) == 2
    assert [m["role"] for m in s["history"]] == ["user", "assistant"]


def test_undo_n_clamps_to_oldest_turn(server, session_with_history, db):
    """/undo with N larger than the number of user turns backs up to the oldest."""
    sid, session_key, s, agent = session_with_history
    resp = _call(server, "command.dispatch", session_id=sid, name="undo", arg="99")
    result = resp["result"]
    assert result["message"] == "question 1"
    assert len(s["history"]) == 0


def test_undo_rejects_invalid_count(server, session_with_history):
    sid, _, _, _ = session_with_history
    resp = _call(server, "command.dispatch", session_id=sid, name="undo", arg="abc")
    assert "error" in resp
    assert "invalid count" in resp["error"]["message"].lower()


def test_undo_soft_deletes_rows_in_db(server, session_with_history, db):
    sid, session_key, _, _ = session_with_history
    _call(server, "command.dispatch", session_id=sid, name="undo", arg="")
    # All rows still present
    all_rows = db.get_messages(session_key, include_inactive=True)
    assert len(all_rows) == 6
    # 2 inactive (the "question 3" row + its trailing "answer 3").
    active = [r for r in all_rows if r["active"] == 1]
    assert len(active) == 4
    # rewind_count bumped
    sess = db.get_session(session_key)
    assert sess["rewind_count"] == 1


def test_undo_notifies_memory_provider(server, session_with_history):
    sid, session_key, _, agent = session_with_history
    _call(server, "command.dispatch", session_id=sid, name="undo", arg="")
    agent._memory_manager.on_session_switch.assert_called_once()
    args, kwargs = agent._memory_manager.on_session_switch.call_args
    assert args[0] == session_key
    assert kwargs["rewound"] is True
    assert kwargs["reset"] is False


def test_undo_refuses_when_session_busy(server, session_with_history):
    sid, _, s, _ = session_with_history
    s["running"] = True
    resp = _call(server, "command.dispatch", session_id=sid, name="undo", arg="")
    assert "error" in resp
    assert "busy" in resp["error"]["message"].lower()


def test_undo_errors_when_no_active_session(server):
    resp = _call(server, "command.dispatch", session_id="no-such-sid", name="undo", arg="")
    assert "error" in resp
    assert "no active session" in resp["error"]["message"].lower()


def test_undo_in_pending_input_commands(server):
    """Registry sanity: /undo must be in _PENDING_INPUT_COMMANDS so
    slash.exec rejects it and the TUI falls through to command.dispatch."""
    assert "undo" in server._PENDING_INPUT_COMMANDS
