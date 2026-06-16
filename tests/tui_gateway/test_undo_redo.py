from __future__ import annotations

import importlib
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import hermes_undo
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
        hermes_undo.clear_state()
        importlib.reload(mod)


@pytest.fixture()
def db(server, hermes_home):
    session_db = SessionDB(db_path=hermes_home / "state.db")
    server._db = session_db
    hermes_undo._session_db = session_db
    hermes_undo.clear_state()
    yield session_db
    session_db.close()
    hermes_undo.clear_state()


def _session(server, db, sid="sid-redo", session_key="tui-redo"):
    db.create_session(session_key, source="tui")
    session = {
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "agent": MagicMock(_memory_manager=MagicMock()),
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid] = session
    return sid, session


def _call(server, method, **params):
    return server._methods[method](1, params)


def test_session_redo_round_trip_restores_active_rows_and_history(server, db):
    sid, session = _session(server, db)
    ids = [
        db.append_message(session["session_key"], "user", "q1"),
        db.append_message(session["session_key"], "assistant", "a1"),
        db.append_message(session["session_key"], "user", "q2"),
        db.append_message(session["session_key"], "assistant", "a2"),
    ]
    session["history"] = db.get_messages_as_conversation(session["session_key"])
    before = [row["id"] for row in db.get_messages(session["session_key"])]

    undo = _call(server, "session.undo", session_id=sid, n=1)
    assert undo["result"]["rewound_ids"] == [ids[-1]]
    redo = _call(server, "session.redo", session_id=sid)

    assert redo["result"] == {
        "reactivated_count": 1,
        "new_tail_id": ids[-1],
        "prefill_text": None,
    }
    assert [row["id"] for row in db.get_messages(session["session_key"])] == before
    assert session["history"] == db.get_messages_as_conversation(session["session_key"])


def test_session_redo_busy_guard_code_4009(server, db):
    sid, session = _session(server, db, sid="sid-busy", session_key="tui-busy")
    session["running"] = True

    resp = _call(server, "session.redo", session_id=sid)

    assert resp["error"]["code"] == 4009
    assert "busy" in resp["error"]["message"]


def test_prompt_submit_clears_redo_stack_on_user_send(server, db, monkeypatch):
    sid, session = _session(server, db, sid="sid-send", session_key="tui-send")
    db.append_message(session["session_key"], "user", "q")
    db.append_message(session["session_key"], "assistant", "a")
    hermes_undo.undo(session["session_key"], 1)
    hermes_undo.redo(session["session_key"], 1)
    state = hermes_undo.get_state(session["session_key"])
    assert state.redo_stack

    monkeypatch.setattr(server, "_start_agent_build", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_start_inflight_turn", lambda sess, text: None)

    class NoThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

    monkeypatch.setattr(server.threading, "Thread", NoThread)

    resp = _call(server, "prompt.submit", session_id=sid, text="new branch")

    assert resp["result"]["status"] == "streaming"
    assert state.redo_stack == []
