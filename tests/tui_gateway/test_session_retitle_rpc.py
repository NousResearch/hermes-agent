from __future__ import annotations

import contextlib
import importlib
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
    monkeypatch.setattr(mod, "_hermes_home", home)
    yield mod
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()


def _session(server):
    sid = f"sid-retitle-{uuid.uuid4().hex}"
    key = f"stored-retitle-{uuid.uuid4().hex}"
    history = [
        {"role": "user", "content": "Fix the title generator"},
        {"role": "assistant", "content": "I found the context-selection bug."},
    ]
    entry = {
        "session_key": key,
        "history": history,
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
        "agent": None,
    }
    server._sessions[sid] = entry
    return sid, key, history


def test_retitle_rpc_is_registered_as_long_handler(server, monkeypatch):
    sid, key, history = _session(server)
    db = MagicMock()

    @contextlib.contextmanager
    def session_db(_session):
        yield db

    monkeypatch.setattr(server, "_session_db", session_db)

    with patch("agent.session_retitle.retitle_session", return_value="Fix title generator") as retitle:
        response = server._methods["session.retitle"](77, {"session_id": sid})

    assert "session.retitle" in server._LONG_HANDLERS
    assert response["result"] == {"title": "Fix title generator"}
    retitle.assert_called_once_with(db, key, history)


def test_retitle_rpc_reports_missing_context(server, monkeypatch):
    sid, _, _ = _session(server)
    db = MagicMock()

    @contextlib.contextmanager
    def session_db(_session):
        yield db

    monkeypatch.setattr(server, "_session_db", session_db)

    with patch("agent.session_retitle.retitle_session", return_value=None):
        response = server._methods["session.retitle"](78, {"session_id": sid})

    assert response["error"]["code"] == 4018
    assert "conversation context" in response["error"]["message"]
