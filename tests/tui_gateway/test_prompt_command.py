"""/prompt (alias /compose) on GUI clients.

The CLI and TUI compose the next prompt in $EDITOR (``hermes_cli.commands``). A GUI
client has no editor, so the gateway answers with a ``prefill`` dispatch that seeds
the composer, the same directive /undo uses. It must never reach the slash worker
or print the system prompt (the removed pre-#6752 /prompt semantics).
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with patch.dict("sys.modules", {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()}):
        mod = importlib.import_module("tui_gateway.server")
    methods = dict(mod._methods)
    yield mod
    mod._methods.clear()
    mod._methods.update(methods)
    mod._sessions.clear()


@pytest.fixture()
def session(server, monkeypatch):
    class _ExplodingWorker:
        def __init__(self, *args, **kwargs):
            raise AssertionError("/prompt must not spawn the slash worker")

    agent = MagicMock()
    agent.ephemeral_system_prompt = "SECRET SYSTEM PROMPT"
    agent._cached_system_prompt = "SECRET SYSTEM PROMPT"
    server._sessions["sid"] = {
        "session_key": "key", "agent": agent, "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": False, "cols": 80,
    }
    monkeypatch.setattr(server, "_SlashWorker", _ExplodingWorker)
    return server._sessions["sid"]


def _call(server, method, params):
    return server.handle_request({"id": "1", "method": method, "params": {"session_id": "sid", **params}})


@pytest.mark.parametrize("command", ["prompt", "compose"])
def test_slash_exec_prefills_the_composer(server, session, command):
    resp = _call(server, "slash.exec", {"command": f"/{command} draft this  for me"})
    assert resp["result"] == {"type": "prefill", "message": "draft this  for me"}
    assert "SECRET" not in str(resp)


def test_command_dispatch_prefills_and_bare_prompt_keeps_the_draft(server, session):
    assert _call(server, "command.dispatch", {"name": "compose", "arg": "hi"})["result"] == {
        "type": "prefill", "message": "hi"}
    bare = _call(server, "command.dispatch", {"name": "prompt", "arg": ""})["result"]
    # An empty prefill never overwrites the client's draft; the notice explains usage.
    assert bare["type"] == "prefill" and bare["message"] == ""
    assert "/prompt <text>" in bare["notice"]
    assert "SECRET" not in str(bare)


def test_prompt_is_allowed_while_a_turn_runs(server, session):
    session["running"] = True
    assert _call(server, "slash.exec", {"command": "/prompt next"})["result"]["type"] == "prefill"
