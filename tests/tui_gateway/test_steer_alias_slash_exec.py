"""``/s`` (alias of ``/steer``) must reach command.dispatch via slash.exec.

Before the fix slash.exec compared the raw alias against ``_PENDING_INPUT_COMMANDS``,
so ``/s`` fell through to the slash-worker subprocess. That worker has no running
agent and no reader for its ``_pending_input`` queue: it printed "No agent running;
queued as next turn" and the steer text was silently lost.
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
    __import__("tui_gateway.server_requests", fromlist=["x"]).reset_for_tests()


def _session(server, running: bool):
    agent = MagicMock()
    agent.steer.return_value = True
    worker = MagicMock()
    worker.run.side_effect = AssertionError("/s must not reach the slash worker")
    s = {"agent": agent, "running": running, "session_key": "k-steer", "history": [],
         "history_lock": threading.Lock(), "slash_worker": worker}
    server._sessions["sid-steer"] = s
    return s, agent


@pytest.mark.parametrize("alias", ["/s", "/S", "/steer"])
def test_steer_alias_reaches_running_agent(server, alias):
    _s, agent = _session(server, running=True)
    res = server._methods["slash.exec"]("r1", {"session_id": "sid-steer", "command": f"{alias} go left"})
    assert "error" not in res, res
    agent.steer.assert_called_once_with("go left")


def test_steer_alias_idle_becomes_next_turn(server):
    _s, agent = _session(server, running=False)
    res = server._methods["slash.exec"]("r1", {"session_id": "sid-steer", "command": "/s go left"})
    assert res["result"]["type"] == "send" and res["result"]["message"] == "go left"
    agent.steer.assert_not_called()
