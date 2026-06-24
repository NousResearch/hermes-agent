"""Tests for /learn routing in tui_gateway.

The TUI routes ``/learn`` through ``command.dispatch`` (not ``slash.exec``)
because the CLI's ``_handle_learn_command`` queues the learn prompt onto
``_pending_input``, which the slash-worker subprocess has no reader for.
Instead we handle ``/learn`` directly in the server and return a
``{"type": "send", "message": ...}`` payload the TUI client uses to submit
the learn prompt as a normal agent turn.
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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


@pytest.fixture()
def session(server):
    sid = "sid-test"
    session_key = "tui-learn-session-1"
    s = {
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid] = s
    return sid, session_key, s


def _call(server, method, **params):
    handler = server._methods[method]
    return handler(1, params)


# ── _PENDING_INPUT_COMMANDS guard ─────────────────────────────────────


def test_learn_in_pending_input_commands(server):
    """Guard: /learn must be in _PENDING_INPUT_COMMANDS so slash.exec routes
    it to command.dispatch instead of the slash worker subprocess."""
    assert "learn" in server._PENDING_INPUT_COMMANDS


# ── command.dispatch /learn ───────────────────────────────────────────


def test_command_dispatch_learn_returns_send(server, session):
    """command.dispatch /learn must return a 'send' dispatch with the learn
    prompt as the message, so the Desktop frontend submits it as a turn."""
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="learn", arg="how to use the terminal tool", session_id=sid)
    assert "result" in r
    assert r["result"]["type"] == "send"
    assert "message" in r["result"]
    assert "learn" in r["result"]["message"].lower() or "skill" in r["result"]["message"].lower()


def test_command_dispatch_learn_bare_uses_conversation(server, session):
    """command.dispatch /learn with no arg should build a prompt referencing
    the current conversation."""
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="learn", arg="", session_id=sid)
    assert "result" in r
    assert r["result"]["type"] == "send"
    assert "conversation" in r["result"]["message"].lower() or "skill" in r["result"]["message"].lower()


# ── slash.exec /learn routing ─────────────────────────────────────────


def test_slash_exec_routes_learn_to_command_dispatch(server, session):
    """slash.exec must route /learn directly to command.dispatch internally
    instead of the slash worker subprocess, so the learn prompt reaches
    the agent as a normal turn."""
    sid, _, _ = session
    r = _call(server, "slash.exec", command="learn how to use read_file", session_id=sid)
    assert "result" in r
    assert r["result"]["type"] == "send"
    assert "message" in r["result"]
    assert "learn" in r["result"]["message"].lower() or "skill" in r["result"]["message"].lower()
