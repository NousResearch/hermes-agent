"""Tests for /learn handling in tui_gateway.

The TUI/desktop app routes slash commands through ``slash.exec`` (the slash
worker subprocess) by default. ``/learn``'s CLI handler
(``_handle_learn_command``) seeds the kickoff prompt onto the agent's input
queue, which the slash-worker subprocess has no reader for — so when ``learn``
was missing from ``_PENDING_INPUT_COMMANDS`` the worker ran the handler, the
"Learning a skill..." line printed, and the prompt was dropped: nothing
happened. ``/learn`` must instead route to ``command.dispatch``, which returns
a ``{"type": "send", "message": <build_learn_prompt(arg)>}`` payload the TUI
fires as a normal agent turn (same path as /goal, /queue, /retry).

Regression guard for that gap.
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
    s = {
        "session_key": "tui-learn-session-1",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid] = s
    return sid, s


def _call(server, method, **params):
    handler = server._methods[method]
    return handler(1, params)


# ── command.dispatch /learn ───────────────────────────────────────────


def test_learn_returns_send_with_built_prompt(server, session):
    sid, _ = session
    r = _call(
        server,
        "command.dispatch",
        name="learn",
        arg="the auth flow in ~/projects/acme",
        session_id=sid,
    )
    result = r["result"]
    assert result["type"] == "send"
    # The message is the standards-guided prompt, not the raw user text.
    assert "skill_manage" in result["message"]
    assert "~/projects/acme" in result["message"]


def test_bare_learn_falls_back_to_the_conversation(server, session):
    sid, _ = session
    r = _call(server, "command.dispatch", name="learn", arg="", session_id=sid)
    result = r["result"]
    assert result["type"] == "send"
    assert "conversation" in result["message"].lower()


# ── slash.exec /learn routing ─────────────────────────────────────────


def test_slash_exec_routes_learn_to_command_dispatch(server, session):
    """slash.exec must route /learn to command.dispatch internally rather
    than to the slash worker. The worker would run the CLI handler, which
    queues onto _pending_input — a queue the worker subprocess can't drain —
    so the prompt would be silently dropped ("Learning..." prints, nothing
    runs)."""
    sid, _ = session
    r = _call(server, "slash.exec", command="learn what we just did", session_id=sid)
    assert "result" in r
    assert r["result"]["type"] == "send"
    assert "skill_manage" in r["result"]["message"]


def test_pending_input_commands_includes_learn(server):
    """Guard: _PENDING_INPUT_COMMANDS must list 'learn' — removing it sends
    /learn to the slash worker and silently drops the prompt."""
    assert "learn" in server._PENDING_INPUT_COMMANDS
