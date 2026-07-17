"""Tests for the hermes serve WebSocket relaying Codex app-server tool events.

On the Codex app-server route, native tools execute *inside* Codex, so the
authoritative ``tool_start_callback`` / ``tool_complete_callback`` never fire.
The only tool signal that reaches ``_on_tool_progress`` is the
``tool.started`` / ``tool.completed`` progress events bridged from the Codex
event stream. These must be relayed to WS clients (tool.start / tool.complete)
on that route, while remaining suppressed/ignored on the normal route (where
the authoritative emitters already fire and would otherwise be duplicated).

Covers issue #66360 Layers 1+2 (Layer 3, webSearch bridging, is already fixed
on main).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(return_value="/tmp/hermes_test_codex_ws")
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        import importlib

        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()


@pytest.fixture()
def emits(server, monkeypatch):
    captured: list = []
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, sid, payload=None: captured.append((event, sid, payload)),
    )
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda sid: True)
    return captured


def _seed_session(server, sid: str, api_mode: str) -> None:
    """Register a session whose agent reports the given ``api_mode``."""
    agent = MagicMock()
    agent.api_mode = api_mode
    server._sessions[sid] = {"agent": agent, "tool_started_at": {}}


# ── Layer 2: tool.started is relayed on the Codex route ───────────────

def test_codex_route_tool_started_emits_tool_start(server, emits):
    _seed_session(server, "sid-cx", "codex_app_server")
    server._on_tool_progress(
        "sid-cx", "tool.started", "web_search", "searching the web", {"query": "x"}
    )
    assert len(emits) == 1
    event, sid, payload = emits[0]
    assert event == "tool.start"
    assert sid == "sid-cx"
    assert payload["name"] == "web_search"


def test_normal_route_tool_started_still_suppressed(server, emits):
    """Guard: the normal route must not duplicate the authoritative tool.start."""
    _seed_session(server, "sid-norm", "")
    server._on_tool_progress(
        "sid-norm", "tool.started", "terminal", "running cmd", {"command": "ls"}
    )
    assert emits == []


# ── Layer 1: tool.completed is relayed on the Codex route ─────────────

def test_codex_route_tool_completed_emits_tool_complete_with_result(server, emits):
    _seed_session(server, "sid-cx", "codex_app_server")
    server._on_tool_progress(
        "sid-cx", "tool.completed", "web_search", None, None,
        result=json.dumps({"urls": ["https://example.com"]}),
        duration=1.25, is_error=False,
    )
    assert len(emits) == 1
    event, sid, payload = emits[0]
    assert event == "tool.complete"
    assert sid == "sid-cx"
    assert payload["name"] == "web_search"
    # JSON-string result is parsed, mirroring _on_tool_complete.
    assert payload["result"] == {"urls": ["https://example.com"]}
    assert payload["duration_s"] == 1.25


def test_codex_route_tool_completed_error_flag_forwarded(server, emits):
    _seed_session(server, "sid-cx", "codex_app_server")
    server._on_tool_progress(
        "sid-cx", "tool.completed", "commandExecution", None, None,
        result="[error] boom", is_error=True,
    )
    assert len(emits) == 1
    event, _sid, payload = emits[0]
    assert event == "tool.complete"
    assert payload["is_error"] is True
    # Non-JSON result string is preserved verbatim.
    assert payload["result"] == "[error] boom"


def test_normal_route_tool_completed_still_ignored(server, emits):
    """Guard: the normal route already emits the authoritative tool.complete."""
    _seed_session(server, "sid-norm", "")
    server._on_tool_progress(
        "sid-norm", "tool.completed", "terminal", None, None, result="ok"
    )
    assert emits == []


# ── Robustness: unknown api_mode / missing session do not crash ────────

def test_no_session_tool_started_does_not_crash(server, emits):
    server._on_tool_progress(
        "orphan-sid", "tool.started", "web_search", "p", {}
    )
    assert emits == []


def test_no_session_tool_completed_does_not_crash(server, emits):
    server._on_tool_progress(
        "orphan-sid", "tool.completed", "web_search", None, None, result="ok"
    )
    assert emits == []
