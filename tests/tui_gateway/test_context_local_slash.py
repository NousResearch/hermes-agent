"""``/context`` must be answered IN-PROCESS for a local (non-compute-host) session.

The slash worker is a CLI subprocess whose ``self.agent`` is built lazily by the
first chat prompt and never by a slash command, so its /context answers
"(._.) No active agent -- send a message first." for ANY local session — active
or idle (#93280). The gateway already renders the view from the live session
(``_format_live_context_output``), with the desktop gauge's breakdown engine when
a live agent exists; this pins that routing so the command never falls through
to the worker for a local session.

Supersedes the narrower #81266, which only served the agentless case and pinned
the live-agent fallthrough (the bug) as intended behaviour.
"""

from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace
from unittest.mock import patch

from tui_gateway import server


def _live_session(agent, messages=2):
    return {
        "agent": agent,
        "session_key": "sk-1",
        "session_id": "sid-1",
        "history": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ][:messages],
        "history_lock": threading.Lock(),
        "running": False,
    }


def _no_db():
    """Render from in-memory history: no state.db needed in the test."""
    return contextlib.nullcontext()


def test_context_local_session_with_live_agent(monkeypatch):
    """THE bug: a live local session's /context fell through to the worker CLI
    (lazy agent, always None) and answered 'No active agent'."""
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: False)
    monkeypatch.setattr(server, "_session_db", _no_db)

    out = server._live_slash_command_output("sid-1", _live_session(object()), "context", "")

    assert out is not None, "/context fell through to the CLI worker for a local session"
    assert "No active agent" not in out
    assert "Conversation:" in out


def test_context_local_session_agentless(monkeypatch):
    """Agentless (idle/resumed) local session: same in-process path (#81266 case)."""
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: False)
    monkeypatch.setattr(server, "_session_db", _no_db)

    out = server._live_slash_command_output("sid-1", _live_session(None), "context", "")

    assert out is not None
    assert "No active agent" not in out
    assert "Conversation:" in out


def test_context_compute_host_route_unchanged(monkeypatch):
    """Guard: the isolated/compute-host route the gate was built for still works."""
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: True)
    monkeypatch.setattr(server, "_session_db", _no_db)

    out = server._live_slash_command_output("sid-1", _live_session(object()), "context", "")

    assert out is not None
    assert "Conversation:" in out


def test_tools_local_still_falls_through(monkeypatch):
    """Scope guard: /tools consults the agent, so it keeps the old routing."""
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: False)

    out = server._live_slash_command_output("sid-1", _live_session(None), "tools", "")

    assert out is None  # worker path, unchanged


def test_context_breakdown_engine_used_with_live_agent(monkeypatch):
    """With a live agent, the output carries the gauge engine's 'Context window'
    line (same compute_session_context_breakdown the popover RPC uses)."""
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: False)
    monkeypatch.setattr(server, "_session_db", _no_db)

    payload = {
        "categories": [
            {"id": "system_prompt", "label": "System prompt", "tokens": 8000, "color": "dim"},
            {"id": "conversation", "label": "Conversation", "tokens": 82000, "color": "normal"},
        ],
        "context_max": 131072,
        "context_percent": 68,
        "context_used": 90000,
        "context_source": "provider_usage",
        "context_estimated": False,
        "estimated_total": 90000,
        "model": "test-model",
    }
    with patch("agent.context_breakdown.compute_session_context_breakdown", return_value=payload):
        out = server._live_slash_command_output("sid-1", _live_session(object()), "context", "")

    assert out is not None
    assert "Context window:" in out
    assert "90,000" in out  # rendered from the breakdown payload, not the usage mirror


def test_context_all_arg_reaches_details(monkeypatch):
    """``/context all`` expands per-skill / per-toolset costs (compute_context_details)."""
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: False)
    monkeypatch.setattr(server, "_session_db", _no_db)

    payload = {
        "categories": [{"id": "conversation", "label": "Conversation", "tokens": 1000, "color": "normal"}],
        "context_max": 10000,
        "context_percent": 10,
        "context_used": 1000,
        "context_source": "provider_usage",
        "context_estimated": False,
        "estimated_total": 1000,
        "model": "test-model",
    }
    details = {"skills": [], "toolsets": []}
    with (
        patch("agent.context_breakdown.compute_session_context_breakdown", return_value=payload) as m_breakdown,
        patch("agent.context_breakdown.compute_context_details", return_value=details) as m_details,
    ):
        out = server._live_slash_command_output("sid-1", _live_session(object()), "context", "all")

    assert out is not None
    m_breakdown.assert_called_once()
    m_details.assert_called_once()
