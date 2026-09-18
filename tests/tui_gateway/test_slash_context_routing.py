"""Slash /context routing: non-isolated sessions must answer from the live gateway session.

The slash worker spawns its own HermesCLI that resumes the session but never builds an AIAgent
(lazy init on first chat message), so a /context routed there can only print "no active agent" —
after forking a full MCP-fleet worker to say it. /context is therefore answered live: the full
breakdown when the in-process agent exists, otherwise the DB/usage summary (the compute-host path).

Every test drives the real JSON-RPC entry point (``server.handle_request``); handler bodies are
rebound onto ``server.py``'s globals by ``method_ctx.HandlerRegistry.install()``, so calling a
handler function directly would bypass the path the gateway actually executes.
"""

from __future__ import annotations

import importlib
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB

SESSION_ID = "sid-ctx-route"
SESSION_KEY = "tui-ctx-route-1"


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def server(hermes_home):
    # Mocks are scoped to the initial import only (see tests/tui_gateway/test_protocol.py).
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
    # Restore in place instead of clear+reload (see tests/tui_gateway/test_session_profile_db.py).
    mod._methods.clear()
    mod._methods.update(methods)
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()
    mod._db = None


@pytest.fixture()
def launch_db(server, hermes_home):
    """The launch profile's state.db, wired in as the ``_get_db()`` handle."""
    db = SessionDB(db_path=hermes_home / "state.db")
    server._db = db
    return db


def _fake_agent():
    """SimpleNamespace with explicit None/zero attrs: MagicMock auto-attrs are truthy and would
    masquerade as usage anchors, memory stores, etc. in the breakdown engine."""
    return types.SimpleNamespace(
        model="test-model",
        tools=[
            {"type": "function", "function": {"name": "terminal", "description": "run"}}
        ],
        context_compressor=types.SimpleNamespace(
            context_length=200_000, last_prompt_tokens=1500
        ),
        _memory_store=None,
    )


def _register(server, *, agent, history):
    session = {
        "session_key": SESSION_KEY,
        "history": list(history),
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "agent": agent,
        "attached_images": [],
        "image_counter": 0,
        "cols": 120,
    }
    server._sessions[SESSION_ID] = session
    return session


def _recording_worker(srv, monkeypatch, spawned):
    class _Worker:
        def __init__(self, *a, **k):
            spawned.append(1)

        def run(self, cmd):
            return f"worker answered {cmd}"

        def close(self):
            pass

    monkeypatch.setattr(srv, "_SlashWorker", _Worker)


def _exec(server, command):
    resp = server.handle_request({
        "id": "1",
        "method": "slash.exec",
        "params": {"command": command, "session_id": SESSION_ID},
    })
    assert not resp.get("error"), f"slash.exec failed: {resp.get('error')}"
    return str(resp["result"]["output"])


HISTORY = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi there"},
]
PROMPT_PARTS = {
    "stable": "identity and guidance\n<available_skills>\n  demo:\n    - hello: hi\n</available_skills>",
    "context": "",
    "volatile": "Current time: now",
}


def test_context_non_isolated_answers_live_breakdown(server, monkeypatch):
    """A live in-process agent means /context renders the full breakdown WITHOUT forking a worker."""
    spawned = []
    _recording_worker(server, monkeypatch, spawned)
    _register(server, agent=_fake_agent(), history=HISTORY)

    with patch(
        "agent.system_prompt.build_system_prompt_parts", return_value=PROMPT_PARTS
    ):
        output = _exec(server, "/context")

    assert spawned == [], "non-isolated /context must not delegate to the slash worker"
    assert "(._.) No active agent" not in output
    # Full breakdown rendered by the real engine over the live agent's prompt parts.
    assert "Model: test-model" in output
    assert "Estimated usage by category" in output
    assert (
        "System prompt" in output
        and "Tool definitions" in output
        and "Skills" in output
    )
    assert "Context window:" in output and "200,000" in output
    # Unexpanded form points at /context all instead of listing details.
    assert "Use /context all for per-skill and per-toolset costs." in output


def test_context_all_includes_details(server, monkeypatch):
    """``/context all`` wires the per-skill / per-toolset tables through the real renderer."""
    spawned = []
    _recording_worker(server, monkeypatch, spawned)
    _register(server, agent=_fake_agent(), history=HISTORY)
    details = {
        "skills": [{"name": "demo", "index_tokens": 10, "skill_md_tokens": 120}],
        "toolsets": [{"toolset": "core", "tool_count": 1, "schema_tokens": 50}],
    }

    with (
        patch(
            "agent.system_prompt.build_system_prompt_parts", return_value=PROMPT_PARTS
        ),
        patch("agent.context_breakdown.compute_context_details", return_value=details),
    ):
        output = _exec(server, "/context all")

    assert spawned == []
    assert "Toolsets by schema cost" in output and "core" in output
    assert "Skills by cost" in output and "demo" in output
    # The expanded form replaces the hint line.
    assert "Use /context all for per-skill and per-toolset costs." not in output


def test_context_without_agent_answers_live_summary(server, monkeypatch):
    """No agent yet (fresh session): still answered live with the DB/usage summary — no worker fork,
    no 'no active agent' dead end."""
    spawned = []
    _recording_worker(server, monkeypatch, spawned)
    _register(server, agent=None, history=HISTORY)

    output = _exec(server, "/context")

    assert spawned == [], (
        "agentless /context must not fork a slash worker to print an error"
    )
    assert "(._.) No active agent" not in output
    assert "Conversation: 2 messages" in output
    assert "user: 1" in output and "assistant: 1" in output


def test_context_compute_host_session_reads_db_not_local_agent(
    server, launch_db, monkeypatch
):
    """On a compute host the local agent (if any) is stale truth: /context must report the DB
    transcript and never run the breakdown engine against it."""
    db = launch_db
    db.create_session(SESSION_KEY, source="tui")
    for i in range(1, 4):
        db.append_message(SESSION_KEY, "user", f"question {i}")
        db.append_message(SESSION_KEY, "assistant", f"answer {i}")
    # In-memory history is deliberately empty: the point of the db read is to rebuild the
    # transcript for a session whose turns ran on the child host.
    _register(server, agent=_fake_agent(), history=[])

    calls = []

    def _boom(*a, **k):
        calls.append(1)
        raise AssertionError(
            "breakdown engine must not run against a stale local agent"
        )

    monkeypatch.setattr(
        "agent.context_breakdown.compute_session_context_breakdown", _boom
    )
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: True)

    output = _exec(server, "/context")

    assert calls == []
    assert "Conversation: 6 messages" in output
    assert "user: 3" in output and "assistant: 3" in output
