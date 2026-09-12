"""``/context`` must answer for an idle session, not report "No active agent".

The gateway already renders a full context view from persisted state
(``_format_live_context_output``), but the routing gate only reached it for
compute-host sessions, i.e. when ``dashboard.process_isolation.turn_isolation``
is on. With the default config the command fell through to the CLI handler,
which requires ``self.agent`` and answers "No active agent -- send a message
first." for a session that plainly has messages.
"""

from __future__ import annotations

import threading

import pytest

from tui_gateway import server


def _session(*, agent=None, messages=2):
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ][:messages]
    return {
        "session_key": "sk-1",
        "history": history,
        "history_lock": threading.Lock(),
        "agent": agent,
    }


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    """Render from in-memory history so the test needs no state.db."""
    monkeypatch.setattr(server, "_get_db", lambda: None)


def test_context_answers_for_an_idle_session(monkeypatch):
    """The bug: an agentless session got None here and fell through to the CLI."""
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: False)

    out = server._live_slash_command_output("sid", _session(agent=None), "context", "")

    assert out is not None, "/context fell through to the CLI handler for an idle session"
    assert "No active agent" not in out
    assert "Conversation:" in out


def test_context_still_reaches_the_cli_handler_when_an_agent_is_live(monkeypatch):
    """Guard: a live agent keeps the richer CLI rendering, unchanged."""
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: False)

    out = server._live_slash_command_output(
        "sid", _session(agent=object()), "context", ""
    )

    assert out is None


def test_compute_host_context_path_is_unchanged(monkeypatch):
    """Guard: the isolated/compute-host route this gate was built for still works."""
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: True)

    out = server._live_slash_command_output(
        "sid", _session(agent=object()), "context", ""
    )

    assert out is not None
    assert "Conversation:" in out


def test_tools_is_not_widened_by_this_change(monkeypatch):
    """Scope guard: /tools consults the agent, so it must keep its old routing."""
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: False)

    out = server._live_slash_command_output("sid", _session(agent=None), "tools", "")

    assert out is None


def test_context_without_a_session_still_declines(monkeypatch):
    """No session dict means nothing to render from; the caller keeps its path."""
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: False)

    assert server._live_slash_command_output("sid", None, "context", "") is None


def test_serves_persisted_read_predicate():
    assert server._serves_persisted_read({"agent": None}, "context") is True
    assert server._serves_persisted_read({"agent": object()}, "context") is False
    assert server._serves_persisted_read({"agent": None}, "tools") is False
    assert server._serves_persisted_read(None, "context") is False
