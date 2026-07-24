"""Contract test: tui_gateway._set_session_context must inject the live
session id into HERMES_SESSION_ID so terminal/execute_code subprocesses can
read the current session's id.

Regression for the bug where _set_session_context called set_session_vars
WITHOUT session_id, leaving the contextvar as "" (explicitly empty). Because
the session-context bridge treats an explicit "" as authoritative and does NOT
fall back to os.environ, every terminal command in a dashboard/TUI/web session
saw an empty HERMES_SESSION_ID even though agent_init had set it via
set_current_session_id().
"""
import pytest

from gateway.session_context import (
    get_session_env,
    _VAR_MAP,
    _UNSET,
)
import tui_gateway.server as server


@pytest.fixture(autouse=True)
def _reset_contextvars():
    """Reset all session contextvars to _UNSET between tests.

    In production each asyncio.Task/worker thread gets a fresh context copy
    where the defaults are _UNSET. In tests functions share one thread
    context, so a value set by test A would leak into test B without this.
    """
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)


class _FakeAgent:
    def __init__(self, session_id):
        self.session_id = session_id


def _install_session(monkeypatch, *, session_key, agent_session_id, source="cli"):
    """Register a fake session in server._sessions for the duration of a test."""
    sess = {
        "session_key": session_key,
        "source": source,
        "agent": _FakeAgent(agent_session_id) if agent_session_id is not None else None,
        "cwd": "/home/user",
    }
    monkeypatch.setattr(server, "_sessions", {session_key: sess}, raising=False)
    return sess


def test_set_session_context_injects_agent_session_id(monkeypatch):
    """HERMES_SESSION_ID must equal the live agent.session_id after binding."""
    _install_session(
        monkeypatch, session_key="skey-abc", agent_session_id="20260722_deadbeef"
    )

    server._set_session_context("skey-abc", ui_session_id="ui-123")

    assert get_session_env("HERMES_SESSION_ID") == "20260722_deadbeef"


def test_set_session_context_falls_back_to_session_key(monkeypatch):
    """When the agent has no session_id yet, fall back to the session_key
    (never leave HERMES_SESSION_ID empty for an identified session)."""
    _install_session(monkeypatch, session_key="skey-xyz", agent_session_id=None)

    server._set_session_context("skey-xyz")

    assert get_session_env("HERMES_SESSION_ID") == "skey-xyz"


def test_set_session_context_unknown_key_uses_key(monkeypatch):
    """An ephemeral/unknown session_key (not in _sessions) still binds the key
    itself rather than leaving the id empty."""
    monkeypatch.setattr(server, "_sessions", {}, raising=False)

    server._set_session_context("ephemeral-key")

    assert get_session_env("HERMES_SESSION_ID") == "ephemeral-key"
