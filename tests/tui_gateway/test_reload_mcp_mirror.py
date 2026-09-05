"""``/reload-mcp`` live-session mirror (``_mirror_reload_mcp``).

The slash worker runs in its own subprocess, so a worker-run ``/reload-mcp``
reloads the WORKER's throwaway MCP pool — the live session's pool only changes
through the ``_SLASH_MIRRORS`` hook in ``tui_gateway/methods_slash.py``. That
mirror used to call ``agent.reload_mcp_tools()``, a method no agent ever
implemented, making the whole path a silent no-op. These tests pin the real
behavior: shutdown → reprobe → discover → refresh the live agent's snapshot,
plus the running-turn guard.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tui_gateway import server


@pytest.fixture()
def reload_env(monkeypatch):
    """Fake the reload pipeline; count calls."""
    calls = {"shutdown": 0, "reprobe": 0, "discover": 0, "refresh": 0}

    monkeypatch.setattr(
        "tools.mcp_tool_lifecycle.shutdown_mcp_servers",
        lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1))
    monkeypatch.setattr(
        "tools.mcp_tool_agent.reprobe_tool_availability",
        lambda: calls.__setitem__("reprobe", calls["reprobe"] + 1))
    monkeypatch.setattr(
        "tools.mcp_tool_discovery.discover_mcp_tools",
        lambda: calls.__setitem__("discover", calls["discover"] + 1))

    def fake_refresh(agent, *, enabled_override=None, quiet_mode=False, **_kw):
        calls["refresh"] += 1
        calls["refresh_agent"] = agent
        calls["refresh_override"] = enabled_override

    monkeypatch.setattr("tools.mcp_tool_agent.refresh_agent_mcp_tools", fake_refresh)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda platform=None: ["mcp"])
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_session_info", lambda agent, session: {"fake": True})
    return calls


def _make_session(running: bool = False):
    agent = MagicMock()
    agent.enabled_toolsets = ["mcp"]
    return {"agent": agent, "running": running, "session_key": "k", "history": []}


def test_mirror_reloads_pools_and_refreshes_live_agent(reload_env):
    sid = "sid-reload-mirror"
    session = _make_session()
    server._sessions[sid] = session
    try:
        output = server._mirror_slash_side_effects(sid, session, "/reload-mcp")
    finally:
        server._sessions.pop(sid, None)

    assert output == "", output
    assert reload_env["shutdown"] == 1
    assert reload_env["reprobe"] == 1
    assert reload_env["discover"] == 1
    assert reload_env["refresh"] == 1
    assert reload_env["refresh_agent"] is session["agent"]
    # enabled_override re-resolves toolsets so a server enabled in config this
    # session is picked up (parity with the CLI worker and the reload.mcp RPC).
    assert reload_env["refresh_override"] == ["mcp"]


def test_mirror_rejects_while_turn_is_running(reload_env):
    sid = "sid-reload-busy"
    session = _make_session(running=True)
    server._sessions[sid] = session
    try:
        output = server._mirror_slash_side_effects(sid, session, "/reload-mcp")
    finally:
        server._sessions.pop(sid, None)

    assert "session busy" in output
    assert reload_env["shutdown"] == 0
    assert reload_env["refresh"] == 0


def test_mirror_without_agent_still_reloads_the_shared_pool(reload_env):
    # The MCP pool is gateway-global, not per-agent: a session with no agent
    # yet still needs the pool rebuilt (its future turns read from it). The
    # RPC's session refresh tolerates agent=None.
    sid = "sid-reload-noagent"
    session = {"agent": None, "running": False, "session_key": "k", "history": []}
    server._sessions[sid] = session
    try:
        output = server._mirror_slash_side_effects(sid, session, "/reload-mcp")
    finally:
        server._sessions.pop(sid, None)

    assert output == ""
    assert reload_env["shutdown"] == 1
    assert reload_env["discover"] == 1
