"""reload.mcp must refresh EVERY live session's cached agent tools.

The handler rebuilt the process-global MCP pool but refreshed only the
requesting session's agent snapshot — and when ``params['session_id']`` was
missing/unknown (desktop callers pass ``activeSessionId ?? undefined``) it
refreshed NO agent while still answering ``{'status': 'reloaded'}``. Every
other live session kept its stale ``agent.tools`` until ``/new``.

Invariant: after reload.mcp, every live session's agent reflects the new tool
registry, and each session gets its own ``session.info`` push.
"""

from __future__ import annotations

import threading
import types

import pytest

import model_tools
import tools.mcp_tool as mcp_tool
from tools import mcp_tool_discovery as _mcp_discovery
from tools import mcp_tool_lifecycle as _mcp_lifecycle
import tui_gateway.server as srv


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


class _Recorder:
    """Session transport that captures every frame written to it."""

    def __init__(self):
        self.frames = []

    def write(self, obj) -> bool:
        self.frames.append(obj)
        return True


def _agent(tool_names):
    a = types.SimpleNamespace()
    a.tools = [_tool(n) for n in tool_names]
    a.valid_tool_names = set(tool_names)
    a.enabled_toolsets = None
    a.disabled_toolsets = None
    a.model = "test-model"
    a.provider = "test"
    a.api_key = "test-key"
    a.session_id = ""
    a.reasoning_config = None
    a.service_tier = None
    a.context_compressor = None
    a._session_db = None
    a._cached_system_prompt = ""
    a._bot_mode_protocol = False
    return a


def _live_session(sid: str, agent, transport) -> dict:
    return {
        "agent": agent,
        "session_key": sid,
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "inflight_turn": None,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cwd": "",
        "cols": 80,
        "source": "tui",
        "profile_home": None,
        "transport": transport,
    }


@pytest.fixture()
def reload_env(monkeypatch, tmp_path):
    """Stub the global pool rebuild; keep the real per-agent refresh path."""
    defs = [_tool(n) for n in ("read_file", "terminal", "mcp_new_server_tool")]
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: list(defs))
    # get_tool_definitions is also reachable through tools.mcp_tool; keep both in step.
    monkeypatch.setattr(mcp_tool, "get_tool_definitions", lambda **kw: list(defs), raising=False)
    monkeypatch.setattr(_mcp_lifecycle, "shutdown_mcp_servers", lambda: None)
    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", lambda: None)
    monkeypatch.setattr(srv, "_compute_mcp_rev", lambda: "rev-a")

    saved = (srv._mcp_reload_gen, srv._mcp_reload_loaded_rev)
    srv._mcp_reload_gen = 0
    srv._mcp_reload_loaded_rev = ""
    yield
    srv._mcp_reload_gen, srv._mcp_reload_loaded_rev = saved


@pytest.fixture()
def two_sessions(reload_env):
    """Two live sessions whose agents hold a stale pre-reload snapshot."""
    agents = {sid: _agent(["read_file", "terminal"]) for sid in ("s1", "s2")}
    transports = {sid: _Recorder() for sid in agents}
    for sid, agent in agents.items():
        srv._sessions[sid] = _live_session(sid, agent, transports[sid])
    yield agents, transports
    for sid in agents:
        srv._sessions.pop(sid, None)


def _info_sids(transport: _Recorder) -> list:
    return [f["params"]["session_id"] for f in transport.frames
            if f.get("method") == "event" and f.get("params", {}).get("type") == "session.info"]


def test_reload_mcp_refreshes_every_live_session(two_sessions):
    """A reload requested by s1 must rebuild s2's snapshot too — the pool is
    process-global, so a per-requester refresh leaves siblings stale."""
    agents, transports = two_sessions

    result = srv._methods["reload.mcp"](1, {"session_id": "s1", "confirm": True})

    assert result["result"]["status"] == "reloaded"
    for sid, agent in agents.items():
        assert "mcp_new_server_tool" in agent.valid_tool_names
        assert any(t["function"]["name"] == "mcp_new_server_tool" for t in agent.tools)
        assert _info_sids(transports[sid]) == [sid]


@pytest.mark.parametrize("params", [{"confirm": True}, {"session_id": "no-such-session", "confirm": True}],
                         ids=["missing", "unknown"])
def test_reload_mcp_without_requester_session_still_refreshes(two_sessions, params):
    """Desktop callers can send no/unknown session_id; the reload must not
    silently refresh zero agents while reporting success."""
    agents, transports = two_sessions

    result = srv._methods["reload.mcp"](1, params)

    assert result["result"]["status"] == "reloaded"
    for sid, agent in agents.items():
        assert "mcp_new_server_tool" in agent.valid_tool_names
        assert _info_sids(transports[sid]) == [sid]
