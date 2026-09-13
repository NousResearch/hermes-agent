"""reload.mcp refreshes the tool snapshot of EVERY live session, not only the requester's.

The MCP pool is process-global while ``agent.tools`` is per-agent: a reload that refreshes only
``params["session_id"]`` leaves sibling sessions on stale tools (and refreshes nothing at all when
the id is absent or unknown, while still answering ``reloaded``).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from tools import mcp_tool_agent as _mcp_agent
from tools import mcp_tool_discovery as _mcp_discovery
from tools import mcp_tool_lifecycle as _mcp_lifecycle
import tui_gateway.server as srv


@pytest.fixture()
def reload_env(monkeypatch):
    refreshed: list[str] = []
    monkeypatch.setattr(_mcp_lifecycle, "shutdown_mcp_servers", lambda: None)
    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", lambda: None)
    monkeypatch.setattr(_mcp_agent, "refresh_agent_mcp_tools",
                        lambda agent, **_kw: refreshed.append(agent.name) or set())
    monkeypatch.setattr(srv, "_compute_mcp_rev", lambda: "rev-a")
    monkeypatch.setattr(srv, "_emit", lambda *_a, **_k: True)
    monkeypatch.setattr(srv, "_session_info", lambda agent, session=None: {})
    monkeypatch.setattr(srv, "_mcp_reload_gen", 0)
    monkeypatch.setattr(srv, "_mcp_reload_loaded_rev", "")

    def _session(name):
        return {"agent": SimpleNamespace(name=name), "history": [], "history_lock": threading.RLock(), "running": False}

    monkeypatch.setattr(srv, "_sessions", {
        "A": _session("agent-A"), "B": _session("agent-B"),
        "lazy": {"agent": None, "history_lock": threading.RLock()},
    })
    return refreshed


def test_reload_from_one_session_refreshes_every_live_agent(reload_env):
    resp = srv._methods["reload.mcp"](1, {"session_id": "A", "confirm": True})

    assert resp["result"]["status"] == "reloaded"
    assert sorted(reload_env) == ["agent-A", "agent-B"]


def test_reload_without_session_id_still_refreshes_live_agents(reload_env):
    resp = srv._methods["reload.mcp"](1, {"confirm": True})

    assert resp["result"]["status"] == "reloaded"
    assert sorted(reload_env) == ["agent-A", "agent-B"]
