"""reload.mcp refreshes the tool snapshot of EVERY live session, not only the requester's.

The MCP pool is process-global while ``agent.tools`` is per-agent: a reload that refreshes only
``params["session_id"]`` leaves sibling sessions on stale tools (and refreshes nothing at all when
the id is absent or unknown, while still answering ``reloaded``).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import hermes_constants
from agent.secret_scope import current_secret_scope
from tools import mcp_tool_agent as _mcp_agent
from tools import mcp_tool_discovery as _mcp_discovery
from tools import mcp_tool_lifecycle as _mcp_lifecycle
import tui_gateway.server as srv


@pytest.fixture()
def reload_env(monkeypatch, tmp_path):
    refreshed: list[str] = []
    discovered_homes: list[str] = []
    monkeypatch.setattr(_mcp_lifecycle, "shutdown_mcp_servers", lambda: None)
    scoped_homes: list[str] = []

    def _discover():
        discovered_homes.append(hermes_constants.hermes_home_key())
        if current_secret_scope() is not None:
            scoped_homes.append(hermes_constants.hermes_home_key())

    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", _discover)
    monkeypatch.setattr(_mcp_agent, "refresh_agent_mcp_tools",
                        lambda agent, **_kw: refreshed.append(agent.name) or set())
    monkeypatch.setattr(srv, "_compute_mcp_rev", lambda: "rev-a")
    monkeypatch.setattr(srv, "_emit", lambda *_a, **_k: True)
    monkeypatch.setattr(srv, "_session_info", lambda agent, session=None: {})
    monkeypatch.setattr(srv, "_mcp_reload_gen", 0)
    monkeypatch.setattr(srv, "_mcp_reload_loaded_rev", "")

    def _session(name, profile_home=None):
        return {"agent": SimpleNamespace(name=name), "history": [], "history_lock": threading.RLock(),
                "running": False, "profile_home": profile_home}

    profile_b = tmp_path / "profile-b"
    profile_b.mkdir()
    monkeypatch.setattr(srv, "_sessions", {
        "A": _session("agent-A"), "B": _session("agent-B", profile_home=str(profile_b)),
        "lazy": {"agent": None, "history_lock": threading.RLock()},
    })
    return SimpleNamespace(refreshed=refreshed, discovered_homes=discovered_homes,
                           scoped_homes=scoped_homes, profile_b=profile_b)


def test_reload_from_one_session_refreshes_every_live_agent(reload_env):
    resp = srv._methods["reload.mcp"](1, {"session_id": "A", "confirm": True})

    assert resp["result"]["status"] == "reloaded"
    assert sorted(reload_env.refreshed) == ["agent-A", "agent-B"]


def test_reload_without_session_id_still_refreshes_live_agents(reload_env):
    resp = srv._methods["reload.mcp"](1, {"confirm": True})

    assert resp["result"]["status"] == "reloaded"
    assert sorted(reload_env.refreshed) == ["agent-A", "agent-B"]


def test_reload_rediscovers_under_each_live_profile_scope(reload_env):
    """The unscoped shutdown tears down every profile's servers; discovery under the ambient home
    alone would leave a secondary-profile session refreshing against a registry that never
    regained its overlay, so it loses its MCP tools until its own reload."""
    srv._methods["reload.mcp"](1, {"session_id": "A", "confirm": True})

    assert hermes_constants.hermes_home_key() in reload_env.discovered_homes
    assert hermes_constants.hermes_home_key(reload_env.profile_b) in reload_env.discovered_homes


def test_reload_rediscovers_the_launch_profile_under_its_own_secret_scope(reload_env):
    """The launch profile's servers resolve connect-time credentials through ``get_secret`` too:
    rediscovered unscoped, they park with UnscopedSecretError once the process multiplexes while
    the RPC still answers "reloaded" (#113746). Every rediscovery, launch home included, is scoped."""
    srv._methods["reload.mcp"](1, {"session_id": "A", "confirm": True})

    assert sorted(set(reload_env.scoped_homes)) == sorted(set(reload_env.discovered_homes))
    assert hermes_constants.hermes_home_key() in reload_env.scoped_homes
