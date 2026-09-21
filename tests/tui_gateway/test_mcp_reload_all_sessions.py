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
    def refresh(agent, **_kw):
        refreshed.append(agent.name)
        agent.tools = ["fresh-tool"]
        return set()

    monkeypatch.setattr(_mcp_agent, "refresh_agent_mcp_tools", refresh)
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
    sessions = {
        "A": _session("agent-A"), "B": _session("agent-B", profile_home=str(profile_b)),
        "lazy": {"agent": None, "history_lock": threading.RLock()},
    }
    monkeypatch.setattr(srv, "_sessions", sessions)
    return SimpleNamespace(
        refreshed=refreshed, discovered_homes=discovered_homes,
        scoped_homes=scoped_homes, profile_b=profile_b, sessions=sessions)


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


def test_running_session_consumes_only_latest_reload_once_at_idle_boundary(reload_env):
    session_b = reload_env.sessions["B"]
    session_b["running"] = True
    session_b["agent"].tools = ["turn-frozen-tool"]

    srv._methods["reload.mcp"](1, {"session_id": "A", "confirm": True})
    srv._methods["reload.mcp"](2, {"session_id": "A", "confirm": True})

    assert reload_env.refreshed.count("agent-B") == 0
    assert session_b["agent"].tools == ["turn-frozen-tool"]
    assert session_b[srv._MCP_RELOAD_PENDING_GENERATION] == 2

    session_b["running"] = False
    assert srv._apply_pending_mcp_reload("B", session_b) is True
    assert reload_env.refreshed.count("agent-B") == 1
    assert session_b["agent"].tools == ["fresh-tool"]
    assert srv._MCP_RELOAD_PENDING_GENERATION not in session_b
    assert srv._apply_pending_mcp_reload("B", session_b) is False
    assert reload_env.refreshed.count("agent-B") == 1


def test_deferred_reload_failure_stays_pending_for_next_idle_boundary(reload_env, monkeypatch):
    session_b = reload_env.sessions["B"]
    session_b["running"] = True
    srv._methods["reload.mcp"](1, {"session_id": "A", "confirm": True})
    session_b["running"] = False

    monkeypatch.setattr(
        _mcp_agent, "refresh_agent_mcp_tools",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("refresh failed")))
    assert srv._apply_pending_mcp_reload("B", session_b) is False
    assert session_b[srv._MCP_RELOAD_PENDING_GENERATION] == 1

    monkeypatch.setattr(
        _mcp_agent, "refresh_agent_mcp_tools",
        lambda agent, **_kw: reload_env.refreshed.append(agent.name) or set())
    assert srv._apply_pending_mcp_reload("B", session_b) is True
    assert reload_env.refreshed.count("agent-B") == 1
    assert srv._MCP_RELOAD_PENDING_GENERATION not in session_b


def test_unmarked_turn_boundary_never_waits_for_global_reload_lock(monkeypatch):
    session = {"history_lock": threading.RLock(), "running": False}

    class ContendedReloadLock:
        def __enter__(self):
            raise AssertionError("unmarked turn tried to acquire the global reload lock")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(srv, "_mcp_reload_lock", ContendedReloadLock())

    assert srv._apply_pending_mcp_reload("B", session) is False


def test_early_turn_admission_release_applies_deferred_reload(monkeypatch):
    session = {"history_lock": threading.RLock(), "running": True}
    applied = []

    def reject(*_args, **_kwargs):
        session["running"] = False
        return None

    monkeypatch.setattr(srv, "_admit_prompt_turn", reject)
    monkeypatch.setattr(
        srv, "_apply_pending_mcp_reload",
        lambda sid, live: applied.append((sid, live.get("running"))) or True)

    assert srv._run_prompt_submit(1, "B", session, "hello") is False
    assert applied == [("B", False)]


def test_followup_dispatch_exception_releases_before_deferred_reload(monkeypatch):
    session = {"history_lock": threading.RLock(), "running": True}
    applied = []
    monkeypatch.setattr(srv, "_emit", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        srv, "_run_prompt_submit",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("turn failed")))
    monkeypatch.setattr(
        srv, "_apply_pending_mcp_reload",
        lambda sid, live: applied.append((sid, live.get("running"))) or True)

    srv._dispatch_followup_turn(1, "B", session, "hello", "test follow-up")

    assert session["running"] is False
    assert applied == [("B", False)]


def test_reload_rediscovers_the_launch_profile_under_its_own_secret_scope(reload_env):
    """The launch profile's servers resolve connect-time credentials through ``get_secret`` too:
    rediscovered unscoped, they park with UnscopedSecretError once the process multiplexes while
    the RPC still answers "reloaded" (#113746). Every rediscovery, launch home included, is scoped."""
    srv._methods["reload.mcp"](1, {"session_id": "A", "confirm": True})

    assert sorted(set(reload_env.scoped_homes)) == sorted(set(reload_env.discovered_homes))
    assert hermes_constants.hermes_home_key() in reload_env.scoped_homes
