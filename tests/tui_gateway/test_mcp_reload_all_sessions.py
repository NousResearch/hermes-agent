"""reload.mcp must refresh EVERY live session's cached agent tools.

The handler rebuilt the process-global MCP pool but refreshed only the
requesting session's agent snapshot — and when ``params['session_id']`` was
missing/unknown (desktop callers pass ``activeSessionId ?? undefined``) it
refreshed NO agent while still answering ``{'status': 'reloaded'}``. Every
other live session kept its stale ``agent.tools`` until ``/new``.

Invariants: after reload.mcp, every live session's agent reflects the new tool
registry; each refreshed session gets its own ``session.info`` push; compute-host
sessions are forwarded to the host (its own reload.mcp fans out there); running
sessions defer to the next turn boundary; a replaced session is never refreshed
or emitted through; a failed session reports partial instead of a success-shaped
stale session.info.
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


def _live_session(sid: str, agent, transport, **overrides) -> dict:
    session = {
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
    session.update(overrides)
    return session


@pytest.fixture()
def reload_env(monkeypatch):
    """Stub the global pool rebuild; keep the real per-agent refresh path."""
    defs = [_tool(n) for n in ("read_file", "terminal", "mcp_new_server_tool")]
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: list(defs))
    # get_tool_definitions is also reachable through tools.mcp_tool; keep both in step.
    monkeypatch.setattr(mcp_tool, "get_tool_definitions", lambda **kw: list(defs), raising=False)
    monkeypatch.setattr(_mcp_lifecycle, "shutdown_mcp_servers", lambda **kw: None)
    monkeypatch.setattr(_mcp_discovery, "discover_mcp_tools", lambda *a, **kw: [])
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
    assert result["result"]["sessions_refreshed"] == 2
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


def test_reload_mcp_defers_running_session_to_turn_boundary(reload_env, monkeypatch):
    """A mid-turn sibling must not have its tools swapped under it: the refresh
    is queued (``pending_mcp_refresh``) and applied at the next turn start."""
    running_agent, idle_agent = _agent(["read_file"]), _agent(["read_file"])
    running_t, idle_t = _Recorder(), _Recorder()
    srv._sessions["busy"] = _live_session("busy", running_agent, running_t, running=True)
    srv._sessions["idle"] = _live_session("idle", idle_agent, idle_t)
    try:
        result = srv._methods["reload.mcp"](1, {"session_id": "idle", "confirm": True})

        assert result["result"]["sessions_refreshed"] == 1
        assert result["result"]["sessions_deferred"] == 1
        assert "mcp_new_server_tool" not in running_agent.valid_tool_names
        assert srv._sessions["busy"]["pending_mcp_refresh"] is True
        assert _info_sids(running_t) == []  # no success-shaped info for a session not yet refreshed
        assert "mcp_new_server_tool" in idle_agent.valid_tool_names

        srv._apply_pending_mcp_refresh("busy", srv._sessions["busy"])

        assert "mcp_new_server_tool" in running_agent.valid_tool_names
        assert "pending_mcp_refresh" not in srv._sessions["busy"]
        assert _info_sids(running_t) == ["busy"]
    finally:
        srv._sessions.pop("busy", None)
        srv._sessions.pop("idle", None)


def test_reload_mcp_skips_session_replaced_after_snapshot(reload_env, monkeypatch):
    """A sid removed/replaced between the _sessions snapshot and its refresh
    must not be mutated or emitted through the replacement's transport."""
    old_agent, new_agent = _agent(["read_file"]), _agent(["read_file"])
    old_t, new_t = _Recorder(), _Recorder()
    old_sess = _live_session("s1", old_agent, old_t)
    srv._sessions["s1"] = old_sess

    real_uses_host = srv._session_uses_compute_host
    calls = {"n": 0}

    def _swap_on_check(sess):
        # Call 1 is the requester's compute-host check (pre-snapshot); call 2 is the
        # loop's — swap there to simulate the replace landing after the snapshot.
        calls["n"] += 1
        if sess is old_sess and calls["n"] > 1:
            srv._sessions["s1"] = _live_session("s1", new_agent, new_t)
        return real_uses_host(sess)

    monkeypatch.setattr(srv, "_session_uses_compute_host", _swap_on_check)
    try:
        result = srv._methods["reload.mcp"](1, {"session_id": "s1", "confirm": True})

        assert result["result"]["status"] == "reloaded"
        assert "mcp_new_server_tool" not in old_agent.valid_tool_names
        assert "mcp_new_server_tool" not in new_agent.valid_tool_names
        assert _info_sids(old_t) == [] and _info_sids(new_t) == []
    finally:
        srv._sessions.pop("s1", None)


def test_reload_mcp_reports_partial_failure(two_sessions, monkeypatch):
    """One session's refresh raising must not fail the reload or the siblings:
    the failure lands in sessions_failed and no session.info is emitted for it."""
    agents, transports = two_sessions

    import tools.mcp_tool_agent as mcp_agent_mod
    real_refresh = mcp_agent_mod.refresh_agent_mcp_tools

    def _flaky(agent, **kw):
        if agent is agents["s2"]:
            raise RuntimeError("boom")
        return real_refresh(agent, **kw)

    monkeypatch.setattr(mcp_agent_mod, "refresh_agent_mcp_tools", _flaky)

    result = srv._methods["reload.mcp"](1, {"session_id": "s1", "confirm": True})

    assert result["result"]["status"] == "reloaded"
    assert result["result"]["sessions_refreshed"] == 1
    assert result["result"]["sessions_failed"] == {"s2": "boom"}
    assert "mcp_new_server_tool" in agents["s1"].valid_tool_names
    assert "mcp_new_server_tool" not in agents["s2"].valid_tool_names
    assert _info_sids(transports["s1"]) == ["s1"]
    assert _info_sids(transports["s2"]) == []


class _FakeSupervisor:
    def __init__(self, ack):
        self.ack = ack
        self.calls = []

    def reload_mcp(self, sid, *, request_id=None):
        self.calls.append(sid)
        return self.ack


def _host_env(reload_env, monkeypatch, ack):
    """Turn isolation on; one host-owned session + one local session."""
    monkeypatch.setattr(srv, "_turn_isolation_enabled", lambda cfg=None: True)
    supervisor = _FakeSupervisor(ack)
    monkeypatch.setattr(srv, "_get_compute_host_supervisor", lambda *a, **k: supervisor)
    host_t, local_t = _Recorder(), _Recorder()
    host_agent, local_agent = _agent(["read_file"]), _agent(["read_file"])
    srv._sessions["host-s"] = _live_session(
        "host-s", host_agent, host_t, _compute_host_active=True)
    srv._sessions["local-s"] = _live_session("local-s", local_agent, local_t)
    return supervisor, host_agent, local_agent, host_t, local_t


def _drop_host_env():
    srv._sessions.pop("host-s", None)
    srv._sessions.pop("local-s", None)


def test_reload_mcp_forwards_to_compute_host_sessions(reload_env, monkeypatch):
    """A local requester with a compute-host sibling: the host session is NOT
    refreshed in-process (its agent lives in the host) — one forward covers it,
    and the host's own reload.mcp fans out to every session it owns."""
    supervisor, host_agent, local_agent, host_t, local_t = _host_env(
        reload_env, monkeypatch,
        ack={"type": "reload_mcp.ack", "response": {"result": {"status": "reloaded"}}})
    try:
        result = srv._methods["reload.mcp"](1, {"session_id": "local-s", "confirm": True})

        assert result["result"]["status"] == "reloaded"
        assert result["result"]["compute_host_sessions"] == ["host-s"]
        assert supervisor.calls == ["host-s"]
        assert "mcp_new_server_tool" in local_agent.valid_tool_names
        assert "mcp_new_server_tool" not in host_agent.valid_tool_names
        assert _info_sids(local_t) == ["local-s"]
        assert _info_sids(host_t) == []
    finally:
        _drop_host_env()


def test_reload_mcp_host_requester_also_reloads_local_sessions(reload_env, monkeypatch):
    """The old early return skipped local siblings when the requester was a
    compute-host session; now the forward happens AND the local pool reloads."""
    supervisor, host_agent, local_agent, host_t, local_t = _host_env(
        reload_env, monkeypatch,
        ack={"type": "reload_mcp.ack", "response": {"result": {"status": "reloaded"}}})
    try:
        result = srv._methods["reload.mcp"](1, {"session_id": "host-s", "confirm": True})

        assert result["result"]["status"] == "reloaded"
        assert result["result"]["turn_isolation"] is True
        assert supervisor.calls == ["host-s"]
        assert "mcp_new_server_tool" in local_agent.valid_tool_names
        assert "mcp_new_server_tool" not in host_agent.valid_tool_names
    finally:
        _drop_host_env()


def test_reload_mcp_host_ack_error_is_not_success(reload_env, monkeypatch):
    """An ack wrapper is not proof: a child JSON-RPC error inside
    reload_mcp.ack must surface as compute_host_error, not silent success."""
    supervisor, host_agent, local_agent, host_t, local_t = _host_env(
        reload_env, monkeypatch,
        ack={"type": "reload_mcp.ack",
             "response": {"error": {"code": 5015, "message": "discovery blew up"}}})
    try:
        result = srv._methods["reload.mcp"](1, {"session_id": "local-s", "confirm": True})

        assert result["result"]["status"] == "reloaded"
        assert "discovery blew up" in result["result"]["compute_host_error"]
        assert "mcp_new_server_tool" in local_agent.valid_tool_names
    finally:
        _drop_host_env()


def test_reload_mcp_discovers_per_session_profile(reload_env, monkeypatch, tmp_path):
    """The unscoped shutdown tears down every profile's connections, so
    rediscovery must run once per live session's profile scope — ambient-only
    discovery would leave non-ambient profiles' servers dead."""
    import hermes_constants

    profile_home = tmp_path / "profile-b"
    profile_home.mkdir()
    homes_seen = []
    monkeypatch.setattr(
        _mcp_discovery, "discover_mcp_tools",
        lambda *a, **kw: homes_seen.append(hermes_constants.hermes_home_key()) or [])

    agent_a, agent_b = _agent(["read_file"]), _agent(["read_file"])
    srv._sessions["pa"] = _live_session("pa", agent_a, _Recorder())
    srv._sessions["pb"] = _live_session("pb", agent_b, _Recorder(), profile_home=str(profile_home))
    try:
        result = srv._methods["reload.mcp"](1, {"session_id": "pa", "confirm": True})

        assert result["result"]["status"] == "reloaded"
        assert "mcp_new_server_tool" in agent_a.valid_tool_names
        assert "mcp_new_server_tool" in agent_b.valid_tool_names
        ambient = hermes_constants.hermes_home_key()
        assert ambient in homes_seen
        assert hermes_constants.hermes_home_key(profile_home) in homes_seen
    finally:
        srv._sessions.pop("pa", None)
        srv._sessions.pop("pb", None)
