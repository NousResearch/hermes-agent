"""The quiet-mode ``get_tool_definitions`` memo must not outlive the check_fn verdicts it filtered on.

An MCP server whose connection dropped (check_fn False) and came back without re-registering its
tools left the memo holding the omission: ``tool_search`` / ``tool_describe`` read the memo, so the
recovered server stayed invisible for the life of the process, ``/new`` included.
"""
import json
from collections import OrderedDict

import pytest

import model_tools
from model_tools import get_tool_definitions
from tests.module_clock import install_module_clock
from tools import registry as registry_mod
from tools.mcp_tool_agent import reprobe_tool_availability
from tools.registry import invalidate_check_fn_cache, registry

MAIL = "mcp__probe_mail__list_messages"
ANCHOR = "mcp__probe_anchor__ping"
BRIDGE_KW = {"quiet_mode": True, "skip_tool_search_assembly": True}
BOTH = ["mcp-probe-mail", "mcp-probe-anchor"]


def _schema(name, description):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {}}}


def _names(defs):
    return {d["function"]["name"] for d in defs}


@pytest.fixture
def mail_state():
    """A recovering MCP-style tool (mail) next to an always-healthy one (anchor)."""
    state = {"up": False, "probes": 0}

    def mail_connected():
        state["probes"] += 1
        return state["up"]

    registry.register(name=MAIL, toolset="mcp-probe-mail",
                      schema=_schema(MAIL, "List messages in the probe mailbox"),
                      handler=lambda args, **kw: "MAIL_OK", check_fn=mail_connected)
    registry.register(name=ANCHOR, toolset="mcp-probe-anchor",
                      schema=_schema(ANCHOR, "Ping the anchor server"),
                      handler=lambda args, **kw: "ANCHOR_OK", check_fn=lambda: True)
    reprobe_tool_availability()
    yield state
    registry.deregister(MAIL)
    registry.deregister(ANCHOR)
    reprobe_tool_availability()


@pytest.fixture
def clock(monkeypatch):
    """One fake monotonic clock for the verdict cache and the definitions memo."""
    module_clock = install_module_clock(monkeypatch, registry_mod)
    monkeypatch.setattr(model_tools, "time", module_clock)
    return module_clock


def test_reporter_reproducer_explicit_invalidation_reaches_quiet_memo(mail_state):
    kwargs = {"enabled_toolsets": ["mcp-probe-mail"], **BRIDGE_KW}
    assert get_tool_definitions(**kwargs) == []

    mail_state["up"] = True
    invalidate_check_fn_cache()  # e.g. hermes_platform.declaration.on_change

    assert _names(get_tool_definitions(**kwargs)) == {MAIL}


def test_false_to_true_recovers_on_natural_ttl_expiry(mail_state, clock):
    kwargs = {"enabled_toolsets": BOTH, **BRIDGE_KW}
    assert _names(get_tool_definitions(**kwargs)) == {ANCHOR}

    mail_state["up"] = True  # the server reconnected; nothing re-registered, nothing invalidated
    clock.now += registry_mod._CHECK_FN_TTL_SECONDS - 1
    assert _names(get_tool_definitions(**kwargs)) == {ANCHOR}  # the False verdict is still within its TTL

    clock.now += 2
    assert _names(get_tool_definitions(**kwargs)) == {ANCHOR, MAIL}


def test_memo_hits_within_verdict_ttl(mail_state, clock, monkeypatch):
    computes = []
    real_compute = model_tools._compute_tool_definitions

    def counting_compute(*args, **kwargs):
        computes.append(1)
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(model_tools, "_compute_tool_definitions", counting_compute)
    kwargs = {"enabled_toolsets": BOTH, **BRIDGE_KW}
    for _ in range(5):
        get_tool_definitions(**kwargs)
        clock.now += 5
    assert len(computes) == 1
    assert mail_state["probes"] == 1

    clock.now += registry_mod._CHECK_FN_TTL_SECONDS
    get_tool_definitions(**kwargs)
    get_tool_definitions(**kwargs)
    assert len(computes) == 2
    assert mail_state["probes"] == 2


def test_true_to_false_respects_ttl_then_failure_grace(mail_state, clock):
    ttl, grace = registry_mod._CHECK_FN_TTL_SECONDS, registry_mod._CHECK_FN_FAILURE_GRACE_SECONDS
    kwargs = {"enabled_toolsets": BOTH, **BRIDGE_KW}
    mail_state["up"] = True
    start = clock.now
    assert MAIL in _names(get_tool_definitions(**kwargs))

    mail_state["up"] = False  # connection dropped
    clock.now = start + ttl - 1
    assert MAIL in _names(get_tool_definitions(**kwargs))  # cached True verdict
    assert mail_state["probes"] == 1

    clock.now = start + ttl + 1
    assert MAIL in _names(get_tool_definitions(**kwargs))  # failure inside the grace window: a flake
    clock.now += 1
    assert MAIL in _names(get_tool_definitions(**kwargs))
    assert mail_state["probes"] == 2  # a list built on the grace verdict is reused for a few seconds

    clock.now = start + grace + 1
    assert _names(get_tool_definitions(**kwargs)) == {ANCHOR}  # grace over: the failure is honored


def test_restricted_toolset_never_gains_the_recovered_tool(mail_state, clock):
    restricted = {"enabled_toolsets": ["mcp-probe-anchor"], **BRIDGE_KW}
    assert _names(get_tool_definitions(**restricted)) == {ANCHOR}
    mail_state["up"] = True
    clock.now += registry_mod._CHECK_FN_TTL_SECONDS + 1
    invalidate_check_fn_cache()
    assert _names(get_tool_definitions(**restricted)) == {ANCHOR}
    result, redispatch = model_tools._dispatch_bridge_tool(
        "tool_call", {"calls": [{"name": MAIL, "arguments": {}}]}, ["mcp-probe-anchor"], None)
    assert redispatch is None and "error" in json.loads(result)


def test_profile_scoped_verdicts_stay_separate(mail_state, monkeypatch):
    profile = {"current": "/profiles/a"}
    verdicts = {"/profiles/a": True, "/profiles/b": False}
    mail_state["up"] = True
    for module in (registry_mod, model_tools):
        monkeypatch.setattr(module, "check_fn_cache_scope", lambda: profile["current"])
    registry.register(name=MAIL, toolset="mcp-probe-mail", schema=_schema(MAIL, "List messages"),
                      handler=lambda args, **kw: "MAIL_OK", check_fn=lambda: verdicts[profile["current"]])
    kwargs = {"enabled_toolsets": BOTH, **BRIDGE_KW}
    assert MAIL in _names(get_tool_definitions(**kwargs))
    profile["current"] = "/profiles/b"
    assert _names(get_tool_definitions(**kwargs)) == {ANCHOR}
    profile["current"] = "/profiles/a"
    assert MAIL in _names(get_tool_definitions(**kwargs))

    monkeypatch.setattr(model_tools, "check_fn_cache_scope", lambda: registry_mod.CHECK_FN_CACHE_BYPASS)
    model_tools._clear_tool_defs_cache()
    get_tool_definitions(**kwargs)
    assert not model_tools._tool_defs_cache  # unresolved profile identity still bypasses the memo


def _tool_results(server):
    """``{tool_call_id: content}`` for every tool message the agent sent back."""
    out = {}
    for body in server.main_requests():
        for msg in body.get("messages", []):
            if msg.get("role") == "tool":
                out[msg.get("tool_call_id")] = msg.get("content")
    return out


def test_long_running_agent_bridge_recovers_after_mcp_reconnect(mail_state, clock):
    """Composed: one AIAgent across turns, real bridge dispatch, the mail server recovering
    between turns without re-registering (the reconnect path keeps its registered names)."""
    from run_agent import AIAgent
    from tests.fakes.fake_llm_provider import MODEL_ID, FakeLLMServer, Text, ToolCall

    script = [
        ToolCall("tool_call", {"calls": [{"name": ANCHOR, "arguments": {}}]},
                 parallel=[("tool_describe", {"names": [MAIL]})]),
        Text("turn one done"),
        ToolCall("tool_search", {"queries": ["probe mailbox messages"]},
                 parallel=[("tool_describe", {"names": [MAIL]})]),
        ToolCall("tool_call", {"calls": [{"name": MAIL, "arguments": {}}]}),
        Text("turn two done"),
    ]
    with FakeLLMServer(script) as server:
        agent = AIAgent(model=MODEL_ID, provider="openai", api_key="unused-test-key",
                        base_url=server.base_url, enabled_toolsets=BOTH, quiet_mode=True,
                        skip_context_files=True, skip_memory=True, save_trajectories=False)
        bridge = {(t.get("function") or {}).get("name") for t in agent.tools}
        assert {"tool_search", "tool_describe", "tool_call"} <= bridge

        first = agent.run_conversation("turn one")
        tools_after_turn_one = json.dumps(agent.tools, sort_keys=True)
        turn_one = list(_tool_results(server).values())
        assert any("ANCHOR_OK" in c for c in turn_one), turn_one
        assert any(MAIL in c and "not_found" in c for c in turn_one), turn_one

        mail_state["up"] = True
        clock.now += registry_mod._CHECK_FN_TTL_SECONDS + 1  # the False verdict has expired
        agent.run_conversation("turn two", conversation_history=first.get("messages"))
        turn_two = list(_tool_results(server).values())[len(turn_one):]

    search, describe, call = turn_two
    assert MAIL in search, search
    assert "not_found" not in describe and "List messages in the probe mailbox" in describe, describe
    assert "MAIL_OK" in call, call
    # The recovery reached the live catalog only; the conversation's tools[] prefix did not move.
    assert json.dumps(agent.tools, sort_keys=True) == tools_after_turn_one


@pytest.fixture
def direct_tools():
    """Tool Search off, so the MCP tools sit in tools[] itself. Written once: a config write
    moves the memo key on its own and would hide a stale entry."""
    from hermes_cli.config import atomic_config_write, get_config_path

    atomic_config_write(get_config_path(), {"tools": {"tool_search": {"enabled": "off"}}})


def _direct_agent():
    from run_agent import AIAgent

    return AIAgent(model="test-model", provider="openai", api_key="unused-test-key",
                   base_url="http://127.0.0.1:9/v1", enabled_toolsets=BOTH, quiet_mode=True,
                   skip_context_files=True, skip_memory=True, save_trajectories=False)


def test_direct_tool_list_stays_frozen_mid_session_and_a_new_session_sees_the_recovery(
        mail_state, clock, direct_tools):
    """Without Tool Search the tools sit in tools[], which is frozen for the session (#101367):
    a check_fn flip alone never changes it between turns. ``/new`` and ``/reload-mcp`` are the
    consented refresh points, and both must see the recovered server."""
    from tools.mcp_tool_agent import refresh_agent_mcp_tools

    agent = _direct_agent()
    assert ANCHOR in _names(agent.tools) and MAIL not in _names(agent.tools)
    sent = json.dumps(agent.tools, sort_keys=True)
    grace = registry_mod._CHECK_FN_FAILURE_GRACE_SECONDS

    for up in (True, False, True):  # recover, flap, recover: no registry change, no consent
        mail_state["up"] = up
        clock.now += grace + 1
        assert refresh_agent_mcp_tools(agent, quiet_mode=True, preserve_prefix=True) == set()
        assert json.dumps(agent.tools, sort_keys=True) == sent

    assert MAIL in _names(_direct_agent().tools)  # /new: a fresh session takes the live availability

    reprobe_tool_availability()  # /reload-mcp
    assert refresh_agent_mcp_tools(agent, quiet_mode=True) == {MAIL}


def test_direct_tool_list_still_appends_a_late_registration_at_the_tail(mail_state, clock, direct_tools):
    """The freeze holds back check_fn flips only: a server that registers mid-session (the
    registry generation moves) still lands at the tail between turns, as on main."""
    from tools.mcp_tool_agent import refresh_agent_mcp_tools

    late = "mcp__probe_anchor__aaa_late"
    agent = _direct_agent()
    sent = list(agent.tools)
    registry.register(name=late, toolset="mcp-probe-anchor", schema=_schema(late, "Late tool"),
                      handler=lambda args, **kw: "LATE_OK", check_fn=lambda: True)
    try:
        assert refresh_agent_mcp_tools(agent, quiet_mode=True, preserve_prefix=True) == {late}
        assert agent.tools[:-1] == sent and agent.tools[-1]["function"]["name"] == late
    finally:
        registry.deregister(late)


def test_grace_verdict_reuse_bounds_failing_probes_and_ends_with_the_grace(mail_state, clock):
    """A failing probe served as a grace True must not re-run on every catalog read: a slow or
    timing-out probe would block each ``tool_call`` for its full duration. The list built on it
    is reused for a few seconds, and never past the end of the grace window."""
    ttl, grace = registry_mod._CHECK_FN_TTL_SECONDS, registry_mod._CHECK_FN_FAILURE_GRACE_SECONDS
    kwargs = {"enabled_toolsets": BOTH, **BRIDGE_KW}
    mail_state["up"] = True
    start = clock.now
    assert MAIL in _names(get_tool_definitions(**kwargs))
    mail_state["up"] = False

    clock.now = start + ttl + 1
    before = mail_state["probes"]
    for _ in range(15):  # 15 reads over 7.5 s, all inside the grace window
        assert MAIL in _names(get_tool_definitions(**kwargs))
        clock.now += 0.5
    assert mail_state["probes"] - before == 2  # one probe per 5 s, not one per read

    clock.now = start + grace - 1
    assert MAIL in _names(get_tool_definitions(**kwargs))
    clock.now = start + grace + 0.5
    assert _names(get_tool_definitions(**kwargs)) == {ANCHOR}  # the reuse stops where the grace does


BRIDGE = {"tool_search", "tool_describe", "tool_call"}


@pytest.fixture
def tool_search_on():
    """Written once, before any agent is built (see ``direct_tools``)."""
    from hermes_cli.config import atomic_config_write, get_config_path

    atomic_config_write(get_config_path(), {"tools": {"tool_search": {"enabled": "on"}}})


def _agent(enabled_toolsets):
    from run_agent import AIAgent

    return AIAgent(model="test-model", provider="openai", api_key="unused-test-key",
                   base_url="http://127.0.0.1:9/v1", enabled_toolsets=enabled_toolsets, quiet_mode=True,
                   skip_context_files=True, skip_memory=True, save_trajectories=False)


def test_bridge_tools_stay_out_of_a_live_session_when_its_only_deferrable_tools_recover(
        mail_state, clock, tool_search_on):
    """Tool Search on, and the session's only deferrable tools were down when it was built, so it
    has no bridge. Their recovery is a check_fn flip: it must not add ``tool_search`` /
    ``tool_describe`` / ``tool_call`` to the live tools[]. ``/new`` gets them."""
    from tools.mcp_tool_agent import refresh_agent_mcp_tools

    selection = ["mcp-probe-mail", "clarify"]
    agent = _agent(selection)
    assert not BRIDGE & _names(agent.tools)
    sent = json.dumps(agent.tools, sort_keys=True)

    mail_state["up"] = True
    clock.now += registry_mod._CHECK_FN_TTL_SECONDS + 1
    assert refresh_agent_mcp_tools(agent, quiet_mode=True, preserve_prefix=True) == set()
    assert json.dumps(agent.tools, sort_keys=True) == sent

    assert BRIDGE <= _names(_agent(selection).tools)  # /new


class _ApprovedLateAttempt:
    def poll(self):
        return {"status": "approved", "error": None, "tools": [], "discovery_error": None}


def test_selection_widened_by_a_late_connection_appends_with_the_generation_unchanged(
        mail_state, clock, direct_tools, monkeypatch):
    """``adopt_late_connections`` widens the selection to a server another session already
    connected in this process: nothing registers, the generation stays put, and the server's
    tools must still append between turns, as on main."""
    from hermes_constants import hermes_home_key
    from tools.connectors import mcp as connectors_mcp
    from tools.mcp_tool_agent import refresh_agent_mcp_tools

    late = "mcp__probe_late__fetch"
    registry.register(name=late, toolset="mcp-probe_late", schema=_schema(late, "Fetch from the late server"),
                      handler=lambda args, **kw: "LATE_OK", check_fn=lambda: True)
    registry.register_toolset_alias("probe_late", "mcp-probe_late")
    agent = _direct_agent()
    agent.session_id = "late-session"
    sent = list(agent.tools)
    generation = registry._generation
    monkeypatch.setitem(connectors_mcp._LATE_ATTEMPTS, (hermes_home_key(), "late-session"),
                        {"probe_late": _ApprovedLateAttempt()})
    monkeypatch.setattr("tools.mcp_tool_config._load_mcp_config", lambda: {"probe_late": {"command": "probe"}})
    monkeypatch.setattr("tools.mcp_tool_discovery.register_mcp_servers", lambda servers: None)  # connected already
    try:
        assert connectors_mcp.adopt_late_connections(agent) == ["probe_late"]
        assert registry._generation == generation
        assert refresh_agent_mcp_tools(agent, quiet_mode=True, preserve_prefix=True) == {late}
        assert agent.tools[:-1] == sent and agent.tools[-1]["function"]["name"] == late
    finally:
        registry.deregister(late)


def test_registration_landing_during_the_refresh_build_appends_in_that_refresh(
        mail_state, clock, direct_tools, monkeypatch):
    """The hold decision reads the generation after the build: a server that registers while the
    refresh is building is a late registration, not a check_fn flip, and lands this turn."""
    from tools.mcp_tool_agent import refresh_agent_mcp_tools

    late = "mcp__probe_anchor__aaa_midbuild"
    agent = _direct_agent()
    sent = list(agent.tools)
    real_build = model_tools.get_tool_definitions

    def build_while_a_server_registers(*args, **kwargs):
        if registry.get_entry(late) is None:
            registry.register(name=late, toolset="mcp-probe-anchor", schema=_schema(late, "Mid-build tool"),
                              handler=lambda args, **kw: "MID_OK", check_fn=lambda: True)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(model_tools, "get_tool_definitions", build_while_a_server_registers)
    try:
        assert refresh_agent_mcp_tools(agent, quiet_mode=True, preserve_prefix=True) == {late}
        assert agent.tools[:-1] == sent and agent.tools[-1]["function"]["name"] == late
    finally:
        registry.deregister(late)


class _PinStore:
    """The session row's tools[] pin, JSON round-tripped like the session DB stores it."""

    def __init__(self):
        self.pins = {}

    def update_session_tool_names(self, session_id, pin):
        self.pins[session_id] = json.loads(json.dumps(pin))


def _session_agent(store, session_id):
    agent = _direct_agent()
    agent._session_db, agent.session_id = store, session_id
    return agent


def test_pin_restore_in_the_same_process_holds_a_check_fn_flip(mail_state, clock, direct_tools):
    """An agent rebuilt from the session pin in the same process (gateway agent-cache eviction,
    ``/model``, a fresh agent per turn) is the same conversation: a tool that is new only because
    its check_fn flipped stays out, the same rule the between-turns refresh applies."""
    from tools.mcp_tool_agent import persist_agent_tool_names, refresh_agent_mcp_tools, restore_agent_tool_prefix

    store = _PinStore()
    persist_agent_tool_names(_session_agent(store, "pinned-session"))
    pin = store.pins["pinned-session"]
    assert MAIL not in _names(pin["tools"])

    mail_state["up"] = True
    clock.now += registry_mod._CHECK_FN_TTL_SECONDS + 1
    rebuilt = _session_agent(store, "pinned-session")
    assert MAIL in _names(rebuilt.tools)  # the fresh build sees the recovery
    restore_agent_tool_prefix(rebuilt, pin)
    assert rebuilt.tools == pin["tools"] and MAIL not in rebuilt.valid_tool_names
    assert refresh_agent_mcp_tools(rebuilt, quiet_mode=True, preserve_prefix=True) == set()
    assert rebuilt.tools == pin["tools"]


def test_pin_restore_still_appends_a_late_registration_and_a_restore_in_a_new_process(
        mail_state, clock, direct_tools, monkeypatch):
    """Guards for the hold above: a real registration since the pin still appends, and a process
    that did not write the pin (``--resume`` in a new process, a fresh worker) cannot tell a flip
    from a registration, so it appends as on main."""
    from tools import mcp_tool_agent
    from tools.mcp_tool_agent import persist_agent_tool_names, restore_agent_tool_prefix

    store = _PinStore()
    persist_agent_tool_names(_session_agent(store, "pinned-session"))
    pin = store.pins["pinned-session"]
    late = "mcp__probe_anchor__aaa_late"
    registry.register(name=late, toolset="mcp-probe-anchor", schema=_schema(late, "Late tool"),
                      handler=lambda args, **kw: "LATE_OK", check_fn=lambda: True)
    try:
        rebuilt = _session_agent(store, "pinned-session")
        restore_agent_tool_prefix(rebuilt, pin)
        assert rebuilt.tools[:-1] == pin["tools"] and rebuilt.tools[-1]["function"]["name"] == late
    finally:
        registry.deregister(late)

    persist_agent_tool_names(_session_agent(store, "resumed-session"))
    pin = store.pins["resumed-session"]
    mail_state["up"] = True
    clock.now += registry_mod._CHECK_FN_TTL_SECONDS + 1
    monkeypatch.setattr(mcp_tool_agent, "_PINNED_SNAPSHOTS", OrderedDict(), raising=False)  # wrote no pin
    rebuilt = _session_agent(store, "resumed-session")
    restore_agent_tool_prefix(rebuilt, pin)
    assert rebuilt.tools[:-1] == pin["tools"] and rebuilt.tools[-1]["function"]["name"] == MAIL
