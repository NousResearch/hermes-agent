"""The quiet-mode ``get_tool_definitions`` memo must not outlive the check_fn verdicts it filtered on.

An MCP server whose connection dropped (check_fn False) and came back without re-registering its
tools left the memo holding the omission: ``tool_search`` / ``tool_describe`` read the memo, so the
recovered server stayed invisible for the life of the process, ``/new`` included.
"""
import json

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
    assert mail_state["probes"] == 3  # the grace verdict is not memoized: every call re-probes

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
