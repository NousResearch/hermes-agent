"""A required MCP guard is the last authorization seam, not an optional hook."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace
import asyncio
import hashlib
import json
import threading
import time
from types import MappingProxyType, SimpleNamespace

import pytest

from gateway.trusted_turn_origin import TrustedTurnOrigin, bind_turn_origin, current_origin
from tools import mcp_dispatch_guard as guard
from tools import mcp_tool_handlers as handlers


POLICY = {
    "server": "knowledge", "plugin": "identity-binding", "profile": "assistant",
    "platform": "slack", "timeout_seconds": 0.1,
    "cron": {"job_id_sha256": [hashlib.sha256(b"morning").hexdigest()],
             "read_only_tools": ["search"]},
}


def _origin(*, user="U1", channel="C1", message="100.1", platform="slack"):
    return TrustedTurnOrigin(
        kind="gateway", platform=platform, profile="assistant", transport_profile="assistant",
        workspace_id="T1", channel_id=channel, user_id=user, message_ts=message,
        thread_ts=None, admission_id="admission", admitted=True, internal=False,
        claims=MappingProxyType({"identity-binding": MappingProxyType({"user": user})}),
    )


@pytest.fixture
def configured(monkeypatch):
    from hermes_cli import config, plugins, profiles

    monkeypatch.setattr(config, "load_config_readonly", lambda: {"mcp": {"required_dispatch_guards": [POLICY]}})
    monkeypatch.setattr(profiles, "current_profile_name", lambda default=None: "assistant")
    manager = SimpleNamespace(
        _plugins={"identity-binding": SimpleNamespace(enabled=True)},
        _required_mcp_dispatch_guards={"knowledge": ("identity-binding", lambda **kw: kw["args"])},
    )
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    return manager


def test_unconfigured_server_keeps_existing_dispatch(configured):
    args = {"query": "hello"}
    assert guard.admit_mcp_dispatch("other", "search", args) == (args, None)


def test_registered_required_guard_denies_if_profile_policy_row_disappears(configured, monkeypatch):
    from hermes_cli import config

    monkeypatch.setattr(config, "load_config_readonly", lambda: {"mcp": {"required_dispatch_guards": []}})
    with bind_turn_origin(_origin()):
        args, denial = guard.admit_mcp_dispatch("knowledge", "search", {})
    assert args is None and "policy is missing" in denial
    assert guard.admit_mcp_dispatch("other", "search", {}) == ({}, None)


def test_unavailable_registry_denies_without_raising_when_policy_row_is_missing(configured, monkeypatch):
    from hermes_cli import config, plugins

    monkeypatch.setattr(config, "load_config_readonly", lambda: {"mcp": {"required_dispatch_guards": []}})
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: (_ for _ in ()).throw(RuntimeError("unavailable")))
    args, denial = guard.admit_mcp_dispatch("knowledge", "search", {})
    assert args is None and "registry is unavailable" in denial


def test_missing_origin_and_missing_plugin_fail_closed(configured):
    args, denial = guard.admit_mcp_dispatch("knowledge", "search", {"query": "x"})
    assert args is None and "no call was sent" in denial
    configured._plugins.clear()
    with bind_turn_origin(_origin()):
        args, denial = guard.admit_mcp_dispatch("knowledge", "search", {"query": "x"})
    assert args is None and "no call was sent" in denial


@pytest.mark.parametrize("bad", [None, [], {"query": float("nan")}])
def test_invalid_callback_result_denies(configured, bad):
    configured._required_mcp_dispatch_guards["knowledge"] = ("identity-binding", lambda **kw: bad)
    with bind_turn_origin(_origin()):
        args, denial = guard.admit_mcp_dispatch("knowledge", "search", {"query": "x"})
    assert args is None and "no call was sent" in denial


def test_callback_failure_or_timeout_denies(configured):
    def fails(**kw):
        raise RuntimeError("oops")

    configured._required_mcp_dispatch_guards["knowledge"] = ("identity-binding", fails)
    with bind_turn_origin(_origin()):
        assert guard.admit_mcp_dispatch("knowledge", "search", {})[0] is None

    blocked = threading.Event()
    configured._required_mcp_dispatch_guards["knowledge"] = (
        "identity-binding", lambda **kw: blocked.wait(1) or kw["args"])
    with bind_turn_origin(_origin()):
        start = time.monotonic()
        args, denial = guard.admit_mcp_dispatch("knowledge", "search", {})
    blocked.set()
    assert args is None and denial and time.monotonic() - start < 0.5


@pytest.mark.parametrize("channel", ["C1", "D1"])
def test_current_slack_channel_or_dm_reaches_guard(configured, channel):
    seen = []

    def canonical(**kw):
        origin = current_origin()
        seen.append((origin.channel_id, origin.user_id, kw["tool_name"]))
        return {**kw["args"], "_meta": {"slack_user": origin.user_id}}

    configured._required_mcp_dispatch_guards["knowledge"] = ("identity-binding", canonical)
    with bind_turn_origin(_origin(channel=channel)):
        args, denial = guard.admit_mcp_dispatch("knowledge", "search", {"query": "x"})
    assert denial is None and args["_meta"]["slack_user"] == "U1"
    assert seen == [(channel, "U1", "search")]


def test_concurrent_queued_origins_do_not_cross(configured):
    def canonical(**kw):
        origin = current_origin()
        return {**kw["args"], "actor": origin.user_id, "message": origin.message_ts}

    configured._required_mcp_dispatch_guards["knowledge"] = ("identity-binding", canonical)

    def call(user, message):
        with bind_turn_origin(_origin(user=user, message=message)):
            return guard.admit_mcp_dispatch("knowledge", "search", {})[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(lambda row: call(*row), [("U1", "1"), ("U2", "2")]))
    assert (a["actor"], a["message"]) == ("U1", "1")
    assert (b["actor"], b["message"]) == ("U2", "2")


def test_spoofed_or_mismatched_origin_denied_before_callback(configured):
    called = []
    configured._required_mcp_dispatch_guards["knowledge"] = ("identity-binding", lambda **kw: called.append(kw))
    for origin in (replace(_origin(), transport_profile="other"), replace(_origin(), platform="discord"),
                   replace(_origin(), admitted=False), replace(_origin(), internal=True)):
        with bind_turn_origin(origin):
            assert guard.admit_mcp_dispatch("knowledge", "search", {"user": "U2"})[0] is None
    assert called == []


def test_ambiguous_cron_context_does_not_bypass_current_slack_guard(configured, monkeypatch):
    from cron import execution_identity

    cron = SimpleNamespace(profile="assistant", source="builtin", scheduled_instant="2026-10-22T00:00:00Z",
                           execution_id="exec", job_id="morning")
    monkeypatch.setattr(execution_identity, "current_cron_execution", lambda: cron)
    monkeypatch.setattr(handlers, "_tool_is_read_only", lambda *_: True)
    configured._required_mcp_dispatch_guards.clear()
    with bind_turn_origin(_origin()):
        assert guard.admit_mcp_dispatch("knowledge", "search", {})[0] is None


def test_required_guard_is_final_authority_after_advisory_argument_edits(configured, monkeypatch):
    seen = []

    def canonical(**kw):
        # The input represents arguments after an optional pre_tool_call hook
        # changed them. The mandatory guard replaces the spoofed actor.
        return {**kw["args"], "actor": current_origin().user_id}

    configured._required_mcp_dispatch_guards["knowledge"] = ("identity-binding", canonical)
    server = SimpleNamespace(_rpc_lock=asyncio.Lock(), _pending_call_context=None,
                             _mark_session_proven=None)

    @asynccontextmanager
    async def track(*a, **kw):
        yield

    async def wire(_server, _server_name, _tool_name, args, _progress):
        seen.append(args)
        return {"ok": True}

    monkeypatch.setattr(handlers, "_trust_gate_check", lambda *a, **kw: None)
    monkeypatch.setattr(handlers, "_check_circuit_breaker", lambda *a: None)
    monkeypatch.setattr(handlers, "_acquire_call_server", lambda *a: (server, None))
    monkeypatch.setattr(handlers, "_track_inflight_rpc", track)
    monkeypatch.setattr(handlers, "_call_tool_racing_stdio_death", wire)
    monkeypatch.setattr(handlers, "_render_call_tool_result", lambda *a: "ok")
    monkeypatch.setattr(handlers, "_dispatch", lambda _name, _server, _op, call, *_a, **_kw: asyncio.run(call()))
    with bind_turn_origin(_origin(user="U1")):
        assert handlers._make_tool_handler("knowledge", "create_document", 5)(
            {"title": "draft", "actor": "U2"}) == "ok"
    assert seen == [{"title": "draft", "actor": "U1"}]


def test_scheduler_exception_needs_exact_occurrence_job_tool_and_readonly(configured, monkeypatch):
    from cron import execution_identity

    cron = SimpleNamespace(profile="assistant", source="builtin", scheduled_instant="2026-10-22T00:00:00Z",
                           execution_id="exec", job_id="morning")
    monkeypatch.setattr(execution_identity, "current_cron_execution", lambda: cron)
    monkeypatch.setattr(handlers, "_tool_is_read_only", lambda server, tool: tool == "search")
    assert guard.admit_mcp_dispatch("knowledge", "search", {"query": "x"}) == ({"query": "x"}, None)
    for spoof in ({"_meta": {"slack": {"user_id": "U1"}}}, {"slack_user_id": "U1"},
                  {"query": [{"channel_id": "C1"}]}):
        assert guard.admit_mcp_dispatch("knowledge", "search", spoof)[0] is None
    for change in ({"job_id": "other"}, {"source": "direct"}, {"scheduled_instant": None},
                   {"profile": "other"}, {"execution_id": ""}):
        monkeypatch.setattr(execution_identity, "current_cron_execution", lambda c=change: SimpleNamespace(**{**vars(cron), **c}))
        assert guard.admit_mcp_dispatch("knowledge", "search", {})[0] is None
    monkeypatch.setattr(execution_identity, "current_cron_execution", lambda: cron)
    assert guard.admit_mcp_dispatch("knowledge", "create_document", {})[0] is None
    monkeypatch.setattr(handlers, "_tool_is_read_only", lambda server, tool: False)
    assert guard.admit_mcp_dispatch("knowledge", "search", {})[0] is None


def test_native_write_denial_stops_before_trust_or_transport(configured, monkeypatch):
    def unexpected(*a, **kw):
        raise AssertionError("MCP call reached trust, charge or transport")

    monkeypatch.setattr(handlers, "_trust_gate_check", unexpected)
    monkeypatch.setattr(handlers, "_acquire_call_server", unexpected)
    result = handlers._make_tool_handler("knowledge", "create_document", 5)({"title": "x"})
    assert "no call was sent" in result


def test_utility_dispatch_is_guarded_too(configured, monkeypatch):
    from tools import mcp_tool_discovery
    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call",
                        lambda *a: (_ for _ in ()).throw(AssertionError("utility reached transport")))
    result = handlers._make_read_resource_handler("knowledge", 5)({"uri": "x"})
    assert "no call was sent" in result


def test_safe_worker_optional_hook_bypass_cannot_bypass_required_guard(configured, monkeypatch):
    from agent import safe_worker_policy
    monkeypatch.setattr(safe_worker_policy, "safe_worker_enabled", lambda: True)
    assert guard.admit_mcp_dispatch("knowledge", "search", {})[0] is None


def test_registration_is_profile_local_and_refuses_duplicate(monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager
    from hermes_cli.plugins_manifest import PluginManifest

    manager = PluginManager(scope_key="/tmp/required-guard-test")
    ctx = PluginContext(PluginManifest(name="identity-binding", version="1.0"), manager)
    callback = lambda **kw: kw["args"]
    handle = ctx.register_required_mcp_dispatch_guard(server="knowledge", callback=callback)
    assert manager._required_mcp_dispatch_guards["knowledge"] == ("identity-binding", callback)
    with pytest.raises(ValueError):
        ctx.register_required_mcp_dispatch_guard(server="knowledge", callback=callback)
    handle.dispose()
    assert "knowledge" not in manager._required_mcp_dispatch_guards
