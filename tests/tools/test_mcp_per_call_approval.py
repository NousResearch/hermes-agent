"""ClixRx per-call consent through real registry and CLI/gateway approval paths.

All transport sessions are fake; configuration and grants live in the test home.
"""

import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.mcp_catalog import _build_server_config, _parse_manifest
from tools import approval, approval_prompt, mcp_tool, mcp_tool_costs
from tools import mcp_tool_handlers as handlers, mcp_tool_registration as registration
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.registry import ToolRegistry


@pytest.fixture
def transport(monkeypatch):
    for name in ("_servers", "_server_trust_levels", "_server_approval_policies",
                 "_tool_read_only_hints", "_server_cost_policies", "_server_error_counts"):
        monkeypatch.setattr(mcp_tool, name, {})
    monkeypatch.setattr(approval, "_session_approved", {})
    monkeypatch.setattr(approval, "_permanent_approved", set())
    monkeypatch.setattr(approval, "_permanent_approved_by_home", {})
    monkeypatch.setattr(approval, "_permanent_baseline_by_home", {})
    monkeypatch.setattr(approval, "_session_yolo", set())
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(mcp_tool_costs, "_spent", {})
    monkeypatch.setattr(mcp_tool_costs, "_ask_locks", {})
    monkeypatch.setattr(mcp_tool_costs, "_unscoped_moved_to", None)
    result = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")],
                             isError=False, structuredContent=None)
    session = SimpleNamespace(call_tool=AsyncMock(return_value=result))
    server = SimpleNamespace(session=session, _rpc_lock=None)
    mcp_tool._servers["clixrx"] = server

    def run(coro_or_factory, timeout=30):
        async def call():
            server._rpc_lock = asyncio.Lock()
            return await (coro_or_factory() if callable(coro_or_factory) else coro_or_factory)
        return asyncio.run(call())

    monkeypatch.setattr("tools.mcp_tool_loop._run_on_mcp_loop", run)
    return session


def _configure(config=None, *, read_only=False):
    if config is None:
        path = Path(__file__).parents[2] / "optional-mcps/clixrx/manifest.yaml"
        config = _build_server_config(_parse_manifest(path), None)
    registration._record_tool_trust_metadata("clixrx", config, [
        SimpleNamespace(name="lookup", annotations={"readOnlyHint": read_only})])
    registry = ToolRegistry()
    registry.register(name="test_clixrx_lookup", toolset="mcp-clixrx",
                      schema={"name": "test_clixrx_lookup", "parameters": {"type": "object"}},
                      handler=handlers._make_tool_handler("clixrx", "lookup", 30))
    return registry


@contextmanager
def _surface(kind, choices, monkeypatch):
    cards = []
    tokens = set_session_vars(platform="telegram" if kind == "gateway" else "", session_key="chat")
    key = set_current_session_key("chat")
    answers = iter(choices)

    def cli_answer(command, description, **kwargs):
        cards.append(dict(kwargs, command=command, description=description))
        return next(answers)

    def gateway_answer(data):
        cards.append(dict(data))
        approval.resolve_gateway_approval("chat", next(answers), request_id=data["request_id"])

    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: cli_answer)
    monkeypatch.setattr(approval_prompt._ctx, "_no_user_can_answer", lambda: False)
    if kind == "gateway":
        approval.register_gateway_notify("chat", gateway_answer)
    try:
        yield cards
    finally:
        if kind == "gateway":
            approval.unregister_gateway_notify("chat")
        reset_current_session_key(key)
        clear_session_vars(tokens)


@pytest.mark.parametrize("surface", ["cli", "gateway"])
@pytest.mark.parametrize("grant", ["session", "always", "session_yolo", "process_yolo", "off"])
@pytest.mark.parametrize("read_only", [False, True])
def test_each_clixrx_call_asks_and_denial_never_dispatches(
        transport, monkeypatch, surface, grant, read_only):
    from hermes_cli.config import get_config_path, save_config
    registry = _configure(read_only=read_only)
    if grant == "session_yolo":
        approval.enable_session_yolo("chat")
    elif grant == "process_yolo":
        monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    elif grant == "off":
        save_config({"approvals": {"mode": "off"}})
    else:
        approval._persist_choice("chat", grant, ["mcp:clixrx/lookup"])
    config_path = get_config_path()
    before = config_path.read_bytes() if config_path.exists() else None
    sessions_before = {key: set(value) for key, value in approval._session_approved.items()}
    monkeypatch.setattr(approval, "is_approved", lambda *a: pytest.fail("per-call policy read grants"))

    # Even a stale surface returning a broader choice grants only this call.
    with _surface(surface, ["session", "always", "deny"], monkeypatch) as cards:
        if grant in ("session_yolo", "process_yolo", "off"):
            assert approval.is_approval_bypass_active() is True
        for expected in (1, 2):
            assert '"result": "ok"' in registry.dispatch("test_clixrx_lookup", {})
            assert transport.call_tool.await_count == expected
        denied = registry.dispatch("test_clixrx_lookup", {})
    assert "did not approve" in denied
    assert transport.call_tool.await_count == 2
    assert len(cards) == 3
    assert all(card["allow_session"] is False and card["allow_permanent"] is False for card in cards)
    assert approval._session_approved == sessions_before
    assert (config_path.read_bytes() if config_path.exists() else None) == before


@pytest.mark.parametrize("surface", ["cli", "gateway"])
@pytest.mark.parametrize("choice", ["session", "always"])
def test_default_policy_keeps_scoped_grants(transport, monkeypatch, surface, choice):
    registry = _configure({"trust": "untrusted"})
    with _surface(surface, [choice], monkeypatch) as cards:
        for _ in range(2):
            assert '"result": "ok"' in registry.dispatch("test_clixrx_lookup", {})
    assert transport.call_tool.await_count == 2
    assert len(cards) == 1
    assert cards[0].get("allow_session", True) is True and cards[0]["allow_permanent"] is True
    assert approval.is_approved("chat", "mcp:clixrx/lookup") is True


@pytest.mark.parametrize("surface", ["cli", "gateway"])
def test_per_call_over_limit_consent_does_not_remember_or_raise_budget(transport, monkeypatch, surface):
    registry = _configure({"trust": "untrusted", "approval_policy": "per_call",
                           "tool_costs": {"lookup": 40}, "spend_limit": 30})
    with _surface(surface, ["always", "once", "deny"], monkeypatch) as cards:
        registry.dispatch("test_clixrx_lookup", {})
        registry.dispatch("test_clixrx_lookup", {})
        denied = registry.dispatch("test_clixrx_lookup", {})
    assert transport.call_tool.await_count == 2
    assert len(cards) == 3
    assert all("spend limit" in card["command"] for card in cards)
    assert "used about 80 credits" in denied
    assert all(card["allow_session"] is False and card["allow_permanent"] is False for card in cards)
    assert approval._session_approved == {}


@pytest.mark.parametrize("value", [None, False, "", "typo", [], {}])
def test_invalid_runtime_policy_fails_closed(transport, monkeypatch, value):
    registry = _configure({"trust": "full", "approval_policy": value}, read_only=True)
    with _surface("cli", ["deny"], monkeypatch) as cards:
        assert "did not approve" in registry.dispatch("test_clixrx_lookup", {})
    assert len(cards) == 1
    transport.call_tool.assert_not_awaited()


def test_shared_connection_keeps_consuming_profiles_policy(transport, monkeypatch):
    from tools.mcp_tool_scope import rekey_unscoped_connections
    _configure()
    rekey_unscoped_connections("owner")
    registration._record_scope_trust("clixrx", {"trust": "full"}, "other")
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: "owner")
    assert handlers._requires_per_call_approval("clixrx") is True
    monkeypatch.setattr(mcp_tool, "_mcp_registry_scope", lambda: "other")
    assert handlers._requires_per_call_approval("clixrx") is False
    registration._record_scope_trust("clixrx", {"approval_policy": "per_call"}, "other")
    assert handlers._requires_per_call_approval("clixrx") is True
