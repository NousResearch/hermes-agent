"""Unanswerable MCP approvals must fail closed without inventing a refusal (#144369)."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from tools import approval, approval_prompt, mcp_tool, mcp_tool_discovery, mcp_tool_handlers
from tools.approval_context import reset_current_session_key, set_current_session_key
from tools.mcp_tool_scope import _server_key
from tools.registry import ToolRegistry


def test_event_subscription_without_notify_is_unanswered(monkeypatch):
    """The sibling subscription gate must not turn the shared cancel verdict into a refusal."""
    from tools import mcp_tool_events

    tokens = set_session_vars(platform="telegram", session_key="event-consent")
    key_token = set_current_session_key("event-consent")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "0")
    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call",
                        lambda name: SimpleNamespace(session=object()))
    dispatch = Mock(side_effect=AssertionError("subscription must not reach MCP"))
    monkeypatch.setattr(mcp_tool_handlers, "_dispatch", dispatch)
    registry = ToolRegistry()
    registry.register(name="subscribe", toolset="mcp_test", schema={"name": "subscribe", "parameters": {}},
                      handler=mcp_tool_events._make_subscribe_event_handler("srv", 30))
    try:
        raw = registry.dispatch("subscribe", {"name": "mail_received"})
        assert isinstance(raw, str)
        error = json.loads(raw)["error"]
        assert "no approval answer" in error
        assert "NOT created" in error
        assert "The user did not approve" not in error
        dispatch.assert_not_called()
    finally:
        reset_current_session_key(key_token)
        clear_session_vars(tokens)


@pytest.mark.parametrize("route", [
    "single_query", "cron", "webhook", "missing_notify", "notify_failed",
    "dispatch_error", "cli_error", "timeout", "withdrawn", "deny",
])
def test_trust_gate_reports_only_real_denials(monkeypatch, route):
    """Drive real routing and gateway queue settlement through the registry, before any MCP I/O."""
    session_key = "unanswered-consent"
    platform = "" if route == "cli_error" else "webhook" if route == "webhook" else "telegram"
    tokens = set_session_vars(platform=platform, session_key=session_key,
                              cron_session="1" if route == "cron" else "")
    key_token = set_current_session_key(session_key)
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1" if route == "single_query" else "0")
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "0")
    monkeypatch.setenv("HERMES_YOLO_MODE", "0")
    monkeypatch.setitem(mcp_tool._server_trust_levels, _server_key("srv"), "untrusted")
    connect = Mock(side_effect=AssertionError("approval must block before connecting"))
    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call", connect)
    panel = Mock(side_effect=RuntimeError("panel unavailable"))
    monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", panel)
    if route == "timeout":
        monkeypatch.setattr(approval_prompt._ctx, "_get_approval_timeout", lambda: 0)
    if route == "dispatch_error":
        monkeypatch.setattr(approval_prompt._gw, "_await_gateway_decision",
                            Mock(side_effect=RuntimeError("dispatch unavailable")))

    def notify(data):
        if route == "notify_failed":
            raise RuntimeError("client cannot display approval")
        if route == "withdrawn":
            approval.unregister_gateway_notify(session_key)
        elif route == "deny":
            assert approval.resolve_gateway_approval(session_key, "deny", request_id=data["request_id"]) == 1

    notifier = Mock(side_effect=notify)
    if route != "missing_notify":
        approval.register_gateway_notify(session_key, notifier)
    consent = Mock(wraps=approval_prompt.request_elicitation_consent)
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent", consent)
    registry = ToolRegistry()
    registry.register(name="write", toolset="mcp_test", schema={"name": "write", "parameters": {}},
                      handler=mcp_tool_handlers._make_tool_handler("srv", "write", 30))
    try:
        raw = registry.dispatch("write", {"message": "draft"})
        assert isinstance(raw, str)
        result = json.loads(raw)
        assert "error" in result
        connect.assert_not_called()
        consent.assert_called_once()
        if route == "deny":
            assert "The user did not approve" in result["error"]
        else:
            assert "The user did not approve" not in result["error"]
            assert "no approval answer" in result["error"].lower()
            assert "NOT run" in result["error"]
            assert "Do not retry" in result["error"]
        if route in ("single_query", "cron", "webhook", "missing_notify"):
            notifier.assert_not_called()
            panel.assert_not_called()
        assert not approval._gateway_queues.get(session_key)
    finally:
        approval.unregister_gateway_notify(session_key)
        reset_current_session_key(key_token)
        clear_session_vars(tokens)


@pytest.mark.parametrize("route", ["single_query", "missing_notify", "notify_failed", "dispatch_error"])
def test_consent_unavailable_is_not_decline(monkeypatch, route):
    """The shared consent API distinguishes no answer from an explicit no."""
    tokens = set_session_vars(platform="telegram", session_key="consent-verdict")
    key_token = set_current_session_key("consent-verdict")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1" if route == "single_query" else "0")
    if route == "dispatch_error":
        monkeypatch.setattr(approval_prompt._gw, "_await_gateway_decision",
                            Mock(side_effect=RuntimeError("dispatch unavailable")))
    if route != "missing_notify":
        approval.register_gateway_notify("consent-verdict", Mock(side_effect=RuntimeError("no client")))
    try:
        assert approval_prompt.request_elicitation_consent("write", "Approve this write") == "cancel"
    finally:
        approval.unregister_gateway_notify("consent-verdict")
        reset_current_session_key(key_token)
        clear_session_vars(tokens)
