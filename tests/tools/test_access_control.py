"""Tests for scoped connected-account access control."""

import json

from model_tools import handle_function_call
from tools.access_control import evaluate_access, infer_mcp_operation
from tools.registry import registry


def test_evaluate_access_uses_read_only_platform_profile_for_writes():
    decision = evaluate_access(
        "ha_call_service",
        {"service": "homeassistant", "account": "homeassistant", "operation": "write"},
        platform="cron",
        config={
            "access_control": {
                "enabled": True,
                "default_scope": "full",
                "platform_profiles": {"cron": "read-only"},
            }
        },
    )

    assert decision.allowed is False
    assert decision.scope == "read-only"
    assert decision.platform == "cron"


def test_evaluate_access_allows_account_override_for_write():
    decision = evaluate_access(
        "mcp_github_create_issue",
        {"service": "mcp", "account": "mcp.github", "operation": "write"},
        platform="cron",
        config={
            "access_control": {
                "enabled": True,
                "default_scope": "full",
                "platform_profiles": {"cron": "read-only"},
                "accounts": {"mcp.github": {"scope": "full"}},
            }
        },
    )

    assert decision.allowed is True
    assert decision.scope == "full"


def test_infer_mcp_operation_fails_closed_for_mutating_verbs():
    assert infer_mcp_operation("mcp_github_create_issue") == "write"
    assert infer_mcp_operation("mcp_docs_read_page") == "read"
    assert infer_mcp_operation("mcp_unknown_sync") == "write"


def test_registry_hides_static_tool_when_scope_denies_it(monkeypatch):
    tool_name = "test_scope_hidden_tool"
    original = registry._tools.copy()
    try:
        registry.register(
            name=tool_name,
            toolset="testing",
            schema={
                "name": tool_name,
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda args, **kwargs: json.dumps({"ok": True}),
            access_fn=lambda args, **kwargs: {
                "service": "homeassistant",
                "account": "homeassistant",
                "operation": "write",
            },
            access_static=True,
        )
        monkeypatch.setattr(
            "tools.access_control._get_access_control_config",
            lambda config=None: {
                "enabled": True,
                "default_scope": "full",
                "platform_profiles": {"cron": "read-only", "cli": "full"},
            },
        )

        hidden = registry.get_definitions({tool_name}, quiet=True, platform="cron")
        shown = registry.get_definitions({tool_name}, quiet=True, platform="cli")

        assert hidden == []
        assert shown[0]["function"]["name"] == tool_name
    finally:
        registry._tools = original


def test_handle_function_call_returns_scope_violation_without_running_handler(monkeypatch):
    tool_name = "test_scope_blocked_tool"
    original = registry._tools.copy()
    called = {"value": False}
    try:
        registry.register(
            name=tool_name,
            toolset="testing",
            schema={
                "name": tool_name,
                "description": "blocked tool",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda args, **kwargs: called.__setitem__("value", True) or json.dumps({"ok": True}),
            access_fn=lambda args, **kwargs: {
                "service": "mcp",
                "account": "mcp.github",
                "operation": "write",
            },
            access_static=True,
        )
        monkeypatch.setattr(
            "tools.access_control._get_access_control_config",
            lambda config=None: {
                "enabled": True,
                "default_scope": "full",
                "platform_profiles": {"cron": "read-only"},
            },
        )

        result = json.loads(handle_function_call(tool_name, {}, platform="cron"))

        assert result["error_type"] == "scope_violation"
        assert called["value"] is False
    finally:
        registry._tools = original
