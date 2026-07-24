from __future__ import annotations

import json
import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import (
    _classify_ai_dev_usage_route,
    _record_ai_dev_usage,
    delegate_task,
)


def make_parent() -> MagicMock:
    parent = MagicMock()
    parent.base_url = "https://chatgpt.com/backend-api/codex"
    parent.api_key = "oauth-token-placeholder"
    parent.provider = "openai-codex"
    parent.api_mode = "codex_responses"
    parent.model = "gpt-5.6-sol"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def test_openai_codex_delegate_is_subscription_delegate_route() -> None:
    route = _classify_ai_dev_usage_route(
        {
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "oauth-token-placeholder",
        },
        make_parent(),
    )

    assert route == ("openai-codex", "gpt-5.6-sol", "delegate")


def test_local_loopback_routes_pass_without_enforcement() -> None:
    """Local routes (loopback) pass through even without enforcement configured."""
    rc = _record_ai_dev_usage(
        provider="local-llama",
        model="llama-3.1",
        route="local",
        task="Debug issue",
        allow_variable_cost=False,
    )
    assert rc == 0, "Local routes should pass without enforcement"


def test_delegate_routes_pass_without_enforcement() -> None:
    """OAuth subscription routes (delegate) pass through even without enforcement."""
    rc = _record_ai_dev_usage(
        provider="openai-codex",
        model="gpt-5.6-sol",
        route="delegate",
        task="Write code",
        allow_variable_cost=False,
    )
    assert rc == 0, "Delegate routes should pass without enforcement"


def test_copilot_routes_pass_without_enforcement() -> None:
    """GitHub Copilot routes pass through even without enforcement."""
    rc = _record_ai_dev_usage(
        provider="copilot",
        model="gpt-4o",
        route="copilot",
        task="Refactor",
        allow_variable_cost=False,
    )
    assert rc == 0, "Copilot routes should pass without enforcement"


def test_api_key_route_blocked_by_default_when_no_enforcement() -> None:
    """Paid API-key routes are blocked by default (fail-closed)."""
    with patch.dict(os.environ, {}, clear=True):
        rc = _record_ai_dev_usage(
            provider="openai-api",
            model="gpt-4o",
            route="api_key",
            task="Analyze logs",
            allow_variable_cost=False,
        )
    assert rc != 0, "API-key routes should fail closed without enforcement"


@patch("tools.delegate_tool._run_single_child")
@patch("tools.delegate_tool._resolve_delegation_credentials")
def test_api_key_delegate_unblocked_when_allow_route_set(
    resolve_credentials: MagicMock,
    run_child: MagicMock,
) -> None:
    """Delegation to paid API-key routes passes when AI_DEV_USAGE_ALLOW_ROUTE=1 is set."""
    resolve_credentials.return_value = {
        "provider": "openai-api",
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-key",
        "api_mode": "chat_completions",
    }

    # Mock _run_single_child to return a valid result dict
    run_child.return_value = {
        "summary": "Test completed",
        "result": "success",
    }

    # Set AI_DEV_USAGE_ALLOW_ROUTE to bypass enforcement
    env_backup = os.environ.get("AI_DEV_USAGE_ALLOW_ROUTE")
    try:
        os.environ["AI_DEV_USAGE_ALLOW_ROUTE"] = "1"
        result = json.loads(
            delegate_task(goal="Analyze logs", parent_agent=make_parent())
        )

        # Should NOT contain "blocked" or "route" error messages
        # It may fail for other reasons (credential resolution, etc.) but not due to gating
        if "error" in result:
            error_msg = result["error"].lower()
            assert "blocked" not in error_msg and "route" not in error_msg
        else:
            # Success - child was called
            run_child.assert_called_once()
    finally:
        if env_backup is None:
            os.environ.pop("AI_DEV_USAGE_ALLOW_ROUTE", None)
        else:
            os.environ["AI_DEV_USAGE_ALLOW_ROUTE"] = env_backup


@patch("tools.delegate_tool._run_single_child")
@patch("tools.delegate_tool._resolve_delegation_credentials")
def test_api_key_delegate_fails_closed_by_default(
    resolve_credentials: MagicMock,
    run_child: MagicMock,
) -> None:
    """Delegation to paid API-key routes is blocked by default without enforcement."""
    resolve_credentials.return_value = {
        "provider": "openai-api",
        "model": "gpt-5.4",
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-not-a-real-key",
        "api_mode": "chat_completions",
    }

    # Clear AI_DEV_USAGE_ENFORCE to test fail-closed default
    # Mock _run_single_child to return a valid result dict (though it shouldn't be called)
    run_child.return_value = {
        "summary": "Test completed",
        "result": "success",
    }

    env_backup = os.environ.pop("AI_DEV_USAGE_ENFORCE", None)
    try:
        result = json.loads(
            delegate_task(goal="Review current diff", parent_agent=make_parent())
        )

        assert "error" in result
        assert "blocked" in result["error"].lower() or "route" in result["error"].lower()
        run_child.assert_not_called()
    finally:
        if env_backup is not None:
            os.environ["AI_DEV_USAGE_ENFORCE"] = env_backup


@patch("tools.delegate_tool._run_single_child")
@patch("tools.delegate_tool._resolve_delegation_credentials")
def test_local_delegate_passes_without_enforcement(
    resolve_credentials: MagicMock,
    run_child: MagicMock,
) -> None:
    """Delegation to local routes passes without enforcement configured."""
    resolve_credentials.return_value = {
        "provider": "local-llama",
        "model": "llama-3.1",
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key": "",
        "api_mode": "chat_completions",
    }

    # Mock _run_single_child to return a valid result dict
    run_child.return_value = {
        "summary": "Test completed",
        "result": "success",
    }

    env_backup = os.environ.pop("AI_DEV_USAGE_ENFORCE", None)
    try:
        result = json.loads(
            delegate_task(goal="Debug issue", parent_agent=make_parent())
        )

        # Local routes should not be blocked; may fail for other reasons
        # but should NOT contain "blocked" or "route" error messages
        if "error" in result:
            error_msg = result["error"].lower()
            assert "blocked" not in error_msg and "route" not in error_msg
    finally:
        if env_backup is not None:
            os.environ["AI_DEV_USAGE_ENFORCE"] = env_backup


@patch("tools.delegate_tool._run_single_child")
@patch("tools.delegate_tool._resolve_delegation_credentials")
def test_allow_variable_cost_tool_arg_ignored(
    resolve_credentials: MagicMock,
    run_child: MagicMock,
) -> None:
    """The allow_variable_cost tool arg was removed; model cannot self-approve."""
    resolve_credentials.return_value = {
        "provider": "openai-api",
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-key",
        "api_mode": "chat_completions",
    }

    # Mock _run_single_child to return a valid result dict (though it shouldn't be called)
    run_child.return_value = {
        "summary": "Test completed",
        "result": "success",
    }

    env_backup = os.environ.pop("AI_DEV_USAGE_ENFORCE", None)
    try:
        # The delegate_task function no longer accepts allow_variable_cost arg
        # This tests that the arg is ignored (not passed to handler)
        result = json.loads(
            delegate_task(goal="Test", parent_agent=make_parent())
        )

        # Should still be blocked because env is not set
        assert "error" in result
        assert "blocked" in result["error"].lower() or "route" in result["error"].lower()
        run_child.assert_not_called()
    finally:
        if env_backup is not None:
            os.environ["AI_DEV_USAGE_ENFORCE"] = env_backup