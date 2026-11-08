"""Tests for the request_gateway_restart tool."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from tools.gateway_restart_tool import (
    _check_restart_policy,
    _handle_request_gateway_restart,
    _is_authorized_foreground_turn,
    _resolve_current_source,
    check_fn,
)
from tools.registry import registry


def test_restart_tool_registered_at_top_level():
    """Verify tool is registered in gateway toolset and exposed in _HERMES_CORE_TOOLS."""
    from toolsets import _HERMES_CORE_TOOLS

    entry = registry.get_entry("request_gateway_restart")
    assert entry is not None
    assert entry.name == "request_gateway_restart"
    assert entry.toolset == "gateway"
    assert "description" in entry.schema["function"]
    assert "request_gateway_restart" in _HERMES_CORE_TOOLS


def test_foreground_authorization_guards(monkeypatch):
    """Verify invocation is denied outside gateway, in cron, or in subagent context."""
    # 1. Outside Gateway process
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    assert not _is_authorized_foreground_turn()
    assert not check_fn()
    res = json.loads(_handle_request_gateway_restart({}))
    assert not res["success"]
    assert "restricted" in res["error"]

    # 2. Inside Gateway but missing session provenance
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    assert not _is_authorized_foreground_turn()

    # 3. Inside Gateway with session provenance -> True
    with patch("gateway.session_context.get_session_env") as mock_env:
        mock_env.side_effect = lambda k: {
            "HERMES_SESSION_PLATFORM": "telegram",
            "HERMES_SESSION_CHAT_ID": "12345",
            "HERMES_SESSION_CHAT_TYPE": "dm",
        }.get(k)
        assert _is_authorized_foreground_turn()
        assert check_fn()

    # 4. In cron execution -> False
    with patch("gateway.session_context.get_session_env") as mock_env:
        mock_env.side_effect = lambda k: {
            "HERMES_CRON_AUTO_DELIVER_PLATFORM": "telegram",
            "HERMES_SESSION_PLATFORM": "telegram",
            "HERMES_SESSION_CHAT_ID": "12345",
        }.get(k)
        assert not _is_authorized_foreground_turn()

    # 5. In delegated subagent -> False
    with patch("gateway.session_context.get_session_env") as mock_env:
        mock_env.side_effect = lambda k: {
            "HERMES_SESSION_PLATFORM": "telegram",
            "HERMES_SESSION_CHAT_ID": "12345",
        }.get(k)
        with patch("agent.delegation_context.is_delegated_child_process_context", return_value=True):
            assert not _is_authorized_foreground_turn()

    # 6. In Kanban worker -> False
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    with patch("gateway.session_context.get_session_env") as mock_env:
        mock_env.side_effect = lambda k: {
            "HERMES_SESSION_PLATFORM": "telegram",
            "HERMES_SESSION_CHAT_ID": "12345",
        }.get(k)
        assert not _is_authorized_foreground_turn()


def test_slash_access_policy_check():
    """Verify restart policy honors admin/slash permissions."""
    runner = MagicMock()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="user-1")

    with patch("gateway.slash_access.policy_for_source") as mock_pfs:
        policy = MagicMock()
        policy.enabled = True
        policy.is_admin.return_value = False
        policy.can_run.return_value = False
        mock_pfs.return_value = policy

        denial = _check_restart_policy(runner, source)
        assert denial is not None
        assert "Not allowed" in denial

        policy.can_run.return_value = True
        assert _check_restart_policy(runner, source) is None


def test_resolve_current_source():
    """Verify session ContextVars are correctly parsed into SessionSource."""
    with patch("gateway.session_context.get_session_env") as mock_env:
        mock_env.side_effect = lambda k: {
            "HERMES_SESSION_PLATFORM": "telegram",
            "HERMES_SESSION_CHAT_ID": "chat_999",
            "HERMES_SESSION_CHAT_TYPE": "dm",
            "HERMES_SESSION_USER_ID": "user_888",
            "HERMES_SESSION_MESSAGE_ID": "msg_777",
        }.get(k)
        source = _resolve_current_source()
        assert source is not None
        assert source.platform == Platform.TELEGRAM
        assert source.chat_id == "chat_999"
        assert source.user_id == "user_888"
        assert source.message_id == "msg_777"


def test_handle_request_gateway_restart_execution(monkeypatch):
    """Verify tool executes and dispatches restart to GatewayRunner."""
    monkeypatch.setenv("_HERMES_GATEWAY", "1")

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
        message_id="msg-1",
    )

    runner = MagicMock()
    runner._running = True
    runner._gateway_loop = MagicMock()

    from concurrent.futures import Future
    fut = Future()
    fut.set_result((True, "Restarting gateway..."))

    with patch("tools.gateway_restart_tool._is_authorized_foreground_turn", return_value=True):
        with patch("tools.gateway_restart_tool._resolve_current_source", return_value=source):
            with patch("gateway.run._gateway_runner_ref", return_value=runner):
                with patch("tools.gateway_restart_tool._check_restart_policy", return_value=None):
                    with patch("tools.gateway_restart_tool.safe_schedule_threadsafe", return_value=fut):
                        raw = _handle_request_gateway_restart({"reason": "test restart"})
                        res = json.loads(raw)
                        assert res["success"] is True
                        assert "Restarting" in res["message"]
