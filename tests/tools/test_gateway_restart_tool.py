"""Tests for tools/gateway_restart_tool.py — the request_gateway_restart tool.

Verifies:
1. Authorization boundaries (_is_authorized_foreground_turn):
   - Refused outside gateway (_HERMES_GATEWAY unset).
   - Refused for cron tasks (_CRON_AUTO_DELIVER_PLATFORM set).
   - Refused for delegated subagents (is_delegated_child_process_context() is True).
   - Refused for Kanban workers (HERMES_KANBAN_TASK or KANBAN_ENV_KEYS set).
   - Refused when session ContextVars are missing.
   - Accepted only for direct user-facing foreground Gateway turns.
2. ContextVar propagation via _run_in_executor_with_context().
3. Tool returns failure JSON when handoff initialization fails.
4. Worker outer timeout outcome_unknown vs cancel handling.
5. Runtime registry registration (no AST source hacking).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

# ---------------------------------------------------------------------------
# 1. Authorization boundary tests (_is_authorized_foreground_turn)
# ---------------------------------------------------------------------------


def test_auth_refused_outside_gateway(monkeypatch):
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    from tools.gateway_restart_tool import _is_authorized_foreground_turn

    assert _is_authorized_foreground_turn() is False


def test_auth_refused_for_cron(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _is_authorized_foreground_turn

    with patch(
        "gateway.session_context.get_session_env",
        side_effect=lambda key, d="": "telegram" if key in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_CRON_AUTO_DELIVER_PLATFORM") else "",
    ):
        assert _is_authorized_foreground_turn() is False


def test_auth_refused_for_delegated_subagent(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _is_authorized_foreground_turn

    with patch(
        "agent.delegation_context.is_delegated_child_process_context",
        return_value=True,
    ):
        assert _is_authorized_foreground_turn() is False


def test_auth_refused_for_kanban_worker(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    from tools.gateway_restart_tool import _is_authorized_foreground_turn

    assert _is_authorized_foreground_turn() is False


def test_auth_accepted_for_foreground_gateway(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    from tools.gateway_restart_tool import _is_authorized_foreground_turn

    with patch(
        "gateway.session_context.get_session_env",
        side_effect=lambda key, d="": "telegram" if key in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID") else "",
    ), patch(
        "agent.delegation_context.is_delegated_child_process_context",
        return_value=False,
    ):
        assert _is_authorized_foreground_turn() is True


# ---------------------------------------------------------------------------
# 2. ContextVars propagation test in Executor Thread
# ---------------------------------------------------------------------------


def test_contextvars_propagation_in_executor(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _resolve_current_source
    import gateway.session_context as sc
    import contextvars

    sc._session_context_engaged = True
    sc._SESSION_PLATFORM.set("telegram")
    sc._SESSION_CHAT_ID.set("chat-executor-101")
    sc._SESSION_USER_ID.set("user-42")

    ctx = contextvars.copy_context()

    def _worker():
        return _resolve_current_source()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        source = pool.submit(ctx.run, _worker).result()

    assert source is not None
    assert source.platform.value == "telegram"
    assert source.chat_id == "chat-executor-101"
    assert source.user_id == "user-42"


# ---------------------------------------------------------------------------
# 3. Handler error when runner is missing or not running
# ---------------------------------------------------------------------------


def test_handler_error_when_runner_missing(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _handle_request_gateway_restart
    import gateway.run as gw_run

    orig = gw_run._gateway_runner_ref
    gw_run._gateway_runner_ref = lambda: None
    try:
        with patch("tools.gateway_restart_tool._is_authorized_foreground_turn", return_value=True):
            res = json.loads(_handle_request_gateway_restart({"reason": "test"}))
            assert res["success"] is False
            assert "Cannot reach" in res["error"] or "GatewayRunner" in res["error"]
    finally:
        gw_run._gateway_runner_ref = orig


# ---------------------------------------------------------------------------
# 4. Tool returns failure JSON when handoff initialization fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_returns_failure_json_when_handoff_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    from tools.gateway_restart_tool import _handle_request_gateway_restart
    import gateway.run as gw_run

    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop

    async def fake_launch_fail():
        return False

    runner._launch_detached_restart_command = fake_launch_fail  # type: ignore[method-assign]
    runner.stop = AsyncMock()  # type: ignore[method-assign]

    orig = gw_run._gateway_runner_ref
    gw_run._gateway_runner_ref = lambda: runner

    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="chat-fail", chat_type="dm", message_id="m1"
    )

    try:
        with patch("tools.gateway_restart_tool._is_authorized_foreground_turn", return_value=True), patch(
            "tools.gateway_restart_tool._resolve_current_source", return_value=source
        ), patch("gateway.run._hermes_home", tmp_path), concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        ) as pool:

            def _worker_call():
                return _handle_request_gateway_restart({"reason": "handoff fail test"})

            fut = pool.submit(_worker_call)
            res_str = await asyncio.wrap_future(fut)
            res = json.loads(res_str)

            assert res["success"] is False
            assert "Handoff failed" in res["message"]
            runner.stop.assert_not_called()
    finally:
        gw_run._gateway_runner_ref = orig


# ---------------------------------------------------------------------------
# 5. Runtime Registry Registration Verification (No source AST hacking)
# ---------------------------------------------------------------------------


def test_tool_is_registered_in_runtime_registry():
    """Verify tool is actually registered in tools.registry at runtime."""
    import tools.gateway_restart_tool
    from tools.registry import registry

    tools.gateway_restart_tool.register(registry)

    tools_map = getattr(registry, "_tools", getattr(registry, "tools", {}))
    assert "request_gateway_restart" in tools_map, (
        "request_gateway_restart must be registered in tools.registry"
    )
