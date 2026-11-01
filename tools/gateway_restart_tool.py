"""Tool for requesting safe Gateway restart from natural-language user turns.

Gated by _HERMES_GATEWAY=1 env var and restricted exclusively to direct user-facing
foreground Gateway turns. Refuses invocation from cron jobs, delegated subagents,
Kanban workers, or CLI sessions. Reuses the same slash-access policy that /restart
uses, so a user who cannot run /restart cannot run this tool either.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from agent.async_utils import safe_schedule_threadsafe
from tools.registry import registry

logger = logging.getLogger(__name__)

_SCHEMA = {
    "type": "function",
    "function": {
        "name": "request_gateway_restart",
        "description": (
            "Request a safe restart of the running Hermes Gateway process when the user "
            "explicitly and unambiguously requests a restart via natural language "
            "(e.g., 'please restart the gateway', 'reboot yourself'). "
            "This tool initiates a graceful drain and restart handoff. "
            "Do not execute external shell commands to terminate the process."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Short explanation for the restart request.",
                },
            },
            "required": [],
        },
    },
}


def _is_authorized_foreground_turn() -> bool:
    """Return True if running inside a Gateway process for a direct user-facing turn.

    Refuses:
    1. Outside Gateway process (_HERMES_GATEWAY != "1").
    2. Delegated child subagent execution.
    3. Kanban workers (HERMES_KANBAN_TASK or any KANBAN_ENV_KEYS present in env).
    4. Scheduled cron jobs (HERMES_CRON_AUTO_DELIVER_PLATFORM ContextVar set).
    5. ContextVars missing platform or chat_id (e.g. background tasks or CLI).
    """
    if os.environ.get("_HERMES_GATEWAY") != "1":
        return False

    # Check delegated child subagent context
    try:
        from agent.delegation_context import is_delegated_child_process_context
        if is_delegated_child_process_context():
            return False
    except Exception:
        pass

    # Check Kanban worker environment markers
    try:
        from agent.delegation_context import KANBAN_ENV_KEYS
        for k in KANBAN_ENV_KEYS:
            if os.environ.get(k):
                return False
    except Exception:
        if os.environ.get("HERMES_KANBAN_TASK"):
            return False

    # Check session ContextVars
    try:
        from gateway.session_context import get_session_env
        if get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM"):
            return False
        platform = get_session_env("HERMES_SESSION_PLATFORM")
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID")
        if not platform or not chat_id:
            return False
    except Exception:
        return False

    return True


def _check_restart_policy(runner: Any, source: Any) -> Optional[str]:
    """Reuse the /restart slash-access policy."""
    if source is None or runner is None:
        return None
    try:
        from gateway.slash_access import policy_for_source as _policy_for_source
        policy = _policy_for_source(getattr(runner, "config", None), source)
        if not policy.enabled:
            return None
        user_id = getattr(source, "user_id", None) or ""
        if not policy.is_admin(user_id) and not policy.can_run(user_id, "restart"):
            return (
                "Not allowed to run /restart on this scope. "
                "Ask an admin to grant restart access."
            )
    except Exception:
        return None
    return None


def check_fn() -> bool:
    """Check function called by tool registry to determine tool visibility."""
    if not _is_authorized_foreground_turn():
        return False
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
        source = _resolve_current_source()
        if runner is not None and source is not None:
            denial = _check_restart_policy(runner, source)
            if denial is not None:
                return False
    except Exception:
        pass
    return True


def _resolve_current_source() -> Optional[Any]:
    """Reconstruct SessionSource from session ContextVars."""
    try:
        from gateway.config import Platform
        from gateway.session import SessionSource
        from gateway.session_context import get_session_env

        plat_str = get_session_env("HERMES_SESSION_PLATFORM")
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID")
        if not plat_str or not chat_id:
            return None

        try:
            platform = Platform(plat_str)
        except Exception:
            platform = None

        return SessionSource(
            platform=platform,
            chat_id=chat_id,
            chat_type=get_session_env("HERMES_SESSION_CHAT_TYPE") or "dm",
            user_id=get_session_env("HERMES_SESSION_USER_ID") or None,
            thread_id=get_session_env("HERMES_SESSION_THREAD_ID") or None,
            message_id=get_session_env("HERMES_SESSION_MESSAGE_ID") or None,
            scope_id=get_session_env("HERMES_SESSION_SCOPE_ID") or None,
        )
    except Exception as e:
        logger.debug("Failed to resolve current SessionSource from ContextVars: %s", e)
        return None


def _handle_request_gateway_restart(args: dict[str, Any]) -> str:
    """Handler executed in a ThreadPoolExecutor worker thread during agent turn."""
    if not _is_authorized_foreground_turn():
        return json.dumps(
            {
                "success": False,
                "error": "request_gateway_restart is restricted to direct user-facing Gateway turns.",
            },
            ensure_ascii=False,
        )

    from gateway.run import _gateway_runner_ref

    runner = _gateway_runner_ref()
    if runner is None or not getattr(runner, "_running", False):
        return json.dumps(
            {
                "success": False,
                "error": "Cannot reach running GatewayRunner instance.",
            },
            ensure_ascii=False,
        )

    gw_loop = getattr(runner, "_gateway_loop", None)
    if gw_loop is None:
        return json.dumps(
            {
                "success": False,
                "error": "Gateway event loop is not available.",
            },
            ensure_ascii=False,
        )

    source = _resolve_current_source()
    if source is None:
        return json.dumps(
            {
                "success": False,
                "error": "Could not resolve session source provenance.",
            },
            ensure_ascii=False,
        )

    denial = _check_restart_policy(runner, source)
    if denial is not None:
        return json.dumps(
            {"success": False, "error": denial},
            ensure_ascii=False,
        )

    reason = str(args.get("reason") or "Agent requested restart via natural language").strip()

    coro = runner.dispatch_gateway_restart(
        source=source,
        reason=reason,
        origin="agent_tool",
    )

    future = safe_schedule_threadsafe(coro, gw_loop)
    if future is None:
        return json.dumps(
            {
                "success": False,
                "error": "Failed to schedule restart request onto event loop.",
            },
            ensure_ascii=False,
        )

    try:
        success, message = future.result(timeout=10.0)
    except TimeoutError:
        return json.dumps(
            {
                "success": False,
                "outcome": "outcome_unknown",
                "message": "Restart outcome could not be confirmed. Do not retry immediately.",
                "can_retry": False,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps(
            {
                "success": False,
                "error": f"Gateway restart request failed: {exc}",
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "success": success,
            "message": message,
        },
        ensure_ascii=False,
    )


# Top-level registry scan contract
registry.register(
    toolset="gateway",
    name="request_gateway_restart",
    schema=_SCHEMA,
    handler=_handle_request_gateway_restart,
    check_fn=check_fn,
)
