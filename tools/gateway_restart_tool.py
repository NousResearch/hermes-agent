"""Tool for requesting safe Gateway restart from natural-language user turns.

Gated by _HERMES_GATEWAY=1 env var and restricted exclusively to direct user-facing
foreground Gateway turns. Refuses invocation from cron jobs, delegated subagents,
Kanban workers, or CLI sessions. Reuses the same slash-access policy that /restart
uses, so a user who can't run /restart can't run this tool either.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from agent.async_utils import safe_schedule_threadsafe

logger = logging.getLogger(__name__)

_SCHEMA = {
    "type": "function",
    "function": {
        "name": "request_gateway_restart",
        "description": (
            "Request a safe restart of the running Hermes Gateway process when the user "
            "explicitly and unambiguously requests a restart via natural language "
            "(e.g., 'please restart the gateway', 'reboot yourself'). "
            "This tool initiates an out-of-band detached handoff and drains current turns. "
            "DO NOT execute shell-level commands like 'schtasks', 'taskkill', or 'hermes gateway restart'."
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
    2. Delegated subagent execution (is_delegated_child_process_context() is True).
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
    """Reuse the /restart slash-access policy.

    Returns a denial message string if the user is not allowed to run the
    ``restart`` slash command on this scope, else None. When the operator has
    not configured ``allow_admin_from`` / ``user_allowed_commands`` for the
    scope, ``policy.enabled`` is False and the request is allowed (matches
    the existing /restart backward-compat semantics).
    """
    if source is None or runner is None:
        return None
    try:
        from gateway.slash_access import policy_for_source as _policy_for_source
    except Exception:
        return None
    try:
        policy = _policy_for_source(getattr(runner, "config", None), source)
    except Exception:
        return None
    if not policy.enabled:
        return None
    user_id = getattr(source, "user_id", None) or ""
    if not policy.is_admin(user_id) and not policy.can_run(user_id, "restart"):
        return (
            "Not allowed to run /restart on this scope. "
            "Ask an admin to grant restart access."
        )
    return None


def check_fn() -> bool:
    """Check function called by tool registry to determine tool visibility."""
    if not _is_authorized_foreground_turn():
        return False
    # If a runner is alive, also re-verify the slash-access policy so the model
    # never sees the tool when the user is unauthorized to run /restart.
    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
        source = _resolve_current_source()
        if runner is not None and source is not None:
            denial = _check_restart_policy(runner, source)
            if denial is not None:
                return False
    except Exception:
        # If policy lookup fails for any reason, do not silently hide the tool;
        # the handler performs a second, more authoritative check.
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


def _handle_request_gateway_restart(args: Dict[str, Any]) -> str:
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
    if gw_loop is None or not gw_loop.is_running():
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

    # Authoritative re-check of the /restart slash-access policy using the
    # resolved source. The check_fn already ran but we re-verify inside the
    # handler so a window-shift (e.g. config reload) between visibility and
    # invocation can't be exploited.
    denial = _check_restart_policy(runner, source)
    if denial is not None:
        return json.dumps(
            {"success": False, "error": denial},
            ensure_ascii=False,
        )

    reason = str(args.get("reason") or "Agent requested restart via natural language").strip()

    # Schedule dispatch_gateway_restart onto the Gateway main event loop
    coro = runner.dispatch_gateway_restart(
        source=source,
        reason=reason,
        origin="agent_tool",
    )

    try:
        future = safe_schedule_threadsafe(coro, gw_loop)
    except Exception as exc:
        return json.dumps(
            {
                "success": False,
                "error": f"Failed to schedule restart request: {exc}",
            },
            ensure_ascii=False,
        )

    # Outer worker timeout: 10.0s (longer than inner 8.0s coordinator timeout).
    # Per the restart-transaction contract: when the outer worker cannot confirm
    # the restart outcome within the 10s budget, we MUST return outcome_unknown
    # — never a "Gateway remains active" / "can_retry=True" — because by then the
    # transaction may have committed (and the gateway may already be shutting
    # down) without the worker seeing the result. We do NOT inspect shared
    # runner state to make a finer-grained judgment; the runner-level
    # _restart_transaction field can race with other concurrent requests, and
    # future.cancel() is not authoritative proof of abort.
    try:
        success, message = future.result(timeout=10.0)
    except TimeoutError:
        # Best-effort: do not block the worker further. The handoff (if any)
        # continues independently of this worker's return value. We do not
        # call future.cancel() because that races with the actual coordinator
        # and may swallow a committed ack.
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


def register(registry: Any) -> None:
    """Register the tool with the global tool registry."""
    registry.register(
        toolset="gateway",
        name="request_gateway_restart",
        schema=_SCHEMA,
        handler=_handle_request_gateway_restart,
        check_fn=check_fn,
    )
