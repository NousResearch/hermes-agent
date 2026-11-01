"""Best-effort chat continuation after a detached MCP sign-in finishes."""

import contextvars
import logging
import sys
from threading import Thread
from typing import Any, Callable

from gateway.session_context import async_delivery_supported, get_session_env, session_is_messaging_surface

logger = logging.getLogger(__name__)
_WAIT_SECONDS = 900
SIGN_IN_WAKE_TEXT = (
    "[Hermes gateway notice, not typed by the user: the MCP server '{name}' finished signing in "
    "and its tools are connected now. If the user's earlier request in this conversation needed "
    "it and is not finished yet, continue it without asking them to confirm, and do not repeat "
    "anything already done. Otherwise tell them briefly that {name} is connected.]"
)


def _runner_live(runner) -> bool:
    loop = getattr(runner, "_gateway_loop", None)
    return bool(getattr(runner, "_running", False) and loop is not None and loop.is_running())


def _wait_and_wake(runner, route, name, attempt, still_pending) -> None:
    try:
        if not attempt.flow.wait_worker_done(_WAIT_SECONDS):
            return
        snapshot = attempt.poll()
        if snapshot["status"] != "approved" or snapshot.get("discovery_error"):
            return
        if not _runner_live(runner) or not still_pending(name, attempt):
            return
        from agent.async_utils import safe_schedule_threadsafe
        from gateway.wake import wake_route

        contextvars.Context().run(
            safe_schedule_threadsafe,
            wake_route(runner, route, SIGN_IN_WAKE_TEXT.format(name=name),
                       lambda: still_pending(name, attempt), producer="mcp-sign-in-wake"),
            runner._gateway_loop, logger=logger, log_message="MCP sign-in wake scheduling failed",
        )
    except Exception:
        logger.warning("MCP sign-in wake worker failed", exc_info=True)


def arm_sign_in_wakes(attempts: dict[str, Any], still_pending: Callable[[str, Any], bool]) -> bool:
    armed = False
    try:
        if not async_delivery_supported() or not session_is_messaging_surface() or get_session_env("HERMES_CRON_SESSION"):
            return False
        runner_ref = getattr(sys.modules.get("gateway.run"), "_gateway_runner_ref", None)
        runner = runner_ref() if callable(runner_ref) else None
        if not _runner_live(runner):
            return False
        route = {field: get_session_env("HERMES_SESSION_" + field.upper()) for field in (
            "platform", "chat_id", "chat_type", "thread_id", "user_id", "user_name", "scope_id")}
        route.update(session_key=get_session_env("HERMES_SESSION_KEY"),
                     parent_session_id=get_session_env("HERMES_SESSION_ID"))
        if not route["session_key"]:
            return False
        for name, attempt in attempts.items():
            if attempt.auth_url and attempt.poll()["status"] == "pending":
                Thread(target=_wait_and_wake, args=(runner, route, name, attempt, still_pending),
                       daemon=True, name="mcp-sign-in-wake").start()
                armed = True
    except Exception:
        logger.warning("MCP sign-in wake could not be armed", exc_info=True)
    return armed
