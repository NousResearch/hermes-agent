"""schedule_wake - arm a one-shot self-wake deadline for this session (#122444).

Inspired by ChatGPT Work's dots, which "decide when to pause and wake up to continue work"
instead of needing a fixed schedule for every follow-up. The orchestrator's own alarm: at
``fires_at`` the session's owning driver (classic-CLI idle hook, TUI/Desktop session-owner
poller, or the messaging gateway's wake watcher for a wake armed inside a chat) injects
``prompt`` as a plain user turn, so a long delegation/gating session re-enters its loop
without human polling. One-shot only (recurring wake-ups are ``/heartbeat``'s job); refused
for subagent sessions (a child's session has no idle loop of its own) and hidden from
surfaces that cannot fire it (cron and one-shot ``-q`` runs end with the turn, the API server
hands the next turn to the client) - a wake armed there would be armed-but-dead, the exact
failure class #122444 reports.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Optional

from tools.registry import registry, tool_error

# Same floor as heartbeat: re-entering more often than once a minute is a busy-loop,
# not a wake. The issue's alarm ladders sleep ~15 min; nothing legitimate is faster.
MIN_WAKE_DELAY_SECONDS = 60

# Platforms whose turns nobody re-enters after they end: a wake armed there never fires.
_NO_WAKE_DRIVER_PLATFORMS = frozenset({"api_server", "kanban", "webhook", "msgraph_webhook"})


def check_schedule_wake_requirements() -> bool:
    """Offer the tool only where a driver will fire the wake: not in cron runs (the session
    ends with the job), not in Kanban workers / one-shot API turns (the client owns the next
    turn, see ``async_delivery_supported``), and not under a stateless channel."""
    if os.environ.get("HERMES_KANBAN_TASK"):
        return False
    try:
        from gateway.session_context import async_delivery_supported, get_session_env

        if not async_delivery_supported():
            return False
        if get_session_env("HERMES_CRON_SESSION", ""):
            return False
        platform = str(get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower()
        return platform not in _NO_WAKE_DRIVER_PLATFORMS
    except ImportError:
        return True


def _resolve_fires_at(args: dict) -> float:
    delay = args.get("delay_secs")
    at_iso = (args.get("at_iso") or "").strip()
    if delay is None and not at_iso:
        raise ValueError("either delay_secs or at_iso is required")
    if delay is not None and at_iso:
        raise ValueError("pass only one of delay_secs or at_iso")
    if delay is not None:
        return time.time() + float(delay)
    fires_at = datetime.fromisoformat(at_iso).timestamp()
    if fires_at < time.time():
        raise ValueError("at_iso is in the past")
    return fires_at


def _gateway_route() -> Dict[str, str]:
    """The messaging chat this turn runs in (empty for CLI/TUI/Desktop), captured at arm time so
    the gateway's wake watcher can re-enter the same chat after a restart - same shape as the
    ``/loop`` route (``gateway/slash_commands_goals.py``)."""
    try:
        from gateway.session_context import get_session_env, session_is_messaging_surface

        if not session_is_messaging_surface():
            return {}
        route = {
            "platform": get_session_env("HERMES_SESSION_PLATFORM", ""),
            "chat_id": get_session_env("HERMES_SESSION_CHAT_ID", ""),
            "chat_type": get_session_env("HERMES_SESSION_CHAT_TYPE", ""),
            "thread_id": get_session_env("HERMES_SESSION_THREAD_ID", ""),
            "user_id": get_session_env("HERMES_SESSION_USER_ID", ""),
            "user_name": get_session_env("HERMES_SESSION_USER_NAME", ""),
            "profile": get_session_env("HERMES_SESSION_PROFILE", ""),
        }
        return {k: str(v) for k, v in route.items() if v}
    except ImportError:
        return {}


def schedule_wake_tool(args: dict, session_id: Optional[str] = None) -> str:
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return tool_error("schedule_wake needs a non-empty prompt: the message injected when the wake fires.")
    if str(args.get("recurring", False)).lower() in {"1", "true", "yes"}:
        return tool_error("schedule_wake is one-shot; for recurring wake-ups use /heartbeat.")
    from agent.delegation_context import is_delegated_child_context

    if is_delegated_child_context():
        return tool_error(
            "schedule_wake arms the CALLING session's idle loop; a subagent session has no loop "
            "to fire it. The orchestrating session must arm its own wake."
        )
    try:
        fires_at = _resolve_fires_at(args)
    except (ValueError, TypeError, OverflowError) as exc:
        return tool_error(f"schedule_wake: {exc}")
    if fires_at - time.time() < MIN_WAKE_DELAY_SECONDS - 1:  # 1s slack for clock reads
        return tool_error(
            f"schedule_wake: delay must be at least {MIN_WAKE_DELAY_SECONDS}s "
            "(faster re-entry than once a minute is a busy-loop, not a wake)."
        )
    sid = session_id or os.environ.get("HERMES_SESSION_ID", "")
    if not sid:
        return tool_error("schedule_wake: no session id in this context; run from a session.")
    from hermes_cli.wake import WakeBudgetExhausted, schedule_wake

    try:
        state = schedule_wake(sid, prompt, fires_at, route=_gateway_route())
    except WakeBudgetExhausted as exc:
        return tool_error(f"schedule_wake refused: {exc}")
    except ValueError as exc:
        return tool_error(f"schedule_wake failed: {exc}")
    return json.dumps(
        {"success": True, "session_id": sid, "fires_at": fires_at, "prompt": prompt,
         "fires_so_far": state.fire_count},
        ensure_ascii=False,
    )


SCHEDULE_WAKE_SCHEMA = {
    "name": "schedule_wake",
    "description": (
        "Arm a ONE-SHOT self-wake for this session: at the deadline the prompt below is "
        "injected as a user message and the session's loop re-enters with no human input - "
        "use it to keep a long orchestration (delegation batches, gated pipelines, waiting on "
        "an external process) alive instead of sleeping forever when nothing else will wake you. "
        "Requires delay_secs (seconds from now) or at_iso (ISO timestamp), both >= 60s out; "
        "exactly one wake per session (latest wins); fires once and is consumed, so re-arm it "
        "from the wake turn while work is still outstanding (a per-session fire budget caps "
        "runaway re-arming). For recurring wake-ups use /heartbeat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Message injected as a user turn when the wake fires.",
            },
            "delay_secs": {
                "type": "number",
                "description": "Seconds from now until the wake fires (min 60).",
            },
            "at_iso": {
                "type": "string",
                "description": "ISO timestamp for the wake instead of delay_secs (future only).",
            },
            "recurring": {
                "type": "boolean",
                "description": "Must be false/omitted; schedule_wake is one-shot (use /heartbeat for recurring).",
            },
        },
        "required": ["prompt"],
    },
}


registry.register(
    name="schedule_wake",
    toolset="wake",
    schema=SCHEDULE_WAKE_SCHEMA,
    handler=lambda args, **kw: schedule_wake_tool(args, session_id=kw.get("session_id") or None),
    check_fn=check_schedule_wake_requirements,
)
