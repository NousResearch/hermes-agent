"""
Nudge scheduling tool for agents.

Allows agents to schedule wake-up reminders for themselves that persist
across gateway restarts. Unlike cron jobs, nudges don't delegate work -
they simply wake the agent to continue processing with full context.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

SCHEDULE_NUDGE_SCHEMA = {
    "name": "schedule_nudge",
    "description": (
        "Schedule a wake-up reminder for yourself that persists across gateway restarts. "
        "When the scheduled time arrives, you will receive a synthetic message and can "
        "continue processing with your full context intact.\n\n"
        "Use this when you need to:\n"
        "- Wait for something to complete (e.g., infrastructure deployment)\n"
        "- Check on a long-running process later\n"
        "- Continue a workflow after a delay\n"
        "- Implement a 'loop' that survives container restarts\n\n"
        "Schedule formats:\n"
        "- One-time: '5m' (5 minutes), '2h' (2 hours), '1d' (1 day), or ISO datetime\n"
        "- Recurring: 'every 5m', 'every 30s', 'every 2h', 'every 1d'\n\n"
        "RECURRING NUDGES are recommended for robustness:\n"
        "- They automatically reschedule after firing\n"
        "- If container restarts, they'll fire again at the next interval\n"
        "- Cancel them with cancel_nudge() when done\n\n"
        "Unlike cron jobs, no sub-agent is spawned - you simply wake up and continue."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "schedule": {
                "type": "string",
                "description": (
                    "When to fire the nudge. One-time: '5m', '2h', '2025-06-02T14:30:00'. "
                    "Recurring (recommended): 'every 5m', 'every 30s', 'every 2h', 'every 1d'"
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional message/context to include when the nudge fires. "
                    "This helps you remember what you were waiting for."
                ),
            },
            "name": {
                "type": "string",
                "description": "Optional name for this nudge (for identification in listings)",
            },
        },
        "required": ["schedule"],
    },
}

LIST_NUDGES_SCHEMA = {
    "name": "list_nudges",
    "description": "List all scheduled nudges for the current session.",
    "parameters": {
        "type": "object",
        "properties": {
            "include_fired": {
                "type": "boolean",
                "description": "Include already-fired nudges in the list",
                "default": False,
            },
        },
    },
}

CANCEL_NUDGE_SCHEMA = {
    "name": "cancel_nudge",
    "description": "Cancel a scheduled nudge by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "nudge_id": {
                "type": "string",
                "description": "ID of the nudge to cancel",
            },
        },
        "required": ["nudge_id"],
    },
}


def _get_current_session_info() -> tuple[Optional[str], Optional[str]]:
    """Get current session ID and key from gateway context.

    Returns:
        Tuple of (session_id, session_key) or (None, None) if not available
    """
    # Try environment variables first (set by gateway)
    session_id = os.getenv("HERMES_SESSION_ID")
    session_key = os.getenv("HERMES_SESSION_KEY")

    if session_id and session_key:
        return session_id, session_key

    # Try gateway context vars (more reliable in gateway context)
    try:
        from gateway.session_context import get_session_env

        # Build session key from available info
        platform = get_session_env("HERMES_SESSION_PLATFORM", "")
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")

        if platform and chat_id:
            # Check if session_key was explicitly set
            stored_key = get_session_env("HERMES_SESSION_KEY", "")
            if stored_key:
                session_key = stored_key
            else:
                # Construct session key from platform and chat_id
                session_key = f"agent:main:{platform}:dm:{chat_id}"

            # session_id is not available from context vars by default
            # The nudge will still work but won't verify session_id
            return None, session_key
    except Exception:
        pass

    return None, None


def schedule_nudge(
    schedule: str,
    context: Optional[str] = None,
    name: Optional[str] = None,
    task_id: str = None,
) -> str:
    """Schedule a wake-up nudge for the current agent session.

    Nudges are lightweight reminders that wake up the agent at a scheduled time.
    Unlike cron jobs, nudges don't spawn sub-agents - the main agent simply
    wakes up and continues processing with its full context intact.

    Use this when you need to:
    - Wait for something to complete (e.g., infrastructure deployment)
    - Check on a long-running process later
    - Continue a workflow after a delay
    - Implement a "loop" that persists across container restarts

    IMPORTANT: For robustness against container restarts, use RECURRING nudges
    ("every 5m") instead of one-time nudges ("5m"). Recurring nudges automatically
    reschedule after firing, so if the container restarts while you're working,
    you'll still get woken up at the next interval.

    Args:
        schedule: When to fire the nudge. Supported formats:
            - One-time: "5m" (5 minutes), "2h" (2 hours), "1d" (1 day), ISO datetime
            - Recurring (recommended): "every 5m", "every 30s", "every 2h", "every 1d"
        context: Optional message/context to include when the nudge fires.
            This will be shown to the agent when it wakes up.
        name: Optional name for this nudge (for identification in listings)
        task_id: Task ID for handler compatibility (unused)

    Returns:
        JSON string with result information
    """
    del task_id

    try:
        from nudges import create_nudge

        session_id, session_key = _get_current_session_info()

        # Require at least session_key; session_id is optional
        # (but recommended for session-reset detection)
        if not session_key:
            return json.dumps({
                "success": False,
                "error": (
                    "Cannot determine current session. "
                    "Nudges can only be scheduled from within an active agent session "
                    "in a gateway context (not CLI or subagent)."
                ),
            }, indent=2)

        nudge = create_nudge(
            session_id=session_id or "",
            session_key=session_key,
            schedule=schedule,
            context=context,
            name=name,
        )

        if not nudge:
            return json.dumps({
                "success": False,
                "error": f"Failed to parse schedule: {schedule}. Use format like '5m', 'every 5m', '2h', '1d', or ISO datetime.",
            }, indent=2)

        return json.dumps({
            "success": True,
            "nudge_id": nudge["id"],
            "name": nudge["name"],
            "fire_at": nudge["fire_at"],
            "is_recurring": nudge.get("is_recurring", False),
            "context": nudge.get("context", ""),
            "message": f"Nudge '{nudge['name']}' scheduled for {nudge['fire_at']}",
        }, indent=2)

    except Exception as e:
        logger.exception("Failed to schedule nudge: %s", e)
        return json.dumps({
            "success": False,
            "error": f"Failed to schedule nudge: {e}",
        }, indent=2)


def list_nudges(
    include_fired: bool = False,
    task_id: str = None,
) -> str:
    """List nudges for the current session.

    Args:
        include_fired: Include already-fired nudges in the list
        task_id: Task ID for handler compatibility (unused)

    Returns:
        JSON string with list of nudges
    """
    del task_id

    try:
        from nudges import list_nudges as _list_nudges

        session_id, _ = _get_current_session_info()

        nudges = _list_nudges(
            session_id=session_id,
            include_fired=include_fired,
        )

        formatted = []
        for nudge in nudges:
            is_recurring = nudge.get("is_recurring", False)
            formatted.append({
                "nudge_id": nudge["id"],
                "name": nudge["name"],
                "schedule": nudge.get("schedule", ""),
                "is_recurring": is_recurring,
                "fire_at": nudge["fire_at"],
                "context": nudge.get("context", "")[:100] + "..." if len(nudge.get("context", "")) > 100 else nudge.get("context", ""),
                "fired": nudge.get("fired", False),
                "fire_count": nudge.get("fire_count", 0) if is_recurring else None,
            })

        return json.dumps({
            "success": True,
            "count": len(formatted),
            "nudges": formatted,
        }, indent=2)

    except Exception as e:
        logger.exception("Failed to list nudges: %s", e)
        return json.dumps({
            "success": False,
            "error": f"Failed to list nudges: {e}",
        }, indent=2)


def cancel_nudge(
    nudge_id: str,
    task_id: str = None,
) -> str:
    """Cancel a scheduled nudge.

    Args:
        nudge_id: ID of the nudge to cancel
        task_id: Task ID for handler compatibility (unused)

    Returns:
        JSON string with result
    """
    del task_id

    try:
        from nudges import delete_nudge, get_nudge

        session_id, _ = _get_current_session_info()
        nudge = get_nudge(nudge_id)

        if not nudge:
            return json.dumps({
                "success": False,
                "error": f"Nudge {nudge_id} not found",
            }, indent=2)

        if session_id and nudge.get("session_id") != session_id:
            return json.dumps({
                "success": False,
                "error": "Cannot cancel nudge: it belongs to a different session",
            }, indent=2)

        if delete_nudge(nudge_id):
            return json.dumps({
                "success": True,
                "message": f"Nudge {nudge_id} cancelled",
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": f"Failed to cancel nudge {nudge_id}",
            }, indent=2)

    except Exception as e:
        logger.exception("Failed to cancel nudge: %s", e)
        return json.dumps({
            "success": False,
            "error": f"Failed to cancel nudge: {e}",
        }, indent=2)


# =============================================================================
# Registry
# =============================================================================

from tools.registry import registry


def _check_nudge_requirements() -> bool:
    """Nudge tool is always available (no external dependencies)."""
    return True


def _handle_schedule_nudge(args: dict, **kwargs) -> str:
    """Handler for schedule_nudge tool."""
    return schedule_nudge(
        schedule=args.get("schedule", ""),
        context=args.get("context"),
        name=args.get("name"),
        task_id=kwargs.get("task_id"),
    )


def _handle_list_nudges(args: dict, **kwargs) -> str:
    """Handler for list_nudges tool."""
    return list_nudges(
        include_fired=args.get("include_fired", False),
        task_id=kwargs.get("task_id"),
    )


def _handle_cancel_nudge(args: dict, **kwargs) -> str:
    """Handler for cancel_nudge tool."""
    return cancel_nudge(
        nudge_id=args.get("nudge_id", ""),
        task_id=kwargs.get("task_id"),
    )


registry.register(
    name="schedule_nudge",
    toolset="nudge",
    schema=SCHEDULE_NUDGE_SCHEMA,
    handler=_handle_schedule_nudge,
    check_fn=_check_nudge_requirements,
    emoji="⏰",
)

registry.register(
    name="list_nudges",
    toolset="nudge",
    schema=LIST_NUDGES_SCHEMA,
    handler=_handle_list_nudges,
    check_fn=_check_nudge_requirements,
    emoji="📋",
)

registry.register(
    name="cancel_nudge",
    toolset="nudge",
    schema=CANCEL_NUDGE_SCHEMA,
    handler=_handle_cancel_nudge,
    check_fn=_check_nudge_requirements,
    emoji="❌",
)
