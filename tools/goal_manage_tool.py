"""
Goal management tool for Hermes Agent.

Exposes a ``goal_manage`` tool so the agent itself can signal completion
of a standing goal, pause/resume, or check status — analogous to how the
``cronjob`` tool gives the agent lifecycle control over scheduled tasks.

When the agent calls ``goal_manage(action="complete")``, the standing goal
is closed immediately and the judge retry loop stops, avoiding the problem
where the agent says "done" but the judge sees "no progress" and retries up
to ``goals.max_turns`` times (issue #31718).
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level GoalManager injection
# ---------------------------------------------------------------------------
# The CLI and gateway each hold one GoalManager per live session.  Setting
# ``_goal_manager`` on init lets the tool handler operate without needing
# direct access to session internals from the tool registry dispatch path.
# This mirrors how the cronjob tool imports cron modules directly.
_goal_manager: Optional["GoalManager"] = None  # noqa: F821


def set_goal_manager(mgr: "GoalManager") -> None:  # noqa: F821
    """Set the active GoalManager for this session.

    Called by HermesCLI and GatewayRunner during session init so the
    goal_manage tool dispatches against the correct session state.
    """
    global _goal_manager
    _goal_manager = mgr


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

GOAL_MANAGE_SCHEMA = {
    "name": "goal_manage",
    "description": (
        "Manage standing goals.  Call ``complete`` when you have finished "
        "all the work for your current standing goal — this signals the "
        "system to close the goal and stop the judge retry loop.  You can "
        "also pause, resume, clear, or check status."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["complete", "pause", "resume", "clear", "status"],
                "description": (
                    "``complete`` — mark the current standing goal as done "
                    "and exit the retry loop.  ``pause`` — suspend the goal "
                    "without clearing it.  ``resume`` — reactivate a paused "
                    "goal.  ``clear`` — remove the goal entirely.  "
                    "``status`` — return a one-liner about the current goal."
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    "Optional explanation for the action (e.g. \"All "
                    "milestones met\").  Stored in goal history."
                ),
            },
        },
        "required": ["action"],
    },
}


def _tool_error(msg: str, *, success: bool = False) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False) if not success else json.dumps({"result": msg}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def goal_manage(
    action: str,
    reason: Optional[str] = None,
    task_id: str = None,
) -> str:
    """Handle goal_manage tool calls."""
    del task_id  # unused — kept for handler signature compatibility

    mgr = _goal_manager
    if mgr is None:
        return _tool_error(
            "GoalManager is not available in this session.  "
            "Standing goals are only supported in CLI and gateway sessions."
        )

    action = (action or "").strip().lower()

    if action == "complete":
        if not mgr.has_goal():
            return _tool_error("No active standing goal to complete.")
        mgr.mark_done(reason or "completed by agent")
        return json.dumps({
            "result": f"Goal completed: {reason or 'completed by agent'}"
        }, ensure_ascii=False)

    elif action == "pause":
        if not mgr.has_goal():
            return _tool_error("No active standing goal to pause.")
        mgr.pause(reason or "paused by agent")
        return json.dumps({
            "result": f"Goal paused: {reason or 'paused by agent'}"
        }, ensure_ascii=False)

    elif action == "resume":
        if not mgr.is_active():
            return _tool_error("No paused goal to resume.")
        mgr.resume()
        return json.dumps({"result": "Goal resumed."}, ensure_ascii=False)

    elif action == "clear":
        mgr.clear()
        return json.dumps({"result": "Goal cleared."}, ensure_ascii=False)

    elif action == "status":
        return json.dumps({"result": mgr.status_line()}, ensure_ascii=False)

    else:
        return _tool_error(f"Unknown action: {action!r}.  Valid actions: complete, pause, resume, clear, status.")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry  # noqa: E402

registry.register(
    name="goal_manage",
    toolset="hermes-cli",
    schema=GOAL_MANAGE_SCHEMA,
    handler=lambda args, **kw: goal_manage(
        action=args.get("action", ""),
        reason=args.get("reason"),
        task_id=kw.get("task_id"),
    ),
    description="Manage standing goals: complete, pause, resume, clear, or check status.",
    emoji="🎯",
)
