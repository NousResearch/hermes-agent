#!/usr/bin/env python3
"""One-shot self-nudge tool for gateway sessions.

Lets the agent arm a single in-memory timer that will later inject a hidden
continuation turn back into the same session. This is lighter and safer than
creating a cron job for short-lived follow-up work.
"""

import json

from tools.registry import registry


SELF_NUDGE_SCHEMA = {
    "name": "self_nudge",
    "description": (
        "Arm a one-time self-nudge timer for the current session. When the "
        "timer fires, Hermes injects a hidden follow-up turn back into this "
        "same session. Use this instead of cron for short-lived reminders or "
        "follow-up checks. Only one self-nudge is kept per session; arming a "
        "new one replaces the previous timer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "delay_seconds": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "How many seconds to wait before firing the one-time "
                    "hidden follow-up turn."
                ),
            },
            "note": {
                "type": "string",
                "description": (
                    "Optional private reminder to inject into the hidden turn. "
                    "Example: 'Check whether the deploy finished and report "
                    "back if it failed.'"
                ),
            },
        },
        "required": ["delay_seconds"],
    },
}


def check_self_nudge_requirements() -> bool:
    """Tool availability is filtered by platform in model_tools."""
    return True


def self_nudge_tool(args, **kwargs):
    """Stub dispatcher for the registry.

    Real handling happens in run_agent.py because it needs the live session's
    gateway callback.
    """
    return json.dumps(
        {"error": "self_nudge must be handled by the agent loop"},
        ensure_ascii=False,
    )


registry.register(
    name="self_nudge",
    toolset="user_updates",
    schema=SELF_NUDGE_SCHEMA,
    handler=self_nudge_tool,
    check_fn=check_self_nudge_requirements,
    description="Arm a one-shot hidden self-reminder for the current gateway session.",
    emoji="⏰",
)
