"""Runtime inspection tools.

Read-only access to structured runtime trace events emitted by Hermes routing
and execution paths.  This is intended for debugging whether named agents,
runners, model/provider selection, and continuity behaved as expected.
"""

from __future__ import annotations

import json
from typing import Optional

from tools.registry import registry


def runtime_inspect(
    session_id: Optional[str] = None,
    limit: int = 20,
    agent_name: Optional[str] = None,
) -> str:
    """Return recent runtime trace events as JSON."""
    try:
        from agent.runtime_trace import read_runtime_events

        try:
            bounded_limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            bounded_limit = 20
        events = read_runtime_events(
            session_id=session_id,
            limit=bounded_limit,
            agent_name=agent_name,
        )
        return json.dumps(
            {
                "success": True,
                "session_id": session_id,
                "agent_name": agent_name,
                "count": len(events),
                "events": events,
                "hint": "Runtime trace is best-effort and redacts secret-like fields; CLI command argv and prompts are intentionally not logged.",
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


RUNTIME_INSPECT_SCHEMA = {
    "name": "runtime_inspect",
    "description": (
        "Inspect recent runtime trace events for debugging named-agent routing, "
        "runner selection, session continuity, and execution results. Read-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Optional Hermes session id to filter events by.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of recent events to return (default 20, max 100).",
                "default": 20,
            },
            "agent_name": {
                "type": "string",
                "description": "Optional named-agent filter, e.g. 'code-architect'.",
            },
        },
        "required": [],
    },
}


registry.register(
    name="runtime_inspect",
    toolset="debugging",
    schema=RUNTIME_INSPECT_SCHEMA,
    handler=lambda args, **kw: runtime_inspect(
        session_id=args.get("session_id"),
        limit=args.get("limit", 20),
        agent_name=args.get("agent_name"),
    ),
    description="Inspect recent runtime trace events for agent/model routing debugging",
    emoji="🧭",
)
