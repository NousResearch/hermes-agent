"""Audit query tool — lets the agent inspect its own audit trail.

Registered as an agent-loop tool (like todo, memory, session_search).
The agent can call this to answer user questions about recent activity,
errors, performance, and security events.
"""

import json
from typing import Any


def audit_query(
    action: str = "summary",
    hours: float = 24,
    event_type: str = None,
    search: str = None,
    limit: int = 20,
) -> str:
    """Query the audit log and return formatted results.

    Args:
        action: "summary", "problems", "list", "types"
        hours: Time window in hours (default: 24)
        event_type: Filter by event type (tool_call, api_error, etc.)
        search: Full-text search query
        limit: Max events to return (default: 20)

    Returns:
        JSON string with results
    """
    try:
        from agent.audit import get_audit_logger
        audit = get_audit_logger()
    except Exception as e:
        return json.dumps({"error": f"Audit system unavailable: {e}"})

    if action == "summary":
        result = audit.summary(last_hours=hours)
        if not result:
            return json.dumps({"message": "No audit events recorded."})
        return json.dumps(result, default=str)

    elif action == "problems":
        problems = audit.detect_problems(last_hours=hours)
        if not problems:
            return json.dumps({"message": f"No problems detected in the last {hours} hours."})
        return json.dumps({"problems": problems}, default=str)

    elif action == "types":
        event_types = audit.get_event_types()
        if not event_types:
            return json.dumps({"message": "No audit events recorded."})
        return json.dumps({"types": [{"type": t, "count": c} for t, c in event_types]})

    elif action == "list":
        kwargs = {"last_hours": hours, "limit": limit}
        if event_type:
            kwargs["event_type"] = event_type
        if search:
            events = audit.search(search, limit=limit)
        else:
            events = audit.query(**kwargs)
        if not events:
            return json.dumps({"message": "No matching events."})
        # Trim large fields for readability
        for ev in events:
            if ev.get("tool_args") and len(str(ev["tool_args"])) > 200:
                ev["tool_args"] = str(ev["tool_args"])[:200] + "..."
            if ev.get("tool_result_preview") and len(str(ev["tool_result_preview"])) > 200:
                ev["tool_result_preview"] = str(ev["tool_result_preview"])[:200] + "..."
            if ev.get("context") and len(str(ev["context"])) > 200:
                ev["context"] = str(ev["context"])[:200] + "..."
        return json.dumps({"events": events, "count": len(events)}, default=str)

    else:
        return json.dumps({"error": f"Unknown action: {action}. Use: summary, problems, list, types"})


# Tool definition for the agent
AUDIT_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "audit_query",
        "description": (
            "Query the audit log to inspect recent agent activity, errors, performance, "
            "and security events. Use this when the user asks about errors, what happened, "
            "system health, or recent activity."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["summary", "problems", "list", "types"],
                    "description": "summary: overview stats. problems: detected issues. list: raw events. types: available event types.",
                },
                "hours": {
                    "type": "number",
                    "description": "Time window in hours (default: 24)",
                },
                "event_type": {
                    "type": "string",
                    "description": "Filter by type: tool_call, tool_error, api_call, api_error, session_start, session_end, auth_refresh, approval_result, etc.",
                },
                "search": {
                    "type": "string",
                    "description": "Full-text search across errors, tools, context",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max events to return (default: 20)",
                },
            },
        },
    },
}
