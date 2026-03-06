"""Honcho tool for querying user context via dialectic reasoning.

Registers ``query_user_context`` -- an LLM-callable tool that asks Honcho
about the current user's history, preferences, goals, and communication
style. The session key is injected at runtime by the agent loop via
``set_session_context()``.
"""

import json
import logging

logger = logging.getLogger(__name__)

# ── Module-level state (injected by AIAgent at init time) ──

_session_manager = None  # HonchoSessionManager instance
_session_key: str | None = None  # Current session key (e.g., "telegram:123456")


def set_session_context(session_manager, session_key: str) -> None:
    """Register the active Honcho session manager and key.

    Called by AIAgent.__init__ when Honcho is enabled.
    """
    global _session_manager, _session_key
    _session_manager = session_manager
    _session_key = session_key


def clear_session_context() -> None:
    """Clear session context (for testing or shutdown)."""
    global _session_manager, _session_key
    _session_manager = None
    _session_key = None


# ── Tool schema ──

HONCHO_TOOL_SCHEMA = {
    "name": "query_user_context",
    "description": (
        "Query Honcho to retrieve relevant context about the user based on their "
        "history and preferences. Use this when you need to understand the user's "
        "background, preferences, past interactions, or goals. This helps you "
        "personalize your responses and provide more relevant assistance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A natural language question about the user. Examples: "
                    "'What are this user's main goals?', "
                    "'What communication style does this user prefer?', "
                    "'What topics has this user discussed recently?', "
                    "'What is this user's technical expertise level?'"
                ),
            }
        },
        "required": ["query"],
    },
}


# ── Tool handler ──

def _handle_query_user_context(args: dict, **kw) -> str:
    """Execute the Honcho context query."""
    query = args.get("query", "")
    if not query:
        return json.dumps({"error": "Missing required parameter: query"})

    if not _session_manager or not _session_key:
        return json.dumps({"error": "Honcho is not active for this session."})

    try:
        result = _session_manager.get_user_context(_session_key, query)
        return json.dumps({"result": result})
    except Exception as e:
        logger.error("Error querying Honcho user context: %s", e)
        return json.dumps({"error": f"Failed to query user context: {e}"})


# ── Availability check ──

def _check_honcho_available() -> bool:
    """Tool is only available when Honcho is active."""
    return _session_manager is not None and _session_key is not None


# ── Registration ──

from tools.registry import registry

registry.register(
    name="query_user_context",
    toolset="honcho",
    schema=HONCHO_TOOL_SCHEMA,
    handler=_handle_query_user_context,
    check_fn=_check_honcho_available,
)
