"""Honcho tools for user context retrieval.

Registers three complementary tools, ordered by capability:

  honcho_context   — dialectic Q&A (LLM-powered, direct answers)
  honcho_search        — semantic search (fast, no LLM, raw excerpts)
  honcho_profile       — peer card (fast, no LLM, structured facts)

Use honcho_context when you need Honcho to synthesize an answer.
Use honcho_search or honcho_profile when you want raw data to reason
over yourself.

The session key is injected at runtime by the agent loop via
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


# ── Availability check ──

def _check_honcho_available() -> bool:
    """Tool is only available when Honcho is active."""
    return _session_manager is not None and _session_key is not None


def _resolve_session_context(**kwargs):
    """Prefer the calling agent's session context over module-global fallback."""
    session_manager = kwargs.get("honcho_manager") or _session_manager
    session_key = kwargs.get("honcho_session_key") or _session_key
    return session_manager, session_key


# ── honcho_profile ──

_PROFILE_SCHEMA = {
    "name": "honcho_profile",
    "description": (
        "Retrieve the user's peer card from Honcho — a curated list of key facts "
        "about them (name, role, preferences, communication style, patterns). "
        "Fast, no LLM reasoning, minimal cost. "
        "Use this at conversation start or when you need a quick factual snapshot. "
        "Use honcho_context instead when you need Honcho to synthesize an answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def _handle_honcho_profile(args: dict, **kw) -> str:
    session_manager, session_key = _resolve_session_context(**kw)
    if not session_manager or not session_key:
        return json.dumps({"error": "Honcho is not active for this session."})
    try:
        card = session_manager.get_peer_card(session_key)
        if not card:
            return json.dumps({"result": "No profile facts available yet. The user's profile builds over time through conversations."})
        return json.dumps({"result": card})
    except Exception as e:
        logger.error("Error fetching Honcho peer card: %s", e)
        return json.dumps({"error": f"Failed to fetch profile: {e}"})


# ── honcho_search ──

_SEARCH_SCHEMA = {
    "name": "honcho_search",
    "description": (
        "Semantic search over Honcho's stored context about the user. "
        "Returns raw excerpts ranked by relevance to your query — no LLM synthesis. "
        "Cheaper and faster than honcho_context. "
        "Good when you want to find specific past facts and reason over them yourself. "
        "Use honcho_context when you need a direct synthesized answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in Honcho's memory (e.g. 'programming languages', 'past projects', 'timezone').",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Token budget for returned context (default 800, max 2000).",
            },
        },
        "required": ["query"],
    },
}


def _handle_honcho_search(args: dict, **kw) -> str:
    query = args.get("query", "")
    if not query:
        return json.dumps({"error": "Missing required parameter: query"})
    session_manager, session_key = _resolve_session_context(**kw)
    if not session_manager or not session_key:
        return json.dumps({"error": "Honcho is not active for this session."})
    max_tokens = min(int(args.get("max_tokens", 800)), 2000)
    try:
        result = session_manager.search_context(session_key, query, max_tokens=max_tokens)
        if not result:
            return json.dumps({"result": "No relevant context found."})
        return json.dumps({"result": result})
    except Exception as e:
        logger.error("Error searching Honcho context: %s", e)
        return json.dumps({"error": f"Failed to search context: {e}"})


# ── honcho_context (dialectic — LLM-powered) ──

_QUERY_SCHEMA = {
    "name": "honcho_context",
    "description": (
        "Ask Honcho a natural language question and get a synthesized answer. "
        "Uses Honcho's LLM (dialectic reasoning) — higher cost than honcho_profile or honcho_search. "
        "Can query about any peer: the user (default), the AI assistant, or any named peer. "
        "Examples: 'What are the user's main goals?', 'What has hermes been working on?', "
        "'What is the user's technical expertise level?'"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A natural language question.",
            },
            "peer": {
                "type": "string",
                "description": "Which peer to query about: 'user' (default) or 'ai'. Omit for user.",
            },
        },
        "required": ["query"],
    },
}


def _handle_honcho_context(args: dict, **kw) -> str:
    query = args.get("query", "")
    if not query:
        return json.dumps({"error": "Missing required parameter: query"})
    session_manager, session_key = _resolve_session_context(**kw)
    if not session_manager or not session_key:
        return json.dumps({"error": "Honcho is not active for this session."})
    peer_target = args.get("peer", "user")
    try:
        result = session_manager.dialectic_query(session_key, query, peer=peer_target)
        return json.dumps({"result": result or "No result from Honcho."})
    except Exception as e:
        logger.error("Error querying Honcho context: %s", e)
        return json.dumps({"error": f"Failed to query context: {e}"})


# ── honcho_conclude ──

_CONCLUDE_SCHEMA = {
    "name": "honcho_conclude",
    "description": (
        "Write a conclusion about the user back to Honcho's memory. "
        "Conclusions are persistent facts that build the user's profile — "
        "preferences, corrections, clarifications, project context, or anything "
        "the user tells you that should be remembered across sessions. "
        "Use this when the user explicitly states a preference, corrects you, "
        "or shares something they want remembered. "
        "Examples: 'User prefers dark mode', 'User's project uses Python 3.11', "
        "'User corrected: their name is spelled Eri not Eric'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {
                "type": "string",
                "description": "A factual statement about the user to persist in memory.",
            }
        },
        "required": ["conclusion"],
    },
}


def _handle_honcho_conclude(args: dict, **kw) -> str:
    conclusion = args.get("conclusion", "")
    if not conclusion:
        return json.dumps({"error": "Missing required parameter: conclusion"})
    session_manager, session_key = _resolve_session_context(**kw)
    if not session_manager or not session_key:
        return json.dumps({"error": "Honcho is not active for this session."})
    try:
        ok = session_manager.create_conclusion(session_key, conclusion)
        if ok:
            return json.dumps({"result": f"Conclusion saved: {conclusion}"})
        return json.dumps({"error": "Failed to save conclusion."})
    except Exception as e:
        logger.error("Error creating Honcho conclusion: %s", e)
        return json.dumps({"error": f"Failed to save conclusion: {e}"})


# ── Registration ──

from tools.registry import registry

registry.register(
    name="honcho_profile",
    toolset="honcho",
    schema=_PROFILE_SCHEMA,
    handler=_handle_honcho_profile,
    check_fn=_check_honcho_available,
    emoji="🔮",
)

registry.register(
    name="honcho_search",
    toolset="honcho",
    schema=_SEARCH_SCHEMA,
    handler=_handle_honcho_search,
    check_fn=_check_honcho_available,
    emoji="🔮",
)

registry.register(
    name="honcho_context",
    toolset="honcho",
    schema=_QUERY_SCHEMA,
    handler=_handle_honcho_context,
    check_fn=_check_honcho_available,
    emoji="🔮",
)

registry.register(
    name="honcho_conclude",
    toolset="honcho",
    schema=_CONCLUDE_SCHEMA,
    handler=_handle_honcho_conclude,
    check_fn=_check_honcho_available,
    emoji="🔮",
)
