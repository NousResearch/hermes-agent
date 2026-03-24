"""RetainDB tools for Hermes' native deep-memory integration."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import date, datetime

logger = logging.getLogger(__name__)

_session_manager = None
_session_key: str | None = None


def set_session_context(session_manager, session_key: str) -> None:
    global _session_manager, _session_key
    _session_manager = session_manager
    _session_key = session_key


def clear_session_context() -> None:
    global _session_manager, _session_key
    _session_manager = None
    _session_key = None


def _check_retaindb_available() -> bool:
    return _session_manager is not None and _session_key is not None


def _resolve_session_context(**kwargs):
    session_manager = kwargs.get("retaindb_manager") or _session_manager
    session_key = kwargs.get("retaindb_session_key") or _session_key
    return session_manager, session_key


def _json_default(value):
    """Serialize a few common Python/runtime types used by RetainDB helpers."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_response(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=_json_default)


_PROFILE_SCHEMA = {
    "name": "retaindb_profile",
    "description": (
        "Retrieve a compact stable profile for this user from RetainDB. "
        "Use this for preferences, working style, and durable user facts."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def _handle_retaindb_profile(args: dict, **kwargs) -> str:
    session_manager, session_key = _resolve_session_context(**kwargs)
    if not session_manager or not session_key:
        return _json_response({"error": "RetainDB is not active for this session."})
    try:
        return _json_response({"result": session_manager.get_profile(session_key)})
    except Exception as exc:
        logger.error("RetainDB profile lookup failed: %s", exc)
        return _json_response({"error": f"Failed to fetch profile: {exc}"})


_SEARCH_SCHEMA = {
    "name": "retaindb_search",
    "description": (
        "Search RetainDB memory for facts related to a query. "
        "Use this when you need deeper cross-session recall beyond the injected turn context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {
                "type": "integer",
                "description": "How many matching memories to return (default 8, max 20).",
            },
        },
        "required": ["query"],
    },
}


def _handle_retaindb_search(args: dict, **kwargs) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return _json_response({"error": "Missing required parameter: query"})
    session_manager, session_key = _resolve_session_context(**kwargs)
    if not session_manager or not session_key:
        return _json_response({"error": "RetainDB is not active for this session."})
    top_k = max(1, min(int(args.get("top_k", 8)), 20))
    try:
        return _json_response({"result": session_manager.search(session_key, query, top_k=top_k)})
    except Exception as exc:
        logger.error("RetainDB search failed: %s", exc)
        return _json_response({"error": f"Failed to search RetainDB: {exc}"})


_CONTEXT_SCHEMA = {
    "name": "retaindb_context",
    "description": (
        "Ask RetainDB for the best compact 'what matters now' memory context for this query. "
        "Use this when you need a synthesized memory block rather than raw search results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What context to retrieve."},
        },
        "required": ["query"],
    },
}


def _handle_retaindb_context(args: dict, **kwargs) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return _json_response({"error": "Missing required parameter: query"})
    session_manager, session_key = _resolve_session_context(**kwargs)
    if not session_manager or not session_key:
        return _json_response({"error": "RetainDB is not active for this session."})
    try:
        return _json_response({"result": session_manager.get_context(session_key, query)})
    except Exception as exc:
        logger.error("RetainDB context lookup failed: %s", exc)
        return _json_response({"error": f"Failed to fetch context: {exc}"})


_REMEMBER_SCHEMA = {
    "name": "retaindb_remember",
    "description": (
        "Persist an explicit fact, preference, correction, or instruction into RetainDB. "
        "Use this only for durable information that should survive future sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact or preference to remember."},
            "memory_type": {
                "type": "string",
                "description": "Optional type: factual, preference, event, relationship, opinion, goal, instruction.",
            },
            "importance": {
                "type": "number",
                "description": "Optional importance from 0.0 to 1.0.",
            },
        },
        "required": ["content"],
    },
}


def _handle_retaindb_remember(args: dict, **kwargs) -> str:
    content = str(args.get("content") or "").strip()
    if not content:
        return _json_response({"error": "Missing required parameter: content"})
    session_manager, session_key = _resolve_session_context(**kwargs)
    if not session_manager or not session_key:
        return _json_response({"error": "RetainDB is not active for this session."})
    memory_type = str(args.get("memory_type") or "factual").strip() or "factual"
    importance = float(args.get("importance", 0.7))
    try:
        return _json_response(
            {
                "result": session_manager.remember(
                    session_key,
                    content,
                    memory_type=memory_type,
                    importance=importance,
                )
            }
        )
    except Exception as exc:
        logger.error("RetainDB remember failed: %s", exc)
        return _json_response({"error": f"Failed to remember: {exc}"})


_FORGET_SCHEMA = {
    "name": "retaindb_forget",
    "description": (
        "Delete a specific RetainDB memory by id. "
        "Use this when the user explicitly asks to forget or remove stored information."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "The memory id to delete."},
        },
        "required": ["memory_id"],
    },
}


def _handle_retaindb_forget(args: dict, **kwargs) -> str:
    memory_id = str(args.get("memory_id") or "").strip()
    if not memory_id:
        return _json_response({"error": "Missing required parameter: memory_id"})
    session_manager, _session = _resolve_session_context(**kwargs)
    if not session_manager:
        return _json_response({"error": "RetainDB is not active for this session."})
    try:
        return _json_response({"result": session_manager.forget(memory_id)})
    except Exception as exc:
        logger.error("RetainDB forget failed: %s", exc)
        return _json_response({"error": f"Failed to forget memory: {exc}"})


from tools.registry import registry

registry.register(
    name="retaindb_profile",
    toolset="retaindb",
    schema=_PROFILE_SCHEMA,
    handler=_handle_retaindb_profile,
    check_fn=_check_retaindb_available,
)

registry.register(
    name="retaindb_search",
    toolset="retaindb",
    schema=_SEARCH_SCHEMA,
    handler=_handle_retaindb_search,
    check_fn=_check_retaindb_available,
)

registry.register(
    name="retaindb_context",
    toolset="retaindb",
    schema=_CONTEXT_SCHEMA,
    handler=_handle_retaindb_context,
    check_fn=_check_retaindb_available,
)

registry.register(
    name="retaindb_remember",
    toolset="retaindb",
    schema=_REMEMBER_SCHEMA,
    handler=_handle_retaindb_remember,
    check_fn=_check_retaindb_available,
)

registry.register(
    name="retaindb_forget",
    toolset="retaindb",
    schema=_FORGET_SCHEMA,
    handler=_handle_retaindb_forget,
    check_fn=_check_retaindb_available,
)
