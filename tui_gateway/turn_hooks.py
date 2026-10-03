"""Profile-scoped lifecycle hooks for turns served by the JSON-RPC backend."""

import asyncio
import logging

from gateway.hooks import ProfileHookRegistries

logger = logging.getLogger(__name__)
_hooks = ProfileHookRegistries()


def _emit(event: str, context: dict) -> None:
    # The prompt worker is synchronous. asyncio.run preserves its ContextVars for
    # async handlers, including the home, credential and terminal scopes.
    try:
        asyncio.run(_hooks.emit(event, context))
    except Exception:
        logger.exception("Turn hook %s failed", event)


def start_turn(sid: str, session: dict, agent, message) -> dict:
    context = {
        "platform": session.get("source") or getattr(agent, "platform", "") or "tui",
        "user_id": session.get("auth_user_id") or getattr(agent, "user_id", "") or "",
        "chat_id": sid,
        "thread_id": "",
        "chat_type": "dm",
        "session_id": getattr(agent, "session_id", None) or session.get("session_key") or "",
        "message": message[:500] if isinstance(message, str) else "",
    }
    _emit("agent:start", context)
    return context


def end_turn(context: dict, session: dict, agent, result) -> None:
    result = result if isinstance(result, dict) else {}
    response = result.get("final_response") or ""
    _emit("agent:end", {
        **context,
        # Compression can rotate the durable id. Memory extractors must read the
        # committed turn from the live tip, rather than its pre-compression parent.
        "session_id": getattr(agent, "session_id", None) or session.get("session_key") or context["session_id"],
        "response": response[:500] if isinstance(response, str) else "",
        "model": result.get("model") or getattr(agent, "model", "") or "",
        "provider": result.get("provider") or getattr(agent, "provider", "") or "",
    })
