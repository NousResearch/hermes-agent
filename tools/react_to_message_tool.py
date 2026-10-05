#!/usr/bin/env python3
"""Agent emoji reaction affordance.

Desktop sessions persist reactions in SessionDB for in-app rendering. Gateway sessions with a
live platform adapter react natively to the triggering inbound message, so Telegram/Discord users
see the tapback on their message instead of a redundant text acknowledgement.
"""

import contextlib
import inspect
import json

from gateway.session_context import get_session_env
from tools import desktop_ui
from tools.registry import registry, tool_error


def _is_desktop_reactions_session() -> bool:
    return desktop_ui.user_enabled("message_reactions", default=False)


def _current_platform_target():
    """Return ``(platform_name, chat_id, message_id)`` for the active gateway turn."""
    return (
        (get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower(),
        (get_session_env("HERMES_SESSION_CHAT_ID", "") or "").strip(),
        (get_session_env("HERMES_SESSION_MESSAGE_ID", "") or "").strip(),
    )


def _live_adapter_for_current_platform(platform_name: str):
    try:
        from gateway.config import Platform
        from tools.send_message_senders import _live_adapter

        platform = Platform(platform_name)
        _runner, adapter = _live_adapter(platform)
        return adapter
    except Exception:
        return None


async def _apply_native_reaction(adapter, platform_name: str, chat_id: str, message_id: str, emoji: str) -> bool:
    """Apply or clear a reaction through the live platform adapter."""
    async def _call(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    if platform_name == "telegram":
        set_reaction = getattr(adapter, "_set_reaction", None)
        if callable(set_reaction):
            return bool(await _call(set_reaction, chat_id, message_id, emoji or None))

    if emoji:
        for name in ("add_reaction", "_add_reaction"):
            fn = getattr(adapter, name, None)
            if callable(fn):
                try:
                    result = await _call(fn, chat_id=chat_id, message_id=message_id, emoji=emoji)
                except TypeError:
                    result = await _call(fn, chat_id, message_id, emoji)
                return bool(result.get("success", True) if isinstance(result, dict) else result)
        return False

    for name in ("remove_reaction", "_remove_reaction"):
        fn = getattr(adapter, name, None)
        if callable(fn):
            try:
                result = await _call(fn, chat_id=chat_id, message_id=message_id)
            except TypeError:
                result = await _call(fn, chat_id, message_id)
            return bool(result.get("success", True) if isinstance(result, dict) else result)
    return False


def _native_reactions_available() -> bool:
    platform_name, chat_id, message_id = _current_platform_target()
    if not platform_name or not chat_id or not message_id:
        return False
    adapter = _live_adapter_for_current_platform(platform_name)
    if adapter is None:
        return False
    if platform_name == "telegram" and callable(getattr(adapter, "_set_reaction", None)):
        return True
    return any(callable(getattr(adapter, name, None)) for name in (
        "add_reaction", "_add_reaction", "remove_reaction", "_remove_reaction"))


def _react_native(emoji: str) -> str | None:
    platform_name, chat_id, message_id = _current_platform_target()
    if not platform_name or not chat_id or not message_id:
        return None
    adapter = _live_adapter_for_current_platform(platform_name)
    if adapter is None:
        return None
    try:
        from model_tools import _run_async

        ok = _run_async(_apply_native_reaction(adapter, platform_name, chat_id, message_id, emoji))
    except Exception as exc:
        return tool_error(f"Failed to set the platform reaction: {exc}")
    if not ok:
        return tool_error(f"Platform '{platform_name}' does not support reacting to the current message.")
    return json.dumps({
        "success": True,
        "platform": platform_name,
        "chat_id": chat_id,
        "message_id": message_id,
        "emoji": emoji,
        "native": True,
    }, ensure_ascii=False)


def _open_session_db():
    """Open the SessionDB for the profile owning this turn, or ``None``."""
    try:
        from hermes_state_registry import acquire
        return acquire()
    except Exception:
        return None


def react_to_message_tool(emoji: str, message_row_id=None, messages_back=None) -> str:
    """Attach (or with an empty ``emoji`` retract) the agent's reaction."""
    emoji = (emoji or "").strip()
    if message_row_id is None and messages_back is None:
        native_result = _react_native(emoji)
        if native_result is not None:
            return native_result

    session_key = get_session_env("HERMES_SESSION_KEY", "") or get_session_env("HERMES_SESSION_ID", "")
    if not session_key:
        return tool_error("No active session — reactions need a persisted conversation.")
    db = _open_session_db()
    if db is None:
        return tool_error("Session storage is unavailable.")
    try:
        row_id, target_role = message_row_id, "user"
        if row_id is None:
            # Default: the latest user message; `messages_back` steps to earlier user turns
            # (ids aren't visible to the model; "two messages ago" is how a person thinks).
            back = max(0, int(messages_back or 0))
            row_id = db.latest_message_row_id(session_key, role="user", offset=back)
            if row_id is None:
                return tool_error(f"No user message found {back} back." if back else "No user message to react to yet.")
        else:
            target_role = db.get_message_role(session_key, int(row_id)) or "user"
        try:
            reactions = db.set_message_reaction(session_key, int(row_id), emoji or None, author="agent")
        except Exception as exc:
            return tool_error(f"Failed to set the reaction: {exc}")
        if reactions is None:
            return tool_error(f"Message {row_id} is not part of this conversation.")
        # Paint it live; a missing bridge (non-desktop) is not an error — the reaction is
        # persisted. `role` lets the renderer match a live message without a durable row id.
        with contextlib.suppress(Exception):
            desktop_ui.emit("message.reaction", {"row_id": int(row_id), "reactions": reactions, "role": target_role})
        return json.dumps({"success": True, "row_id": int(row_id), "reactions": reactions}, ensure_ascii=False)
    finally:
        with contextlib.suppress(Exception):
            from hermes_state_registry import release_or_close
            release_or_close(db)


def check_react_requirements() -> bool:
    """Available in desktop reaction sessions or gateway turns with native reaction support."""
    return _is_desktop_reactions_session() or _native_reactions_available()


REACT_TO_MESSAGE_SCHEMA = {
    "name": "react_to_message",
    "description": (
        "React to the user's current message with a single emoji, the way you'd tapback in iMessage. "
        "Reach for it when a reaction is what a person would do: something funny gets "
        "a 😂, warmth gets a ❤️, a plan you're on board with gets a 👍 — then just "
        "carry on with whatever the message actually needs. If a reaction says it "
        "all, it can BE the reply (skip the redundant 'sounds good!' turn). Use it "
        "like a person would: occasionally, when felt — not on every message, and "
        "never as a status signal. NEVER narrate or explain a reaction ('I reacted "
        "with...', 'Reacting now') — the emoji appearing on the bubble is the whole "
        "point, and commentary kills it. In messaging platforms, defaults to the inbound "
        "message that triggered this turn. In desktop sessions, defaults to the user's most recent message. "
        "One reaction per message: a different emoji replaces yours, an empty string "
        "retracts it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "emoji": {
                "type": "string",
                "description": (
                    "The emoji to react with (e.g. '❤️', '😂', '👍'). Pass an empty "
                    "string to remove your reaction."
                ),
            },
            "message_row_id": {
                "type": "integer",
                "description": (
                    "Optional. The specific message to react to. Omit to react to the "
                    "user's latest message, which is almost always what you want."
                ),
            },
            "messages_back": {
                "type": "integer",
                "description": (
                    "Optional. React to an EARLIER user message: 1 = the one before "
                    "the latest, 2 = two before, and so on. For when something lands "
                    "late — the joke you only got after answering."
                ),
            },
        },
        "required": ["emoji"],
    },
}


registry.register(
    name="react_to_message", toolset="reactions", schema=REACT_TO_MESSAGE_SCHEMA,
    handler=lambda args, **kw: react_to_message_tool(
        emoji=args.get("emoji", ""), message_row_id=args.get("message_row_id"),
        messages_back=args.get("messages_back")),
    check_fn=check_react_requirements, emoji="💛",
)
