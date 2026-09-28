#!/usr/bin/env python3
"""Agent emoji reaction in the Hermes desktop app: the counterpart to the user's tapback
(same store, one-per-author, ``author="agent"``). Lives in the ``desktop_ui`` toolset so it
costs nothing elsewhere (adapters expose reactions via ``send_message(action="react")``);
defaults to the triggering message and emits ``message.reaction`` for live painting."""

import contextlib
import json

from gateway.session_context import get_session_env
from tools import desktop_ui
from tools.registry import registry, tool_error


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
    """Opt-in flag (Settings → Appearance); ``desktop_ui`` already restricts to GUI sessions."""
    return desktop_ui.user_enabled("message_reactions", default=False)


REACT_TO_MESSAGE_SCHEMA = {
    "name": "react_to_message",
    "description": (
        "React to a message with a single emoji, the way you'd tapback in iMessage. "
        "Reach for it when a reaction is what a person would do: something funny gets "
        "a 😂, warmth gets a ❤️, a plan you're on board with gets a 👍 — then just "
        "carry on with whatever the message actually needs. If a reaction says it "
        "all, it can BE the reply (skip the redundant 'sounds good!' turn). Use it "
        "like a person would: occasionally, when felt — not on every message, and "
        "never as a status signal. NEVER narrate or explain a reaction ('I reacted "
        "with...', 'Reacting now') — the emoji appearing on the bubble is the whole "
        "point, and commentary kills it. Defaults to the user's most recent message. "
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
    name="react_to_message", toolset="desktop_ui", schema=REACT_TO_MESSAGE_SCHEMA,
    handler=lambda args, **kw: react_to_message_tool(
        emoji=args.get("emoji", ""), message_row_id=args.get("message_row_id"),
        messages_back=args.get("messages_back")),
    check_fn=check_react_requirements, emoji="💛",
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'env_var_enabled': ('utils', 'env_var_enabled'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
