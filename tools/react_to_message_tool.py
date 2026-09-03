#!/usr/bin/env python3
"""Let the agent react to a message with an emoji in the Hermes desktop app.

The conversational counterpart to the user's tapback: the same reaction store,
the same one-per-author semantics, just written with ``author="agent"``.

Lives in the ``desktop_ui`` toolset (like the other GUI affordances) so it costs
nothing on every other surface — the platform adapters already expose reactions
through ``send_message(action="react")``, and this is the desktop's equivalent.

Defaults to the message that triggered this turn (the photon precedent: the
model shouldn't have to thread row ids through tool calls), and emits
``message.reaction`` so the renderer paints it without waiting for a resume.
"""

import json

from gateway.session_context import get_session_env
from tools import desktop_ui
from tools.registry import registry, tool_error
from utils import env_var_enabled


def _open_session_db():
    """Open the SessionDB for the profile owning this turn, or ``None``."""
    try:
        from hermes_state import get_shared_session_db

        return get_shared_session_db()
    except Exception:
        return None


def _react_to_message_with_db(
    emoji: str,
    message_row_id=None,
    messages_back=None,
    *,
    db,
    session_key: str,
) -> str:
    """Attach (or with an empty ``emoji`` retract) the agent's reaction."""
    if not session_key:
        return tool_error("No active session — reactions need a persisted conversation.")

    row_id = message_row_id
    target_role = "user"
    if row_id is None:
        # Default target: the latest user message. `messages_back` steps to
        # earlier user turns (1 = the one before, etc.) for retroactive
        # reactions — quoting text would be ambiguous, ids aren't visible to
        # the model, but "two messages ago" is how a person thinks about it.
        back = max(0, int(messages_back or 0))
        row_id = db.latest_message_row_id(session_key, role="user", offset=back)
        if row_id is None:
            return tool_error(
                f"No user message found {back} back." if back else "No user message to react to yet."
            )
    else:
        row = db.get_message_role(session_key, int(row_id))
        target_role = row or "user"

    try:
        reactions = db.set_message_reaction(
            session_key, int(row_id), emoji or None, author="agent"
        )
    except Exception as exc:
        return tool_error(f"Failed to set the reaction: {exc}")

    if reactions is None:
        return tool_error(f"Message {row_id} is not part of this conversation.")

    # Paint it live. A missing bridge (non-desktop surface) is not an error —
    # the reaction is persisted either way and shows on the next load.
    # `role` lets the renderer match a live message that doesn't know its
    # durable row id yet (it only learns rowId on resume).
    try:
        desktop_ui.emit(
            "message.reaction",
            {"row_id": int(row_id), "reactions": reactions, "role": target_role},
        )
    except Exception:
        pass

    return json.dumps(
        {"success": True, "row_id": int(row_id), "reactions": reactions}, ensure_ascii=False
    )


def react_to_message_tool(emoji: str, message_row_id=None, messages_back=None) -> str:
    """Attach (or with an empty ``emoji`` retract) the agent's reaction."""
    emoji = (emoji or "").strip()
    session_key = get_session_env("HERMES_SESSION_KEY", "") or get_session_env(
        "HERMES_SESSION_ID", ""
    )

    if not session_key:
        return tool_error("No active session — reactions need a persisted conversation.")

    db = _open_session_db()
    if db is None:
        return tool_error("Session storage is unavailable.")

    try:
        return _react_to_message_with_db(
            emoji,
            message_row_id,
            messages_back,
            db=db,
            session_key=session_key,
        )
    finally:
        try:
            from hermes_state import release_or_close
            release_or_close(db)
        except Exception:
            pass


def check_react_requirements() -> bool:
    """Opt-in feature flag — surface eligibility is the toolset's job.

    ``desktop_ui`` already restricts this to GUI sessions. What's left is the
    user's own toggle (Settings → Appearance).
    """
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
    name="react_to_message",
    toolset="desktop_ui",
    schema=REACT_TO_MESSAGE_SCHEMA,
    handler=lambda args, **kw: react_to_message_tool(
        emoji=args.get("emoji", ""),
        message_row_id=args.get("message_row_id"),
        messages_back=args.get("messages_back"),
    ),
    check_fn=check_react_requirements,
    emoji="💛",
)
