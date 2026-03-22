"""Blooio reaction tool.

Adds or removes a reaction on a message in the current Blooio chat.
This tool is only available during Blooio messaging sessions.
"""

import json
import os
from urllib.parse import quote

import requests

from tools.registry import registry


REACTION_TOOL_SCHEMA = {
    "name": "react_to_message",
    "description": (
        "Add or remove a reaction on a message in the current Blooio chat. "
        "Use this when a lightweight reaction is more appropriate than a full reply, "
        "or when a reaction complements a normal reply."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reaction": {
                "type": "string",
                "description": (
                    "Reaction to add or remove. Must start with '+' to add or '-' to remove. "
                    "Examples: '+love', '-love', '+like', '+laugh', '+question', '+👍'."
                ),
            },
            "message_id": {
                "type": "string",
                "description": (
                    "Optional Blooio message ID to react to. If omitted, Hermes reacts to the "
                    "current inbound message when available, otherwise uses '-1' for the latest "
                    "inbound message in this chat."
                ),
            },
            "direction": {
                "type": "string",
                "enum": ["inbound", "outbound"],
                "description": (
                    "Optional direction filter used only when message_id is a relative index such as '-1'. "
                    "Defaults to 'inbound'."
                ),
            },
        },
        "required": ["reaction"],
    },
}


def _blooio_reaction_available() -> bool:
    """Return True when a Blooio session is active and the API key exists."""
    return bool(
        os.getenv("HERMES_SESSION_PLATFORM", "").strip().lower() == "blooio"
        and os.getenv("HERMES_SESSION_CHAT_ID", "").strip()
        and os.getenv("BLOOIO_API_KEY", "").strip()
    )


def _default_message_target() -> tuple[str, str]:
    """Return (message_id, direction) for the current Blooio session."""
    current_message_id = os.getenv("HERMES_SESSION_MESSAGE_ID", "").strip()
    if current_message_id:
        return current_message_id, ""
    return "-1", "inbound"


def _normalize_reaction(value: str) -> str:
    reaction = str(value or "").strip()
    if len(reaction) < 2 or reaction[0] not in {"+", "-"}:
        raise ValueError("reaction must start with '+' or '-' and include a reaction value")
    return reaction


def react_to_message_tool(args, **_kw):
    """Handle Blooio reactions for the current chat."""
    if not _blooio_reaction_available():
        return json.dumps({
            "error": "react_to_message is only available inside an active Blooio chat session"
        })

    try:
        reaction = _normalize_reaction(args.get("reaction", ""))
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    chat_id = os.getenv("HERMES_SESSION_CHAT_ID", "").strip()
    message_id = str(args.get("message_id") or "").strip()
    direction = str(args.get("direction") or "").strip().lower()

    if not message_id:
        message_id, default_direction = _default_message_target()
        if not direction:
            direction = default_direction

    if direction and direction not in {"inbound", "outbound"}:
        return json.dumps({"error": "direction must be 'inbound' or 'outbound'"})

    payload = {"reaction": reaction}
    if direction and message_id.startswith("-"):
        payload["direction"] = direction

    api_key = os.getenv("BLOOIO_API_KEY", "").strip()
    api_base = (os.getenv("BLOOIO_BASE_URL") or "https://backend.blooio.com/v2/api").rstrip("/")
    url = (
        f"{api_base}/chats/{quote(chat_id, safe='')}/messages/"
        f"{quote(message_id, safe='')}/reactions"
    )

    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
    except Exception as exc:
        return json.dumps({"error": f"Blooio reaction request failed: {exc}"})

    body_text = response.text or ""
    try:
        data = response.json() if body_text else {}
    except Exception:
        data = {"raw": body_text}

    if response.status_code != 200:
        message = body_text or data.get("error") or f"HTTP {response.status_code}"
        return json.dumps({"error": f"Blooio reaction failed ({response.status_code}): {message}"})

    return json.dumps({
        "success": True,
        "platform": "blooio",
        "chat_id": chat_id,
        "message_id": str(data.get("message_id") or message_id),
        "reaction": data.get("reaction") or reaction[1:],
        "action": data.get("action") or ("add" if reaction.startswith("+") else "remove"),
    })


registry.register(
    name="react_to_message",
    toolset="messaging",
    schema=REACTION_TOOL_SCHEMA,
    handler=react_to_message_tool,
    check_fn=_blooio_reaction_available,
    requires_env=["BLOOIO_API_KEY"],
    emoji="💬",
)
