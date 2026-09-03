#!/usr/bin/env python3
"""Recall (unsend) one of OUR OWN iMessage bubbles over Photon.

iMessage supports recalling a message you sent — the bubble vanishes
from both sides with the native "You unsent a message" notice. This
tool wraps the sidecar's ``/unsend-message`` route (which calls
spectrum-ts ``space.unsend()``). It is an opt-in tool in the
``photon_tools`` toolset, exposed only when a live Photon adapter is
running in this process (check_fn), so it costs zero tokens on
every other install.

Only messages sent by the bot can be unsent; iMessage rejects
attempts to unsend inbound/foreign messages. The handler validates
the chat and message id, defaults to the most recent message the bot
sent in the chat when message_id is omitted, and returns a JSON
string (the registry's normal result shape).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_TOOLSET = "photon_tools"

SCHEMA: Dict[str, Any] = {
    "name": "photon_unsend",
    "description": (
        "Unsend (recall) one of YOUR OWN recent iMessage messages on the "
        "Photon platform — for example a garbled or mistaken bubble you "
        "just sent. Only works on messages YOU sent (never the user's "
        "messages) that are still in the session's memory. Omit message_id "
        "to recall your most recent sent message in the chat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message_id": {
                "type": "string",
                "description": (
                    "The id of the message to recall. Omit to recall your "
                    "most recent sent message in the chat."
                ),
            },
            "chat_id": {
                "type": "string",
                "description": (
                    "Optional chat (space GUID like 'any;-;+1555…' or bare "
                    "E.164). Defaults to the conversation currently being "
                    "answered."
                ),
            },
        },
        "required": [],
    },
}


def _live_photon_adapter():
    """Return the running Photon adapter, or None.

    Mirrors ``photon_poll_tool`` / ``photon_react_tool``.
    """
    try:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
    except Exception:
        return None
    if runner is None:
        return None
    try:
        from gateway.config import Platform

        return runner.adapters.get(Platform.PHOTON)
    except Exception:
        return None


async def _resolve_chat_id(explicit: str) -> str:
    chat_id = (explicit or "").strip()
    if chat_id:
        return chat_id
    try:
        from gateway.session_context import get_session_env

        chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "").strip()
    except Exception:
        chat_id = ""
    if chat_id:
        return chat_id
    from gateway.config import load_gateway_config

    config = load_gateway_config()
    home = config.get_home_channel("photon")
    return home.chat_id


async def _last_sent_message_id(chat_id: str) -> str | None:
    """Most recent message WE sent in chat, via the adapter's tracker."""
    adapter = _live_photon_adapter()
    if adapter is None:
        return None
    # Prefer the adapter's own sent-tracker; fall back to sidecar /last-sent.
    per_chat = getattr(adapter, "_last_sent_by_chat", None)
    if per_chat:
        norm = getattr(adapter, "_normalize_chat_key", None)
        key = norm(chat_id) if norm else chat_id
        return per_chat.get(key)
    # Sidecar fallback would need the runtime record — not needed for core
    # tools since the adapter tracks its own sent ids.
    return None


def _photon_unsend_check() -> bool:
    """Availability check: tool only exists when live Photon adapter in-process."""
    return _live_photon_adapter() is not None


async def _photon_unsend(args: Dict[str, Any], **kw) -> str:
    message_id = str(args.get("message_id") or "").strip()
    chat_id = str(args.get("chat_id") or "").strip()

    def err(msg: str) -> str:
        return tool_error(msg)

    try:
        chat_id = await _resolve_chat_id(chat_id)
    except Exception as e:
        return err(f"No chat context available: {e}")

    if not message_id:
        message_id = await _last_sent_message_id(chat_id)
        if not message_id:
            return err(
                "No message_id given and no recently-sent message found to recall."
            )

    adapter = _live_photon_adapter()
    if adapter is None:
        return err("No live Photon adapter in this process — unsend requires the "
                   "gateway to be running here.")

    result = await adapter.unsend(chat_id, message_id)
    if getattr(result, "success", True) and not getattr(result, "error", None):
        return json.dumps(
            {"success": True, "unsent": getattr(result, "message_id", message_id)},
            ensure_ascii=False,
        )
    return err(
        getattr(result, "error", None)
        or f"Unsend failed (HTTP {getattr(result, 'raw_response', '?')})."
    )


registry.register(
    name="photon_unsend",
    toolset=_TOOLSET,
    schema=SCHEMA,
    handler=lambda args, **kw: _photon_unsend(args, **kw),
    check_fn=_photon_unsend_check,
    is_async=True,
    emoji="🗑️",
)