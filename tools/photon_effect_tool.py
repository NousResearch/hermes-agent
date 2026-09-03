#!/usr/bin/env python3
"""Native iMessage bubble/screen effects over Photon.

iMessage bubbles support native send effects (slam, loud, gentle,
invisible ink, confetti, fireworks, balloons, heart, lasers,
celebration, sparkles, spotlight, echo). Like ``photon_poll_tool``,
this is a narrow, genuinely native affordance — it delivers the
recipient's message with a built-in animation instead of extra text,
so it is safe to expose as an opt-in toolset that only exists when a
live Photon adapter is running in this process.

The effects route lives behind the local sidecar ``/send-effect``
route — see the Photon overlay docs for the patch that wires it.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_TOOLSET = "photon_tools"

_EFFECTS = (
    "slam",
    "loud",
    "gentle",
    "invisible",
    "confetti",
    "fireworks",
    "balloons",
    "heart",
    "lasers",
    "celebration",
    "sparkles",
    "spotlight",
    "echo",
)

SCHEMA: Dict[str, Any] = {
    "name": "photon_effect",
    "description": (
        "Send an iMessage with a native bubble/screen effect (Photon "
        "platform). USE RARELY — only for moments that genuinely warrant "
        "one: 'confetti'/'fireworks'/'celebration' for big wins, 'slam' or "
        "'loud' for emphatic statements, 'invisible' for spoilers/secrets "
        "the recipient must tap to reveal, 'heart' for affection. Most "
        "messages deserve NO effect."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The message text to send with the effect.",
            },
            "effect": {
                "type": "string",
                "enum": list(_EFFECTS),
                "description": (
                    "Effect name: slam, loud, gentle, invisible, confetti, "
                    "fireworks, balloons, heart, lasers, celebration, "
                    "sparkles, spotlight, echo."
                ),
            },
            "chat_id": {
                "type": "string",
                "description": (
                    "Optional chat to send to (space GUID like 'any;-;+1555…' "
                    "or bare E.164). Defaults to the chat of the message "
                    "currently being answered."
                ),
            },
        },
        "required": ["text", "effect"],
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


def _photon_effect_check() -> bool:
    """Availability check (runs at schema-assembly time): the tool only
    exists when a live Photon adapter is in this process."""
    return _live_photon_adapter() is not None


async def _photon_effect(args: Dict[str, Any], **kw) -> str:
    text = str(args.get("text") or "").strip()
    effect = str(args.get("effect") or "").strip().lower()
    if not text or not effect:
        return tool_error("Both 'text' and 'effect' are required.")
    if effect not in _EFFECTS:
        return tool_error(
            f"Unknown effect '{effect}'. Choose one of: {', '.join(_EFFECTS)}."
        )

    adapter = _live_photon_adapter()
    if adapter is None:
        return tool_error(
            "No live Photon adapter in this process — effects require the "
            "gateway to be running here."
        )

    chat_id = await _resolve_chat_id(str(args.get("chat_id") or ""))

    result = await adapter.send_effect(chat_id, text, effect)
    if getattr(result, "success", True) and not getattr(result, "error", None):
        return json.dumps(
            {
                "success": True,
                "effect": effect,
                "message_id": getattr(result, "message_id", None),
            },
            ensure_ascii=False,
        )
    return tool_error(
        getattr(result, "error", None)
        or f"Effect send failed (HTTP {getattr(result, 'raw_response', '?')})."
    )


registry.register(
    name="photon_effect",
    toolset=_TOOLSET,
    schema=SCHEMA,
    handler=lambda args, **kw: _photon_effect(args, **kw),
    check_fn=_photon_effect_check,
    is_async=True,
    emoji="🎉",
)