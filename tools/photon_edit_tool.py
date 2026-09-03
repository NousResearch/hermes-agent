#!/usr/bin/env python3
"""Edit one of OUR OWN iMessage bubbles over Photon.

iMessage supports editing a message you sent within a 15-minute window
(max 5 edits): the bubble's text is replaced in place, with a subtle
"Edited" label. This tool wraps the sidecar's ``/edit`` route (which
calls spectrum-ts ``space.send(spectrumEdit(...))``). It is an opt-in
tool in the ``photon_tools`` toolset, exposed only when a live Photon
adapter is running in this process (check_fn), so it costs zero tokens
on every other install.

Only messages sent by the bot can be edited, and only within Apple's
15-minute / 5-edit window; iMessage rejects other cases. Handlers
return JSON strings (the registry's normal result shape).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_TOOLSET = "photon_tools"

SCHEMA: Dict[str, Any] = {
    "name": "photon_edit",
    "description": (
        "Edit one of YOUR OWN recent iMessage messages on the Photon "
        "platform — replace the text of a bubble you sent with corrected "
        "text, in place (Apple shows a subtle 'Edited' label). Only works "
        "on messages YOU sent within the last 15 minutes (max 5 edits per "
        "message). Never edit the user's messages. Pass both message_id "
        "and the new text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message_id": {
                "type": "string",
                "description": "The id of the message to edit.",
            },
            "text": {
                "type": "string",
                "description": "The new text the bubble should read.",
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
        "required": ["message_id", "text"],
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


def _photon_edit_check() -> bool:
    """Availability check: tool only exists when live Photon adapter in-process."""
    return _live_photon_adapter() is not None


async def _photon_edit(args: Dict[str, Any], **kw) -> str:
    message_id = str(args.get("message_id") or "").strip()
    text = str(args.get("text") or "").strip()
    chat_id = str(args.get("chat_id") or "").strip()

    def err(msg: str) -> str:
        return tool_error(msg)

    if not message_id or not text:
        return err("Both 'message_id' and 'text' are required.")

    adapter = _live_photon_adapter()
    if adapter is None:
        return err(
            "No live Photon adapter in this process — edit requires the "
            "gateway to be running here."
        )

    try:
        chat_id = await _resolve_chat_id(chat_id)
    except Exception as e:
        return err(f"No chat context available: {e}")

    result = await adapter.edit_message(chat_id, message_id, text)
    if getattr(result, "success", True) and not getattr(result, "error", None):
        return json.dumps(
            {"success": True, "edited": message_id},
            ensure_ascii=False,
        )
    return err(
        getattr(result, "error", None)
        or f"Edit failed (HTTP {getattr(result, 'raw_response', '?')})."
    )


registry.register(
    name="photon_edit",
    toolset=_TOOLSET,
    schema=SCHEMA,
    handler=lambda args, **kw: _photon_edit(args, **kw),
    check_fn=_photon_edit_check,
    is_async=True,
    emoji="✏️",
)