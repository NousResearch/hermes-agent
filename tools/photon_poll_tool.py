#!/usr/bin/env python3
"""Native iMessage polls over Photon.

``send_message`` is deliberately not agent-callable for cross-platform
reach (see ``send_message_tool.py``); a poll is a narrower, genuinely
native affordance — an orange iMessage poll bubble where the recipient
taps an option instead of typing. That same narrow surface makes it
safe to expose: it delivers no new text and the recipient's choice
streams back as a ``poll_option`` event, which the Photon adapter
turns into the answer resolving a pending ``clarify``.

Like ``photon_react_tool`` this ships as its own opt-in toolset: the
tool only exists when a live Photon adapter is running in this process
(the same state ``send_poll`` needs). On every other install the
toolset is empty and costs zero prompt tokens. The handler still
re-checks and returns a clear error if the gateway went away
mid-session.

The native poll (and the option-tap workflow it enables) lives behind
the local sidecar ``/send-poll`` route — see the Photon overlay docs
for the patch that wires it.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_TOOLSET = "photon_tools"

SCHEMA: Dict[str, Any] = {
    "name": "photon_poll",
    "description": (
        "Send an interactive iMessage poll card (Photon platform). Use when "
        "the user faces a genuine either/or choice worth tapping — picking "
        "between options, voting, deciding. NOT for open questions. Provide "
        "a short title and 2-4 concise options."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short question shown above the options.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "description": "2-4 concise answer options.",
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
        "required": ["title", "options"],
    },
}


def _live_photon_adapter():
    """Return the running Photon adapter, or None.

    Mirrors ``send_message_tool._send_via_adapter`` and
    ``photon_react_tool._live_photon_adapter``: polls route through the
    gateway's live adapter (which owns the sidecar connection).
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


def _photon_poll_check() -> bool:
    """Availability check (runs at schema-assembly time): the tool only
    exists when a live Photon adapter is in this process. Mirrors
    photon_react_tool._photon_react_check."""
    return _live_photon_adapter() is not None


async def _photon_poll(args: Dict[str, Any], **kw) -> str:
    title = str(args.get("title") or "").strip()
    options = [str(o).strip() for o in (args.get("options") or []) if str(o).strip()]
    if not title:
        return tool_error("A 'title' is required.")
    if len(options) < 2:
        return tool_error("Polls need at least two non-empty options.")

    adapter = _live_photon_adapter()
    if adapter is None:
        return tool_error(
            "No live Photon adapter in this process — polls require the "
            "gateway to be running here."
        )

    chat_id = await _resolve_chat_id(str(args.get("chat_id") or ""))

    result = await adapter.send_poll(chat_id, title, options)
    if getattr(result, "success", True) and not getattr(result, "error", None):
        return json.dumps(
            {"success": True, "title": title, "options": len(options)},
            ensure_ascii=False,
        )
    return tool_error(
        getattr(result, "error", None)
        or f"Poll send failed (HTTP {getattr(result, 'raw_response', '?')})."
    )


registry.register(
    name="photon_poll",
    toolset=_TOOLSET,
    schema=SCHEMA,
    handler=lambda args, **kw: _photon_poll(args, **kw),
    check_fn=_photon_poll_check,
    is_async=True,
    emoji="📊",
)