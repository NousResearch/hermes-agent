#!/usr/bin/env python3
"""Selective emoji reactions for Photon / iMessage.

``send_message`` is deliberately not agent-callable (see the note at the
bottom of ``send_message_tool.py``): an agent that can fire cross-platform
messages whenever it likes is an agent one prompt injection away from
spamming every connected channel.  Reactions are a much smaller surface —
they never deliver new content to anyone — but the same worry applies in a
milder form: an agent tapbacking every message is noisy.

This module exposes the middle ground as its own opt-in toolset:

* The tool only *exists* when the operator turned reactions on for the
  Photon platform (``PHOTON_REACTIONS=true``) **and** a live Photon adapter
  is running in this process — the same state ``add_reaction`` needs.  On
  every other install the toolset is empty and costs zero prompt tokens.
* Even when offered, the tool description tells the model to react
  sparingly and pick emoji by context ("vibes"), mirroring how a human
  uses tapbacks.  The guardrail is policy-first; the code just refuses to
  load where the feature was never enabled.

The conversational counterpart is ``react_to_message`` (desktop_ui); this
is the messaging-platform equivalent, scoped to Photon because iMessage is
where tapbacks are a first-class native affordance.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_TOOLSET = "photon_react"

SCHEMA: Dict[str, Any] = {
    "name": "photon_react",
    "description": (
        "React to an iMessage with a tapback emoji (Photon platform). "
        "USE SPARINGLY — react only to messages that genuinely warrant it "
        "(good news, a joke worth laughing at, a big win, explicit thanks); "
        "most messages deserve NO reaction. Pick the emoji by the vibe of "
        "the message: 🔥 hype, 😂 funny, ❤️ warm, 💀 hilarious-shock, 👍 "
        "acknowledge, 🙌 celebrate. ❤️👍👎😂‼️❓ render as native Apple "
        "tapbacks; any other emoji renders as a custom-emoji reaction "
        "(iOS 17+). Pass message_id to target a specific message, or omit "
        "it to react to the most recent inbound message in the chat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "emoji": {
                "type": "string",
                "description": "The emoji to react with (e.g. '🔥').",
            },
            "chat_id": {
                "type": "string",
                "description": (
                    "Optional chat to react in (space GUID like "
                    "'any;-;+1555...' or bare E.164). Defaults to the chat "
                    "of the message currently being answered."
                ),
            },
            "message_id": {
                "type": "string",
                "description": (
                    "Optional id of the message to react to. Omit to target "
                    "the most recent inbound message in the chat."
                ),
            },
            "remove": {
                "type": "boolean",
                "description": "True to retract our tapback instead of adding one.",
            },
        },
        "required": ["emoji"],
    },
}


def _live_photon_adapter():
    """Return the running Photon adapter, or None.

    Mirrors ``send_message_tool._send_via_adapter``: reactions need the
    gateway's live message-id tracking (``_last_inbound_by_chat``), so a
    standalone sender fallback cannot help here.
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


def _reactions_enabled() -> bool:
    """Same gate as the lifecycle tapbacks in the Photon adapter."""
    return os.getenv("PHOTON_REACTIONS", "false").strip().lower() in {
        "true", "1", "yes", "on",
    }


def _photon_react_check() -> bool:
    """Availability check (runs at schema-assembly time).

    Enforces the dual gate documented at module level: the operator flag
    is on AND a live Photon adapter exists in this process.  The adapter
    probe is the same in-process helper the handler uses — a runner-ref
    lookup, no I/O — so startup never blocks on a dead sidecar.  The
    handler still re-checks and returns a clear error if the gateway went
    away mid-session.
    """
    return _reactions_enabled() and _live_photon_adapter() is not None


async def _photon_react(args: Dict[str, Any], **kw) -> Dict[str, Any]:
    emoji = str(args.get("emoji") or "").strip()
    remove = bool(args.get("remove", False))
    if not emoji and not remove:
        return tool_error("An 'emoji' is required (or pass remove=true).")

    adapter = _live_photon_adapter()
    if adapter is None:
        return tool_error(
            "No live Photon adapter in this process — reactions require the "
            "gateway to be running here."
        )

    chat_id = str(args.get("chat_id") or "").strip()
    if not chat_id:
        # Honor the promise in the schema: default to the chat of the
        # message currently being answered (the gateway tracks this per
        # conversation), falling back to the Photon home channel DM.
        try:
            from gateway.session_context import get_session_env

            chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "").strip()
        except Exception:
            chat_id = ""
    if not chat_id:
        try:
            from gateway.config import load_gateway_config

            config = load_gateway_config()
            home = config.get_home_channel("photon")
            chat_id = home.chat_id
        except Exception:
            return tool_error(
                "No chat_id given and no current-chat context or Photon "
                "home channel available."
            )

    message_id: Optional[str] = str(args.get("message_id") or "").strip() or None

    if remove:
        result = await adapter.remove_reaction(chat_id, message_id)
    else:
        result = await adapter.add_reaction(chat_id, emoji, message_id)
    return result


registry.register(
    name="photon_react",
    toolset=_TOOLSET,
    schema=SCHEMA,
    handler=lambda args, **kw: _photon_react(args, **kw),
    check_fn=_photon_react_check,
    is_async=True,
    emoji="🔥",
)
