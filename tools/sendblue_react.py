"""Sendblue tapback reaction tool.

Exposes a single-purpose ``sendblue_react`` tool that lets the agent
send iMessage tapback reactions (love, like, dislike, laugh, emphasize,
question) to the most recent inbound message in the current Sendblue
chat. The adapter holds an in-memory per-chat last-inbound handle cache;
this tool resolves it implicitly so the model doesn't need to track
opaque message_handles.

Only registered when the active session platform is Sendblue.
"""

from __future__ import annotations

import json
import os
from typing import Any

from gateway.session_context import get_session_env
from model_tools import _run_async
from tools.registry import registry

REACTION_TYPES = ["love", "like", "dislike", "laugh", "emphasize", "question"]

SENDBLUE_REACT_SCHEMA = {
    "name": "sendblue_react",
    "description": (
        "Send an iMessage tapback reaction (love/like/dislike/laugh/"
        "emphasize/question) to the most recent inbound message in the "
        "current Sendblue chat. Use sparingly — reactions are best for "
        "acknowledgement, not as a primary reply."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reaction": {
                "type": "string",
                "enum": REACTION_TYPES,
                "description": "Which tapback to send.",
            },
        },
        "required": ["reaction"],
    },
}


def _check_sendblue_react_available() -> bool:
    platform = get_session_env("HERMES_SESSION_PLATFORM", "") or os.getenv(
        "HERMES_SESSION_PLATFORM", ""
    )
    return platform.strip().lower() == "sendblue"


def _handle_sendblue_react(args: dict, **_kw) -> str:
    reaction = str(args.get("reaction") or "").strip().lower()
    if reaction not in REACTION_TYPES:
        return json.dumps(
            {"error": f"Invalid reaction. Use one of: {', '.join(REACTION_TYPES)}"}
        )

    chat_id = (
        get_session_env("HERMES_SESSION_CHAT_ID", "")
        or os.getenv("HERMES_SESSION_CHAT_ID", "")
    ).strip()
    if not chat_id:
        return json.dumps({"error": "No active Sendblue chat"})

    try:
        from gateway.run import _gateway_runner_ref
        runner = _gateway_runner_ref()
    except Exception:
        runner = None
    if runner is None:
        return json.dumps({"error": "Gateway runner not available"})

    adapter = None
    try:
        adapter = runner.adapters.get("sendblue")
    except Exception:
        adapter = None
    if adapter is None:
        return json.dumps({"error": "Sendblue adapter not running"})

    ok = _run_async(adapter.send_reaction(chat_id, reaction))
    return json.dumps(
        {"success": bool(ok), "chat_id": chat_id, "reaction": reaction}
    )


registry.register(
    name="sendblue_react",
    toolset="sendblue",
    schema=SENDBLUE_REACT_SCHEMA,
    handler=_handle_sendblue_react,
    check_fn=_check_sendblue_react_available,
    emoji="💬",
)
