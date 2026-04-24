#!/usr/bin/env python3
"""
Human approval tool - gateway-mediated owner approval for sensitive actions.

This is like clarify(), but the approval target can be a DIFFERENT chat/platform
from the original requester. The platform layer owns the actual delivery and
waiting logic; this module just defines the schema + validation.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from tools.registry import registry, tool_error


def request_human_approval_tool(
    question: str,
    target: str,
    timeout_seconds: Optional[int] = None,
    metadata: Optional[dict[str, Any]] = None,
    callback: Optional[Callable[..., str]] = None,
) -> str:
    if not question or not str(question).strip():
        return tool_error("question is required")
    if not target or not str(target).strip():
        return tool_error("target is required")
    if timeout_seconds is not None:
        try:
            timeout_seconds = int(timeout_seconds)
        except Exception:
            return tool_error("timeout_seconds must be an integer")
        if timeout_seconds < 1 or timeout_seconds > 3600:
            return tool_error("timeout_seconds must be between 1 and 3600")
    else:
        timeout_seconds = 600

    if metadata is not None and not isinstance(metadata, dict):
        return tool_error("metadata must be an object when provided")

    if callback is None:
        return json.dumps(
            {"error": "Human approval tool is not available in this execution context."},
            ensure_ascii=False,
        )

    try:
        return callback(
            question=str(question).strip(),
            target=str(target).strip(),
            timeout_seconds=timeout_seconds,
            metadata=metadata or {},
        )
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to request human approval: {exc}"},
            ensure_ascii=False,
        )


HUMAN_APPROVAL_SCHEMA = {
    "name": "request_human_approval",
    "description": (
        "Request approval from a human in another chat/platform before proceeding. "
        "Use this for suspicious, sensitive, or policy-gated actions where the "
        "approver is not the current requester. The tool sends a structured approval "
        "request to the target, waits for an approve/deny reply tied to a unique approval ID, "
        "and returns the decision. Prefer this over plain send_message when you actually need "
        "a tracked decision rather than a fire-and-forget notification."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What needs approval. Include enough context for the approver to decide safely.",
            },
            "target": {
                "type": "string",
                "description": (
                    "Where to send the approval request. Same format as send_message target, "
                    "for example 'telegram', 'telegram:987654321', or 'discord:123:456'."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3600,
                "description": "How long to wait for approval before timing out. Default 600.",
            },
            "metadata": {
                "type": "object",
                "description": (
                    "Optional structured context for the approval message and audit trail, "
                    "for example requester platform/user/chat/excerpt."
                ),
                "additionalProperties": True,
            },
        },
        "required": ["question", "target"],
    },
}


registry.register(
    name="request_human_approval",
    toolset="clarify",
    schema=HUMAN_APPROVAL_SCHEMA,
    handler=lambda args, **kw: request_human_approval_tool(
        question=args.get("question", ""),
        target=args.get("target", ""),
        timeout_seconds=args.get("timeout_seconds"),
        metadata=args.get("metadata"),
        callback=kw.get("callback"),
    ),
    check_fn=lambda: True,
    emoji="🛂",
)
