"""Explicit completion signal for agent workflows."""

from __future__ import annotations

import json
from typing import Any

from tools.registry import registry


FINISH_SCHEMA = {
    "name": "finish",
    "description": (
        "Explicitly signal that the current task is done or blocked. "
        "Use status='done' after the requested work is complete. "
        "Use status='blocked' when human input or an external dependency is required."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["done", "blocked"],
                "description": "done = completed; blocked = cannot continue without external input",
            },
            "summary": {
                "type": "string",
                "description": "Concise completion or blocker summary.",
            },
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Observable proof supporting the finish claim.",
            },
        },
        "required": ["status", "summary", "evidence"],
    },
}


def finish_tool(status: str, summary: str, evidence: list[str]) -> str:
    if status not in {"done", "blocked"}:
        return json.dumps(
            {
                "success": False,
                "error": "status must be 'done' or 'blocked'",
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "success": True,
            "status": status,
            "summary": summary,
            "evidence": evidence,
        },
        ensure_ascii=False,
    )


def _handle_finish(args: dict, **kwargs: Any) -> str:
    evidence = args.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    return finish_tool(
        status=args.get("status", ""),
        summary=args.get("summary", ""),
        evidence=[str(item) for item in evidence],
    )


registry.register(
    name="finish",
    toolset="finish",
    schema=FINISH_SCHEMA,
    handler=_handle_finish,
)
