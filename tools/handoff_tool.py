"""tools/handoff_tool.py — write-only session handoff document preparation.

Writes a handoff document to disk so a human can start a fresh session and
manually paste/reference it. This tool is deliberately write-only: it does
NOT reset, restart, or trigger `/new`, and it does NOT inject its content
into any future turn or session. Continuation is always a manual, human-
initiated step.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error, tool_result

HANDOFF_SCHEMA = {
    "name": "handoff",
    "description": (
        "Write a handoff document to disk describing the current session's state "
        "so a human can manually continue the work in a fresh session. This tool "
        "only writes a file — it does not reset the session and does not inject "
        "anything into a future turn. After writing, tell the user to run /new "
        "and reference (or paste) the handoff file themselves."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["write"],
                "description": "The only supported action: write the handoff document to disk.",
            },
            "content": {
                "type": "string",
                "description": "The handoff document content (markdown) to write to disk.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Optional destination. An absolute path is used as-is. A bare "
                    "filename is written under the handoffs/ directory inside the "
                    "Hermes home. If omitted, a UTC-timestamped filename is used."
                ),
            },
        },
        "required": ["action", "content"],
    },
}


def _resolve_handoff_path(path: str | None) -> Path:
    """Resolve the target path for a handoff document.

    - Absolute ``path`` is used as-is.
    - Bare-filename ``path`` is written under ``get_hermes_home()/handoffs/``.
    - No ``path`` defaults to a UTC-timestamped filename under the same directory.
    """
    handoffs_dir = get_hermes_home() / "handoffs"
    if path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return handoffs_dir / candidate
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return handoffs_dir / f"{timestamp}-handoff.md"


def _handoff_write(content: str, path: str | None = None) -> str:
    """Write a handoff document to disk. Returns a JSON tool_result/tool_error string."""
    if not content or not content.strip():
        return tool_error("content is required and cannot be empty.")

    target = _resolve_handoff_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    resolved = str(target.resolve())
    instructions = (
        f"Handoff written to {resolved}. Run /new and reference this file "
        "(or paste its content) to continue in a fresh session."
    )
    return tool_result({
        "success": True,
        "path": resolved,
        "instructions": instructions,
    })


def handoff(args: dict, **kwargs) -> str:
    """Dispatch handler for the `handoff` tool. Only `action="write"` is supported —
    the schema enum permits no other value, but this defensive check keeps the
    handler correct even if something upstream constructs a call by hand."""
    action = args.get("action")
    if action != "write":
        return tool_error("Unsupported action for handoff tool; only 'write' is supported.")
    return _handoff_write(content=args.get("content", ""), path=args.get("path"))


registry.register(
    name="handoff",
    toolset="handoff",
    schema=HANDOFF_SCHEMA,
    handler=lambda args, **kw: handoff(args, **kw),
    emoji="📝",
)
