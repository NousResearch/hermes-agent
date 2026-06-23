#!/usr/bin/env python3
"""Session Summary Tool — Running session summary for context reduction.

Maintains a compact running summary of the current session in
``~/.hermes/sessions/<session_id>/running_summary.md``.  Three actions:

  write   — overwrite with new content
  append  — add a timestamped entry (``## HH:MM\\ncontent\\n``)
  read    — return current summary (or empty string if none exists)

The agent writes incremental summaries every few turns so that when
context compression fires (or ``protect_last_n`` is set low), the
summary file preserves the session's key decisions, errors, and facts.
The agent reads it on demand instead of loading full transcripts.

Design:
- Plain markdown file — human-readable, survives compression, no schema
- Profile-safe via ``get_hermes_home()``
- No character limits — the agent is responsible for keeping it compact
- Append entries are timestamped for chronological context
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

SUMMARY_FILENAME = "running_summary.md"


def _resolve_summary_path(session_id: str) -> Path:
    """Return the absolute path to the running summary file for *session_id*."""
    return get_hermes_home() / "sessions" / session_id / SUMMARY_FILENAME


def _now_timestamp() -> str:
    """Return a compact timestamp for append entries (HH:MM format)."""
    return datetime.now().strftime("%H:%M")


def session_summary(
    action: str,
    session_id: Optional[str] = None,
    content: Optional[str] = None,
) -> str:
    """Maintain a running session summary.

    Args:
        action: One of ``write``, ``append``, ``read``.
        session_id: The session ID (required for all actions).
        content: The content to write or append (required for write/append).

    Returns:
        JSON string with ``success`` and either ``content`` (read) or
        ``path`` (write/append).  On error, ``success=False`` with ``error``.
    """
    if not session_id:
        return json.dumps({
            "success": False,
            "error": "session_id is required for all actions.",
        })

    path = _resolve_summary_path(session_id)

    if action == "read":
        if path.exists():
            text = path.read_text(encoding="utf-8")
            return json.dumps({"success": True, "content": text})
        return json.dumps({"success": True, "content": ""})

    if action == "write":
        if content is None:
            return json.dumps({
                "success": False,
                "error": "content is required for write action.",
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return json.dumps({"success": True, "path": str(path)})

    if action == "append":
        if content is None:
            return json.dumps({
                "success": False,
                "error": "content is required for append action.",
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = _now_timestamp()
        entry = f"\n## {ts}\n{content}\n"
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            # Avoid double newline when file is empty
            if existing.strip():
                entry = f"\n{entry}"
            else:
                entry = entry.lstrip()
            path.write_text(existing + entry, encoding="utf-8")
        else:
            path.write_text(entry.lstrip(), encoding="utf-8")
        return json.dumps({"success": True, "path": str(path)})

    return json.dumps({
        "success": False,
        "error": f"Unknown action: {action!r}. Valid actions: write, append, read.",
    })


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SESSION_SUMMARY_SCHEMA = {
    "name": "session_summary",
    "description": (
        "Maintain a compact running summary of the current session. "
        "Use this to preserve key decisions, errors found, files changed, "
        "and facts learned across turns — especially when context compression "
        "is configured with a low protect_last_n. The summary survives "
        "compression and can be read back on demand instead of loading full "
        "transcripts.\\n\\n"
        "ACTIONS:\\n"
        "- write: overwrite the summary with new content. Use for a full refresh.\\n"
        "- append: add a timestamped entry. Use for incremental updates every "
        "few turns.\\n"
        "- read: return the current summary. Use when you need context beyond "
        "the last few verbatim turns.\\n\\n"
        "Keep entries compact — this is a running summary, not a transcript. "
        "Focus on decisions, errors, changed files, and durable facts. Skip "
        "transient tool output and routine progress."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["write", "append", "read"],
                "description": "The action to perform: write (overwrite), append (add timestamped entry), read (return current summary).",
            },
            "session_id": {
                "type": "string",
                "description": "The session ID to read/write the summary for. Required for all actions.",
            },
            "content": {
                "type": "string",
                "description": "The content to write or append. Required for write and append actions.",
            },
        },
        "required": ["action", "session_id"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry, tool_error  # noqa: E402

registry.register(
    name="session_summary",
    toolset="session_search",
    schema=SESSION_SUMMARY_SCHEMA,
    handler=lambda args, **kw: session_summary(
        action=args.get("action", ""),
        session_id=args.get("session_id"),
        content=args.get("content"),
    ),
)
