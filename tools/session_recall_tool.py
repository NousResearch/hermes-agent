#!/usr/bin/env python3
"""
Session Recall Tool - Raw Transcript Access

Reads raw session JSONL files directly for full, uncompressed conversation
recall. Complements session_search (FTS5 + LLM summaries) by providing
exact transcript access without information loss.

Two modes:
  - conversation: Filters out tool calls/results/system messages, returning
    only human-readable user↔assistant exchanges. Best for context recovery.
  - full: Returns everything including tool calls, tool results, and system
    messages. Best for debugging or understanding exact tool usage.

Use cases:
  - Recovering context lost to compression in long sessions
  - Finding exact commands, error messages, or code from past sessions
  - Reviewing what tools were called and with what arguments
  - Cross-referencing when session_search summaries lack detail
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from hermes_constants import get_hermes_home


def _get_sessions_dir() -> Path:
    """Return the sessions directory path."""
    return Path(get_hermes_home()) / "sessions"


def _list_session_files(sessions_dir: Path, limit: int = 20) -> List[dict]:
    """List available session JSONL files, most recent first."""
    files = sorted(sessions_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    results = []
    for f in files[:limit]:
        # Parse session ID from filename: YYYYMMDD_HHMMSS_HASH.jsonl
        stem = f.stem
        try:
            date_part = stem[:8]
            time_part = stem[9:15]
            dt = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            date_str = "unknown"

        # Read first user message as preview
        preview = ""
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        msg = json.loads(line)
                        if msg.get("role") == "user" and msg.get("content"):
                            preview = msg["content"][:100]
                            break
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

        # Count lines
        try:
            with open(f, "r", encoding="utf-8") as fh:
                line_count = sum(1 for _ in fh)
        except OSError:
            line_count = 0

        results.append({
            "session_id": stem,
            "date": date_str,
            "preview": preview,
            "messages": line_count,
            "file": str(f),
        })
    return results


def _read_session_file(
    filepath: Path,
    mode: str = "conversation",
    tail: Optional[int] = None,
    search: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
) -> dict:
    """
    Read and filter a session JSONL file.

    Args:
        filepath: Path to the .jsonl file
        mode: 'conversation' (user+assistant text only) or 'full' (everything)
        tail: If set, return only the last N matching messages
        search: If set, only return messages containing this text (case-insensitive)
        offset: Skip first N matching messages (for pagination)
        limit: Max messages to return (default 100, max 500)
    """
    limit = max(1, min(limit, 500))
    messages = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                role = msg.get("role", "")

                # Mode filtering
                if mode == "conversation":
                    if role not in ("user", "assistant"):
                        continue
                    # Skip assistant messages that are only tool calls with no text
                    if role == "assistant":
                        content = msg.get("content") or ""
                        tool_calls = msg.get("tool_calls")
                        if not content.strip() and tool_calls:
                            continue

                # Search filtering
                if search:
                    content = msg.get("content") or ""
                    if search.lower() not in content.lower():
                        # Also check tool call names/args in full mode
                        if mode == "full":
                            tool_calls = msg.get("tool_calls", [])
                            tc_str = json.dumps(tool_calls) if tool_calls else ""
                            if search.lower() not in tc_str.lower():
                                continue
                        else:
                            continue

                # Build clean output message
                out = {"line": line_num, "role": role}

                content = msg.get("content")
                if content:
                    out["content"] = content

                timestamp = msg.get("timestamp")
                if timestamp:
                    out["timestamp"] = timestamp

                if mode == "full":
                    tool_calls = msg.get("tool_calls")
                    if tool_calls:
                        # Compact representation of tool calls
                        out["tool_calls"] = [
                            {
                                "name": (
                                    tc.get("function", {}).get("name")
                                    or tc.get("name", "?")
                                ),
                                "args": tc.get("function", {}).get("arguments", ""),
                            }
                            for tc in (tool_calls if isinstance(tool_calls, list) else [])
                        ]
                    tool_call_id = msg.get("tool_call_id")
                    if tool_call_id:
                        out["tool_call_id"] = tool_call_id
                    tool_name = msg.get("tool_name")
                    if tool_name:
                        out["tool_name"] = tool_name

                messages.append(out)

    except OSError as e:
        return {"success": False, "error": f"Cannot read file: {e}"}

    total_matching = len(messages)

    # Apply tail
    if tail and tail > 0:
        messages = messages[-tail:]
    else:
        # Apply offset/limit pagination
        messages = messages[offset: offset + limit]

    return {
        "success": True,
        "total_matching": total_matching,
        "returned": len(messages),
        "offset": offset if not tail else max(0, total_matching - (tail or 0)),
        "messages": messages,
    }


def session_recall(
    session_id: str = "",
    mode: str = "conversation",
    tail: int = None,
    search: str = None,
    offset: int = 0,
    limit: int = 100,
) -> str:
    """
    Read raw session transcripts from JSONL files.

    When called without session_id, lists available sessions.
    When called with session_id, reads the transcript with filtering.
    """
    sessions_dir = _get_sessions_dir()
    if not sessions_dir.exists():
        return json.dumps({
            "success": False,
            "error": f"Sessions directory not found: {sessions_dir}",
        })

    # List mode
    if not session_id or not session_id.strip():
        sessions = _list_session_files(sessions_dir, limit=20)
        return json.dumps({
            "success": True,
            "mode": "list",
            "sessions": sessions,
            "count": len(sessions),
            "sessions_dir": str(sessions_dir),
            "message": "Pass a session_id to read its transcript.",
        }, ensure_ascii=False)

    session_id = session_id.strip()

    # Find the file — support both exact filename and partial match
    candidates = list(sessions_dir.glob(f"{session_id}*.jsonl"))
    if not candidates:
        # Try fuzzy: search for the hash part
        candidates = list(sessions_dir.glob(f"*{session_id}*.jsonl"))

    if not candidates:
        return json.dumps({
            "success": False,
            "error": f"No session file found matching '{session_id}'",
        })

    if len(candidates) > 1:
        return json.dumps({
            "success": False,
            "error": f"Multiple sessions match '{session_id}': {[c.stem for c in candidates[:5]]}. Be more specific.",
        })

    filepath = candidates[0]

    # Validate mode
    if mode not in ("conversation", "full"):
        mode = "conversation"

    # Handle non-int params defensively
    if not isinstance(offset, int):
        try:
            offset = int(offset)
        except (TypeError, ValueError):
            offset = 0

    if not isinstance(limit, int):
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 100

    if tail is not None and not isinstance(tail, int):
        try:
            tail = int(tail)
        except (TypeError, ValueError):
            tail = None

    result = _read_session_file(
        filepath=filepath,
        mode=mode,
        tail=tail,
        search=search,
        offset=offset,
        limit=limit,
    )
    result["session_id"] = filepath.stem
    result["mode"] = mode

    return json.dumps(result, ensure_ascii=False)


def check_session_recall_requirements() -> bool:
    """Sessions directory must exist."""
    try:
        return _get_sessions_dir().exists()
    except Exception:
        return False


SESSION_RECALL_SCHEMA = {
    "name": "session_recall",
    "description": (
        "Read raw session transcripts from JSONL files for full, uncompressed conversation recall. "
        "Complements session_search (which returns LLM summaries) by providing exact transcript access.\n\n"
        "TWO MODES:\n"
        "1. List sessions (no session_id): Returns available sessions with dates and previews.\n"
        "2. Read transcript (with session_id): Returns the raw conversation with filtering options.\n\n"
        "FILTERING:\n"
        "- mode='conversation' (default): Only user↔assistant text — filters out tool calls, "
        "tool results, and system messages. Best for context recovery.\n"
        "- mode='full': Everything including tool calls and results. Best for debugging.\n"
        "- search: Only return messages containing specific text.\n"
        "- tail: Return only the last N messages (useful for recent context).\n"
        "- offset/limit: Pagination for large sessions.\n\n"
        "USE THIS when:\n"
        "- session_search summaries lack the detail you need\n"
        "- You need exact commands, code, or error messages from a past session\n"
        "- You need to recover context lost to compression\n"
        "- You want to see the full tool call sequence from a past session"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": (
                    "Session ID or partial match (e.g. '20260507_183403_d44b98' or just 'd44b98'). "
                    "Omit to list available sessions."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["conversation", "full"],
                "description": (
                    "'conversation' (default): user + assistant text only, filters out tool calls/system. "
                    "'full': everything including tool calls, tool results, system messages."
                ),
            },
            "search": {
                "type": "string",
                "description": "Filter messages containing this text (case-insensitive).",
            },
            "tail": {
                "type": "integer",
                "description": "Return only the last N matching messages.",
            },
            "offset": {
                "type": "integer",
                "description": "Skip first N matching messages for pagination (default: 0).",
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Max messages to return (default: 100, max: 500).",
                "default": 100,
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="session_recall",
    toolset="session_search",  # Same toolset — they're complementary
    schema=SESSION_RECALL_SCHEMA,
    handler=lambda args, **kw: session_recall(
        session_id=args.get("session_id", ""),
        mode=args.get("mode", "conversation"),
        tail=args.get("tail"),
        search=args.get("search"),
        offset=args.get("offset", 0),
        limit=args.get("limit", 100),
    ),
    check_fn=check_session_recall_requirements,
    emoji="📜",
)
