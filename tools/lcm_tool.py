"""Read-only Hermes LCM recall tools.

These wrap scripts/hermes_lcm.py without enabling a context engine or mutating
state.db. All handlers return JSON strings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from tools.registry import registry

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import hermes_lcm  # type: ignore  # noqa: E402


def check_lcm_requirements() -> bool:
    return hermes_lcm._state_db_path().exists()


def _ns(**kwargs: Any):
    import argparse
    return argparse.Namespace(**kwargs)


def _ok(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _status(args: dict[str, Any], **_: Any) -> str:
    return _ok(hermes_lcm.status(_ns()))


def _grep(args: dict[str, Any], **_: Any) -> str:
    q = str(args.get("query") or "").strip()
    if not q:
        return _ok({"error": "query is required"})
    return _ok(hermes_lcm.grep(_ns(
        query=q,
        all_sessions=True,
        session=args.get("session_id"),
        session_id=args.get("session_id"),
        role=args.get("role"),
        tool_name=args.get("tool_name"),
        since=args.get("since"),
        before=args.get("before"),
        sort=args.get("sort") or "rank",
        limit=int(args.get("limit") or 5),
        max_chars=int(args.get("max_chars") or 800),
    )))


def _describe(args: dict[str, Any], **_: Any) -> str:
    mid = args.get("message_id")
    if mid is not None:
        try:
            mid = int(mid)
        except Exception:
            return _ok({"error": "message_id must be an integer"})
    return _ok(hermes_lcm.describe(_ns(
        message_id=mid,
        session_id=args.get("session_id"),
        tail=args.get("tail"),
        around=args.get("around"),
        window=int(args.get("window") or 3),
        max_chars=int(args.get("max_chars") or 1200),
    )))


def _recall(args: dict[str, Any], **_: Any) -> str:
    q = str(args.get("query") or "").strip()
    if not q:
        return _ok({"error": "query is required"})
    return _ok(hermes_lcm.recall(_ns(
        query=q,
        prompt=args.get("prompt") or "",
        all_sessions=True,
        session=args.get("session_id"),
        role=args.get("role"),
        tool_name=args.get("tool_name"),
        since=args.get("since"),
        before=args.get("before"),
        sort=args.get("sort") or "rank",
        limit=int(args.get("limit") or 10),
        per_session=int(args.get("per_session") or 3),
        window=int(args.get("window") or 1),
        format=args.get("format") or "compact",
        max_chars=int(args.get("max_chars") or 800),
    )))


LCM_STATUS_SCHEMA = {
    "name": "lcm_status",
    "description": "Read-only status for Hermes LCM recall over ~/.hermes/state.db: DB path, counts, FTS availability.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}

LCM_GREP_SCHEMA = {
    "name": "lcm_grep",
    "description": "Read-only bounded search over Hermes session messages. Use for exact old commands, errors, PER IDs, SHAs, and tool outputs.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "session_id": {"type": "string"},
            "role": {"type": "string", "enum": ["system", "user", "assistant", "tool"]},
            "tool_name": {"type": "string"},
            "since": {"type": "string", "description": "e.g. 7d, 12h, ISO datetime, or unix timestamp"},
            "before": {"type": "string"},
            "sort": {"type": "string", "enum": ["rank", "time"], "default": "rank"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
            "max_chars": {"type": "integer", "minimum": 100, "maximum": 6000, "default": 800},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

LCM_DESCRIBE_SCHEMA = {
    "name": "lcm_describe",
    "description": "Read-only bounded context around a message_id, or latest/around messages in a session.",
    "parameters": {
        "type": "object",
        "properties": {
            "message_id": {"type": "integer"},
            "session_id": {"type": "string"},
            "tail": {"type": "integer", "minimum": 1, "maximum": 100},
            "around": {"type": "string"},
            "window": {"type": "integer", "minimum": 0, "maximum": 20, "default": 3},
            "max_chars": {"type": "integer", "minimum": 100, "maximum": 8000, "default": 1200},
        },
        "additionalProperties": False,
    },
}

LCM_RECALL_SCHEMA = {
    "name": "lcm_recall",
    "description": "Read-only grouped extractive recall evidence from Hermes history. No LLM call, no DB writes.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "prompt": {"type": "string"},
            "session_id": {"type": "string"},
            "role": {"type": "string", "enum": ["system", "user", "assistant", "tool"]},
            "tool_name": {"type": "string"},
            "since": {"type": "string"},
            "before": {"type": "string"},
            "sort": {"type": "string", "enum": ["rank", "time"], "default": "rank"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            "per_session": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
            "window": {"type": "integer", "minimum": 0, "maximum": 5, "default": 1},
            "format": {"type": "string", "enum": ["compact", "evidence", "full-json"], "default": "compact"},
            "max_chars": {"type": "integer", "minimum": 100, "maximum": 6000, "default": 800},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

LCM_EXPAND_SCHEMA = {**LCM_RECALL_SCHEMA, "name": "lcm_expand_query", "description": "Alias for lcm_recall: extractive cited recall evidence."}

registry.register(
    name="lcm_status",
    toolset="lcm",
    schema=LCM_STATUS_SCHEMA,
    handler=_status,
    check_fn=check_lcm_requirements,
    description=LCM_STATUS_SCHEMA["description"],
    emoji="🧠",
    max_result_size_chars=50_000,
)
registry.register(
    name="lcm_grep",
    toolset="lcm",
    schema=LCM_GREP_SCHEMA,
    handler=_grep,
    check_fn=check_lcm_requirements,
    description=LCM_GREP_SCHEMA["description"],
    emoji="🔎",
    max_result_size_chars=50_000,
)
registry.register(
    name="lcm_describe",
    toolset="lcm",
    schema=LCM_DESCRIBE_SCHEMA,
    handler=_describe,
    check_fn=check_lcm_requirements,
    description=LCM_DESCRIBE_SCHEMA["description"],
    emoji="📜",
    max_result_size_chars=50_000,
)
registry.register(
    name="lcm_recall",
    toolset="lcm",
    schema=LCM_RECALL_SCHEMA,
    handler=_recall,
    check_fn=check_lcm_requirements,
    description=LCM_RECALL_SCHEMA["description"],
    emoji="🧠",
    max_result_size_chars=50_000,
)
registry.register(
    name="lcm_expand_query",
    toolset="lcm",
    schema=LCM_EXPAND_SCHEMA,
    handler=_recall,
    check_fn=check_lcm_requirements,
    description=LCM_EXPAND_SCHEMA["description"],
    emoji="🧠",
    max_result_size_chars=50_000,
)
