#!/usr/bin/env python3
"""
Tool Failure Logger — persistent, reviewable tool-error journal.

Failures are auto-logged by the agent loop (model_tools.py catch block),
so the agent only calls this tool to update a fix, link related failures,
review stats, or resolve entries.  Manual ``log`` is still available for
failures the auto-logger cannot see (e.g. semantic errors where the tool
returns a valid JSON but the content is wrong).

Storage: ``~/.hermes/tool_failures/failures.jsonl``

Actions:
  log     — manual failure record (auto-log handles most cases)
  update  — add fix description and/or link related failures to a record
  link    — link multiple failure records that share a root cause
  list    — paginate the failure log
  stats   — aggregate by tool + error category
  resolve — mark as fixed, wontfix, or blocked
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from tools._failure_log_store import (
    _MAX_ARGS_LEN,
    _MAX_ERROR_LEN,
    append_record,
    read_all,
    write_records,
)
from tools.registry import registry, tool_error  # noqa: F401

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100
_MAX_FIX_LEN = 300

_VALID_RESOLUTIONS = {"fixed", "wontfix", "blocked"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_ids(ids_raw: Any) -> set | None:
    """Parse ids arg into a set of ints, or None for 'all'."""
    if ids_raw == "all":
        return None
    if isinstance(ids_raw, list):
        target_ids = set()
        for v in ids_raw:
            try:
                target_ids.add(int(v))
            except (ValueError, TypeError):
                pass
        return target_ids if target_ids else None
    if ids_raw is not None:
        try:
            return {int(ids_raw)}
        except (ValueError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------


def _action_log(args: dict, session_id: str = "") -> str:
    tool = (args.get("tool") or "").strip()
    error = (args.get("error") or "").strip()
    if not tool:
        return json.dumps({"e": "tool is required"}, ensure_ascii=False)
    if not error:
        return json.dumps({"e": "error is required"}, ensure_ascii=False)

    args_summary = (args.get("args") or "")[:_MAX_ARGS_LEN]
    fix = (args.get("fix") or "")[:_MAX_FIX_LEN]

    prev_count = len(read_all())

    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "t": tool,
        "e": error[:_MAX_ERROR_LEN],
        "a": args_summary,
        "s": session_id or "",
        "f": fix,
        "l": [],
        "r": "pending",
    }
    # id is assigned atomically by append_record under lock
    saved = append_record(rec)
    return json.dumps({"ok": True, "id": saved["id"], "c": prev_count + 1}, ensure_ascii=False)


def _action_list(args: dict) -> str:
    status = (args.get("status") or "").strip()
    tool_filter = (args.get("tool") or "").strip()
    limit = min(int(args.get("limit", _DEFAULT_LIMIT) or _DEFAULT_LIMIT), _MAX_LIMIT)
    offset = max(int(args.get("offset", 0) or 0), 0)

    records = read_all()
    total = len(records)

    if status:
        records = [r for r in records if r.get("r") == status]
    if tool_filter:
        records = [r for r in records if r.get("t") == tool_filter]

    filtered_total = len(records)
    page = records[offset : offset + limit]

    items = []
    for r in page:
        items.append(
            {
                "id": r["id"],
                "ts": r["ts"],
                "t": r["t"],
                "e": r["e"],
                "a": r.get("a", ""),
                "f": r.get("f", ""),
                "l": r.get("l", []),
                "r": r.get("r", "pending"),
            }
        )

    return json.dumps(
        {
            "items": items,
            "c": filtered_total,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
        ensure_ascii=False,
    )


def _action_stats(args: dict) -> str:  # noqa: ARG001
    records = read_all()
    total = len(records)

    by_tool: Dict[str, Dict[str, Any]] = {}
    # Count by resolution state
    resolved_states: Dict[str, int] = {}
    pending = 0
    error_counts: Dict[str, int] = {}

    for r in records:
        t = r.get("t", "?")
        e = r.get("e", "?")
        rs = r.get("r", "pending")

        if t not in by_tool:
            by_tool[t] = {"c": 0, "fixed": 0, "pending": 0, "blocked": 0, "wontfix": 0, "top": {}}
        by_tool[t]["c"] += 1
        if rs in ("fixed", "wontfix", "blocked"):
            by_tool[t][rs] = by_tool[t].get(rs, 0) + 1
            resolved_states[rs] = resolved_states.get(rs, 0) + 1
        else:
            by_tool[t]["pending"] += 1
            pending += 1

        te_key = e[:60]
        by_tool[t]["top"][te_key] = by_tool[t]["top"].get(te_key, 0) + 1
        error_counts[te_key] = error_counts.get(te_key, 0) + 1

    for t in by_tool:
        by_tool[t]["top"] = dict(
            sorted(by_tool[t]["top"].items(), key=lambda x: x[1], reverse=True)[:3]
        )

    global_top = dict(
        sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    )

    resolved_total = sum(resolved_states.values())

    return json.dumps(
        {
            "total": total,
            "resolved": resolved_total,
            "pending": pending,
            "by_state": resolved_states,
            "tools": len(by_tool),
            "by_tool": by_tool,
            "top_errors": global_top,
        },
        ensure_ascii=False,
    )


def _action_resolve(args: dict) -> str:
    ids_raw = args.get("ids")
    resolution = (args.get("resolution") or "fixed").strip()

    if resolution not in _VALID_RESOLUTIONS:
        return json.dumps(
            {"e": f"resolution must be one of: {', '.join(sorted(_VALID_RESOLUTIONS))}"},
            ensure_ascii=False,
        )

    target_ids = _parse_ids(ids_raw)
    if target_ids is None and ids_raw != "all":
        return json.dumps({"e": "ids is required (list, int, or 'all')"}, ensure_ascii=False)

    status_filter = (args.get("status") or "").strip()
    tool_filter = (args.get("tool") or "").strip()

    records = read_all()
    updated = 0
    records.reverse()  # oldest first for rewrite

    for r in records:
        rid = r.get("id")
        if target_ids is not None and rid not in target_ids:
            continue
        if status_filter and r.get("r") != status_filter:
            continue
        if tool_filter and r.get("t") != tool_filter:
            continue
        r["r"] = resolution
        updated += 1

    if updated == 0:
        return json.dumps({"ok": True, "updated": 0, "hint": "no matching records"}, ensure_ascii=False)

    write_records(records)
    return json.dumps({"ok": True, "updated": updated, "resolution": resolution}, ensure_ascii=False)


def _action_update(args: dict) -> str:
    """Update fix description and/or linked IDs for existing records."""
    ids_raw = args.get("ids")
    target_ids = _parse_ids(ids_raw)
    if target_ids is None:
        return json.dumps({"e": "ids is required (int or list of ints)"}, ensure_ascii=False)

    fix = (args.get("fix") or "").strip()
    link_ids_raw = args.get("link_ids")

    # Parse link_ids
    new_links: List[int] = []
    if link_ids_raw is not None:
        if isinstance(link_ids_raw, list):
            for v in link_ids_raw:
                try:
                    new_links.append(int(v))
                except (ValueError, TypeError):
                    pass
        else:
            try:
                new_links.append(int(link_ids_raw))
            except (ValueError, TypeError):
                pass

    if not fix and not new_links:
        return json.dumps({"e": "provide fix and/or link_ids to update"}, ensure_ascii=False)

    records = read_all()
    updated = 0
    records.reverse()

    for r in records:
        rid = r.get("id")
        if rid not in target_ids:
            continue
        if fix:
            r["f"] = (r.get("f", "") + ("; " if r.get("f") else "") + fix)[:_MAX_FIX_LEN]
        if new_links:
            existing = set(r.get("l", []))
            existing.update(new_links)
            existing.discard(rid)  # don't link to self
            r["l"] = sorted(existing)
        updated += 1

    if updated == 0:
        return json.dumps({"ok": True, "updated": 0, "hint": "no matching records"}, ensure_ascii=False)

    write_records(records)
    return json.dumps({"ok": True, "updated": updated}, ensure_ascii=False)


def _action_link(args: dict) -> str:
    """Link multiple failures together (shared root cause)."""
    ids_raw = args.get("ids")
    if not isinstance(ids_raw, list) or len(ids_raw) < 2:
        return json.dumps({"e": "ids must be a list of 2+ ints"}, ensure_ascii=False)

    target_ids = set()
    for v in ids_raw:
        try:
            target_ids.add(int(v))
        except (ValueError, TypeError):
            pass

    if len(target_ids) < 2:
        return json.dumps({"e": "need at least 2 valid ids"}, ensure_ascii=False)

    records = read_all()
    records.reverse()
    updated = 0

    for r in records:
        rid = r.get("id")
        if rid not in target_ids:
            continue
        existing = set(r.get("l", []))
        # Link to all other targets
        others = target_ids - {rid}
        existing.update(others)
        r["l"] = sorted(existing)
        updated += 1

    if updated < 2:
        return json.dumps({"ok": True, "updated": updated, "hint": "fewer than 2 records matched"}, ensure_ascii=False)

    write_records(records)
    return json.dumps({"ok": True, "updated": updated, "linked": sorted(target_ids)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool entry point
# ---------------------------------------------------------------------------


def tool_failure_log(
    action: str = "",
    tool: str = "",
    error: str = "",
    args: str = "",
    fix: str = "",
    ids: Any = None,
    link_ids: Any = None,
    resolution: str = "fixed",
    status: str = "",
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
    session_id: str = "",
    task_id: Optional[str] = None,
) -> str:
    action = (action or "").strip().lower()

    payload = {
        "tool": tool,
        "error": error,
        "args": args,
        "fix": fix,
        "ids": ids,
        "link_ids": link_ids,
        "resolution": resolution,
        "status": status,
        "limit": limit,
        "offset": offset,
    }

    try:
        if action == "log":
            return _action_log(payload, session_id=session_id)
        elif action == "list":
            return _action_list(payload)
        elif action == "stats":
            return _action_stats(payload)
        elif action == "resolve":
            return _action_resolve(payload)
        elif action == "update":
            return _action_update(payload)
        elif action == "link":
            return _action_link(payload)
        else:
            return json.dumps(
                {"e": f"unknown action: '{action}'. Use: log, update, link, list, stats, resolve"},
                ensure_ascii=False,
            )
    except Exception as exc:
        return json.dumps({"e": f"internal error: {exc}"}, ensure_ascii=False)


def check_requirements() -> bool:
    return True


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

registry.register(
    name="tool_failure_log",
    toolset="tool_failure_log",
    schema={
        "name": "tool_failure_log",
        "description": "Log, update, link, list, and resolve tool failures. Failures are auto-logged — use this to add fixes (update), link related (link), review (list/stats), or resolve. Manual log for semantic failures only.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["log", "update", "link", "list", "stats", "resolve"],
                    "description": "log: record. update: add fix/links. link: connect failures. list/stats: review. resolve: mark fixed/wontfix/blocked.",
                },
                "tool": {
                    "type": "string",
                    "description": "Tool name (for log, or filter for list/resolve).",
                },
                "error": {
                    "type": "string",
                    "description": "Error summary (for log/update). Truncated to 256 chars.",
                },
                "args": {
                    "type": "string",
                    "description": "Key arguments summary (for log). Truncated to 200 chars.",
                },
                "fix": {
                    "type": "string",
                    "description": "Fix/workaround applied (for log/update). Truncated to 300 chars.",
                },
                "ids": {
                    "description": "Target ids: single int, list of ints, or 'all'.",
                },
                "link_ids": {
                    "description": "For update: list of related failure ids to link to this record.",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["fixed", "wontfix", "blocked"],
                    "description": "fixed: permanent fix shipped. wontfix: not worth fixing. blocked: waiting on external dependency.",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "fixed", "wontfix", "blocked"],
                    "description": "Filter by resolution status (for list or resolve with ids='all').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max records for list (default 20, max 100).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset for list (default 0).",
                },
            },
            "required": ["action"],
        },
    },
    handler=lambda args, **kw: tool_failure_log(
        action=args.get("action", ""),
        tool=args.get("tool", ""),
        error=args.get("error", ""),
        args=args.get("args", ""),
        fix=args.get("fix", ""),
        ids=args.get("ids"),
        link_ids=args.get("link_ids"),
        resolution=args.get("resolution", "fixed"),
        status=args.get("status", ""),
        limit=args.get("limit", _DEFAULT_LIMIT),
        offset=args.get("offset", 0),
        session_id=kw.get("session_id", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
    requires_env=[],
)
