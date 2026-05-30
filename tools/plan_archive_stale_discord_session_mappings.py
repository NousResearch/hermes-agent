#!/usr/bin/env python3
"""Create a dry-run archive plan for stale Discord thread mappings."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.list_discord_thread_context_inventory import (  # noqa: E402
    _classify_db_stat_status,
    _entry_session_id,
    _entry_text,
    _read_db_stats,
    _read_sessions,
    _split_discord_display_name,
    _text_or_none,
    _thread_id_from_key,
    list_discord_thread_context_inventory,
)


WARNING = "dry-run metadata-only plan; not a repair, migration, restore, prune, or archive operation"
RECOMMENDED_ACTION = "archive_mapping_only_after_operator_approval"


def plan_archive_stale_discord_session_mappings(
    *,
    state_root: str | Path,
    sessions_json: str | Path | None = None,
    state_db: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return a metadata-only dry-run archive plan for stale Discord mappings."""
    source_root = Path(state_root)
    sessions_path = Path(sessions_json) if sessions_json else source_root / "sessions" / "sessions.json"
    state_db_path = Path(state_db) if state_db else source_root / "state.db"
    inventory = _inventory_for_paths(
        state_root=source_root,
        sessions_json=sessions_path,
        state_db=state_db_path,
    )
    stale_rows = [
        row for row in inventory.get("threads", [])
        if row.get("db_stat_status") == "mapped_session_absent_from_db"
    ]
    if limit is not None:
        stale_rows = stale_rows[: max(0, int(limit))]
    stale_mappings = [_stale_mapping(row) for row in stale_rows]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "warning": WARNING,
        "would_modify_state": False,
        "source_state_root": str(source_root),
        "source_sessions_json_path": str(sessions_path),
        "source_state_db_path": str(state_db_path),
        "total_mapped_discord_thread_sessions": inventory.get("total_discord_thread_sessions", 0),
        "db_status_counts": inventory.get("db_stat_status_counts", {}),
        "stale_mapping_count": inventory.get("db_stat_status_counts", {}).get(
            "mapped_session_absent_from_db",
            0,
        ),
        "included_stale_mapping_count": len(stale_mappings),
        "limit": limit,
        "stale_mappings": stale_mappings,
        "errors": inventory.get("errors", []),
    }


def _inventory_for_paths(
    *,
    state_root: Path,
    sessions_json: Path,
    state_db: Path,
) -> dict[str, Any]:
    default_sessions = state_root / "sessions" / "sessions.json"
    default_state_db = state_root / "state.db"
    if sessions_json == default_sessions and state_db == default_state_db:
        return list_discord_thread_context_inventory(
            state_root=state_root,
            limit=1_000_000,
        )

    return _inventory_from_explicit_paths(sessions_json=sessions_json, state_db=state_db)


def _inventory_from_explicit_paths(*, sessions_json: Path, state_db: Path) -> dict[str, Any]:
    sessions_report, entries, session_errors = _read_sessions(sessions_json)
    db_report, db_stats, db_errors = _read_db_stats(state_db)
    rows = []
    for session_key, entry in entries:
        thread_id = _thread_id_from_key(session_key)
        if not thread_id:
            continue
        mapped_session_id = _entry_session_id(entry)
        origin = entry.get("origin") if isinstance(entry, dict) else {}
        if not isinstance(origin, dict):
            origin = {}
        display_name = _entry_text(entry, "display_name") or _text_or_none(origin.get("chat_name"))
        _, _, thread_name = _split_discord_display_name(display_name)
        stats = db_stats.get(mapped_session_id or "", {})
        transcript_count = stats.get("transcript_message_count")
        status = _classify_db_stat_status(
            db_report=db_report,
            db_stats=db_stats,
            mapped_session_id=mapped_session_id,
            transcript_count=transcript_count,
        )
        rows.append(
            {
                "thread_id": thread_id,
                "expected_session_key": session_key,
                "mapped_session_id": mapped_session_id,
                "db_stat_status": status,
                "thread_name": thread_name,
                "display_name": display_name,
            }
        )
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("db_stat_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "sessions_json": sessions_report,
        "state_db": db_report,
        "total_discord_thread_sessions": len(rows),
        "db_stat_status_counts": status_counts,
        "threads": rows,
        "errors": session_errors + db_errors,
    }


def _stale_mapping(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "thread_id": row.get("thread_id"),
        "thread_name": row.get("thread_name"),
        "display_name": row.get("display_name"),
        "session_key": row.get("expected_session_key"),
        "mapped_stale_session_id": row.get("mapped_session_id"),
        "current_db_status": row.get("db_stat_status"),
        "backup_trace_status": None,
        "recommended_action": RECOMMENDED_ACTION,
    }


def _write_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a dry-run metadata-only stale Discord mapping archive plan.",
    )
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--sessions-json", type=Path)
    parser.add_argument("--state-db", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.write_plan and args.output is None:
        print("--output is required with --write-plan", file=sys.stderr)
        return 2

    plan = plan_archive_stale_discord_session_mappings(
        state_root=args.state_root,
        sessions_json=args.sessions_json,
        state_db=args.state_db,
        limit=args.limit,
    )
    output_plan = dict(plan)
    output_plan["plan_written"] = False
    output_plan["plan_output_path"] = str(args.output) if args.output else None
    if args.write_plan:
        _write_plan(args.output, plan)
        output_plan["plan_written"] = True

    if args.json:
        print(json.dumps(output_plan, indent=2, sort_keys=True))
    else:
        print(_format_table(output_plan))
    return 0


def _format_table(plan: dict[str, Any]) -> str:
    lines = [
        f"stale_mapping_count\t{plan.get('stale_mapping_count', 0)}",
        "thread_id\tmapped_stale_session_id\tthread_name\trecommended_action",
    ]
    for row in plan.get("stale_mappings", []):
        lines.append(
            "\t".join(
                str(row.get(key) or "")
                for key in (
                    "thread_id",
                    "mapped_stale_session_id",
                    "thread_name",
                    "recommended_action",
                )
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
