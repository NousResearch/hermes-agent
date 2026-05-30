#!/usr/bin/env python3
"""Trace stale Discord thread session mappings across backup DBs.

This tool is intentionally metadata-only. It never selects transcript content.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.list_discord_thread_context_inventory import (  # noqa: E402
    list_discord_thread_context_inventory,
)


def trace_discord_session_mappings_across_backups(
    *,
    state_root: str | Path,
    backup_state_dbs: list[str | Path],
    limit: int = 20,
    absent_only: bool = True,
) -> dict[str, Any]:
    """Return metadata-only backup presence for Discord thread session mappings."""
    normalized_backups = [Path(path) for path in backup_state_dbs]
    inventory = list_discord_thread_context_inventory(
        state_root=state_root,
        limit=1_000_000,
    )
    selected = _select_rows(
        inventory.get("threads", []),
        limit=max(0, int(limit)),
        absent_only=absent_only,
    )
    rows = [
        _trace_row(row, normalized_backups)
        for row in selected
    ]
    backup_presence_counts = _backup_presence_counts(rows)
    return {
        "state_root": str(Path(state_root)),
        "sessions_json": inventory.get("sessions_json"),
        "live_state_db": inventory.get("state_db"),
        "backup_state_dbs": [
            {
                "path": str(path),
                "exists": path.exists(),
            }
            for path in normalized_backups
        ],
        "input_total_discord_thread_sessions": inventory.get("total_discord_thread_sessions"),
        "input_db_stat_status_counts": inventory.get("db_stat_status_counts", {}),
        "absent_only": absent_only,
        "limit": max(0, int(limit)),
        "traced_count": len(rows),
        "found_in_any_backup_count": backup_presence_counts["found"],
        "not_found_in_any_backup_count": backup_presence_counts["not_found"],
        "backup_status_counts": backup_presence_counts["statuses"],
        "threads": rows,
        "errors": inventory.get("errors", []),
    }


def _select_rows(rows: list[dict[str, Any]], *, limit: int, absent_only: bool) -> list[dict[str, Any]]:
    filtered = [
        row for row in rows
        if not absent_only or row.get("db_stat_status") == "mapped_session_absent_from_db"
    ]
    return filtered[:limit]


def _trace_row(row: dict[str, Any], backup_state_dbs: list[Path]) -> dict[str, Any]:
    session_id = row.get("mapped_session_id")
    backup_matches = [
        _inspect_backup_db(path, str(session_id or ""))
        for path in backup_state_dbs
    ]
    present_matches = [
        match for match in backup_matches
        if match.get("session_row_exists") is True
    ]
    return {
        "thread_id": row.get("thread_id"),
        "expected_session_key": row.get("expected_session_key"),
        "mapped_session_id": session_id,
        "thread_name": row.get("thread_name"),
        "display_name": row.get("display_name"),
        "live_db_stat_status": row.get("db_stat_status"),
        "absent_from_live_db": row.get("db_stat_status") == "mapped_session_absent_from_db",
        "present_in_backup_db_count": len(present_matches),
        "present_in_backup_dbs": present_matches,
        "backup_checks": backup_matches,
    }


def _inspect_backup_db(path: Path, session_id: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "available": False,
        "has_sessions_table": False,
        "has_messages_table": False,
        "session_row_exists": False,
        "transcript_message_count": None,
        "last_transcript_timestamp": None,
        "status": "db_missing",
    }
    if not path.exists():
        return report
    try:
        conn = _connect_readonly(path)
    except sqlite3.Error as exc:
        report["status"] = "db_unreadable"
        report["error"] = str(exc)
        return report
    try:
        has_sessions = _table_exists(conn, "sessions")
        has_messages = _table_exists(conn, "messages")
        report["available"] = True
        report["has_sessions_table"] = has_sessions
        report["has_messages_table"] = has_messages
        if not has_sessions:
            report["status"] = "session_table_missing"
            return report
        session_row = conn.execute(
            "SELECT id FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if session_row is None:
            report["status"] = "session_absent"
            return report
        report["session_row_exists"] = True
        if not has_messages:
            report["status"] = "message_table_missing"
            return report
        aggregate = conn.execute(
            """
            SELECT COUNT(*) AS transcript_message_count,
                   MAX(timestamp) AS last_transcript_timestamp
            FROM messages
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        report["transcript_message_count"] = int(aggregate["transcript_message_count"] or 0)
        report["last_transcript_timestamp"] = aggregate["last_transcript_timestamp"]
        report["status"] = (
            "matched_with_messages"
            if report["transcript_message_count"] > 0
            else "matched_zero_messages"
        )
        return report
    finally:
        conn.close()


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _backup_presence_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    found = 0
    not_found = 0
    for row in rows:
        if row.get("present_in_backup_db_count", 0) > 0:
            found += 1
        else:
            not_found += 1
        for check in row.get("backup_checks", []):
            status = str(check.get("status") or "unknown")
            statuses[status] = statuses.get(status, 0) + 1
    return {
        "found": found,
        "not_found": not_found,
        "statuses": statuses,
    }


def _format_table(report: dict[str, Any]) -> str:
    lines = [
        "thread_id\tmapped_session_id\tlive_status\tbackup_count\tthread_name"
    ]
    for row in report.get("threads", []):
        lines.append(
            "\t".join(
                str(row.get(key) or "")
                for key in (
                    "thread_id",
                    "mapped_session_id",
                    "live_db_stat_status",
                    "present_in_backup_db_count",
                    "thread_name",
                )
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Trace Discord thread session mappings across explicit backup DBs without content.",
    )
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--backup-state-db", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--include-present", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = trace_discord_session_mappings_across_backups(
        state_root=args.state_root,
        backup_state_dbs=args.backup_state_db,
        limit=args.limit,
        absent_only=not args.include_present,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
