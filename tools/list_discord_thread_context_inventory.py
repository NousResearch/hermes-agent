#!/usr/bin/env python3
"""Metadata-only inventory of Discord thread context mappings."""

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


def list_discord_thread_context_inventory(
    *,
    state_root: str | Path,
    limit: int = 100,
) -> dict[str, Any]:
    """Return mapped Discord thread sessions with metadata only."""
    root = Path(state_root)
    sessions_path = root / "sessions" / "sessions.json"
    state_db_path = root / "state.db"

    sessions_report, entries, session_errors = _read_sessions(sessions_path)
    db_report, db_stats, db_errors = _read_db_stats(state_db_path)

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
        server_name, channel_name, thread_name = _split_discord_display_name(display_name)
        stats = db_stats.get(mapped_session_id or "", {})
        transcript_count = stats.get("transcript_message_count")
        row = {
            "thread_id": thread_id,
            "expected_session_key": session_key,
            "mapped_session_id": mapped_session_id,
            "server_name": server_name,
            "guild_id": _text_or_none(origin.get("guild_id")),
            "channel_name": channel_name,
            "parent_chat_id": _text_or_none(origin.get("parent_chat_id")),
            "thread_name": thread_name,
            "display_name": display_name,
            "transcript_message_count": transcript_count,
            "last_transcript_timestamp": stats.get("last_transcript_timestamp"),
            "exact_orphan_candidate_count": 0,
            "missing_mapping_diagnostic_would_fire": False,
        }
        rows.append(row)

    rows.sort(key=lambda item: (item.get("last_transcript_timestamp") is None, -(item.get("last_transcript_timestamp") or 0)))
    total = len(rows)
    nonzero = sum(1 for row in rows if (row.get("transcript_message_count") or 0) > 0)
    zero = sum(1 for row in rows if row.get("transcript_message_count") == 0)
    missing_names = sum(1 for row in rows if not row.get("thread_name"))
    limited_rows = rows[: max(0, int(limit))]
    return {
        "sessions_json": sessions_report,
        "state_db": db_report,
        "total_discord_thread_sessions": total,
        "nonzero_transcript_sessions": nonzero,
        "zero_transcript_sessions": zero,
        "mapped_sessions_with_missing_names": missing_names,
        "limit": max(0, int(limit)),
        "threads": limited_rows,
        "errors": session_errors + db_errors,
    }


def _read_sessions(path: Path) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]], list[str]]:
    report = {
        "path": str(path),
        "exists": path.exists(),
        "loaded": False,
        "entry_count": 0,
    }
    if not path.exists():
        return report, [], []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return report, [], [f"sessions_json_unreadable: {exc}"]
    if not isinstance(data, dict):
        return report, [], ["sessions_json_invalid: expected object at top level"]
    report["loaded"] = True
    report["entry_count"] = len(data)
    entries = [
        (str(key), value)
        for key, value in data.items()
        if isinstance(value, dict) and _thread_id_from_key(str(key))
    ]
    return report, entries, []


def _read_db_stats(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    report = {
        "path": str(path),
        "exists": path.exists(),
        "available": False,
    }
    if not path.exists():
        return report, {}, []
    try:
        conn = _connect_readonly(path)
    except sqlite3.Error as exc:
        return report, {}, [f"state_db_unreadable: {exc}"]
    try:
        if not _table_exists(conn, "sessions"):
            return report, {}, ["state_db_missing_sessions_table"]
        report["available"] = True
        rows = conn.execute(
            """
            SELECT s.id,
                   COALESCE(s.message_count, 0) AS message_count,
                   COUNT(m.id) AS transcript_message_count,
                   MAX(m.timestamp) AS last_transcript_timestamp
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            GROUP BY s.id
            """
        ).fetchall() if _table_exists(conn, "messages") else conn.execute(
            "SELECT id, COALESCE(message_count, 0) AS message_count FROM sessions"
        ).fetchall()
        stats: dict[str, dict[str, Any]] = {}
        for row in rows:
            transcript_count = (
                int(row["transcript_message_count"] or 0)
                if "transcript_message_count" in row.keys()
                else int(row["message_count"] or 0)
            )
            stats[str(row["id"])] = {
                "message_count": int(row["message_count"] or 0),
                "transcript_message_count": transcript_count,
                "last_transcript_timestamp": (
                    row["last_transcript_timestamp"]
                    if "last_transcript_timestamp" in row.keys()
                    else None
                ),
            }
        return report, stats, []
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


def _thread_id_from_key(session_key: str) -> str | None:
    parts = session_key.split(":")
    if len(parts) >= 6 and parts[2] == "discord" and parts[3] == "thread":
        return parts[5]
    return None


def _entry_session_id(entry: dict[str, Any]) -> str | None:
    value = entry.get("session_id")
    return str(value) if value else None


def _entry_text(entry: dict[str, Any], key: str) -> str | None:
    return _text_or_none(entry.get(key))


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _split_discord_display_name(display_name: str | None) -> tuple[str | None, str | None, str | None]:
    if not display_name:
        return None, None, None
    parts = [part.strip() for part in display_name.split(" / ") if part.strip()]
    if len(parts) >= 3:
        return parts[0], parts[1], " / ".join(parts[2:])
    return None, None, display_name


def _format_table(report: dict[str, Any]) -> str:
    lines = [
        "thread_id\tmapped_session_id\ttranscript_message_count\tlast_transcript_timestamp\tthread_name"
    ]
    for row in report.get("threads", []):
        lines.append(
            "\t".join(
                str(row.get(key) or "")
                for key in (
                    "thread_id",
                    "mapped_session_id",
                    "transcript_message_count",
                    "last_transcript_timestamp",
                    "thread_name",
                )
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List Discord thread context routing inventory without content.",
    )
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = list_discord_thread_context_inventory(
        state_root=args.state_root,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
