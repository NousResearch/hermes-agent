"""Tests for metadata-only stale Discord mapping backup tracing."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tools.trace_discord_session_mappings_across_backups import (
    main,
    trace_discord_session_mappings_across_backups,
)


PRIVATE_TRANSCRIPT_TEXT = "backup trace must never print this transcript text"


def _write_sessions_json(state_root: Path, entries: dict[str, dict[str, object]]) -> None:
    sessions_dir = state_root / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "sessions.json").write_text(json.dumps(entries), encoding="utf-8")


def _thread_entry(thread_id: str, session_id: str, thread_name: str) -> dict[str, object]:
    key = f"agent:main:discord:thread:{thread_id}:{thread_id}"
    display_name = f"Fake Server / #general / {thread_name}"
    return {
        "session_key": key,
        "session_id": session_id,
        "created_at": "2026-05-30T00:00:00",
        "updated_at": "2026-05-30T00:00:00",
        "platform": "discord",
        "chat_type": "thread",
        "display_name": display_name,
        "origin": {
            "platform": "discord",
            "chat_id": thread_id,
            "chat_type": "thread",
            "thread_id": thread_id,
            "chat_name": display_name,
        },
    }


def _create_state_db(path: Path, *, sessions: list[str], messages: dict[str, int]) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, message_count INTEGER DEFAULT 0)")
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL)"
    )
    for session_id in sessions:
        conn.execute(
            "INSERT INTO sessions (id, message_count) VALUES (?, ?)",
            (session_id, messages.get(session_id, 0)),
        )
        for index in range(messages.get(session_id, 0)):
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, "user", f"{PRIVATE_TRANSCRIPT_TEXT} {index}", 1000 + index),
            )
    conn.commit()
    conn.close()


def test_traces_live_absent_mapping_to_backup_metadata_only(tmp_path):
    live_root = tmp_path / "live"
    live_root.mkdir()
    absent_key = "agent:main:discord:thread:200001:200001"
    present_key = "agent:main:discord:thread:200002:200002"
    _write_sessions_json(
        live_root,
        {
            absent_key: _thread_entry("200001", "old-session", "Thread From Backup"),
            present_key: _thread_entry("200002", "live-session", "Live Thread"),
        },
    )
    _create_state_db(live_root / "state.db", sessions=["live-session"], messages={"live-session": 1})
    backup_db = tmp_path / "backup.db"
    _create_state_db(backup_db, sessions=["old-session"], messages={"old-session": 2})

    report = trace_discord_session_mappings_across_backups(
        state_root=live_root,
        backup_state_dbs=[backup_db],
        limit=10,
    )
    payload = json.dumps(report)
    rows = {row["thread_id"]: row for row in report["threads"]}

    assert report["traced_count"] == 1
    assert report["found_in_any_backup_count"] == 1
    assert report["not_found_in_any_backup_count"] == 0
    assert rows["200001"]["mapped_session_id"] == "old-session"
    assert rows["200001"]["absent_from_live_db"] is True
    assert rows["200001"]["present_in_backup_db_count"] == 1
    assert rows["200001"]["present_in_backup_dbs"][0]["transcript_message_count"] == 2
    assert rows["200001"]["present_in_backup_dbs"][0]["last_transcript_timestamp"] == 1001
    assert PRIVATE_TRANSCRIPT_TEXT not in payload


def test_traces_absent_mapping_not_found_in_backup(tmp_path):
    live_root = tmp_path / "live"
    live_root.mkdir()
    key = "agent:main:discord:thread:200001:200001"
    _write_sessions_json(live_root, {key: _thread_entry("200001", "missing-session", "Missing")})
    _create_state_db(live_root / "state.db", sessions=[], messages={})
    backup_db = tmp_path / "backup.db"
    _create_state_db(backup_db, sessions=["other-session"], messages={"other-session": 1})

    report = trace_discord_session_mappings_across_backups(
        state_root=live_root,
        backup_state_dbs=[backup_db],
        limit=10,
    )
    row = report["threads"][0]

    assert report["found_in_any_backup_count"] == 0
    assert report["not_found_in_any_backup_count"] == 1
    assert row["present_in_backup_db_count"] == 0
    assert row["backup_checks"][0]["status"] == "session_absent"


def test_reports_backup_schema_gap_without_content(tmp_path):
    live_root = tmp_path / "live"
    live_root.mkdir()
    key = "agent:main:discord:thread:200001:200001"
    _write_sessions_json(live_root, {key: _thread_entry("200001", "old-session", "Missing")})
    _create_state_db(live_root / "state.db", sessions=[], messages={})
    backup_db = tmp_path / "backup.db"
    conn = sqlite3.connect(backup_db)
    conn.execute("CREATE TABLE unrelated (id TEXT)")
    conn.commit()
    conn.close()

    report = trace_discord_session_mappings_across_backups(
        state_root=live_root,
        backup_state_dbs=[backup_db],
        limit=10,
    )
    payload = json.dumps(report)

    assert report["backup_status_counts"] == {"session_table_missing": 1}
    assert report["threads"][0]["backup_checks"][0]["has_sessions_table"] is False
    assert PRIVATE_TRANSCRIPT_TEXT not in payload


def test_can_include_present_live_mappings(tmp_path):
    live_root = tmp_path / "live"
    live_root.mkdir()
    key = "agent:main:discord:thread:200001:200001"
    _write_sessions_json(live_root, {key: _thread_entry("200001", "live-session", "Live")})
    _create_state_db(live_root / "state.db", sessions=["live-session"], messages={"live-session": 1})

    report = trace_discord_session_mappings_across_backups(
        state_root=live_root,
        backup_state_dbs=[],
        limit=10,
        absent_only=False,
    )

    assert report["traced_count"] == 1
    assert report["threads"][0]["absent_from_live_db"] is False
    assert report["threads"][0]["live_db_stat_status"] == "matched_with_messages"


def test_cli_json_output_is_metadata_only(tmp_path, capsys):
    live_root = tmp_path / "live"
    live_root.mkdir()
    key = "agent:main:discord:thread:200001:200001"
    _write_sessions_json(live_root, {key: _thread_entry("200001", "old-session", "Missing")})
    _create_state_db(live_root / "state.db", sessions=[], messages={})
    backup_db = tmp_path / "backup.db"
    _create_state_db(backup_db, sessions=["old-session"], messages={"old-session": 1})

    exit_code = main([
        "--state-root",
        str(live_root),
        "--backup-state-db",
        str(backup_db),
        "--limit",
        "5",
        "--json",
    ])

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == 0
    assert payload["traced_count"] == 1
    assert PRIVATE_TRANSCRIPT_TEXT not in output


def test_tool_is_read_only(tmp_path):
    live_root = tmp_path / "live"
    live_root.mkdir()
    key = "agent:main:discord:thread:200001:200001"
    _write_sessions_json(live_root, {key: _thread_entry("200001", "old-session", "Missing")})
    _create_state_db(live_root / "state.db", sessions=[], messages={})
    backup_db = tmp_path / "backup.db"
    _create_state_db(backup_db, sessions=["old-session"], messages={"old-session": 1})
    watched = [
        live_root / "sessions" / "sessions.json",
        live_root / "state.db",
        backup_db,
    ]
    before = {path: path.stat().st_mtime_ns for path in watched}

    trace_discord_session_mappings_across_backups(
        state_root=live_root,
        backup_state_dbs=[backup_db],
        limit=10,
    )

    after = {path: path.stat().st_mtime_ns for path in watched}
    assert after == before
