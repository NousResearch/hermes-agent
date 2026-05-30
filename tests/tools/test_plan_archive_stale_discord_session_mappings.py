"""Tests for the dry-run stale Discord mapping archive planner."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tools.plan_archive_stale_discord_session_mappings import (
    main,
    plan_archive_stale_discord_session_mappings,
)


PRIVATE_TRANSCRIPT_TEXT = "archive planner must never print this transcript text"


def _write_sessions_json(path: Path, entries: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")


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


def _temp_state(tmp_path: Path) -> Path:
    state_root = tmp_path / "state"
    sessions_json = state_root / "sessions" / "sessions.json"
    _write_sessions_json(
        sessions_json,
        {
            "agent:main:discord:thread:200001:200001": _thread_entry(
                "200001", "session-with-messages", "Has Messages"
            ),
            "agent:main:discord:thread:200002:200002": _thread_entry(
                "200002", "session-zero", "Zero Messages"
            ),
            "agent:main:discord:thread:200003:200003": _thread_entry(
                "200003", "stale-session", "Stale Mapping"
            ),
        },
    )
    _create_state_db(
        state_root / "state.db",
        sessions=["session-with-messages", "session-zero"],
        messages={"session-with-messages": 2, "session-zero": 0},
    )
    return state_root


def test_plan_includes_only_mapped_absent_sessions(tmp_path):
    state_root = _temp_state(tmp_path)

    plan = plan_archive_stale_discord_session_mappings(state_root=state_root)
    payload = json.dumps(plan)
    mappings = plan["stale_mappings"]

    assert plan["total_mapped_discord_thread_sessions"] == 3
    assert plan["db_status_counts"] == {
        "matched_with_messages": 1,
        "matched_zero_messages": 1,
        "mapped_session_absent_from_db": 1,
    }
    assert plan["stale_mapping_count"] == 1
    assert mappings == [
        {
            "thread_id": "200003",
            "thread_name": "Stale Mapping",
            "display_name": "Fake Server / #general / Stale Mapping",
            "session_key": "agent:main:discord:thread:200003:200003",
            "mapped_stale_session_id": "stale-session",
            "current_db_status": "mapped_session_absent_from_db",
            "backup_trace_status": None,
            "recommended_action": "archive_mapping_only_after_operator_approval",
        }
    ]
    assert PRIVATE_TRANSCRIPT_TEXT not in payload


def test_plan_accepts_explicit_sessions_json_and_state_db_paths(tmp_path):
    state_root = _temp_state(tmp_path)

    plan = plan_archive_stale_discord_session_mappings(
        state_root=tmp_path / "unused",
        sessions_json=state_root / "sessions" / "sessions.json",
        state_db=state_root / "state.db",
    )

    assert plan["source_state_root"] == str(tmp_path / "unused")
    assert plan["source_sessions_json_path"] == str(state_root / "sessions" / "sessions.json")
    assert plan["source_state_db_path"] == str(state_root / "state.db")
    assert plan["stale_mapping_count"] == 1


def test_plan_contains_review_warning_and_no_repair_claim(tmp_path):
    state_root = _temp_state(tmp_path)

    plan = plan_archive_stale_discord_session_mappings(state_root=state_root)

    assert plan["created_at"]
    assert plan["warning"] == (
        "dry-run metadata-only plan; not a repair, migration, restore, prune, or archive operation"
    )
    assert plan["would_modify_state"] is False


def test_cli_without_write_plan_writes_no_files(tmp_path, capsys):
    state_root = _temp_state(tmp_path)
    output_path = tmp_path / "plan.json"

    exit_code = main([
        "--state-root",
        str(state_root),
        "--json",
        "--limit",
        "10",
        "--output",
        str(output_path),
    ])

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == 0
    assert payload["stale_mapping_count"] == 1
    assert not output_path.exists()
    assert PRIVATE_TRANSCRIPT_TEXT not in output


def test_cli_write_plan_writes_only_requested_output_path(tmp_path, capsys):
    state_root = _temp_state(tmp_path)
    output_path = tmp_path / "review" / "plan.json"
    before_state_files = {
        state_root / "sessions" / "sessions.json": (state_root / "sessions" / "sessions.json").stat().st_mtime_ns,
        state_root / "state.db": (state_root / "state.db").stat().st_mtime_ns,
    }

    exit_code = main([
        "--state-root",
        str(state_root),
        "--json",
        "--write-plan",
        "--output",
        str(output_path),
    ])

    output = capsys.readouterr().out
    payload = json.loads(output)
    after_state_files = {
        state_root / "sessions" / "sessions.json": (state_root / "sessions" / "sessions.json").stat().st_mtime_ns,
        state_root / "state.db": (state_root / "state.db").stat().st_mtime_ns,
    }
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["plan_written"] is True
    assert payload["plan_output_path"] == str(output_path)
    assert written["stale_mapping_count"] == 1
    assert after_state_files == before_state_files
    assert PRIVATE_TRANSCRIPT_TEXT not in output
    assert PRIVATE_TRANSCRIPT_TEXT not in output_path.read_text(encoding="utf-8")


def test_cli_requires_output_when_write_plan_is_set(tmp_path, capsys):
    state_root = _temp_state(tmp_path)

    exit_code = main([
        "--state-root",
        str(state_root),
        "--write-plan",
    ])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--output is required with --write-plan" in captured.err
