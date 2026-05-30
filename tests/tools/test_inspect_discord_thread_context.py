"""Tests for the metadata-only Discord thread context inspector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, build_session_key
from tools.inspect_discord_thread_context import (
    inspect_discord_thread_context,
    main,
)


THREAD_ID = "222222222222222222"
EXPECTED_THREAD_KEY = f"agent:main:discord:thread:{THREAD_ID}:{THREAD_ID}"
PRIVATE_TRANSCRIPT_TEXT = "private transcript text must not be printed"


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    yield store
    if store._db:
        store._db.close()


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=THREAD_ID,
        chat_type="thread",
        user_id="333333333333333333",
        thread_id=THREAD_ID,
        parent_chat_id="111111111111111111",
        guild_id="999999999999999999",
    )


def _create_thread_session(store: SessionStore) -> tuple[str, Path, Path, Path]:
    source = _source()
    assert build_session_key(source) == EXPECTED_THREAD_KEY
    entry = store.get_or_create_session(source)
    store.append_to_transcript(
        entry.session_id,
        {"role": "user", "content": PRIVATE_TRANSCRIPT_TEXT},
    )
    store.append_to_transcript(
        entry.session_id,
        {"role": "assistant", "content": "also private"},
    )
    if store._db:
        store._db.close()
    state_root = store.sessions_dir.parent
    return entry.session_id, state_root, store.sessions_dir / "sessions.json", store._db.db_path


def test_inspector_reports_mapped_thread_metadata_without_content(temp_store):
    session_id, state_root, _sessions_json, _state_db = _create_thread_session(temp_store)

    report = inspect_discord_thread_context(thread_id=THREAD_ID, state_root=state_root)
    payload = json.dumps(report)

    assert report["platform"] == "discord"
    assert report["chat_type"] == "thread"
    assert report["chat_id"] == THREAD_ID
    assert report["thread_id"] == THREAD_ID
    assert report["expected_session_key"] == EXPECTED_THREAD_KEY
    assert report["mapping"]["exists"] is True
    assert report["mapping"]["active_session_id"] == session_id
    assert report["sessions_json"]["exists"] is True
    assert report["active_session"]["message_count"] == 2
    assert report["active_session"]["transcript_message_count"] == 2
    assert report["active_session"]["last_transcript_timestamp"] is not None
    assert report["orphan_summary"]["exact_candidate_count"] == 0
    assert report["diagnostic"]["missing_mapping_diagnostic_would_fire"] is False
    assert PRIVATE_TRANSCRIPT_TEXT not in payload
    assert "also private" not in payload


def test_inspector_reports_missing_mapping_and_orphan_candidate_without_content(temp_store):
    session_id, state_root, sessions_json, _state_db = _create_thread_session(temp_store)
    sessions_json.unlink()

    report = inspect_discord_thread_context(thread_id=THREAD_ID, state_root=state_root)
    payload = json.dumps(report)

    assert report["mapping"]["exists"] is False
    assert report["mapping"]["status"] == "missing"
    assert report["active_session"] is None
    assert report["orphan_summary"]["exact_candidate_count"] == 1
    assert report["exact_orphan_sessions"][0]["session_id"] == session_id
    assert report["exact_orphan_sessions"][0]["message_count"] == 2
    assert report["exact_orphan_sessions"][0]["transcript_message_count"] == 2
    assert report["diagnostic"]["missing_mapping_diagnostic_would_fire"] is True
    assert PRIVATE_TRANSCRIPT_TEXT not in payload
    assert "also private" not in payload


def test_inspector_missing_paths_returns_structured_json_not_traceback(tmp_path):
    report = inspect_discord_thread_context(
        thread_id=THREAD_ID,
        state_root=tmp_path / "missing-root",
    )

    assert report["sessions_json"]["exists"] is False
    assert report["state_db"]["exists"] is False
    assert report["mapping"]["exists"] is False
    assert report["errors"] == []


def test_inspector_is_read_only(temp_store):
    _session_id, state_root, sessions_json, state_db = _create_thread_session(temp_store)
    before = {
        sessions_json: sessions_json.stat().st_mtime_ns,
        state_db: state_db.stat().st_mtime_ns,
    }

    inspect_discord_thread_context(thread_id=THREAD_ID, state_root=state_root)

    after = {
        sessions_json: sessions_json.stat().st_mtime_ns,
        state_db: state_db.stat().st_mtime_ns,
    }
    assert after == before


def test_cli_prints_json_report_without_content(temp_store, capsys):
    _session_id, state_root, _sessions_json, _state_db = _create_thread_session(temp_store)

    exit_code = main([
        "--thread-id",
        THREAD_ID,
        "--state-root",
        str(state_root),
        "--no-content",
    ])

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == 0
    assert payload["expected_session_key"] == EXPECTED_THREAD_KEY
    assert payload["mapping"]["exists"] is True
    assert PRIVATE_TRANSCRIPT_TEXT not in output
    assert "also private" not in output
