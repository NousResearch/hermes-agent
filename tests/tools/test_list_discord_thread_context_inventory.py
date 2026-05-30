"""Tests for the metadata-only Discord thread context inventory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from tools.list_discord_thread_context_inventory import (
    list_discord_thread_context_inventory,
    main,
)


PRIVATE_TRANSCRIPT_TEXT = "inventory must never print this transcript text"


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    yield store
    if store._db:
        store._db.close()


def _source(thread_id: str, *, chat_name: str | None, guild_id: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=thread_id,
        chat_type="thread",
        user_id="333333333333333333",
        thread_id=thread_id,
        parent_chat_id="111111111111111111",
        guild_id=guild_id,
        chat_name=chat_name,
    )


def _create_thread(
    store: SessionStore,
    thread_id: str,
    *,
    chat_name: str | None,
    message_count: int,
    guild_id: str | None = None,
) -> str:
    entry = store.get_or_create_session(
        _source(thread_id, chat_name=chat_name, guild_id=guild_id)
    )
    for index in range(message_count):
        store.append_to_transcript(
            entry.session_id,
            {"role": "user", "content": f"{PRIVATE_TRANSCRIPT_TEXT} {index}"},
        )
    return entry.session_id


def test_inventory_lists_mapped_threads_with_counts_and_names(temp_store):
    alpha_session = _create_thread(
        temp_store,
        "200001",
        chat_name="Fake Server / #general / Thread Alpha",
        message_count=2,
        guild_id="guild-1",
    )
    zero_session = _create_thread(
        temp_store,
        "200002",
        chat_name="Fake Server / #general / Thread Empty",
        message_count=0,
        guild_id="guild-1",
    )
    missing_name_session = _create_thread(
        temp_store,
        "200003",
        chat_name=None,
        message_count=1,
    )
    if temp_store._db:
        temp_store._db.close()

    report = list_discord_thread_context_inventory(state_root=temp_store.sessions_dir.parent)
    payload = json.dumps(report)
    rows = {row["thread_id"]: row for row in report["threads"]}

    assert report["total_discord_thread_sessions"] == 3
    assert report["nonzero_transcript_sessions"] == 2
    assert report["zero_transcript_sessions"] == 1
    assert rows["200001"]["mapped_session_id"] == alpha_session
    assert rows["200001"]["server_name"] == "Fake Server"
    assert rows["200001"]["channel_name"] == "#general"
    assert rows["200001"]["thread_name"] == "Thread Alpha"
    assert rows["200001"]["transcript_message_count"] == 2
    assert rows["200001"]["last_transcript_timestamp"] is not None
    assert rows["200001"]["exact_orphan_candidate_count"] == 0
    assert rows["200001"]["missing_mapping_diagnostic_would_fire"] is False
    assert rows["200002"]["mapped_session_id"] == zero_session
    assert rows["200002"]["transcript_message_count"] == 0
    assert rows["200002"]["last_transcript_timestamp"] is None
    assert rows["200003"]["mapped_session_id"] == missing_name_session
    assert rows["200003"]["server_name"] is None
    assert rows["200003"]["channel_name"] is None
    assert rows["200003"]["thread_name"] is None
    assert PRIVATE_TRANSCRIPT_TEXT not in payload


def test_inventory_limit_and_json_cli_output_without_content(temp_store, capsys):
    _create_thread(temp_store, "200001", chat_name="Fake / #general / A", message_count=1)
    _create_thread(temp_store, "200002", chat_name="Fake / #general / B", message_count=1)
    if temp_store._db:
        temp_store._db.close()

    exit_code = main([
        "--state-root",
        str(temp_store.sessions_dir.parent),
        "--json",
        "--limit",
        "1",
    ])

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == 0
    assert len(payload["threads"]) == 1
    assert payload["total_discord_thread_sessions"] == 2
    assert PRIVATE_TRANSCRIPT_TEXT not in output


def test_inventory_missing_paths_returns_structured_report(tmp_path):
    report = list_discord_thread_context_inventory(state_root=tmp_path / "missing")

    assert report["sessions_json"]["exists"] is False
    assert report["state_db"]["exists"] is False
    assert report["threads"] == []
    assert report["total_discord_thread_sessions"] == 0


def test_inventory_is_read_only(temp_store):
    _create_thread(temp_store, "200001", chat_name="Fake / #general / A", message_count=1)
    if temp_store._db:
        temp_store._db.close()
    sessions_json = temp_store.sessions_dir / "sessions.json"
    state_db = temp_store._db.db_path
    before = {
        sessions_json: sessions_json.stat().st_mtime_ns,
        state_db: state_db.stat().st_mtime_ns,
    }

    list_discord_thread_context_inventory(state_root=temp_store.sessions_dir.parent)

    after = {
        sessions_json: sessions_json.stat().st_mtime_ns,
        state_db: state_db.stat().st_mtime_ns,
    }
    assert after == before
