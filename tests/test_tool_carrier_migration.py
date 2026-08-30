"""Manifest-bounded quarantine tests for legacy leaked tool carriers."""

from __future__ import annotations

import copy
import hashlib
import sqlite3

import pytest

from agent.compaction_display import project_compaction_message_for_display
from hermes_cli.tool_carrier_migration import (
    INVENTORY_SCHEMA,
    LEGACY_CARRIER_METADATA_KEY,
    _online_backup,
    apply_manifest,
    build_inventory,
)
from hermes_state import SessionDB


LEAKED = '[Tool result call_demo]: {"output":"SYNTHETIC_PRIVATE_TOOL_PAYLOAD"}'


def _row(db: SessionDB, session_id: str, *, role: str, content: str, **extra) -> int:
    return int(db.append_message(session_id, role=role, content=content, **extra))


def _proven_manifest(inventory: dict, candidate: dict) -> dict:
    approved = copy.deepcopy(candidate)
    approved["disposition"] = "proven"
    approved["evidence_refs"] = [
        "receipt-message-sha256:synthetic",
        "operator-review:synthetic",
    ]
    return {
        "schema": INVENTORY_SCHEMA,
        "database": inventory["database"],
        "candidates": [approved],
    }


def test_inventory_marks_legacy_shape_ambiguous_without_reclassifying_user_text(
    tmp_path,
):
    db = SessionDB(tmp_path / "state.db")
    session_id = db.create_session("carrier-inventory", source="test")
    leaked_id = _row(db, session_id, role="assistant", content=LEAKED)
    _row(db, session_id, role="user", content=LEAKED)
    _row(db, session_id, role="assistant", content="ordinary assistant answer")

    inventory = build_inventory(tmp_path / "state.db")

    assert inventory["schema"] == INVENTORY_SCHEMA
    assert inventory["counts"] == {"ambiguous": 1, "excluded": 0}
    candidate = inventory["candidates"][0]
    assert candidate["id"] == leaked_id
    assert candidate["role"] == "assistant"
    assert candidate["disposition"] == "ambiguous"
    assert candidate["content_sha256"] == hashlib.sha256(LEAKED.encode()).hexdigest()
    assert "prefix shape is insufficient proof" in candidate["reason"]


def test_apply_requires_manifest_proof_and_is_idempotent(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    session_id = db.create_session("carrier-apply", source="test")
    row_id = _row(db, session_id, role="assistant", content=LEAKED)
    inventory = build_inventory(path)
    candidate = inventory["candidates"][0]

    dry = apply_manifest(
        path,
        {
            "schema": INVENTORY_SCHEMA,
            "database": inventory["database"],
            "candidates": [candidate],
        },
        dry_run=True,
    )
    assert dry == {"changed": 0, "unchanged": 0, "skipped": 1}

    manifest = _proven_manifest(inventory, candidate)
    assert apply_manifest(path, manifest, dry_run=True) == {
        "changed": 0,
        "unchanged": 0,
        "skipped": 0,
        "would_change": 1,
    }
    assert apply_manifest(path, manifest, dry_run=False) == {
        "changed": 1,
        "unchanged": 0,
        "skipped": 0,
    }
    assert apply_manifest(path, manifest, dry_run=False) == {
        "changed": 0,
        "unchanged": 1,
        "skipped": 0,
    }

    row = next(
        message for message in db.get_messages(session_id) if message["id"] == row_id
    )
    assert row["content"] == LEAKED
    assert row["display_kind"] == "hidden"
    marker = row["display_metadata"][LEGACY_CARRIER_METADATA_KEY]
    assert marker["original_content_sha256"] == candidate["content_sha256"]
    assert project_compaction_message_for_display(row) is None


def test_apply_rejects_duplicate_candidates_without_mutating_a_row(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    session_id = db.create_session("carrier-duplicate", source="test")
    row_id = _row(db, session_id, role="assistant", content=LEAKED)
    inventory = build_inventory(path)
    manifest = _proven_manifest(inventory, inventory["candidates"][0])
    manifest["candidates"].append(copy.deepcopy(manifest["candidates"][0]))

    with pytest.raises(ValueError, match="duplicate candidate row"):
        apply_manifest(path, manifest, dry_run=False)

    row = next(
        message for message in db.get_messages(session_id) if message["id"] == row_id
    )
    assert row["display_kind"] is None
    assert row["content"] == LEAKED


def test_manifest_rejects_same_shape_database_replacement(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    session_id = db.create_session("carrier-snapshot", source="test")
    _row(db, session_id, role="assistant", content=LEAKED)
    inventory = build_inventory(path)
    manifest = _proven_manifest(inventory, inventory["candidates"][0])
    db.close()

    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE messages SET content = ? WHERE session_id = ?",
            (LEAKED + " changed", session_id),
        )
        conn.commit()

    with pytest.raises(ValueError, match="database descriptor does not match"):
        apply_manifest(path, manifest, dry_run=True)


def test_online_backup_syncs_backup_file_and_parent_directory(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    destination = tmp_path / "backup" / "state.db"
    db = SessionDB(source)
    db.create_session("backup-proof", source="test")
    db.close()
    fsync_calls = []
    original_fsync = __import__("os").fsync

    def tracked_fsync(fd):
        fsync_calls.append(fd)
        original_fsync(fd)

    monkeypatch.setattr("hermes_cli.tool_carrier_migration.os.fsync", tracked_fsync)
    _online_backup(source, destination)

    assert destination.is_file()
    assert len(fsync_calls) == 2


def test_apply_rejects_content_drift_without_mutating_a_row(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    session_id = db.create_session("carrier-drift", source="test")
    row_id = _row(db, session_id, role="assistant", content=LEAKED)
    inventory = build_inventory(path)
    manifest = _proven_manifest(inventory, inventory["candidates"][0])
    manifest["candidates"][0]["content_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="content hash mismatch"):
        apply_manifest(path, manifest, dry_run=False)

    row = next(
        message for message in db.get_messages(session_id) if message["id"] == row_id
    )
    assert row["display_kind"] is None
    assert row["content"] == LEAKED
