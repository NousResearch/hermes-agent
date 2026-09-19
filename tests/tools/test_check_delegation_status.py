"""Tests for check_delegation_status CLI subcommand.

TDD RED phase: these tests must fail before the implementation exists.

Design decision for Part B: CLI subcommand `hermes kanban delegation-status <id>`
that reads directly from the async_delegations SQLite table in state.db.

Rationale: The Footprint Ladder requires "CLI command + skill" before any new
model-facing tool. Parent orchestrators already call hermes via terminal; adding
a subcommand adds zero model-tool-schema surface. The durable table read bypasses
all LLM self-report and in-memory state, giving mechanical ground truth.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from tools import async_delegation as ad


# ---------------------------------------------------------------------------
# B: query_delegation_status reads the durable async_delegations table
# ---------------------------------------------------------------------------


def test_query_delegation_status_returns_none_for_unknown_id(tmp_path):
    """Missing delegation_id must return None, not raise."""
    db_path = str(tmp_path / "state.db")
    conn = sqlite3.connect(db_path)
    try:
        ad._initialize_schema(conn)
    finally:
        conn.close()
    ad._set_state_db_path_for_tests(db_path)
    result = ad.query_delegation_status("does-not-exist")
    assert result is None


def test_query_delegation_status_missing_database_is_an_error(tmp_path):
    """Observation must not create state.db while answering a read."""
    db_path = tmp_path / "missing" / "state.db"
    ad._set_state_db_path_for_tests(str(db_path))
    with pytest.raises(ad.DelegationStatusReadError):
        ad.query_delegation_status("does-not-exist")
    assert not db_path.exists()


def test_query_delegation_status_corrupt_database_is_an_error(tmp_path):
    """Corruption must not collapse into the same result as an absent id."""
    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"not a sqlite database")
    ad._set_state_db_path_for_tests(str(db_path))
    with pytest.raises(ad.DelegationStatusReadError):
        ad.query_delegation_status("does-not-exist")


def test_query_delegation_status_does_not_initialize_schema(tmp_path):
    """A valid SQLite DB without the table stays unchanged after a failed read."""
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sentinel (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    before = db_path.read_bytes()
    ad._set_state_db_path_for_tests(str(db_path))
    with pytest.raises(ad.DelegationStatusReadError):
        ad.query_delegation_status("does-not-exist")
    assert db_path.read_bytes() == before


def test_query_delegation_status_returns_row_for_known_id(tmp_path):
    """Inserting a row and querying by id must return the durable state."""
    db_path = str(tmp_path / "state.db")
    ad._set_state_db_path_for_tests(db_path)

    # Seed the table directly (simulating a completed delegation)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        ad._initialize_schema(conn)
        now = time.time()
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                state, dispatched_at, updated_at, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "test-delegation-abc",
                "session-key-1",
                "ui-session-1",
                "done",
                now - 60,
                now,
                json.dumps({"status": "done", "summary": "Implemented X"}),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    result = ad.query_delegation_status("test-delegation-abc")

    assert result is not None
    assert result["delegation_id"] == "test-delegation-abc"
    assert result["state"] == "done"


def test_query_delegation_status_returns_result_json_parsed(tmp_path):
    """result_json must be parsed and returned as a dict, not a raw string."""
    db_path = str(tmp_path / "state.db")
    ad._set_state_db_path_for_tests(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        ad._initialize_schema(conn)
        now = time.time()
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                state, dispatched_at, updated_at, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "test-delegation-xyz",
                "session-key-2",
                "ui-session-2",
                "done",
                now - 30,
                now,
                json.dumps({"status": "done", "summary": "Task result summary"}),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    result = ad.query_delegation_status("test-delegation-xyz")
    assert result is not None
    assert isinstance(result.get("result"), dict), (
        "result_json must be returned as a parsed dict, not a raw string"
    )
    assert result["result"]["summary"] == "Task result summary"


def test_query_delegation_status_independent_of_llm_self_report(tmp_path):
    """query_delegation_status must bypass LLM narrative (in-memory records).

    This is the core invariant: a parent can call this and get ground truth
    even when the LLM reports a different status or when _records is empty.
    """
    db_path = str(tmp_path / "state.db")
    ad._set_state_db_path_for_tests(db_path)

    # The in-memory _records dict (LLM self-report path) is empty — no dispatch
    # was made in this process. The durable table says the delegation is done.
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        ad._initialize_schema(conn)
        now = time.time()
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                state, dispatched_at, updated_at, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "recovered-delegation",
                "session-key-3",
                "ui-session-3",
                "done",
                now - 120,
                now,
                json.dumps({"status": "done", "summary": "Recovered result"}),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Confirm in-memory has nothing
    from tools.async_delegation import _records, _records_lock
    with _records_lock:
        assert "recovered-delegation" not in _records

    # Durable read must still return the ground-truth row
    result = ad.query_delegation_status("recovered-delegation")
    assert result is not None
    assert result["state"] == "done"


# ---------------------------------------------------------------------------
# B: CLI integration — hermes kanban delegation-status
# ---------------------------------------------------------------------------


def test_kanban_cli_delegation_status_subcommand_exists():
    """hermes kanban delegation-status must be a registered subcommand."""
    from hermes_cli.kanban import build_parser
    import argparse

    top = argparse.ArgumentParser()
    subs = top.add_subparsers(dest="cmd")
    build_parser(subs)

    # Parse the delegation-status subcommand
    try:
        args = top.parse_args(["kanban", "delegation-status", "test-id-123"])
    except SystemExit:
        pytest.fail(
            "'hermes kanban delegation-status <id>' subcommand is not registered"
        )

    # The parsed args should contain the delegation_id
    assert hasattr(args, "delegation_id") or hasattr(args, "kanban_action"), (
        "delegation-status subcommand must expose the delegation_id argument"
    )


def test_kanban_cli_delegation_status_returns_2_on_read_error(monkeypatch):
    """The CLI contract distinguishes verifier failure from a missing id."""
    import argparse
    from hermes_cli import kanban

    def fail(_delegation_id):
        raise ad.DelegationStatusReadError("read failed")

    monkeypatch.setattr(ad, "query_delegation_status", fail)
    args = argparse.Namespace(delegation_id="abc", json=False)
    assert kanban._cmd_delegation_status(args) == 2
