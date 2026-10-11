"""``session.search``: TUI parity with the dashboard's ``/api/sessions/search``.

Covers the three lanes (id, FTS content, title) and the compression-lineage contract:
dedup keyed by lineage root, hits resolved to the live tip, ``_lineage_root_id`` on the row.
"""

import time

import pytest

import tui_gateway.server as srv
import tui_gateway.methods_session_search  # noqa: F401  (registers the RPC method)
from hermes_state import SessionDB
from tui_gateway.contracts import registry as _contracts


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(srv, "_get_db", lambda: database)
    try:
        yield database
    finally:
        database.close()


def _call(params: dict) -> dict:
    return srv._methods["session.search"](1, params)["result"]


def _seed_compressed_conversation(db) -> None:
    """``root1`` (compression-ended) -> ``tip1`` live continuation, both with messages."""
    t0 = time.time() - 3600
    db.create_session("root1", "cli")
    db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "root1"))
    db.append_message("root1", "user", "help me refactor auth")
    db._conn.execute(
        "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
        (t0 + 1800, "compression", "root1"),
    )
    db.create_session("tip1", "cli", parent_session_id="root1")
    db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 1801, "tip1"))
    db.append_message("tip1", "user", "continuing the refactor")
    db._conn.commit()


def test_blank_query_returns_empty_results(db):
    assert _call({"query": "   "}) == {"results": []}


def test_content_hit_resolves_lineage_tip_with_root(db):
    _seed_compressed_conversation(db)
    results = _call({"query": "refactor"})["results"]
    assert len(results) == 1, results
    row = results[0]
    # Both segments match "refactor*"; one row, carrying the live tip and the root.
    assert row["id"] == "tip1"
    assert row["_lineage_root_id"] == "root1"
    assert row["snippet"]
    assert row["role"] == "user"
    # The dispatcher validates results against the contract (ContractViolation under test
    # isolation); assert it here so a renamed row field fails this file, not an integration run.
    _contracts.check_result(_contracts.METHODS["session.search"], {"results": results})


def test_id_match_and_title_match_lanes(db):
    db.create_session("alp1", "cli")
    db.append_message("alp1", "user", "unrelated words")
    db.set_session_title("alp1", "quarterly planning")

    id_hits = _call({"query": "alp1"})["results"]
    assert [r["id"] for r in id_hits] == ["alp1"]

    # The FTS index only covers message content; "quarterly" lives solely in the title.
    title_hits = _call({"query": "quarterly"})["results"]
    assert [r["id"] for r in title_hits] == ["alp1"]
    assert title_hits[0]["title"] == "quarterly planning"


def test_results_newest_first_overriding_lane_order(db):
    t = time.time()
    db.create_session("old1", "cli")
    db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t - 100000, "old1"))
    db.append_message("old1", "user", "unrelated words")
    db.create_session("mid1", "cli")
    db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t - 50000, "mid1"))
    db.append_message("mid1", "user", "planning the old1 migration")
    db.create_session("new1", "cli")
    db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t - 1000, "new1"))
    db.append_message("new1", "user", "closing out old1 follow-ups")
    db._conn.commit()

    # The query is old1's id and doubles as the shared content keyword: search_sessions_by_id
    # matches the whole query string (a second token would stop the id lane matching at all)
    # and FTS5 MATCH is implicit-AND, so mid1/new1 carry the same token in their messages.
    # The id lane runs first and pins old1 at position 0 before the sort.
    results = _call({"query": "old1"})["results"]
    assert [r["id"] for r in results] == ["new1", "mid1", "old1"], results
    _contracts.check_result(_contracts.METHODS["session.search"], {"results": results})
