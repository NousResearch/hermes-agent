"""``session.list`` must forward the durable lineage root. Regression for #66663.

``hermes_state.list_sessions_rich`` already returns ``_lineage_root_id`` on projected
compression rows, and the REST ``/api/sessions`` path projects it
(``gateway/platforms/api_server.py::_session_response``). The ``session.list`` RPC row
summary dropped the field, so RPC consumers could not group a compressed conversation
with its continuation — the desktop store keys pinning and lineage dedup on it
(``apps/desktop/src/store/session.ts::sessionPinId``).
"""

import time

import pytest

import tui_gateway.server as srv
import tui_gateway.methods_session
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(srv, "_get_db", lambda: database)
    try:
        yield database
    finally:
        database.close()


def _call(method: str, params: dict) -> dict:
    return srv._methods[method](1, params)


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
    db.append_message("tip1", "user", "continuing")
    db._conn.commit()


def test_session_list_projects_lineage_root_id(db):
    """The RPC row summary must forward the projected tip's lineage root, like REST."""
    _seed_compressed_conversation(db)
    rows = _call("session.list", {})["result"]["sessions"]
    by_id = {row["id"]: row for row in rows}
    # The compressed root surfaces as its live continuation tip.
    assert "tip1" in by_id, rows
    # REST already projects this field; the RPC row summary must too (#66663).
    assert by_id["tip1"]["_lineage_root_id"] == "root1"
