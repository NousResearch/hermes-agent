"""``session.archive`` refuses an unconfirmed lineage cascade (#70185).

The CTE flips the whole compression lineage, so archiving via RPC must name the blast
radius (error 4033 + the preview payload) unless the caller passes ``confirm_cascade``.
Unarchive and single-session archives keep working unchanged.
"""

import time

import pytest

import tui_gateway.server as srv
import tui_gateway.methods_session  # noqa: F401  (registers the RPC methods)
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


def _lineage(db: SessionDB) -> None:
    base = time.time() - 1000
    db.create_session("root", source="desktop")
    db.create_session("tip", source="desktop", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression' "
        "WHERE id = 'root'",
        (base, base + 10),
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'tip'", (base + 20,))
    db._conn.commit()


def test_cascade_archive_without_confirm_errors_with_preview(db):
    _lineage(db)

    envelope = _call("session.archive", {"session_id": "tip", "archived": True})

    assert envelope.get("error", {}).get("code") == 4033
    preview = envelope["error"]["data"]["preview"]
    assert preview["cascade_count"] == 2
    assert set(preview["affected_ids"]) == {"root", "tip"}
    assert db.get_session("root")["archived"] == 0  # refused: nothing flipped


def test_cascade_archive_with_confirm_archives_lineage(db):
    _lineage(db)

    envelope = _call(
        "session.archive", {"session_id": "tip", "archived": True, "confirm_cascade": True})

    assert "error" not in envelope, envelope
    assert envelope["result"]["archived"] is True
    assert db.get_session("root")["archived"] == 1


def test_single_session_archive_needs_no_confirm(db):
    db.create_session("solo", source="desktop")

    envelope = _call("session.archive", {"session_id": "solo", "archived": True})

    assert "error" not in envelope, envelope
    assert db.get_session("solo")["archived"] == 1


def test_unarchive_never_gated(db):
    _lineage(db)
    db.set_session_archived("root", True)

    envelope = _call("session.archive", {"session_id": "root", "archived": False})

    assert "error" not in envelope, envelope
    assert db.get_session("root")["archived"] == 0
