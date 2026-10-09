"""RPC-level tests for ``session.archive`` (tui_gateway, #47168).

Desktop archives sessions via ``PATCH /api/sessions/{id}``; the TUI gateway had no
equivalent JSON-RPC method, so TUI clients had to keep every session visible or
delete it permanently. ``session.archive`` mirrors ``session.set_hidden``:

* a DURABLE stored session id / key archives and un-archives through the profile
  db (``SessionDB.set_session_archived``, whole compression lineage);
* a LIVE runtime session id archives its stored key — and a live session with
  NO row yet defers via ``pending_archived`` so ``_ensure_session_db_row`` applies
  the flag when the first prompt persists the row;
* an unknown id still errors.
"""

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


def _seed(db, sid: str) -> None:
    db.create_session(sid, source="desktop")
    db._conn.execute("UPDATE sessions SET message_count = 1 WHERE id = ?", (sid,))
    db._conn.commit()


def test_archive_resolves_stored_id_without_live_session(db):
    """A stored (non-live) session id must be archivable — the session.list picker path."""
    _seed(db, "stored-chat")
    assert srv._find_live_session_by_key("stored-chat") is None

    envelope = _call("session.archive", {"session_id": "stored-chat", "archived": True})
    assert "error" not in envelope, envelope
    assert envelope["result"]["archived"] is True
    assert envelope["result"]["session_key"] == "stored-chat"
    assert db.get_session("stored-chat")["archived"] == 1

    # And back — unarchive through the same durable path.
    envelope = _call("session.archive", {"session_id": "stored-chat", "archived": False})
    assert "error" not in envelope, envelope
    assert envelope["result"]["archived"] is False
    assert db.get_session("stored-chat")["archived"] == 0


def test_archive_accepts_session_key_alias(db):
    """``session_key`` is the documented alias of ``session_id`` (the list rows carry it)."""
    _seed(db, "keyed-chat")
    envelope = _call("session.archive", {"session_key": "keyed-chat", "archived": True})
    assert "error" not in envelope, envelope
    assert db.get_session("keyed-chat")["archived"] == 1


def test_archive_unknown_id_still_errors(db):
    envelope = _call("session.archive", {"session_id": "no-such-session", "archived": True})
    assert envelope.get("error"), envelope


def test_archive_through_the_dispatcher_validates_params(db):
    """The wire path: unknown params are rejected by the contract, and a missing id answers 4006."""
    resp = srv.handle_request({"id": "1", "method": "session.archive", "params": {}})
    assert resp["error"]["code"] == 4006

    resp = srv.handle_request(
        {"id": "2", "method": "session.archive", "params": {"session_id": "x", "bogus": 1}})
    assert resp["error"]["code"] == 4000


def test_archive_live_runtime_session_defers_until_row_exists(db):
    """A live session with no state.db row yet queues ``pending_archived``; the row is born archived."""
    sid = "live-draft"
    srv._sessions[sid] = {"session_key": "20260927_000000_draft", "history": [], "running": False}
    try:
        envelope = _call("session.archive", {"session_id": sid, "archived": True})
        assert "error" not in envelope, envelope
        assert envelope["result"]["session_key"] == "20260927_000000_draft"
        assert db.get_session("20260927_000000_draft") is None  # no row yet

        # The first prompt persists the row; the deferred intent must land with it.
        assert srv._ensure_session_db_row(srv._sessions[sid])
        row = db.get_session("20260927_000000_draft")
        assert row is not None
        assert row["archived"] == 1
    finally:
        srv._sessions.pop(sid, None)
