"""RPC-level tests for the generic hidden-session surface (tui_gateway).

Covers the two seams Bot Mode's "sessions are always hidden" policy leans on:

* ``session.set_hidden`` resolves a DURABLE stored session id when no live
  runtime session matches — plugins reconciling sessions they own (Bot Mode's
  hide sweep) hold stored ids for chats that aren't live right now. The old
  live-only lookup failed those with 4001 and the sweep silently no-opped.
* ``session.list`` honors ``include_hidden`` so owning surfaces (the Bots
  pane's per-profile browser) can still enumerate the rows they hid, while
  every default caller keeps the hidden rows dropped.
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


def test_set_hidden_resolves_stored_id_without_live_session(db):
    """A stored (non-live) session id must be hideable — the sweep path."""
    _seed(db, "stored-chat")
    assert srv._find_live_session_by_key("stored-chat") is None

    envelope = _call("session.set_hidden", {"session_id": "stored-chat", "hidden": True})
    assert "error" not in envelope, envelope
    assert envelope["result"]["hidden"] is True
    assert db.get_session("stored-chat")["hidden"] == 1

    # And back — unhide through the same durable path.
    envelope = _call("session.set_hidden", {"session_id": "stored-chat", "hidden": False})
    assert "error" not in envelope, envelope
    assert db.get_session("stored-chat")["hidden"] == 0


def test_set_hidden_unknown_id_still_errors(db):
    envelope = _call("session.set_hidden", {"session_id": "no-such-session", "hidden": True})
    assert envelope.get("error"), envelope


def test_session_list_include_hidden(db):
    _seed(db, "plain-chat")
    _seed(db, "bot-chat")
    assert db.set_session_hidden("bot-chat", True) is True

    default_rows = _call("session.list", {})["result"]["sessions"]
    assert {s["id"] for s in default_rows} == {"plain-chat"}

    all_rows = _call("session.list", {"include_hidden": True})["result"]["sessions"]
    assert {s["id"] for s in all_rows} == {"plain-chat", "bot-chat"}


def _live_session(key: str) -> dict:
    """Minimal live registry entry — `session.active_list` reads only these fields."""
    return {
        "agent": None, "created_at": 1_000.0, "history": [], "last_active": 1_000.0,
        "pending_hidden": False, "session_key": key, "source": "desktop",
    }


def _live_row(sid: str) -> dict:
    rows = {row["id"]: row for row in _call("session.active_list", {})["result"]["sessions"]}
    return rows[sid]


def test_set_hidden_on_a_live_session_with_a_row_reaches_the_live_payload(db):
    """#50799: hiding a session that ALREADY has a row must reach ``session.active_list``.

    Only the no-row branch wrote the runtime flag, so a live session hidden after its first prompt
    kept reporting ``hidden: false`` — and a client that excludes hidden sessions (the Desktop
    sidebar's live group) would list a session the product deliberately hides, the Bot Chat
    "unconditionally hidden" class. Regression guard for the row-exists flip.
    """
    _seed(db, "live-chat")
    srv._sessions["rt-hidden-after-row"] = _live_session("live-chat")
    try:
        envelope = _call("session.set_hidden", {"session_id": "rt-hidden-after-row", "hidden": True})
        assert "error" not in envelope, envelope
        assert db.get_session("live-chat")["hidden"] == 1
        assert _live_row("rt-hidden-after-row")["hidden"] is True

        # ...and the reverse flip (unhide) is reported too, so a live row never latches hidden.
        _call("session.set_hidden", {"session_id": "rt-hidden-after-row", "hidden": False})
        assert _live_row("rt-hidden-after-row")["hidden"] is False
    finally:
        srv._sessions.pop("rt-hidden-after-row", None)


def test_note_live_session_hidden_mirrors_a_store_only_hide(db):
    """A hide written straight to state.db — the dashboard's PATCH /api/sessions, the messaging
    API server — must reach a session that is live in this process: ``session.active_list`` answers
    from the runtime dict, so a store-only flip would leave it advertising itself as visible."""
    _seed(db, "patched-chat")
    srv._sessions["rt-patched"] = _live_session("patched-chat")
    try:
        assert db.set_session_hidden("patched-chat", True) is True
        assert _live_row("rt-patched")["hidden"] is False  # the store alone does not reach the payload

        srv.note_live_session_hidden("patched-chat", True)

        assert _live_row("rt-patched")["hidden"] is True
    finally:
        srv._sessions.pop("rt-patched", None)
