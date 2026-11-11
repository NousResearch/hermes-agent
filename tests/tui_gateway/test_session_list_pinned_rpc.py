"""Legacy ``session.list`` keeps pinned conversations that aged out of the recency page.

A pin is an explicit keep: REST ``/api/sessions`` and the desktop sidebar back-fill pinned rows
the page missed (``include_pinned``). This handler neither asked for them nor kept them past its
own ``[:limit]``, so a pinned run vanished from Sparky's phone list once newer sessions arrived.
"""

import time

import pytest

import tui_gateway.methods_session  # registers session.list
import tui_gateway.server as srv
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(srv, "_get_db", lambda: database)
    try:
        yield database
    finally:
        database.close()


def _session(db, sid, started):
    db.create_session(sid, source="tui")
    db.append_message(session_id=sid, role="user", content=f"hello from {sid}")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (started, sid))
    db._conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = ?", (started, sid))


def test_pinned_session_older_than_the_page_is_listed(db):
    base = time.time() - 100_000
    _session(db, "pinned_old", base)
    _session(db, "unpinned_old", base + 1)
    db.set_session_pinned("pinned_old", True)
    # More newer rows than the handler over-fetches (200), so the pin also misses the fetched page.
    for n in range(205):
        _session(db, f"new{n:03d}", base + 100 + n)
    db._conn.commit()

    response = srv.handle_request({"id": "1", "method": "session.list",
                                   "params": {"limit": 3, "include_hidden": True}})
    ids = [row["id"] for row in response["result"]["sessions"]]

    assert ids == ["new204", "new203", "new202", "pinned_old"]
