"""``session.list`` must not let a cron flood starve human sessions out of the window.

The RPC over-fetches a fixed recency window (``max(limit * 2, 200)``) and post-filters
only ``{kanban, tool, oneshot}`` — cron was not denied *in the SQL window*, so on a host
minting cron sessions every few minutes the entire window is watcher rows and Discord /
CLI / app chats never reach the filter. Symptom on a production host (mobile app): the
session list showed 196/200 cron rows, the newest human chat from hours earlier was
missing, and platform history looked deleted while live chats kept working.
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


def _seed_session(db, sid: str, source: str, at: float) -> None:
    db.create_session(sid, source)
    db.append_message(sid, "user", f"hello from {sid}")
    db._conn.execute(
        "UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?", (at, at, sid)
    )
    db._conn.commit()


def _flood_with_cron(db, human_at: float, count: int = 250) -> None:
    """``count`` cron sessions, all MORE RECENT than the human sessions."""
    now = time.time()
    for i in range(count):
        _seed_session(db, f"cron{i}", "cron", now - i)
    assert human_at < now - count


def test_session_list_survives_cron_flood(db):
    """Human sessions older than the whole cron flood must still be listed."""
    human_at = time.time() - 86400
    _seed_session(db, "discord1", "discord", human_at)
    _seed_session(db, "gui1", "gui", human_at - 60)
    _flood_with_cron(db, human_at)

    rows = _call("session.list", {"limit": 200})["result"]["sessions"]
    ids = {row["id"] for row in rows}
    sources = {row["source"] for row in rows}

    assert "discord1" in ids, "discord session starved out by cron flood"
    assert "gui1" in ids, "app chat starved out by cron flood"
    assert "cron" not in sources, "cron watcher rows must not crowd the human listing"


def test_session_most_recent_skips_cron_flood(db):
    """Auto-resume must land on the human chat, not the newest cron watcher row."""
    human_at = time.time() - 86400
    _seed_session(db, "discord1", "discord", human_at)
    _flood_with_cron(db, human_at)

    result = _call("session.most_recent", {})["result"]
    assert result["session_id"] == "discord1"
