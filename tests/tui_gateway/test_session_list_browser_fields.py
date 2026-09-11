"""Tests: ``session.list`` rows carry what a per-bot session browser needs.

Why: the desktop Bots pane lists ONE bot's stored chats on demand (context menu
→ "Show sessions"). To order that list by recency and badge the hidden canonical
Bot Chat beside the user's side-chats, every listing row must publish
``last_active`` and ``hidden`` — otherwise the client needs a second round-trip
per row, or falls back to ``started_at`` order, which puts a long-running
Bot Chat at the bottom of its own list.

Contract under test:
- ``include_hidden`` listings return hidden rows flagged ``hidden: true`` and
  visible rows ``hidden: false``.
- ``last_active`` tracks the newest activity, so a row created earlier but
  written to later sorts ahead — it is not a copy of ``started_at``.
"""

from __future__ import annotations

import pytest

import tui_gateway.server as srv


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _db(profile_dir):
    from hermes_state import SessionDB

    return SessionDB(db_path=profile_dir / "state.db")


def _add_session(db, sid, *, title, ts, text, hidden=False):
    db.create_session(sid, "desktop")
    db.append_message(sid, "user", text, timestamp=ts)
    with db._lock:
        db._conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, sid))
    if hidden:
        db.set_session_hidden(sid, True)


def _list(monkeypatch, db, **params):
    monkeypatch.setattr(srv, "_get_db", lambda: db)
    return srv._methods["session.list"](1, params)["result"]["sessions"]


def test_listing_rows_flag_hidden_and_visible(home, monkeypatch):
    db = _db(home)
    _add_session(db, "forever", title="Bot Chat", ts=1000, text="canonical", hidden=True)
    _add_session(db, "side", title="Side thread", ts=2000, text="scratch")

    by_id = {row["id"]: row for row in _list(monkeypatch, db, include_hidden=True)}
    assert by_id["forever"]["hidden"] is True
    assert by_id["side"]["hidden"] is False

    # Without include_hidden the canonical row stays out, unchanged contract.
    assert {row["id"] for row in _list(monkeypatch, db)} == {"side"}
    db.close()


def test_last_active_follows_newest_activity_not_creation(home, monkeypatch):
    db = _db(home)
    _add_session(db, "old-but-busy", title="Bot Chat", ts=1000, text="first", hidden=True)
    _add_session(db, "newer-idle", title="Side thread", ts=2000, text="only message")
    # Later activity on the OLDER session.
    db.append_message("old-but-busy", "assistant", "reply much later", timestamp=5000)

    by_id = {row["id"]: row for row in _list(monkeypatch, db, include_hidden=True)}
    assert by_id["old-but-busy"]["last_active"] > by_id["newer-idle"]["last_active"]
    # And it is the message clock, not a heartbeat that never fired.
    assert by_id["old-but-busy"]["last_active"] == 5000
    db.close()
