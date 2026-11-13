"""Legacy ``session.list`` ``title_prefix``: the match is made before the row limit.

Sparky's task list asks this RPC for its ``Task:`` runs; filtering a recency page client-side
saw 1 of 7 once cron runs filled the page. The prefix keeps its trailing space ("Task: ").
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


def _session(db, sid, started, title=None):
    db.create_session(sid, source="tui")
    db.append_message(session_id=sid, role="user", content=f"hello from {sid}")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (started, sid))
    db._conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = ?", (started, sid))
    if title:
        db.set_session_title(sid, title)


def _list(params):
    response = srv.handle_request({"id": "1", "method": "session.list", "params": params})
    assert "error" not in response, response
    return [row["id"] for row in response["result"]["sessions"]]


def test_title_prefix_returns_matches_older_than_the_limit(db):
    base = time.time() - 10_000
    _session(db, "task_old", base, title="Task: oldest")
    _session(db, "task_new", base + 10, title="Task: newer")
    _session(db, "colon", base + 20, title="Task:no space")  # the trailing space is part of the prefix
    for n in range(5):
        _session(db, f"noise{n}", base + 100 + n)
    db._conn.commit()

    assert not {"task_old", "task_new"} & set(_list({"limit": 3, "include_hidden": True}))

    assert _list({"limit": 3, "include_hidden": True, "title_prefix": "Task: "}) == ["task_new", "task_old"]
