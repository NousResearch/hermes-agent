"""Tests: profiles.list ``worker_session`` (freshest kanban/tool worker).

Why: kanban/tool worker sessions are deny-listed out of every conversation
list, so the desktop Bots roster showed a profile as idle ("3 hr ago")
while its kanban worker had been running for 12+ minutes
(NousResearch/hermes-agent#90268). profiles.list now reports the newest
DENIED row per profile as ``worker_session`` so roster UIs can light
ACTIVE NOW off the worker's ``last_activity_at`` heartbeat.

Contract under test:
- ``worker_session`` is the newest kanban/tool row: id, source, title,
  last_active.
- ``None`` when the profile has no worker sessions.
- ``last_session`` still never includes worker rows (existing deny intact).
- ``include_sessions: false`` skips the field entirely.
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


def _add_session(db, sid, *, source="cli", title="", ts, text):
    db.create_session(sid, source)
    db.append_message(sid, "user", text, timestamp=ts)
    with db._lock:
        db._conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, sid))


def _profiles(params):
    envelope = srv._methods["profiles.list"](1, params)
    return envelope["result"]["profiles"]


def _row(profiles, name):
    return next(p for p in profiles if p["name"] == name)


def test_worker_session_reports_newest_worker_and_keeps_last_session_clean(home):
    db = _db(home)
    _add_session(db, "chat1", source="cli", title="Chat", ts=1000, text="hello")
    _add_session(db, "work-old", source="kanban", title="Task 4", ts=2000, text="work kanban task 4")
    _add_session(db, "work-new", source="kanban", title="Task 9", ts=3000, text="work kanban task 9")
    db.close()

    row = _row(_profiles({}), "default")

    worker = row["worker_session"]
    assert worker is not None
    assert worker["id"] == "work-new"
    assert worker["source"] == "kanban"
    assert worker["last_active"] >= 3000
    # The conversation preview still never surfaces worker rows.
    assert row["last_session"]["id"] == "chat1"


def test_worker_session_none_without_workers(home):
    db = _db(home)
    _add_session(db, "chat1", source="cli", title="Chat", ts=1000, text="hello")
    db.close()

    row = _row(_profiles({}), "default")
    assert row["worker_session"] is None
    assert row["last_session"]["id"] == "chat1"


def test_worker_session_tool_source_counts(home):
    db = _db(home)
    _add_session(db, "sub1", source="tool", title="Subagent", ts=5000, text="delegated run")
    db.close()

    row = _row(_profiles({}), "default")
    assert row["worker_session"]["id"] == "sub1"
    assert row["worker_session"]["source"] == "tool"
    # No human-facing session at all: last_session stays None.
    assert row["last_session"] is None


def test_include_sessions_false_omits_worker_session(home):
    db = _db(home)
    _add_session(db, "work1", source="kanban", title="Task", ts=1000, text="task body")
    db.close()

    row = _row(_profiles({"include_sessions": False}), "default")
    assert "worker_session" not in row
    assert "last_session" not in row
