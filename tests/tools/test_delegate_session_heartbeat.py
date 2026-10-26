"""Retained-child activity is best-effort bookkeeping, never a progress-delivery gate.

Regression for PR #144196's heartbeat review: failures must preserve progress,
back off before retrying, and use the short activity budget under SQLite contention.
"""

import sqlite3
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from tools import delegate_sessions
from tools.delegate_session_store import DelegateSessionStore


@pytest.fixture
def retained_child(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", "cli")
        db.create_session("child", "subagent", parent_session_id="parent",
                          model_config={"_delegate_from": "parent"})
        events = []

        def progress(*args, **kwargs):
            events.append((args, kwargs))
            return "delivered"

        progress._flush = lambda: events.append("flushed")
        child = SimpleNamespace(
            session_id="child", model="test-model", provider="test-provider",
            _session_db=db, _ensure_db_session=lambda: None,
            _delegate_role="leaf", _delegate_depth=1, _delegate_workspace=None,
            _delegate_forked_context=False, tool_progress_callback=progress,
        )
        parent = SimpleNamespace(session_id="parent")
        yield db, parent, child, events
    finally:
        db.close()


def test_failed_heartbeat_preserves_progress_and_throttles_until_recovery(retained_child, monkeypatch):
    db, parent, child, events = retained_child
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(delegate_sessions, "time", SimpleNamespace(monotonic=lambda: clock.now))
    delegate_sessions.bind_child(parent, child, {"goal": "Review"}, extension=None,
                                 options={}, routing_cfg={}, retain=True)
    store = DelegateSessionStore(db)
    before = store.get("parent", "child")["last_activity_at"]
    write = db._write_rowcount
    attempts = []

    def intermittent_write(*args, **kwargs):
        attempts.append(clock.now)
        if len(attempts) == 1:
            clock.now = 102.5  # A slow failed attempt must not trigger an immediate retry.
            raise sqlite3.OperationalError("database is locked")
        return write(*args, **kwargs)

    monkeypatch.setattr(db, "_write_rowcount", intermittent_write)
    for when, expected_attempts in [(101.0, 1), (102.6, 1), (103.6, 2), (103.7, 2)]:
        clock.now = when
        assert child.tool_progress_callback("tool.progress", progress=when) == "delivered"
        assert len(attempts) == expected_attempts
    assert events == [(("tool.progress",), {"progress": when}) for when in (101.0, 102.6, 103.6, 103.7)]
    child.tool_progress_callback._flush()
    assert events[-1] == "flushed"
    row = store.get("parent", "child")
    assert row["last_activity_at"] > before
    assert row["status"] == "running"
    assert store.release("child", child._delegate_session_token)


def test_contended_heartbeat_uses_short_budget_and_keeps_activation_fence(retained_child, monkeypatch):
    db, parent, child, _ = retained_child
    delegate_sessions.bind_child(parent, child, {"goal": "Review"}, extension=None,
                                 options={}, routing_cfg={}, retain=True)
    store = DelegateSessionStore(db)
    token = child._delegate_session_token
    before = store.get("parent", "child")["last_activity_at"]
    budgets = []

    def exhaust_retry_budget(deadline, patience_s):
        budgets.append(patience_s)
        assert 0 < patience_s <= 1, "Activity must not inherit the routine write retry budget"
        return False  # Exercise the real SQLite error without sleeping through its retry window.

    monkeypatch.setattr(db, "_sleep_before_write_retry", exhaust_retry_budget)
    blocker = sqlite3.connect(db.db_path, isolation_level=None)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError):
            store.touch("child", token)
    finally:
        blocker.close()
    assert len(budgets) == 1
    assert store.get("parent", "child")["last_activity_at"] == before
    assert store.release("child", token)
    _, replacement = store.acquire("parent", "child", "Next assignment")
    assert not store.touch("child", token)
    assert store.touch("child", replacement)
    assert store.release("child", replacement)
    assert not store.touch("child", replacement)
