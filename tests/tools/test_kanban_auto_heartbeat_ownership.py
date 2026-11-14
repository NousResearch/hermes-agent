"""Worker activity ownership regression for #74583."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("context", ["delegate", "cron", "descendant", "descendant-root"])
def test_nonowner_activity_preserves_worker_heartbeat(tmp_path, monkeypatch, caplog, context):
    from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from tools import kanban_tools as kt

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("HERMES_DELEGATED_CHILD_CONTEXT", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(key, raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="worker", assignee="worker")
        assert kb.claim_task(conn, tid, ttl_seconds=1, claimer="worker-lock") is not None
        run_id = kb._current_run_id(conn, tid)
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
        monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "worker-lock")
        monkeypatch.setattr(kt, "time", SimpleNamespace(monotonic=lambda: 100.0))
        monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", 0.0)
        monkeypatch.setattr(kt, "_auto_heartbeat_fence_warned", False)
        board = Mock(wraps=kt._board)
        monkeypatch.setattr(kt, "_board", board)
        before = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
        scope = {"delegate": delegated_child_context, "cron": non_dispatcher_owned_context,
                 "descendant": nullcontext, "descendant-root": nullcontext}[context]
        with monkeypatch.context() as child:
            if context.startswith("descendant"):
                child.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1" if context == "descendant" else str(home))
            with scope():
                assert kt.heartbeat_current_worker_from_env() is False
                assert kt.heartbeat_current_worker_from_env() is False
        assert board.call_count == 0
        assert kt._auto_heartbeat_last_attempt == 0.0
        after = kb.get_task(conn, tid)
        assert (after.claim_expires, after.last_heartbeat_at) == (before.claim_expires, before.last_heartbeat_at)
        assert kb.list_events(conn, tid) == events
        warnings = [r for r in caplog.records if "HERMES_DELEGATED_CHILD_CONTEXT" in r.getMessage()]
        assert len(warnings) == (1 if context.startswith("descendant") else 0)
        assert kt.heartbeat_current_worker_from_env() is True
        after = kb.get_task(conn, tid)
        assert after.claim_expires > before.claim_expires
        assert after.last_heartbeat_at is not None
        assert len(kb.list_events(conn, tid)) == len(events) + 1
        assert board.call_count == 1
        assert kt.heartbeat_current_worker_from_env() is False
        assert board.call_count == 1
    finally:
        conn.close()
