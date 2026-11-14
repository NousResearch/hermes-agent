"""Behavioral ownership and fencing contracts for the automatic heartbeat bridge."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def board(tmp_path, monkeypatch):
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
    tid = kb.create_task(conn, title="worker", assignee="worker")
    lock = kb._host_prefix() + "heartbeat-test"
    assert kb.claim_task(conn, tid, ttl_seconds=1, claimer=lock) is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", lock)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(kb._current_run_id(conn, tid)))
    monkeypatch.setattr(kt, "time", SimpleNamespace(monotonic=lambda: 100.0))
    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", 0.0)
    monkeypatch.setattr(kt, "_auto_heartbeat_fence_warned", False)
    yield kb, conn, tid, kt
    conn.close()


@pytest.mark.parametrize("identity,claim_written,heartbeat_written", [
    ("valid", True, True),
    ("wrong-claim", False, True),
    ("stale-run", True, False),
])
def test_bridge_fences_each_operation_independently(board, monkeypatch, identity, claim_written, heartbeat_written):
    kb, conn, tid, kt = board
    if identity == "wrong-claim":
        monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "not-the-owner")
    if identity == "stale-run":
        from hermes_cli import kanban_db_dispatch as kbd

        old_run = kb._current_run_id(conn, tid)
        kbd._set_worker_pid(conn, tid, 98765)
        monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
        monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
        assert kbd.detect_crashed_workers(conn) == [tid]
        assert kb.claim_task(conn, tid, ttl_seconds=1, claimer=kb._host_prefix() + "heartbeat-test") is not None
        assert kb._current_run_id(conn, tid) != old_run
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(old_run))
    before = kb.get_task(conn, tid)
    events = kb.list_events(conn, tid)
    assert kt.heartbeat_current_worker_from_env() is (claim_written and heartbeat_written)
    after = kb.get_task(conn, tid)
    assert (after.claim_expires > before.claim_expires) is claim_written
    assert (after.last_heartbeat_at is not None) is heartbeat_written
    assert len(kb.list_events(conn, tid)) == len(events) + int(heartbeat_written)


def test_permission_error_still_runs_other_operation_and_warns_once(board, monkeypatch, caplog):
    kb, conn, tid, kt = board
    original = kb.heartbeat_claim

    def denied(*args, **kwargs):
        raise PermissionError("board refused claim")

    monkeypatch.setattr(kb, "heartbeat_claim", denied)
    with caplog.at_level("WARNING", logger="tools.kanban_tools"):
        assert kt.heartbeat_current_worker_from_env() is False
        monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", 0.0)
        assert kt.heartbeat_current_worker_from_env() is False
    warnings = [r for r in caplog.records if "board refused claim" in r.getMessage()]
    assert len(warnings) == 1
    assert tid in warnings[0].getMessage()
    assert kb.get_task(conn, tid).last_heartbeat_at is not None
    assert len([e for e in kb.list_events(conn, tid) if e.kind == "heartbeat"]) == 2
    monkeypatch.setattr(kb, "heartbeat_claim", original)


@pytest.mark.parametrize("marker", ["1", "root"])
def test_descendant_marker_suppresses_stop_nudge_and_owner_recovers(tmp_path, monkeypatch, marker):
    from agent.kanban_stop import build_kanban_stop_nudge, kanban_stop_nudge_enabled

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    assert build_kanban_stop_nudge(messages=[]) is not None
    with monkeypatch.context() as child:
        child.setenv("HERMES_DELEGATED_CHILD_CONTEXT", str(tmp_path) if marker == "root" else marker)
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None
    assert kanban_stop_nudge_enabled() is True
    assert build_kanban_stop_nudge(messages=[]) is not None
