"""The worker's ``worker_session_id`` stamp: which heartbeats carry it, and whose id.

``worker_session_id`` was recorded only by ``kanban_complete`` / ``kanban_request_review``
— the graceful path — so a run that crashed, timed out or went stale had no link to the
session that produced it. The stamp moved to the heartbeat, the first moment both the
session id and the run row exist.

The path that matters is the AUTO heartbeat (``_touch_activity`` -> here, every 60s of
agent activity), not the ``kanban_heartbeat`` tool: that tool is optional and model-driven,
and a worker that dies early is the least likely to have called it.

Identity is guarded where liveness is not. ``set_current_session_id`` publishes
HERMES_SESSION_ID into the PROCESS env for every non-delegated agent, so a cron job fired
in-process from a worker leaves its own id there; stamping it would latch a foreign session
under first-write-wins. Liveness must stay unguarded or the dispatcher reclaims a live
worker (see tests/cron/test_cron_kanban_env_isolation.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.delegation_context import non_dispatcher_owned_context
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def worker(monkeypatch, tmp_path):
    """A claimed, running task with this process standing in for its dispatcher worker."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    from tools import kanban_tools as kt
    # Module-global 60s rate limiter; each test drives the beat itself.
    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", 0.0)
    return tid, run_id


def _metadata(run_id: int) -> dict:
    with kbc.connect() as conn:
        row = conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    return kb._json_dict(row["metadata"])


def test_auto_heartbeat_stamps_worker_session_id(worker, monkeypatch):
    """The near-universal path: no explicit tool call, no graceful close."""
    tid, run_id = worker
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-worker")

    from tools import kanban_tools as kt
    assert kt.heartbeat_current_worker_from_env() is True

    assert _metadata(run_id)["worker_session_id"] == "sess-worker"


def test_auto_heartbeat_stamp_survives_a_dispatcher_crash_close(worker, monkeypatch):
    """End to end over both edits: the run the report cares about keeps its link."""
    tid, run_id = worker
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-worker")

    from tools import kanban_tools as kt
    kt.heartbeat_current_worker_from_env()

    with kbc.connect() as conn, kb.write_txn(conn):
        kb._end_run(conn, tid, outcome="crashed", status="crashed",
                    error="pid 1 exited", metadata={"pid": 1, "protocol_violation": True})

    meta = _metadata(run_id)
    assert meta["worker_session_id"] == "sess-worker"
    # The marker _protocol_violation_streak reads is untouched by the merge.
    assert meta["protocol_violation"] is True


def test_in_process_cron_does_not_stamp_its_own_session(worker, monkeypatch):
    """A cron agent inside the worker publishes its id to the process env; the beat must
    still land (liveness) but must not claim the run."""
    tid, run_id = worker
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-cron-job")

    from tools import kanban_tools as kt
    with non_dispatcher_owned_context():
        assert kt.heartbeat_current_worker_from_env() is True

    assert "worker_session_id" not in _metadata(run_id)
    with kbc.connect() as conn:
        row = conn.execute(
            "SELECT last_heartbeat_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert row["last_heartbeat_at"] is not None


def test_worker_session_wins_over_a_later_cron_beat(worker, monkeypatch):
    """First-write-wins is what makes the guard sufficient rather than merely tidy."""
    tid, run_id = worker
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-worker")

    from tools import kanban_tools as kt
    kt.heartbeat_current_worker_from_env()

    monkeypatch.setenv("HERMES_SESSION_ID", "sess-cron-job")
    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", 0.0)
    with non_dispatcher_owned_context():
        kt.heartbeat_current_worker_from_env()

    assert _metadata(run_id)["worker_session_id"] == "sess-worker"


def test_explicit_heartbeat_tool_stamps(worker, monkeypatch):
    tid, run_id = worker
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-worker")

    from tools import kanban_tools as kt
    kt._handle_heartbeat({"note": "still going"})

    assert _metadata(run_id)["worker_session_id"] == "sess-worker"


def test_local_worker_without_session_id_stamps_nothing(worker):
    """A locally-driven worker that bypassed the dispatcher is unaffected."""
    tid, run_id = worker

    from tools import kanban_tools as kt
    kt._handle_heartbeat({})

    assert _metadata(run_id) == {}


def test_stamp_refuses_a_task_this_worker_is_not_scoped_to(worker, monkeypatch):
    """The #19534 ownership boundary is unchanged: a prompt-injected task_id stamps nothing."""
    tid, _ = worker
    monkeypatch.setenv("HERMES_SESSION_ID", "sess-worker")

    from tools import kanban_tools as kt
    assert kt._stamp_worker_session_metadata("some-other-task", None) is None
    assert kt._stamp_worker_session_metadata(tid, None) == {"worker_session_id": "sess-worker"}
