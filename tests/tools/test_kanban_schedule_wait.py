"""``kanban_schedule(wait_for_process=..., until=...)``: what the tool refuses and what it records.

A refused call must leave the card running, so the worker can still act on the
error; only a live, host-local job started under a managed worker can be awaited.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import pytest

from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """An isolated board with one claimed task, seen from that task's worker."""
    return _claimed_task_env(monkeypatch, tmp_path, goal_mode=False)


def _claimed_task_env(monkeypatch, tmp_path, *, goal_mode: bool) -> str:
    from pathlib import Path as _Path

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wait-on-process", assignee="test-worker",
                             body="Must achieve X.", goal_mode=goal_mode)
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


@pytest.fixture
def spawn(monkeypatch, tmp_path):
    """Start a background process, as a managed worker by default; kill leftovers at teardown."""
    from agent import runtime_session_store
    from tools.process_registry import process_registry

    started = []

    def _spawn(command: str, *, managed_worker: bool = True):
        monkeypatch.setattr(runtime_session_store, "_worker_process", managed_worker)
        session = process_registry.spawn_local(command, cwd=str(tmp_path))
        started.append(session.id)
        return session

    yield _spawn
    for sid in started:
        process_registry.kill_process(sid, source="test")


def _future() -> str:
    return datetime.fromtimestamp(time.time() + 3600, UTC).isoformat()


def _schedule(args: dict) -> dict:
    from tools import kanban_tools  # registers the kanban tools
    from tools.registry import registry

    return json.loads(registry.get_entry("kanban_schedule").handler(args))


def _status(tid: str) -> str:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as conn:
        return kb.get_task(conn, tid).status


@pytest.mark.parametrize("args, named", [
    ({"wait_for_process": "proc_feedfacecafe"}, ("until", "deadline")),
    ({"until": "2099-01-01T00:00:00+00:00"}, ("wait_for_process", "timer")),
    ({"wait_for_process": "proc_feedfacecafe", "until": "2099-01-01T00:00:00"}, ("until", "timezone")),
    ({"wait_for_process": "proc_feedfacecafe", "until": "2020-01-01T00:00:00+00:00"}, ("until", "future")),
    ({"wait_for_process": "proc_feedfacecafe", "until": None}, ("wait_for_process", "proc_feedfacecafe")),
], ids=["no-until", "no-process", "naive-until", "past-until", "unknown-session"])
def test_schedule_wait_argument_rejections(worker_env, args, named):
    args = {**args, "until": _future()} if "until" in args and args["until"] is None else args
    out = _schedule(args)

    assert all(word in out.get("error", "") for word in named), out
    assert _status(worker_env) == "running"


def test_schedule_wait_on_finished_process_reports_exit_code(worker_env, spawn):
    from tools.process_registry_output_log import recorded_exit_code

    session = spawn("exit 4")
    wait_until(lambda: recorded_exit_code(session.output_log) is not None, timeout=15, what="the exit file")
    out = _schedule({"wait_for_process": session.id, "until": _future()})

    assert "code 4" in out.get("error", ""), out
    assert _status(worker_env) == "running"


def _unlogged_session(spawn, monkeypatch):
    return spawn("sleep 300", managed_worker=False)


def _sandbox_session(spawn, monkeypatch):
    from tools.process_registry import ProcessSession, process_registry

    live = spawn("sleep 300")
    remote = ProcessSession(id="proc_5a4db0c5e001", command="sleep 300", pid=live.pid,
                            pid_scope="sandbox", output_log=live.output_log)
    monkeypatch.setattr(process_registry, "get", lambda sid: remote if sid == remote.id else None)
    return remote


@pytest.mark.parametrize("make_session", [_unlogged_session, _sandbox_session],
                         ids=["not-started-by-a-worker", "remote-backend"])
def test_schedule_wait_rejects_processes_the_dispatcher_cannot_observe(worker_env, spawn, monkeypatch,
                                                                       make_session):
    session = make_session(spawn, monkeypatch)
    out = _schedule({"wait_for_process": session.id, "until": _future()})

    assert session.id in out.get("error", ""), out
    assert _status(worker_env) == "running"


def test_schedule_wait_refused_for_goal_mode(monkeypatch, tmp_path, spawn):
    tid = _claimed_task_env(monkeypatch, tmp_path, goal_mode=True)
    session = spawn("sleep 300")
    out = _schedule({"wait_for_process": session.id, "until": _future()})

    assert "goal_mode" in out.get("error", ""), out
    assert _status(tid) == "running"


def test_schedule_wait_success_payload(worker_env, spawn):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    session = spawn("sleep 300")
    out = _schedule({"reason": "training run", "wait_for_process": session.id[:12], "until": _future()})

    assert out["status"] == "scheduled", out
    assert out["wait_for_process"] == session.id
    with kbc.connect() as conn:
        [scheduled] = [e for e in kb.list_events(conn, worker_env) if e.kind == "scheduled"]
    assert scheduled.payload["wait"]["kind"] == "process"
    assert scheduled.payload["wait"]["session_id"] == session.id
