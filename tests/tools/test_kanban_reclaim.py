"""Orchestrator reclaim uses the operator flow without granting workers control."""
from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest


@pytest.fixture
def board(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli.config import save_config
    from tools import kanban_tools  # Register the real handlers.

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    save_config({"toolsets": ["kanban"]})
    kb.create_board("reclaim")
    with kbc.connect_closing(board="reclaim") as conn:
        tid = kb.create_task(conn, title="reclaim worker", assignee="default")
        yield kb, conn, tid


@pytest.fixture
def worker_signal(monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as kbd

    signals = []
    alive = {"value": True}

    def signal_worker(pid, sig):
        signals.append((pid, sig))
        alive["value"] = False

    def resolve_signal(signal_fn):
        assert signal_fn is None  # Same default as the CLI and dashboard.
        return signal_worker

    monkeypatch.setattr(kb, "_pid_alive", lambda pid: alive["value"])
    monkeypatch.setattr(kbd, "_kill_fn", resolve_signal)
    return signals


def _reclaim(tid, **args):
    from tools.registry import registry

    return json.loads(registry.dispatch("kanban_reclaim", {
        "task_id": tid, "reason": "orchestrator intervention", "board": "reclaim", **args,
    }))


@pytest.mark.parametrize("park", [False, True])
def test_orchestrator_reclaims_and_optionally_parks(board, worker_signal, park):
    from tools.registry import registry
    from toolsets import resolve_toolset

    kb, conn, tid = board
    names = {row["function"]["name"] for row in registry.get_definitions(
        set(resolve_toolset("kanban")), quiet=True)}
    assert "kanban_reclaim" in names
    assert kb.claim_task(conn, tid)
    conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (12345, tid))
    conn.commit()

    result = _reclaim(tid, **({"park": True} if park else {}))
    assert result == {"ok": True, "task_id": tid, "reclaimed": True,
                      "parked": park, "status_after": "blocked" if park else "ready"}
    assert worker_signal == [(12345, signal.SIGTERM)]
    task = kb.get_task(conn, tid)
    assert task.status == result["status_after"]
    assert task.claim_lock is None
    assert task.claim_expires is None
    assert task.worker_pid is None
    events = kb.list_events(conn, tid)
    reclaimed = next(event for event in events if event.kind == "reclaimed")
    assert reclaimed.payload["reason"] == "orchestrator intervention"
    if park:
        blocked = next(event for event in events if event.kind == "blocked")
        assert blocked.payload["reason"] == "orchestrator intervention"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "blocked"


@pytest.mark.parametrize("park", [False, True])
def test_idle_card_is_not_reclaimed_or_parked(board, worker_signal, park):
    kb, conn, tid = board
    result = _reclaim(tid, park=park)
    assert result == {"ok": True, "task_id": tid, "reclaimed": False,
                      "parked": False, "status_after": "ready"}
    assert kb.get_task(conn, tid).status == "ready"
    assert worker_signal == []


@pytest.mark.parametrize("caller", ["worker", "child"])
def test_reclaim_refuses_workers_and_delegated_children(board, worker_signal, monkeypatch, caller):
    from agent.delegation_context import delegated_child_context
    from tools.registry import registry
    from toolsets import resolve_toolset

    kb, conn, tid = board
    assert kb.claim_task(conn, tid)
    if caller == "worker":
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        result = _reclaim(tid, park=True)
        assert "orchestrator-only" in result.get("error", "")
        names = {row["function"]["name"] for row in registry.get_definitions(
            set(resolve_toolset("kanban")), quiet=True)}
        assert "kanban_reclaim" not in names
    else:
        with delegated_child_context():
            result = _reclaim(tid, park=True)
            assert "child agents are not Kanban run owners" in result.get("error", "")
    assert kb.get_task(conn, tid).status == "running"
    assert kb.get_task(conn, tid).claim_lock is not None
    assert worker_signal == []


def test_reclaimed_review_run_reports_unparked(board, worker_signal):
    kb, conn, tid = board
    assert kb.claim_task(conn, tid)
    assert kb.request_review(conn, tid, summary="ready for review")
    assert kb.claim_review_task(conn, tid)
    result = _reclaim(tid, park=True)
    assert "error" not in result
    assert result["reclaimed"] is True and result["parked"] is False
    assert result["status_after"] == "review"
