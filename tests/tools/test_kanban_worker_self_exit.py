"""Belt for displaced kanban workers: when a card reaches a terminal status
without THIS process closing it, the auto-heartbeat bridge self-terminates
instead of beating on for a run nothing tracks (2026-11-05 zombie-writer
incident; synchronous closer-side termination lives in
``hermes_cli/kanban_db_termination.py``, and this is the worker-side backstop
for dispatchers/closers that cannot signal it — e.g. a cross-host close).

Fail-open by design: only an affirmative terminal read exits; a board error,
a still-running card, a delegated child, or the worker's own completed
terminal close never fires the belt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import kanban_tools


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        yield kb, conn


@pytest.fixture
def fired(monkeypatch):
    """Capture the self-termination instead of SIGTERMing the pytest process."""
    calls = []
    monkeypatch.setattr(
        kanban_tools, "_self_terminate_displaced_worker",
        lambda tid, status: calls.append((tid, status)))
    # Any prior test in this process may have completed a card through the tool.
    monkeypatch.setattr(kanban_tools, "_terminal_close_by_this_worker", False)
    return calls


def _make_card(kb, conn, *, done: bool):
    tid = kb.create_task(conn, title="belt", assignee="worker")
    if done:
        assert kb.complete_task(conn, tid, result="finished") is True
    return tid


def test_belt_fires_on_externally_closed_card(board, fired):
    kb, conn = board
    tid = _make_card(kb, conn, done=True)
    kanban_tools._belt_exit_if_card_closed_elsewhere(kb, conn, tid)
    assert fired == [(tid, "done")]


def test_belt_quiet_while_card_is_running(board, fired):
    kb, conn = board
    tid = _make_card(kb, conn, done=False)
    kb.claim_task(conn, tid)
    kanban_tools._belt_exit_if_card_closed_elsewhere(kb, conn, tid)
    assert fired == []


def test_belt_never_fires_after_this_worker_closed_its_own_card(board, fired, monkeypatch):
    kb, conn = board
    tid = _make_card(kb, conn, done=True)
    monkeypatch.setattr(kanban_tools, "_terminal_close_by_this_worker", True)
    kanban_tools._belt_exit_if_card_closed_elsewhere(kb, conn, tid)
    assert fired == []


def test_belt_is_fenced_for_delegated_children(board, fired):
    kb, conn = board
    tid = _make_card(kb, conn, done=True)
    from agent.delegation_context import delegated_child_context
    with delegated_child_context("child-1"):
        kanban_tools._belt_exit_if_card_closed_elsewhere(kb, conn, tid)
    assert fired == []


def test_heartbeat_bridge_self_exits_when_card_was_force_closed(board, fired, monkeypatch):
    """End-to-end through the real bridge: a running worker whose card is
    force-closed mid-flight exits on its next auto-heartbeat attempt."""
    kb, conn = board
    from hermes_cli import kanban_db_dispatch as kbd
    tid = _make_card(kb, conn, done=False)
    kb.claim_task(conn, tid)
    run_id = kb._current_run_id(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setattr(kanban_tools, "_auto_heartbeat_last_attempt", 0.0)

    # Still running: the bridge beats normally, belt quiet.
    assert kanban_tools.heartbeat_current_worker_from_env() is True
    assert fired == []

    # External force-close (the incident). The env guard keeps the DB-level
    # terminate off this same-process test; the belt is the worker-side answer.
    assert kb.complete_task(conn, tid, result="closed externally", force=True) is True
    monkeypatch.setattr(kanban_tools, "_auto_heartbeat_last_attempt", 0.0)
    assert kanban_tools.heartbeat_current_worker_from_env() is False
    assert fired == [(tid, "done")]
