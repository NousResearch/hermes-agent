"""Failing-first regression tests: judge transport failures must not deadlock a
goal-mode card (kanban_complete / kanban_request_review).

Root cause contract: ``judge_goal`` fails open to ``("continue", ..., transport_failed=True)``
when the judge LLM is unreachable (PermissionDenied/403, auth, DNS). Every handoff gate
mirrors that 5-tuple out of ``judge_goal``; when ``transport_failed`` is True the verdict
is synthetic — NOT evidence that the work is incomplete. The gate must:

1. ``tools.kanban_tools._goal_gate`` — allow the handoff (fail open, logged).
2. ``tools.kanban_tools._goal_verdict_and_reason`` — return
   ``(None, "goal judge transport failure: <exc-type>")`` so handlers render the
   failure honestly (fail-open) without claiming a judge rejection.
3. Spawn an independent verification child task (assignment to the human operator,
   NOT auto-assigned to the worker) so a broken judge cannot trap a completed
   parent — it degrades to a human-verified handoff.

No model/pool changes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Unit: verdict mapping (tools/kanban_tools.py)
# ---------------------------------------------------------------------------


def test_goal_verdict_none_on_transport_failure():
    """``transport_failed=True`` must map to verdict None (fail-open), never 'continue'."""
    from tools import kanban_tools as kt

    verdict, reason = kt._goal_verdict_and_reason(
        lambda **kw: ("continue", "judge error: PermissionDenied", False, None, True))
    assert verdict is None, (
        f"transport failure must fail open (verdict None), got {verdict!r}")
    assert "transport" in (reason or "").lower()


def test_goal_verdict_passes_through_real_judge_verdicts():
    """A reachable judge's verdicts (incl. synthetic-source 'continue') pass through unchanged."""
    from tools import kanban_tools as kt

    verdict, reason = kt._goal_verdict_and_reason(
        lambda **kw: ("continue", "missing verification evidence", False, None, False))
    assert verdict == "continue"
    assert reason == "missing verification evidence"


# ---------------------------------------------------------------------------
# Handler: kanban_complete fails open on transport failure
# ---------------------------------------------------------------------------


def test_complete_transport_failure_fails_open(monkeypatch, tmp_path):
    """Judge PermissionDenied must NOT reject kanban_complete of a finished goal card."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        goal_task_id = kb.create_task(
            conn, title="goal-mode-test", assignee="test-worker",
            body="Must achieve X with verified evidence.", goal_mode=True
        )
        kb.claim_task(conn, goal_task_id)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", goal_task_id)

    # Judge is reachable but its transport fails (the PermissionDenied shape).
    monkeypatch.setattr(kt, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(
        kt, "judge_goal",
        lambda **kw: ("continue", "judge error: PermissionDenied", False, None, True))

    # Child creation must be attempted (independent verification), not a hard reject.
    created = {}
    monkeypatch.setattr(kt, "_spawn_goal_verification_child",
                        lambda conn, tid, reason: created.update(tid=tid, reason=reason) or "t_child")

    out = kt._handle_complete({"summary": "All acceptance criteria verified with evidence."})
    d = json.loads(out)

    assert "error" not in d, f"transport failure must fail open, got: {d}"
    assert created.get("tid") == goal_task_id
    assert "PermissionDenied" in (created.get("reason") or "")
    conn2 = kbc.connect()
    try:
        assert kb.get_task(conn2, goal_task_id).status == "done"
    finally:
        conn2.close()


def test_complete_real_judge_rejection_still_rejects(monkeypatch, tmp_path):
    """A reachable judge's genuine 'continue' still rejects the completion (no bypass)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        goal_task_id = kb.create_task(
            conn, title="goal-mode-test", assignee="test-worker",
            body="Must achieve X with verified evidence.", goal_mode=True
        )
        kb.claim_task(conn, goal_task_id)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", goal_task_id)

    monkeypatch.setattr(kt, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(
        kt, "judge_goal",
        lambda **kw: ("continue", "missing verification evidence", False, None, False))

    out = kt._handle_complete({"summary": "I did some stuff but not X"})
    d = json.loads(out)
    assert "error" in d
    assert "Goal completion rejected by judge" in d["error"]
    assert "missing verification evidence" in d["error"]


# ---------------------------------------------------------------------------
# Unit: verification-child spawning
# ---------------------------------------------------------------------------


def test_verification_child_skips_worker_self_assignment(monkeypatch):
    """The verification child must go to a human operator, never the worker's own profile."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    conn = MagicMock()
    conn.execute.return_value = MagicMock(__getitem__=lambda *_: None)
    monkeypatch.setattr(kb, "create_task", MagicMock(return_value="t_verify"))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")

    tid = kt._spawn_goal_verification_child(conn, "t_parent", "judge error: PermissionDenied")

    assert tid == "t_verify"
    kwargs = kb.create_task.call_args.kwargs
    assert kwargs.get("assignee") == "company"
    assert kwargs.get("assignee") != "test-worker"


def test_verification_assignee_falls_back_to_qa_for_company_worker(monkeypatch):
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_PROFILE", "company")
    monkeypatch.setattr(kt, "load_config", lambda: {})
    assert kt._goal_verification_assignee() == "qa"


def test_request_review_transport_failure_does_not_spawn_redundant_child(monkeypatch):
    from tools import kanban_tools as kt

    task = type("Task", (), {"goal_mode": True, "title": "x", "body": "y"})()
    monkeypatch.setattr(kt, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(
        kt,
        "judge_goal",
        lambda **kw: ("continue", "judge error: PermissionDenied", False, None, True),
    )
    spawned = []
    monkeypatch.setattr(
        kt,
        "_spawn_goal_verification_child",
        lambda *args: spawned.append(args) or "t_child",
    )

    kt._goal_gate("kanban_request_review", task, "t_parent", "evidence", conn=MagicMock())

    assert spawned == []


def test_verification_child_persists_parent_link(tmp_path, monkeypatch):
    """Child creation uses the real DB and links the child back to the parent."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        parent_id = kb.create_task(conn, title="goal-parent", assignee="test-worker",
                                   goal_mode=True)
        child_id = kt._spawn_goal_verification_child(conn, parent_id, "judge error: X")
        child = kb.get_task(conn, child_id)
        assert child is not None, "verification child must exist"
        # Task has no `parents` field; the link lives in task_links — assert it there.
        assert kb.parent_ids(conn, child_id) == [parent_id]
        assert child_id in kb.child_ids(conn, parent_id)
        assert "judge error: X" in (child.body or "")
        assert child.status in ("todo", "ready")
    finally:
        conn.close()


def test_verification_child_persists_parent_link_devtest():
    """Direct (non-monkeypatch) variant against a real tmp DB — same assertions as above."""
    import tempfile
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    old_home = __import__("os").environ.get("HERMES_HOME")
    old_task = __import__("os").environ.get("HERMES_KANBAN_TASK")
    tmp = Path(tempfile.mkdtemp(prefix="goal-judge-devtest-"))
    try:
        (tmp / ".hermes").mkdir()
        __import__("os").environ["HERMES_HOME"] = str(tmp / ".hermes")
        __import__("os").environ.pop("HERMES_KANBAN_TASK", None)
        kb._INITIALIZED_PATHS.clear()
        kb.init_db()
        conn = kbc.connect()
        try:
            parent_id = kb.create_task(conn, title="goal-parent", assignee="test-worker",
                                       goal_mode=True)
            child_id = kt._spawn_goal_verification_child(conn, parent_id, "judge error: X")
            child = kb.get_task(conn, child_id)
            assert child is not None
            assert "judge error: X" in (child.body or "")
            assert child.status in ("todo", "ready")
        finally:
            conn.close()
    finally:
        if old_home is not None:
            __import__("os").environ["HERMES_HOME"] = old_home
        else:
            __import__("os").environ.pop("HERMES_HOME", None)
        if old_task is not None:
            __import__("os").environ["HERMES_KANBAN_TASK"] = old_task
        else:
            __import__("os").environ.pop("HERMES_KANBAN_TASK", None)


# ---------------------------------------------------------------------------
# CLI mirror: hermes_cli/kanban.py _goal_mode_handoff_rejection
# ---------------------------------------------------------------------------


def _fake_goal_task():
    task = MagicMock()
    task.goal_mode = True
    task.title = "goal-mode-test"
    task.body = "Must achieve X with verified evidence."
    return task


def test_cli_mirror_transport_failure_fails_open(monkeypatch):
    """The CLI gate mirror must treat transport_failed=True as fail-open (done),
    never as a judge rejection."""
    from hermes_cli import kanban as kcli

    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda purpose: (MagicMock(), "judge-model"))
    monkeypatch.setattr(
        "hermes_cli.goals.judge_goal",
        lambda **kw: ("continue", "judge error: PermissionDenied", False, None, True))

    verdict, rejection = kcli._goal_mode_handoff_rejection(_fake_goal_task(), "all done")
    assert verdict == "done", f"transport failure must fail open, got {verdict!r}"
    assert rejection is None


def test_cli_mirror_real_rejection_still_rejects(monkeypatch):
    """A reachable judge's genuine 'continue' still rejects through the CLI mirror."""
    from hermes_cli import kanban as kcli

    monkeypatch.setattr(
        "agent.auxiliary_client.get_text_auxiliary_client",
        lambda purpose: (MagicMock(), "judge-model"))
    monkeypatch.setattr(
        "hermes_cli.goals.judge_goal",
        lambda **kw: ("continue", "missing evidence", False, None, False))

    verdict, rejection = kcli._goal_mode_handoff_rejection(_fake_goal_task(), "not done")
    assert verdict == "continue"
    assert rejection == "missing evidence"
