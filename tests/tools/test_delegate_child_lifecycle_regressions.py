"""Regressions for delegated-child lifecycle isolation + the circular review gate.

Audit evidence (default-board t_91706029 / t_666f55e9):

1. A delegate_task investigation child prematurely completed its review-gated
   parent Kanban card — children must never be able to complete / block /
   request-review their parent's card, even by passing the parent's explicit
   ``task_id`` or shelling out to the CLI.
2. A later child ``kanban_block``-ed the parent card. Same class.
3. A goal-mode worker's valid ``kanban_request_review(reviewer="reviewer")``
   was rejected by the goal judge because final acceptance had not already
   happened — a circular demand: requesting review IS how the worker asks for
   acceptance. The judge must evaluate a review handoff against "is the
   implementation ready for a reviewer", not "is the overall goal accepted".
4. A direct (dispatcher-spawned) Kanban worker must still see the full
   lifecycle tool surface — child isolation must not leak into worker context.
5. After a reviewer returns changes, the parent implementation task returns to
   its implementer and the goal loop reports ``changes_requested_by_reviewer``
   without running another turn.

Isolation layers 1–2 have deeper guards already (tools/kanban_tools.py
handler guards, hermes_cli/kanban.py CLI fast-fail, hermes_cli/kanban_db.py
DB-layer write_txn fail-closed); these tests pin the behavior end to end at
each layer so a regression in any one layer goes red.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _python_with_repo_path(code: str) -> str:
    """Shell command running *code* with the repo under test on PYTHONPATH."""
    return (
        f"PYTHONPATH={shlex.quote(str(_REPO_ROOT))} "
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    )


@pytest.fixture
def claimed_parent_task(monkeypatch, tmp_path):
    """Isolated HERMES_HOME with one claimed (running) parent task and the
    dispatcher worker env vars set — simulates a dispatcher-spawned worker
    that is about to delegate an investigation child."""
    home = tmp_path / ".hermes"
    home.mkdir()
    workspace = tmp_path / "parent-workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "parent-worker")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="parent implementation task",
            assignee="parent-worker",
            workspace_kind="scratch",
            workspace_path=str(workspace),
        )
        claim = kb.claim_task(conn, tid)
        assert claim is not None
        run_id = claim.id
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


# ---------------------------------------------------------------------------
# 1) Premature child completion of the parent card
# ---------------------------------------------------------------------------


def test_child_cannot_complete_parent_by_explicit_task_id(claimed_parent_task):
    """A delegated child passing the parent's explicit task_id to
    kanban_complete must be refused at the tool layer (guard runs before
    the task_id defaulting / ownership logic)."""
    from agent.delegation_context import delegated_child_context
    from tools import kanban_tools as kt

    with delegated_child_context():
        raw = kt._handle_complete({
            "task_id": claimed_parent_task,
            "summary": "child finishing the parent's work",
        })

    payload = json.loads(raw)
    assert "error" in payload
    assert "delegate_task child" in payload["error"]

    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        assert kb.get_task(conn, claimed_parent_task).status == "running"
    finally:
        conn.close()


def test_child_cannot_complete_parent_via_cli(claimed_parent_task):
    """The CLI fast-fail layer must refuse a delegated child running
    ``hermes kanban complete`` against the parent card."""
    from agent.delegation_context import delegated_child_context
    from tools.environments.local import LocalEnvironment

    code = (
        "from hermes_cli import kanban; "
        "import argparse; "
        "p=argparse.ArgumentParser(); "
        "sub=p.add_subparsers(dest='cmd'); "
        "kanban.build_parser(sub); "
        f"args=p.parse_args(['kanban','complete','{claimed_parent_task}',"
        "'--summary','child did it']); "
        "raise SystemExit(kanban.kanban_command(args))"
    )
    env = LocalEnvironment(cwd=str(_REPO_ROOT), timeout=30)
    try:
        with delegated_child_context():
            result = env.execute(_python_with_repo_path(code), timeout=30)
    finally:
        env.cleanup()

    assert result["returncode"] == 1
    assert "delegate_task child contexts cannot mutate" in result["output"]

    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        assert kb.get_task(conn, claimed_parent_task).status == "running"
    finally:
        conn.close()


def test_child_db_layer_write_txn_fail_closed(claimed_parent_task):
    """Trust boundary: a delegated child importing kanban_db directly and
    calling complete_task must hit the DB-layer PermissionError, and the
    card must stay untouched."""
    from agent.delegation_context import delegated_child_context
    from hermes_cli import kanban_db as kb

    with delegated_child_context():
        conn = kb.connect()
        try:
            with pytest.raises(PermissionError):
                kb.complete_task(conn, claimed_parent_task, summary="direct db write")
        finally:
            conn.close()

    conn = kb.connect()
    try:
        assert kb.get_task(conn, claimed_parent_task).status == "running"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 2) Accidental child block of the parent card
# ---------------------------------------------------------------------------


def test_child_cannot_block_parent_by_explicit_task_id(claimed_parent_task):
    from agent.delegation_context import delegated_child_context
    from tools import kanban_tools as kt

    with delegated_child_context():
        raw = kt._handle_block({
            "task_id": claimed_parent_task,
            "reason": "child thinks this needs input",
        })

    payload = json.loads(raw)
    assert "error" in payload
    assert "delegate_task child" in payload["error"]

    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        assert kb.get_task(conn, claimed_parent_task).status == "running"
    finally:
        conn.close()


def test_child_cannot_request_review_or_request_changes_on_parent(
    claimed_parent_task,
):
    """kanban_request_review / kanban_request_changes / kanban_heartbeat are
    run-lifecycle mutations too — a child must not touch any of them."""
    from agent.delegation_context import delegated_child_context
    from tools import kanban_tools as kt

    with delegated_child_context():
        for handler, args in (
            (kt._handle_request_review, {"summary": "child hands off"}),
            (kt._handle_request_changes, {"reason": "child wants changes"}),
            (kt._handle_heartbeat, {"note": "child alive"}),
        ):
            raw = handler({"task_id": claimed_parent_task, **args})
            payload = json.loads(raw)
            assert "error" in payload, f"{handler.__name__} must be refused"
            assert "delegate_task child" in payload["error"]

    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        assert kb.get_task(conn, claimed_parent_task).status == "running"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3) Valid review handoff must not be circularly rejected by the judge
# ---------------------------------------------------------------------------


def test_goal_mode_request_review_with_reviewer_is_not_circularly_rejected(
    monkeypatch, tmp_path
):
    """The audit's headline bug: a goal-mode worker calls
    kanban_request_review(reviewer="reviewer") with a solid implementation
    summary, but the judge says CONTINUE because *final acceptance* hasn't
    happened. Review IS the acceptance path — the handoff must go through.

    The judge for a review handoff must be asked a handoff-scoped question
    (is the implementation ready for a reviewer), not scored against the
    card's full acceptance contract.
    """
    from pathlib import Path as _Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="Fix the flaky parser test",
            body="Fix mechanically, add a regression, run scripts/run_tests.sh.",
            assignee="builder",
            goal_mode=True,
        )
        claim = kb.claim_task(conn, tid)
        assert claim is not None
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claim.current_run_id))

    from tools import kanban_tools as kt

    monkeypatch.setattr(kt, "_goal_judge_available", lambda: True)

    captured: dict = {}

    def fake_judge(goal, last_response, **kwargs):
        captured["goal"] = goal
        captured["response"] = last_response
        # Judge says "not done yet" — under the old behavior this
        # circularly rejected the review handoff because acceptance
        # hadn't happened (acceptance is what review produces).
        return "continue", "final acceptance has not happened yet", False, None, False

    monkeypatch.setattr(kt, "judge_goal", fake_judge)

    raw = kt._handle_request_review({
        "summary": (
            "Implemented the fix on branch wt/x; added regression "
            "tests/test_parser.py::test_flaky_case; ran scripts/run_tests.sh: 12 passed."
        ),
        "metadata": {"tests_run": 12},
        "reviewer": "reviewer",
    })
    payload = json.loads(raw)
    assert payload.get("ok") is True, (
        f"valid review handoff must not be circularly rejected: {payload}"
    )

    with kb.connect() as c:
        task = kb.get_task(c, tid)
        assert task.status == "review"
        assert task.assignee == "reviewer"


def test_goal_mode_review_handoff_still_rejects_unachievable(monkeypatch, tmp_path):
    """The gate keeps its teeth for BLOCKED: a judge ruling the goal
    unachievable must still stop a review handoff (#100954)."""
    from pathlib import Path as _Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="Impossible task",
            body="Cannot be done.",
            assignee="builder",
            goal_mode=True,
        )
        claim = kb.claim_task(conn, tid)
        assert claim is not None
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claim.current_run_id))

    from tools import kanban_tools as kt

    monkeypatch.setattr(kt, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(
        kt,
        "judge_goal",
        lambda *a, **k: ("blocked", "repository does not exist", False, None, False),
    )

    payload = json.loads(kt._handle_request_review({"summary": "Looks ready."}))
    assert "error" in payload
    assert "unachievable" in payload["error"]
    with kb.connect() as c:
        assert kb.get_task(c, tid).status == "running"


def test_goal_loop_review_handoff_emits_review_requested_by_worker(monkeypatch):
    """The goal loop must treat a worker's kanban_request_review as a clean
    terminal outcome — implementation ready, awaiting reviewer — without
    consulting the judge for overall acceptance."""
    from hermes_cli import goals

    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *a, **k: pytest.fail("terminal review handoff must not be judged"),
    )
    result = goals.run_kanban_goal_loop(
        task_id="t_review",
        goal_text="implement the thing",
        run_turn=lambda prompt: pytest.fail("must not run another turn"),
        task_status_fn=lambda: "review",
        block_fn=lambda reason: pytest.fail("must not block"),
        first_response="Requested review from reviewer.",
    )
    assert result["outcome"] == "review_requested_by_worker"
    assert result["turns_used"] == 1


def test_delegated_investigation_prompt_uses_child_goal_not_parent_card(
    monkeypatch,
):
    """An investigation-only child is scoped to its delegated goal, not the
    parent Kanban implementation acceptance. The inherited dispatcher task id
    must not cause parent card text/protocol to replace the child's own goal."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    from tools.delegate_tool import _build_child_system_prompt

    child_goal = "Investigate the parser race and return evidence only."
    prompt = _build_child_system_prompt(child_goal)

    assert f"YOUR TASK:\n{child_goal}" in prompt
    assert "work kanban task t_parent" not in prompt
    assert "kanban_complete" not in prompt
    assert "kanban_block" not in prompt


# ---------------------------------------------------------------------------
# 4) Direct workers keep their lifecycle tools
# ---------------------------------------------------------------------------


def test_direct_worker_tools_visible_when_child_context_absent(
    monkeypatch, tmp_path
):
    """Child isolation must be context-scoped: outside delegated_child_context
    (i.e. a genuine dispatcher-spawned worker), the full lifecycle surface is
    visible in the tool schema even with HERMES_KANBAN_TASK inherited in env."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_directworker")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

    import tools.kanban_tools  # noqa: F401 - ensure registered
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    schema = get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    assert {
        "kanban_complete", "kanban_block", "kanban_heartbeat",
        "kanban_comment", "kanban_request_review", "kanban_request_changes",
    } <= names, f"direct worker lost lifecycle tools: {sorted(n for n in names if n.startswith('kanban_'))}"


def test_child_context_hides_lifecycle_tools_but_direct_worker_context_restores_them(
    monkeypatch, tmp_path
):
    """Same process, both states: a child sees zero kanban tools; once the
    child context exits, the direct worker surface returns (the audit's
    'HERMES_DELEGATED_CHILD_CONTEXT hid kanban worker tools' must stay
    scoped to child contexts only)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

    import tools.kanban_tools  # noqa: F401 - ensure registered
    from agent.delegation_context import delegated_child_context
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    with delegated_child_context():
        child_schema = get_tool_definitions(
            enabled_toolsets=["terminal"], quiet_mode=True
        )
    child_names = {
        s["function"].get("name") for s in child_schema if "function" in s
    }
    assert {n for n in child_names if n and n.startswith("kanban_")} == set()

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    worker_schema = get_tool_definitions(
        enabled_toolsets=["terminal"], quiet_mode=True
    )
    worker_names = {
        s["function"].get("name") for s in worker_schema if "function" in s
    }
    assert "kanban_complete" in worker_names
    assert "kanban_block" in worker_names


# ---------------------------------------------------------------------------
# 5) Parent resumption after reviewer changes
# ---------------------------------------------------------------------------


def test_parent_resumes_after_reviewer_changes_then_completes(
    monkeypatch, tmp_path
):
    """Full review round trip on a goal-mode card: implementer hands off,
    reviewer requests changes, card returns to the implementer in ready,
    the goal loop reports changes_requested_by_reviewer (no further turns),
    and the implementer can complete on the retry."""
    from pathlib import Path as _Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="Implement feature",
            body="Do it well.",
            assignee="builder",
            goal_mode=True,
        )
        claim = kb.claim_task(conn, tid)
        assert claim is not None
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claim.current_run_id))

    from tools import kanban_tools as kt

    # Implementer hands off (judge unavailable -> gate fails open, which is
    # the documented behavior for an unconfigured auxiliary judge).
    payload = json.loads(kt._handle_request_review({
        "summary": "Implementation ready for review.",
        "reviewer": "reviewer",
    }))
    assert payload.get("ok") is True

    with kb.connect() as c:
        review = kb.claim_review_task(c, tid, claimer="reviewer:1")
        assert review is not None
    monkeypatch.setenv("HERMES_PROFILE", "reviewer")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))

    changed = json.loads(kt._handle_request_changes({
        "reason": "Add a boundary assertion for empty input.",
    }))
    assert changed["ok"] is True
    assert changed["implementer"] == "builder"

    with kb.connect() as c:
        task = kb.get_task(c, tid)
        assert task.status == "ready"
        assert task.assignee == "builder"

    # The goal loop binds to the ORIGINAL worker run: once the implementer
    # handed off for review, that run's terminal outcome is
    # review_requested — the loop stops cleanly (no further turns, no
    # block). Resumption is a fresh dispatcher run (re-claim below), not
    # this loop reviving.
    from hermes_cli import goals

    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *a, **k: pytest.fail("terminal review verdict must not be judged"),
    )
    def original_run_status():
        with kb.connect() as status_conn:
            return kb.goal_run_status(
                status_conn, tid, expected_run_id=claim.current_run_id
            )

    res = goals.run_kanban_goal_loop(
        task_id=tid,
        goal_text="Implement feature. Do it well.",
        run_turn=lambda prompt: pytest.fail("must not run another turn"),
        task_status_fn=original_run_status,
        block_fn=lambda reason: pytest.fail("must not block"),
        first_response="Requested review.",
    )
    assert res["outcome"] == "review_requested_by_worker"
    assert res["turns_used"] == 1

    # Implementer resumes: re-claim, fix, complete.
    with kb.connect() as c:
        re_claim = kb.claim_task(c, tid, claimer="builder:2")
        assert re_claim is not None
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(re_claim.current_run_id))

    done = json.loads(kt._handle_complete({
        "summary": "Addressed the boundary assertion; tests pass.",
    }))
    assert done.get("ok") is True
    with kb.connect() as c:
        assert kb.get_task(c, tid).status == "done"
