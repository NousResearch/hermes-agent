"""Tests for the Kanban tool surface (tools/kanban_tools.py).

Verifies:
  - Tools are gated on HERMES_KANBAN_TASK: a normal chat session sees
    zero kanban tools in its schema; a worker session sees the kanban set.
  - Each handler's happy path.
  - Error paths (missing required args, bad metadata type, etc).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def test_kanban_tools_hidden_without_env_var(monkeypatch, tmp_path):
    """Normal `hermes chat` sessions (no HERMES_KANBAN_TASK) must have
    zero kanban_* tools in their schema."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.kanban_tools  # ensure registered
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    kanban = {n for n in names if n and n.startswith("kanban_")}
    assert kanban == set(), (
        f"kanban tools leaked into normal chat schema: {kanban}"
    )


# ---------------------------------------------------------------------------
# Handler happy paths
# ---------------------------------------------------------------------------

@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Simulate being a worker: HERMES_HOME isolated, HERMES_KANBAN_TASK set
    after we've created the task."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


def test_show_defaults_to_env_task_id(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_show({})
    d = json.loads(out)
    assert "task" in d
    assert d["task"]["id"] == worker_env
    assert d["task"]["status"] == "running"
    assert "worker_context" in d
    assert "runs" in d


def test_list_filters_tasks(monkeypatch, worker_env):
    """kanban_list gives orchestrators filtered board discovery."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        a = kb.create_task(conn, title="alpha", assignee="factory", priority=5)
        b = kb.create_task(conn, title="beta", assignee="reviewer")
        c = kb.create_task(conn, title="gamma", assignee="factory", tenant="other")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_list({"assignee": "factory", "status": "ready", "limit": 10})
    d = json.loads(out)
    ids = [t["id"] for t in d["tasks"]]
    assert ids == [a, c]
    assert d["count"] == 2
    assert d["tasks"][0]["title"] == "alpha"
    assert d["tasks"][0]["parent_count"] == 0
    assert b not in ids

    tenant_out = kt._handle_list({
        "assignee": "factory",
        "status": "ready",
        "tenant": "other",
    })
    tenant_ids = [t["id"] for t in json.loads(tenant_out)["tasks"]]
    assert tenant_ids == [c]


def test_complete_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_complete({
        "summary": "got the thing done",
        "metadata": {"files": 2},
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["task_id"] == worker_env
    # Verify via kernel
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        run = kb.latest_run(conn, worker_env)
        assert run.outcome == "completed"
        assert run.summary == "got the thing done"
        assert run.metadata == {"files": 2}
    finally:
        conn.close()


def test_complete_retry_with_empty_created_cards_succeeds(worker_env):
    """After a phantom rejection, retrying kanban_complete with
    created_cards=[] (the documented escape hatch) must complete the
    task. Regression for #22923."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    # Hit the gate first.
    rejected = json.loads(kt._handle_complete({
        "summary": "oops",
        "created_cards": ["t_phantomdeadbeef"],
    }))
    assert rejected.get("error")

    # Retry with the escape hatch.
    ok = json.loads(kt._handle_complete({
        "summary": "retry without claims",
        "created_cards": [],
    }))
    assert ok.get("ok") is True

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "done"
    finally:
        conn.close()


def test_complete_goal_mode_rejected_by_judge(monkeypatch, tmp_path):
    """Goal-mode tasks must pass the auxiliary judge before completion.
    Regression for #38367: workers bypassing the judge via early kanban_complete."""
    from pathlib import Path as _Path
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    # Set up isolated HERMES_HOME
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

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

    # Mock the judge to reject the completion. The gate only runs when a
    # judge is reachable, so force the availability probe True as well.
    def mock_judge_goal(goal, last_response, *, timeout=30.0, subgoals=None):
        # Match the real judge_goal contract:
        # (verdict, reason, parse_failed, wait_directive, transport_failed)
        return "continue", "missing verification evidence", False, None, False

    monkeypatch.setattr("tools.kanban_tools.judge_goal", mock_judge_goal)
    monkeypatch.setattr("tools.kanban_tools._goal_judge_available", lambda: True)

    # Attempt to complete should be rejected
    out = kt._handle_complete({"summary": "I did some stuff but not X"})
    d = json.loads(out)
    assert "error" in d
    assert "Goal completion rejected by judge" in d["error"]
    assert "missing verification evidence" in d["error"]
    assert f"parents=[{goal_task_id}]" in d["error"]

    # Verify the task is NOT completed in the DB
    conn2 = kbc.connect()
    try:
        task = kb.get_task(conn2, goal_task_id)
        assert task.status == "running"  # Should still be running, not done
    finally:
        conn2.close()


def test_complete_goal_mode_planner_handoff_includes_blocked_child_graph(
    monkeypatch, tmp_path
):
    """A planner must be judged on its routing handoff, not evidence that can
    only be produced by children still dependency-gated on that handoff."""
    from pathlib import Path as _Path
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "planner")
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as conn:
        planner = kb.create_task(
            conn,
            title="Decompose the implementation",
            body="Decide interfaces and route implementation to specialists.",
            assignee="planner",
            goal_mode=True,
        )
        claimed = kb.claim_task(conn, planner)
        assert claimed is not None
        child = kb.create_task(
            conn,
            title="Implement the decided interface",
            assignee="builder",
            parents=[planner],
        )
    monkeypatch.setenv("HERMES_KANBAN_TASK", planner)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))

    seen_goal = ""

    def graph_aware_judge(goal, last_response, **kwargs):
        nonlocal seen_goal
        seen_goal = goal
        if (
            "dependency-gated until this task completes" in goal
            and child in goal
            and "Implement the decided interface" in goal
        ):
            return "done", "routing handoff satisfies this planner card", False, None, False
        return "continue", "waiting for downstream implementation evidence", False, None, False

    monkeypatch.setattr(kt, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(kt, "judge_goal", graph_aware_judge)

    result = json.loads(kt._handle_complete({
        "summary": "Interfaces decided; implementation routed to the builder child."
    }))

    assert result.get("ok") is True, result
    assert child in seen_goal
    with kb.connect() as conn:
        planner_after = kb.get_task(conn, planner)
        child_after = kb.get_task(conn, child)
        assert planner_after is not None
        assert child_after is not None
        assert planner_after.status == "done"
        assert child_after.status == "ready"


def test_complete_goal_mode_implementation_with_gated_child_stays_running(
    monkeypatch, tmp_path
):
    """A todo child cannot replace this implementation card's own evidence."""
    from pathlib import Path as _Path
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as conn:
        implementation = kb.create_task(
            conn,
            title="Implement the notification guard",
            body="Change the code and run the focused regression tests.",
            assignee="builder",
            goal_mode=True,
        )
        claimed = kb.claim_task(conn, implementation)
        assert claimed is not None
        child = kb.create_task(
            conn,
            title="Run downstream QA",
            assignee="reviewer",
            parents=[implementation],
        )
    monkeypatch.setenv("HERMES_KANBAN_TASK", implementation)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))

    def strict_judge(goal, last_response, **kwargs):
        assert "implementation card" in goal
        assert "not a substitute" in goal
        assert child in goal
        return "continue", "missing implementation and test evidence", False, None, False

    monkeypatch.setattr(kt, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(kt, "judge_goal", strict_judge)

    result = json.loads(kt._handle_complete({
        "summary": "Created the QA child; it will implement and verify the fix."
    }))

    assert "error" in result
    assert "missing implementation and test evidence" in result["error"]
    with kb.connect() as conn:
        implementation_after = kb.get_task(conn, implementation)
        child_after = kb.get_task(conn, child)
        assert implementation_after is not None
        assert child_after is not None
        assert implementation_after.status == "running"
        assert child_after.status == "todo"


def test_block_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_block({"reason": "need clarification"})
    d = json.loads(out)
    assert d["ok"] is True
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "blocked"
    finally:
        conn.close()


def _make_goal_mode_worker_env(monkeypatch, tmp_path):
    """Set up an isolated HERMES_HOME with one claimed goal_mode task,
    matching the pattern used by the kanban_complete judge gate tests."""
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
    conn = kbc.connect()
    try:
        goal_task_id = kb.create_task(
            conn, title="goal-mode-block-test", assignee="test-worker",
            body="Must achieve X.", goal_mode=True,
        )
        kb.claim_task(conn, goal_task_id)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", goal_task_id)
    return goal_task_id


def test_block_goal_mode_rejects_missing_kind(monkeypatch, tmp_path):
    """A goal_mode worker calling kanban_block with no kind must not be able
    to use it as an unguarded escape from the goal loop (Issue #38696,
    sibling of the kanban_complete judge gate / Issue #38367)."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = _make_goal_mode_worker_env(monkeypatch, tmp_path)
    out = kt._handle_block({"reason": "giving up"})
    d = json.loads(out)
    assert "error" in d
    assert "goal_mode" in d["error"]

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()


def test_block_goal_mode_rejects_disallowed_kind(monkeypatch, tmp_path):
    """`capability` / `transient` are valid kinds in general but must not
    let a goal_mode worker exit the loop without going through the judge."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = _make_goal_mode_worker_env(monkeypatch, tmp_path)
    for kind in ("capability", "transient"):
        out = kt._handle_block({"reason": "blocked", "kind": kind})
        d = json.loads(out)
        assert "error" in d, f"kind={kind} should be rejected for goal_mode"

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()


def test_heartbeat_extends_claim_expires(worker_env):
    """The kanban_heartbeat tool MUST extend claim_expires, not just
    update last_heartbeat_at — otherwise long-running workers loop the
    heartbeat tool diligently and still get reclaimed by
    release_stale_claims at DEFAULT_CLAIM_TTL_SECONDS.

    Regression test for the bug where _handle_heartbeat called
    heartbeat_worker but never heartbeat_claim, so claim_expires sat
    static while last_heartbeat_at advanced.
    """
    import time as _time
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    # Rewind claim_expires into the past so any forward movement is
    # unambiguous (avoids time.sleep flakiness).
    conn = kbc.connect()
    try:
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (1, worker_env),
        )
        conn.commit()
        before = conn.execute(
            "SELECT claim_expires FROM tasks WHERE id = ?", (worker_env,)
        ).fetchone()["claim_expires"]
    finally:
        conn.close()
    assert before == 1

    out = kt._handle_heartbeat({"note": "still alive"})
    assert json.loads(out).get("ok") is True

    conn = kbc.connect()
    try:
        after = conn.execute(
            "SELECT claim_expires FROM tasks WHERE id = ?", (worker_env,)
        ).fetchone()["claim_expires"]
    finally:
        conn.close()

    now = int(_time.time())
    # claim_expires should be roughly now + DEFAULT_CLAIM_TTL_SECONDS.
    # We assert a generous floor (now + half the default TTL) to keep the
    # test stable against future TTL changes.
    assert after > before, (
        f"claim_expires did not advance ({before} -> {after}); workers "
        f"would be reclaimed at TTL despite heartbeating"
    )
    assert after >= now + (kb.DEFAULT_CLAIM_TTL_SECONDS // 2), (
        f"claim_expires={after} is suspiciously close to now={now}; "
        f"expected at least now + {kb.DEFAULT_CLAIM_TTL_SECONDS // 2}"
    )


def test_comment_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_comment({
        "task_id": worker_env,
        "body": "hello thread",
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["comment_id"]
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        comments = kb.list_comments(conn, worker_env)
        assert len(comments) == 1
        # Author defaults to HERMES_PROFILE env we set in the fixture
        assert comments[0].author == "test-worker"
        assert comments[0].body == "hello thread"
    finally:
        conn.close()


def test_comment_ignores_caller_supplied_author(worker_env):
    """``args["author"]`` is no longer honored — the author is always
    derived from ``HERMES_PROFILE`` so a worker can't forge a comment
    under an authoritative-looking name like ``hermes-system`` and
    poison the next worker's prompt context. Cross-task commenting
    itself remains unrestricted (see #19713); only the author override
    is removed.
    """
    from tools import kanban_tools as kt
    out = kt._handle_comment({
        "task_id": worker_env, "body": "hi", "author": "hermes-system",
    })
    assert json.loads(out)["ok"]
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        comments = kb.list_comments(conn, worker_env)
        # Author comes from HERMES_PROFILE in the fixture, not the
        # caller-supplied "hermes-system" override.
        assert comments[0].author == "test-worker"
    finally:
        conn.close()


def test_create_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_create({
        "title": "child task",
        "assignee": "peer",
        "parents": [worker_env],
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["task_id"]
    assert d["status"] == "todo"  # parent isn't done yet
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        child = kb.get_task(conn, d["task_id"])
        assert child.title == "child task"
        assert child.assignee == "peer"
    finally:
        conn.close()


def test_link_happy_path(worker_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        a = kb.create_task(conn, title="A", assignee="x")
        b = kb.create_task(conn, title="B", assignee="x")
    finally:
        conn.close()
    from tools import kanban_tools as kt
    out = kt._handle_link({"parent_id": a, "child_id": b})
    d = json.loads(out)
    assert d["ok"] is True


def test_unblock_happy_path(monkeypatch, worker_env):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="blocked", assignee="worker")
        kb.block_task(conn, tid, reason="waiting")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_unblock({"task_id": tid})
    d = json.loads(out)
    assert d["ok"] is True
    assert d["status"] == "ready"

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, tid).status == "ready"
    finally:
        conn.close()


def test_unblock_with_pending_parents_returns_todo(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = kb.create_task(conn, title="child", assignee="worker", parents=[parent])
        conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (child,))
        conn.commit()
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_unblock({"task_id": child})
    d = json.loads(out)
    assert d["ok"] is True
    assert d["status"] == "todo"

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, child).status == "todo"
    finally:
        conn.close()


def test_worker_lifecycle_through_tools(worker_env):
    """Drive the full claim -> heartbeat -> comment -> complete lifecycle
    exclusively through the tools, then verify the DB state matches what
    the dispatcher/notifier expect."""
    from tools import kanban_tools as kt

    # 1. show — worker orientation
    show = json.loads(kt._handle_show({}))
    assert show["task"]["id"] == worker_env

    # 2. heartbeat during long op
    assert json.loads(kt._handle_heartbeat({"note": "warming up"}))["ok"]

    # 3. comment for a future peer
    assert json.loads(kt._handle_comment({
        "task_id": worker_env,
        "body": "note: using stdlib sqlite3 bindings",
    }))["ok"]

    # 4. spawn a child task for follow-up
    child_out = json.loads(kt._handle_create({
        "title": "write integration test",
        "assignee": "qa",
        "parents": [worker_env],
    }))
    assert child_out["ok"]

    # 5. complete with structured handoff
    comp = json.loads(kt._handle_complete({
        "summary": "implemented + spawned QA follow-up",
        "metadata": {"child_task": child_out["task_id"]},
    }))
    assert comp["ok"]

    # Verify final state
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        parent = kb.get_task(conn, worker_env)
        assert parent.status == "done"
        assert parent.current_run_id is None
        run = kb.latest_run(conn, worker_env)
        assert run.outcome == "completed"
        assert run.metadata == {"child_task": child_out["task_id"]}
        # Child is todo (parent just finished, but recompute_ready may
        # have promoted it — complete_task runs recompute internally).
        child = kb.get_task(conn, child_out["task_id"])
        assert child.status == "ready", (
            f"child should be ready after parent done, got {child.status}"
        )
        # Comment is visible
        assert len(kb.list_comments(conn, worker_env)) == 1
        # Heartbeat event recorded
        hb = [e for e in kb.list_events(conn, worker_env) if e.kind == "heartbeat"]
        assert len(hb) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# System-prompt guidance injection
# ---------------------------------------------------------------------------


def test_kanban_guidance_prompt_size_bounded():
    """KANBAN_GUIDANCE is injected into every kanban-capable process's system
    prompt and resolved once at agent init, so its size is a per-worker token
    tax paid on every spawn. Bound it as an invariant, not a change-detector:
    the ceiling (8000 chars, roughly 2000 tokens) leaves headroom above the
    current ~6.2k chars for tight additions, while catching accidental bloat
    (pasted docs, duplicated sections) before it ships to every worker.
    """
    from agent.prompt_builder import KANBAN_GUIDANCE

    assert len(KANBAN_GUIDANCE) < 8000, (
        f"KANBAN_GUIDANCE is {len(KANBAN_GUIDANCE)} chars; it is injected into "
        "every kanban worker's system prompt — trim it or consciously re-bound "
        "this invariant with justification."
    )


def test_kanban_guidance_orchestrator_decision_ownership():
    """The orchestrator section must carry the split-brain prevention
    contract: decisions are made by the orchestrator before fan-out and
    stamped into every dependent card body."""
    from agent.prompt_builder import KANBAN_GUIDANCE

    assert KANBAN_GUIDANCE.count("Decision ownership.") == 1
    assert "Never let two subtree cards decide the same question" in KANBAN_GUIDANCE
    assert "workers cannot see sibling context" in KANBAN_GUIDANCE


# ---------------------------------------------------------------------------
# Worker task-ownership enforcement (regression tests for #19534)
# ---------------------------------------------------------------------------
#
# A worker process has HERMES_KANBAN_TASK set to its own task id. The
# destructive tools (kanban_complete, kanban_block, kanban_heartbeat,
# kanban_unblock) must refuse to operate
# on any OTHER task id, even if the caller supplies an explicit `task_id`
# argument. Workers legitimately call kanban_show / kanban_list /
# kanban_comment / kanban_create / kanban_link on other tasks, so those
# are unrestricted.
#
# Orchestrator profiles (no HERMES_KANBAN_TASK in env) are intentionally
# exempt — their job is routing, and they sometimes close out child
# tasks on behalf of the child.


def test_worker_complete_rejects_foreign_task_id(worker_env):
    """A worker cannot complete a task that isn't its own (#19534)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        other = kb.create_task(conn, title="sibling")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (other,))
        conn.commit()
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_complete({"task_id": other, "summary": "HIJACK"})
    d = json.loads(out)
    assert d.get("ok") is not True
    assert "refusing to mutate" in d.get("error", "")

    # Sibling task must be untouched.
    conn = kbc.connect()
    try:
        assert kb.get_task(conn, other).status == "ready"
    finally:
        conn.close()


def test_worker_can_comment_on_foreign_task(worker_env):
    """Cross-task commenting must remain unrestricted (#19713 policy).

    The author-forgery hardening removed args['author'] but deliberately
    did NOT add an ownership gate to kanban_comment — comments are the
    documented handoff channel between tasks. This test pins that policy
    so a future change accidentally adding ``_enforce_worker_task_ownership``
    to ``_handle_comment`` would fail CI immediately.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        other = kb.create_task(conn, title="sibling")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_comment({
        "task_id": other,
        "body": "handoff: see prior findings before starting",
    })
    d = json.loads(out)
    assert d.get("ok") is True, f"cross-task comment must succeed: {d}"

    # The comment lands on the foreign task, attributed to the worker's
    # HERMES_PROFILE — never to a caller-controlled string.
    conn = kbc.connect()
    try:
        comments = kb.list_comments(conn, other)
        assert len(comments) == 1
        assert comments[0].author == "test-worker"
        assert comments[0].body.startswith("handoff:")
    finally:
        conn.close()


def test_worker_unblock_rejects_foreign_task_id(worker_env):
    """A worker cannot unblock any task — kanban_unblock is orchestrator-only.

    The check fires before the per-task ownership check, so the error
    surface is the orchestrator-only refusal rather than the
    cross-task-ownership refusal. Either is fine — the property we're
    pinning is "worker cannot mutate foreign task via kanban_unblock".
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        other = kb.create_task(conn, title="blocked sibling", assignee="peer")
        kb.block_task(conn, other, reason="waiting")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_unblock({"task_id": other})
    d = json.loads(out)
    err = d.get("error", "")
    assert "orchestrator-only" in err or "refusing to mutate" in err, (
        f"expected worker-rejection error, got {err}"
    )

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, other).status == "blocked"
    finally:
        conn.close()


def test_orchestrator_complete_any_task_allowed(monkeypatch, tmp_path):
    """Orchestrator profiles (no HERMES_KANBAN_TASK) can still complete
    any task via explicit task_id. The check only applies to workers."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pathlib import Path as _P
    monkeypatch.setattr(_P, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="child to close out")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.commit()
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_complete({"task_id": tid, "summary": "orchestrator close"})
    d = json.loads(out)
    assert d.get("ok") is True and d.get("task_id") == tid


# ---------------------------------------------------------------------------
# Optional ``board`` parameter — per-call DB override
# ---------------------------------------------------------------------------
#
# The dispatcher pins the active board via HERMES_KANBAN_BOARD env var,
# but a Telegram-side orchestrator handling multiple boards needs to be
# able to route a single tool call to a specific board's DB without
# restarting Hermes. These tests pin that ``board=<slug>`` argument
# routes each handler to that board's sqlite file, and that omitting
# ``board`` preserves the legacy env-driven resolution.


@pytest.fixture
def multi_board_env(monkeypatch, tmp_path):
    """Isolated Hermes home with two distinct kanban boards seeded.

    Returns ``("default", "alt")`` slugs. The default board has one
    pre-existing task ``seed_default``; ``alt`` has ``seed_alt``. No
    HERMES_KANBAN_TASK is pinned (orchestrator context) — workers test
    the env-task case via the existing ``worker_env`` fixture.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Make sure neither HERMES_KANBAN_DB nor HERMES_KANBAN_BOARD pin a
    # board — the test is specifically about the per-call override.
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "test-orchestrator")
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    # Default board — implicit
    conn = kbc.connect()
    try:
        seed_default = kb.create_task(
            conn, title="seed-default", assignee="worker-d"
        )
    finally:
        conn.close()
    # Alt board — explicit slug routes the connection to a separate DB
    conn = kbc.connect(board="alt")
    try:
        seed_alt = kb.create_task(
            conn, title="seed-alt", assignee="worker-a"
        )
    finally:
        conn.close()
    return {
        "default_seed": seed_default,
        "alt_seed": seed_alt,
        "default_db": kb.kanban_db_path(),
        "alt_db": kb.kanban_db_path(board="alt"),
    }


def test_board_param_none_falls_back_to_env(worker_env):
    """When ``board`` is omitted or None, behaviour is unchanged from
    before this feature — calls land on whatever the env resolves to.
    Regression guard against accidentally rewiring default resolution."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_show({})  # no board, no task_id
    d = json.loads(out)
    assert d["task"]["id"] == worker_env

    out = kt._handle_show({"task_id": worker_env, "board": None})
    d = json.loads(out)
    assert d["task"]["id"] == worker_env

    # Sanity: the env-resolved path is the legacy default DB, NOT an
    # 'alt' board path. Confirms the override path was not silently
    # forced.
    assert kb.kanban_db_path() == kb.kanban_db_path(board="default")


# ---------------------------------------------------------------------------
# kanban_create auto-subscribe behaviour
#
# When a worker calls kanban_create from inside a session that has a
# persistent delivery channel, the originating session should be
# subscribed to the new task's completion/block events automatically.
# - Gateway sessions: HERMES_SESSION_PLATFORM + HERMES_SESSION_CHAT_ID set.
# - TUI sessions: HERMES_SESSION_KEY (or HERMES_SESSION_ID) set, with
#   the platform/chat_id ContextVars intentionally empty.
# - CLI / cron / test sessions: no delivery channel -> no subscription.
# - Config gate kanban.auto_subscribe_on_create: false -> no subscription
#   even when the session has a delivery channel.
# ---------------------------------------------------------------------------

def _list_subs_for_task(task_id):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_notify as kbn
    conn = kbc.connect()
    try:
        return list(kbn.list_notify_subs(conn, task_id))
    finally:
        conn.close()


def _sub_index(subs):
    """Normalise a list of notify-subs (dicts or objects) into dicts
    keyed by platform+chat_id, so assertions work regardless of the
    return shape."""
    out = []
    for s in subs:
        if isinstance(s, dict):
            out.append(s)
        else:
            out.append({
                "platform": getattr(s, "platform", None),
                "chat_id": getattr(s, "chat_id", None),
                "thread_id": getattr(s, "thread_id", None),
                "user_id": getattr(s, "user_id", None),
                "delivery_metadata": getattr(s, "delivery_metadata", None),
                "notifier_profile": getattr(s, "notifier_profile", None),
            })
    return out


def test_create_subscribes_gateway_session(monkeypatch, worker_env):
    """A gateway session (platform + chat_id set) gets auto-subscribed
    to its own kanban_create result, and the response surfaces the
    ``subscribed`` flag so the orchestrator can react."""
    from tools import kanban_tools as kt
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "chat-42")
    monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "thread-7")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "user-9")
    monkeypatch.setenv("HERMES_SESSION_USER_ID_ALT", "alt-user-9")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "forum")

    out = kt._handle_create({
        "title": "auto-sub gateway",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    new_tid = d["task_id"]
    assert d["subscribed"] is True, d

    subs = _sub_index(_list_subs_for_task(new_tid))
    assert len(subs) == 1
    s = subs[0]
    assert s["platform"] == "telegram"
    assert s["chat_id"] == "chat-42"
    assert s["thread_id"] == "thread-7"
    assert s["user_id"] == "user-9"
    assert s["user_id_alt"] == "alt-user-9"
    assert s["chat_type"] == "forum"
    assert s["delivery_mode"] == "notify+wake"


def test_create_subscribes_tui_session_via_session_key(monkeypatch, worker_env):
    """TUI / desktop sessions don't have a platform/chat_id (single
    local channel), but the parent process exports HERMES_SESSION_KEY.
    We should still auto-subscribe, with platform='tui' and
    chat_id=<key>."""
    from tools import kanban_tools as kt
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)
    monkeypatch.setenv("HERMES_SESSION_KEY", "tui-session-abc")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    out = kt._handle_create({
        "title": "auto-sub tui",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    new_tid = d["task_id"]
    assert d["subscribed"] is True, d

    subs = _sub_index(_list_subs_for_task(new_tid))
    assert len(subs) == 1
    assert subs[0]["platform"] == "tui"
    assert subs[0]["chat_id"] == "tui-session-abc"
    assert subs[0]["chat_type"] == "dm"
    assert subs[0]["delivery_mode"] == "notify"


def test_create_malformed_config_fresh_process_fails_closed_for_gateway(tmp_path):
    """A fresh process must distinguish invalid YAML from legitimate defaults."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "kanban:\n  auto_subscribe_on_create: [unterminated\n",
        encoding="utf-8",
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("HERMES_KANBAN_")
        and not key.startswith("HERMES_SESSION_")
    }
    env.update({
        "HERMES_HOME": str(home),
        "HERMES_PROFILE": "fresh-config-probe",
        "HERMES_SESSION_PLATFORM": "telegram",
        "HERMES_SESSION_CHAT_ID": "must-not-persist",
    })
    probe = """
import json
from hermes_cli import kanban_db as kb
from tools import kanban_tools as kt

kb.init_db()
with kb.connect() as conn:
    worker = kb.create_task(conn, title="worker", assignee="fresh-config-probe")
    kb.claim_task(conn, worker)

import os
os.environ["HERMES_KANBAN_TASK"] = worker
result = json.loads(kt._handle_create({"title": "child", "assignee": "peer"}))
with kb.connect() as conn:
    subscriptions = list(kb.list_notify_subs(conn, result["task_id"]))
print(json.dumps({"result": result, "subscriptions": subscriptions}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(__import__("pathlib").Path(__file__).parents[2]),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(completed.stdout)

    assert observed["result"]["ok"] is True, completed.stderr
    assert observed["result"]["subscribed"] is False
    assert observed["subscriptions"] == []


def test_create_malformed_config_fresh_process_does_not_inherit_parent(tmp_path):
    """Fallback defaults must not authorize parent routes in a fresh worker."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "kanban:\n  auto_subscribe_on_create: [unterminated\n",
        encoding="utf-8",
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("HERMES_KANBAN_")
        and not key.startswith("HERMES_SESSION_")
    }
    env.update({
        "HERMES_HOME": str(home),
        "HERMES_PROFILE": "fresh-parent-probe",
    })
    probe = """
import json
import os
from hermes_cli import kanban_db as kb
from tools import kanban_tools as kt

kb.init_db()
with kb.connect() as conn:
    parent = kb.create_task(conn, title="parent", assignee="fresh-parent-probe")
    kb.claim_task(conn, parent)
    kb.add_notify_sub(
        conn,
        task_id=parent,
        platform="telegram",
        chat_id="must-not-inherit",
        notifier_profile="fresh-parent-probe",
    )
os.environ["HERMES_KANBAN_TASK"] = parent
result = json.loads(kt._handle_create({
    "title": "child",
    "assignee": "peer",
    "parents": [parent],
}))
with kb.connect() as conn:
    subscriptions = list(kb.list_notify_subs(conn, result["task_id"]))
print(json.dumps({"result": result, "subscriptions": subscriptions}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(__import__("pathlib").Path(__file__).parents[2]),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(completed.stdout)

    assert observed["result"]["ok"] is True, completed.stderr
    assert observed["result"]["subscribed"] is False
    assert observed["subscriptions"] == []


def test_create_inherits_parent_with_existing_empty_managed_dir(tmp_path):
    """An absent optional managed config must not suppress a valid user opt-in."""
    home = tmp_path / ".hermes"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    (home / "config.yaml").write_text(
        "kanban:\n  auto_subscribe_on_create: true\n",
        encoding="utf-8",
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("HERMES_KANBAN_")
        and not key.startswith("HERMES_SESSION_")
    }
    env.update({
        "HERMES_HOME": str(home),
        "HERMES_MANAGED_DIR": str(managed),
        "HERMES_PROFILE": "empty-managed-parent-probe",
    })
    probe = """
import json
import os
from hermes_cli import kanban_db as kb
from tools import kanban_tools as kt

kb.init_db()
with kb.connect() as conn:
    parent = kb.create_task(conn, title="parent", assignee="empty-managed-parent-probe")
    kb.claim_task(conn, parent)
    kb.add_notify_sub(
        conn,
        task_id=parent,
        platform="telegram",
        chat_id="parent-route",
        notifier_profile="empty-managed-parent-probe",
    )
os.environ["HERMES_KANBAN_TASK"] = parent
result = json.loads(kt._handle_create({
    "title": "child",
    "assignee": "peer",
    "parents": [parent],
}))
with kb.connect() as conn:
    subscriptions = list(kb.list_notify_subs(conn, result["task_id"]))
print(json.dumps({"result": result, "subscriptions": subscriptions}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(__import__("pathlib").Path(__file__).parents[2]),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(completed.stdout)

    assert observed["result"]["ok"] is True, completed.stderr
    assert observed["result"]["subscribed"] is True
    assert [
        (sub["platform"], sub["chat_id"])
        for sub in observed["subscriptions"]
    ] == [("telegram", "parent-route")]


@pytest.mark.parametrize("failure", ["stat", "read"])
def test_create_warm_cache_managed_failure_creates_no_subscription(
    tmp_path, monkeypatch, failure
):
    """Task creation survives stale consent without persisting any candidate route."""
    from pathlib import Path
    from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
    from hermes_cli import config as config_mod
    from hermes_cli import kanban_db as kb
    from hermes_cli import managed_scope
    from tools import kanban_tools as kt

    home = tmp_path / ".hermes"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    (home / "config.yaml").write_text(
        "kanban:\n"
        "  auto_subscribe_on_create: true\n"
        "  headless_notification_home_platform: telegram\n",
        encoding="utf-8",
    )
    managed_config = managed / "config.yaml"
    if failure == "read":
        managed_config.write_text("display:\n  skin: mono\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("HERMES_PROFILE", "warm-cache-probe")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "live-route-must-not-persist")
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    gateway_cfg = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="test-token",
                home_channel=HomeChannel(
                    platform=Platform.TELEGRAM,
                    chat_id="technical-home-must-not-persist",
                    name="Engineering Ops",
                ),
            )
        }
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_cfg)
    managed_scope.invalidate_managed_cache()
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="parent-owner")
        kb.add_notify_sub(
            conn,
            task_id=parent,
            platform="telegram",
            chat_id="parent-route-must-not-persist",
            notifier_profile="parent-owner",
        )
        worker = kb.create_task(conn, title="worker", assignee="warm-cache-probe")
        claimed = kb.claim_task(conn, worker)
        assert claimed is not None
        kb.add_notify_sub(
            conn,
            task_id=worker,
            platform="telegram",
            chat_id="creator-route-must-not-persist",
            notifier_profile="warm-cache-probe",
        )
    monkeypatch.setenv("HERMES_KANBAN_TASK", worker)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    assert config_mod.load_config_with_status()[1] is True

    if failure == "stat":
        real_stat = Path.stat

        def deny_managed_stat(path, *args, **kwargs):
            if path == managed_config:
                raise PermissionError("synthetic managed stat denial")
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", deny_managed_stat)
    else:
        real_open = open

        def deny_managed_read(file, *args, **kwargs):
            if file == managed_config:
                raise PermissionError("synthetic managed read denial")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr("builtins.open", deny_managed_read)
    result = json.loads(kt._handle_create({
        "title": "child",
        "assignee": "peer",
        "parents": [parent],
    }))

    assert result.get("ok") is True, result
    assert result["subscribed"] is False
    with kb.connect() as conn:
        assert kb.get_task(conn, result["task_id"]) is not None
        assert list(kb.list_notify_subs(conn, result["task_id"])) == []
    managed_scope.invalidate_managed_cache()


@pytest.mark.parametrize("surface", ["gateway", "tui"])
def test_create_config_load_failure_fails_closed_for_live_sessions(
    monkeypatch, worker_env, surface
):
    """Unreadable consent config must not create a gateway or TUI route."""
    from tools import kanban_tools as kt

    if surface == "gateway":
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "private-chat")
        monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    else:
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
        monkeypatch.setenv("HERMES_SESSION_KEY", "private-tui-session")
    monkeypatch.setattr(
        kt,
        "load_config_with_status",
        lambda: (_ for _ in ()).throw(OSError("denied")),
    )

    result = json.loads(kt._handle_create({
        "title": f"config failure from {surface}",
        "assignee": "peer",
    }))

    assert result.get("ok") is True, result
    assert result["subscribed"] is False
    assert _list_subs_for_task(result["task_id"]) == []


def test_headless_config_load_failure_does_not_inherit_creator_origin(
    monkeypatch, worker_env
):
    """Unreadable consent config must also stop durable creator inheritance."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    with kb.connect() as conn:
        kb.add_notify_sub(
            conn,
            task_id=worker_env,
            platform="telegram",
            chat_id="private-creator-origin",
            notifier_profile="creator",
            delivery_mode="notify+wake",
        )
    for key in (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_KEY",
        "HERMES_SESSION_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        kt,
        "load_config_with_status",
        lambda: (_ for _ in ()).throw(OSError("denied")),
    )

    result = json.loads(kt._handle_create({
        "title": "headless child with unreadable config",
        "assignee": "peer",
    }))

    assert result.get("ok") is True, result
    assert result["subscribed"] is False
    assert _list_subs_for_task(result["task_id"]) == []


@pytest.mark.parametrize("config_mode", ["load-error", "disabled"])
def test_create_does_not_inherit_parent_when_auto_subscription_unavailable(
    monkeypatch, worker_env, config_mode
):
    """Every automatic route, including parent inheritance, uses one gate."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    with kb.connect() as conn:
        kb.add_notify_sub(
            conn,
            task_id=worker_env,
            platform="telegram",
            chat_id="private-parent-origin",
            notifier_profile="creator",
            delivery_mode="notify+wake",
        )
    for key in (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_KEY",
        "HERMES_SESSION_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    if config_mode == "load-error":
        monkeypatch.setattr(
            kt,
            "load_config_with_status",
            lambda: (_ for _ in ()).throw(OSError("denied")),
        )
    else:
        monkeypatch.setattr(
            kt,
            "load_config_with_status",
            lambda: ({"kanban": {"auto_subscribe_on_create": False}}, True),
        )

    result = json.loads(kt._handle_create({
        "title": f"parent child with {config_mode}",
        "assignee": "peer",
        "parents": [worker_env],
    }))

    assert result.get("ok") is True, result
    assert result["subscribed"] is False
    assert _list_subs_for_task(result["task_id"]) == []


def test_create_inherits_parent_when_auto_subscription_enabled(monkeypatch, worker_env):
    """The consent gate must preserve valid positive parent inheritance."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    with kb.connect() as conn:
        kb.add_notify_sub(
            conn,
            task_id=worker_env,
            platform="telegram",
            chat_id="allowed-parent-origin",
            notifier_profile="creator",
            delivery_mode="notify+wake",
        )
    for key in (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_KEY",
        "HERMES_SESSION_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        kt,
        "load_config_with_status",
        lambda: ({"kanban": {"auto_subscribe_on_create": True}}, True),
    )

    result = json.loads(kt._handle_create({
        "title": "parent child with consent",
        "assignee": "peer",
        "parents": [worker_env],
    }))

    assert result.get("ok") is True, result
    assert result["subscribed"] is True
    subscriptions = _list_subs_for_task(result["task_id"])
    assert [(sub["platform"], sub["chat_id"]) for sub in subscriptions] == [
        ("telegram", "allowed-parent-origin")
    ]


def test_headless_worker_child_inherits_creator_task_origin(monkeypatch, worker_env):
    """A dispatcher-spawned worker has no live gateway session context, but its
    child must retain the originating task's durable notification route."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    with kb.connect() as conn:
        kb.add_notify_sub(
            conn,
            task_id=worker_env,
            platform="telegram",
            chat_id="technical-chat",
            thread_id="incident-thread",
            user_id="operator",
            notifier_profile="engineering",
            delivery_mode="notify+wake",
        )
    for key in (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_THREAD_ID",
        "HERMES_SESSION_USER_ID",
        "HERMES_SESSION_KEY",
        "HERMES_SESSION_ID",
    ):
        monkeypatch.delenv(key, raising=False)

    result = json.loads(kt._handle_create({
        "title": "headless recovery child",
        "assignee": "engineering",
    }))

    assert result.get("ok") is True, result
    assert result["subscribed"] is True
    subs = _list_subs_for_task(result["task_id"])
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "technical-chat"
    assert subs[0]["thread_id"] == "incident-thread"
    assert subs[0]["notifier_profile"] == "engineering"


def test_headless_worker_uses_explicit_technical_home_fallback(
    monkeypatch, worker_env
):
    """An origin-less worker may fall back only to the platform explicitly
    designated for technical kanban notifications."""
    from pathlib import Path
    from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
    from tools import kanban_tools as kt

    home = Path(__import__("os").environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "kanban:\n  headless_notification_home_platform: telegram\n"
    )
    gateway_cfg = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="test-token",
                home_channel=HomeChannel(
                    platform=Platform.TELEGRAM,
                    chat_id="engineering-home",
                    name="Engineering Ops",
                    thread_id="technical-alerts",
                    user_id="oncall-operator",
                ),
            )
        }
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_cfg)
    for key in (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_KEY",
        "HERMES_SESSION_ID",
    ):
        monkeypatch.delenv(key, raising=False)

    result = json.loads(kt._handle_create({
        "title": "headless technical recovery",
        "assignee": "engineering",
    }))

    assert result.get("ok") is True, result
    assert result["subscribed"] is True
    subs = _list_subs_for_task(result["task_id"])
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "engineering-home"
    assert subs[0]["thread_id"] == "technical-alerts"
    assert subs[0]["user_id"] == "oncall-operator"
    assert subs[0]["notifier_profile"] == "test-worker"


@pytest.mark.parametrize(
    ("enabled", "token"),
    [(False, "test-token"), (True, None)],
    ids=["disabled", "unconfigured"],
)
def test_headless_technical_home_requires_connected_platform(
    monkeypatch, worker_env, enabled, token
):
    """A stale home cannot subscribe when its adapter would not connect."""
    from pathlib import Path
    from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
    from tools import kanban_tools as kt

    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "kanban:\n  headless_notification_home_platform: telegram\n"
    )
    gateway_cfg = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=enabled,
                token=token,
                home_channel=HomeChannel(
                    platform=Platform.TELEGRAM,
                    chat_id="stale-home",
                    name="Stale",
                ),
            )
        }
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_cfg)
    for key in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_KEY"):
        monkeypatch.delenv(key, raising=False)

    result = json.loads(kt._handle_create({"title": "no stale route", "assignee": "peer"}))

    assert result.get("ok") is True, result
    assert result["subscribed"] is False
    assert _list_subs_for_task(result["task_id"]) == []


def test_headless_technical_home_requires_matching_home_platform(
    monkeypatch, worker_env
):
    """A home from another platform cannot be persisted under the opt-in one."""
    from pathlib import Path
    from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
    from tools import kanban_tools as kt

    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "kanban:\n  headless_notification_home_platform: telegram\n"
    )
    gateway_cfg = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="telegram-token",
                home_channel=HomeChannel(
                    platform=Platform.SLACK,
                    chat_id="slack-channel-id",
                    name="Malformed cross-platform home",
                ),
            )
        }
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_cfg)
    for key in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_KEY"):
        monkeypatch.delenv(key, raising=False)

    result = json.loads(kt._handle_create({
        "title": "cross-platform route",
        "assignee": "peer",
    }))

    assert result.get("ok") is True, result
    assert result["subscribed"] is False
    assert _list_subs_for_task(result["task_id"]) == []


def test_headless_technical_home_rejects_unproven_relay_fronted_route(
    monkeypatch, worker_env
):
    """A connected Relay alone does not prove a disabled logical route is live."""
    from pathlib import Path
    from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
    from tools import kanban_tools as kt

    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "kanban:\n  headless_notification_home_platform: slack\n"
    )
    gateway_cfg = GatewayConfig(
        platforms={
            Platform.RELAY: PlatformConfig(
                enabled=True, extra={"relay_url": "wss://relay.invalid"}
            ),
            Platform.SLACK: PlatformConfig(
                enabled=False,
                home_channel=HomeChannel(
                    platform=Platform.SLACK,
                    chat_id="relay-era-home",
                    name="Old relay home",
                ),
            ),
        }
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_cfg)
    for key in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_KEY"):
        monkeypatch.delenv(key, raising=False)

    result = json.loads(kt._handle_create({"title": "relay route", "assignee": "peer"}))

    assert result["subscribed"] is False
    assert _list_subs_for_task(result["task_id"]) == []


@pytest.mark.parametrize(
    ("endpoint", "expected_subscribed"),
    [("configured", True), ("missing", False)],
    ids=["connected", "not-connected"],
)
def test_headless_technical_home_uses_only_connected_plugin_platform(
    monkeypatch, worker_env, endpoint, expected_subscribed
):
    """Plugin is eligible only through its registered connectivity contract."""
    from pathlib import Path
    from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry
    from tools import kanban_tools as kt

    platform_name = "kanban-technical-test-plugin"
    entry = PlatformEntry(
        name=platform_name,
        label="Kanban technical test",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        is_connected=lambda cfg: cfg.extra.get("endpoint") == "configured",
    )
    platform_registry.register(entry)
    platform = Platform(platform_name)
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        f"kanban:\n  headless_notification_home_platform: {platform_name}\n"
    )
    gateway_cfg = GatewayConfig(
        platforms={
            platform: PlatformConfig(
                enabled=True,
                extra={"endpoint": endpoint},
                home_channel=HomeChannel(
                    platform=platform,
                    chat_id="plugin-home",
                    name="Plugin ops",
                ),
            )
        }
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_cfg)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    for key in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_KEY"):
        monkeypatch.delenv(key, raising=False)
    try:
        result = json.loads(kt._handle_create({"title": "plugin route", "assignee": "peer"}))
    finally:
        platform_registry.unregister(platform_name)

    assert result["subscribed"] is expected_subscribed
    subs = _list_subs_for_task(result["task_id"])
    expected_routes = [(platform_name, "plugin-home")] if expected_subscribed else []
    assert [(sub["platform"], sub["chat_id"]) for sub in subs] == expected_routes


def test_headless_worker_does_not_guess_from_ordinary_home_channel(
    monkeypatch, worker_env
):
    """A configured messaging home is not implicitly a technical contract;
    without the kanban opt-in it could be a CEO or other unrelated channel."""
    from types import SimpleNamespace
    from gateway.config import HomeChannel, Platform
    from tools import kanban_tools as kt

    gateway_cfg = SimpleNamespace(
        get_home_channel=lambda platform: HomeChannel(
            platform=Platform.TELEGRAM,
            chat_id="unrelated-home",
            name="Unrelated Home",
        )
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_cfg)
    for key in (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_KEY",
        "HERMES_SESSION_ID",
    ):
        monkeypatch.delenv(key, raising=False)

    result = json.loads(kt._handle_create({
        "title": "must remain unsubscribed",
        "assignee": "engineering",
    }))

    assert result.get("ok") is True, result
    assert result["subscribed"] is False
    assert _list_subs_for_task(result["task_id"]) == []


def test_create_does_not_subscribe_in_cli_session(monkeypatch, worker_env):
    """CLI / cron / test sessions have no persistent delivery channel.
    _maybe_auto_subscribe returns False and no row is written."""
    from tools import kanban_tools as kt
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    out = kt._handle_create({
        "title": "no sub cli",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["subscribed"] is False, d

    assert _list_subs_for_task(d["task_id"]) == []


def test_create_respects_auto_subscribe_on_create_false(monkeypatch, worker_env, tmp_path):
    """The config gate kanban.auto_subscribe_on_create=false must
    suppress auto-subscription even when the session has a delivery
    channel. This is the knob that addresses the upstream design
    concern from PR #19718 (reverted in #19721) — users who want
    explicit kanban_notify-subscribe calls per task get that."""
    # worker_env already created <tmp>/.hermes; use a fresh sibling
    # home to avoid mkdir() colliding with the worker's directory.
    home = tmp_path / "gate-home" / ".hermes"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "kanban:\n  auto_subscribe_on_create: false\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "channel-1")

    from tools import kanban_tools as kt
    out = kt._handle_create({
        "title": "no sub gated",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["subscribed"] is False, d

    assert _list_subs_for_task(d["task_id"]) == []


def test_maybe_auto_subscribe_swallows_add_notify_sub_failure(monkeypatch, worker_env):
    """If add_notify_sub itself raises (e.g. DB locked, schema drift),
    _maybe_auto_subscribe must NOT bubble that up and fail the parent
    kanban_create. The function returns False and the parent create
    still succeeds with subscribed=False."""
    from tools import kanban_tools as kt
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "chat-42")

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_notify as kbn

    def _boom(*a, **kw):
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr(kbn, "add_notify_sub", _boom)

    out = kt._handle_create({
        "title": "auto-sub tolerates add_notify_sub failure",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True, d
    assert d["subscribed"] is False, d


# ---------------------------------------------------------------------------
# Attachments — kanban_attach / kanban_attach_url / kanban_attachments
# ---------------------------------------------------------------------------


@pytest.fixture
def allow_private_urls(monkeypatch):
    """Opt the SSRF guard into private/loopback targets for local fixtures.

    Mirrors a user setting HERMES_ALLOW_PRIVATE_URLS on a private network.
    Resets the url_safety process-lifetime cache on both sides so the
    override neither leaks in nor out of the test.
    """
    from tools import url_safety

    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
    url_safety._reset_allow_private_cache()
    yield
    url_safety._reset_allow_private_cache()


def test_attach_url_rejects_non_http_scheme(worker_env):
    from tools import kanban_tools as kt

    out = kt._handle_attach_url({"url": "file:///etc/passwd"})
    d = json.loads(out)
    assert "error" in d
    assert "scheme" in d["error"]


# ---------------------------------------------------------------------------
# kanban_attach_url — SSRF guard (tools/url_safety.is_safe_url per hop)
# ---------------------------------------------------------------------------


@pytest.fixture
def default_url_guard(monkeypatch):
    """Force the SSRF guard to its secure default for this test.

    Clears HERMES_ALLOW_PRIVATE_URLS and resets url_safety's process-lifetime
    cache on both sides so a prior test's opt-in can't leak in.
    """
    from tools import url_safety

    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    url_safety._reset_allow_private_cache()
    yield
    url_safety._reset_allow_private_cache()


def _assert_attach_url_blocked(worker_env, url):
    """Call kanban_attach_url with ``url`` and assert the SSRF guard fired
    (clean tool error, no attachment row, no network fetch needed)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    out = kt._handle_attach_url({"url": url})
    d = json.loads(out)
    assert "error" in d, out
    assert "SSRF" in d["error"] or "blocked" in d["error"].lower(), out
    conn = kbc.connect()
    try:
        assert kb.list_attachments(conn, worker_env) == []
    finally:
        conn.close()


def test_attach_url_blocks_loopback(worker_env, default_url_guard):
    """http://127.0.0.1/ is rejected before any connection is made."""
    _assert_attach_url_blocked(worker_env, "http://127.0.0.1/")


def _fake_public_dns(monkeypatch, mapping):
    """Patch url_safety's getaddrinfo so hostnames in ``mapping`` resolve to
    the given (public) IPs and literal IPs resolve to themselves — no real
    DNS or network traffic."""
    import ipaddress
    import socket as _socket

    real_af, real_sock = _socket.AF_INET, _socket.SOCK_STREAM

    def fake_getaddrinfo(host, *args, **kwargs):
        ip = mapping.get(host)
        if ip is None:
            # Literal IPs pass through; unknown hostnames fail like NXDOMAIN.
            try:
                ipaddress.ip_address(host)
            except ValueError:
                raise _socket.gaierror(f"fake DNS: unknown host {host!r}")
            ip = host
        return [(real_af, real_sock, 6, "", (ip, 0))]

    from tools import url_safety
    monkeypatch.setattr(url_safety.socket, "getaddrinfo", fake_getaddrinfo)


class _FakeStreamResponse:
    def __init__(self, *, status_code=200, headers=None, body=b""):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    @property
    def is_redirect(self):
        return 300 <= self.status_code < 400 and "location" in {
            k.lower() for k in self.headers
        }

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_bytes(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_attach_url_happy_path_public_host(worker_env, default_url_guard, monkeypatch):
    """A public URL passes the guard and the bytes are stored (mocked fetch)."""
    from pathlib import Path

    import httpx

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    _fake_public_dns(monkeypatch, {"files.example.com": "93.184.216.34"})

    payload = b"public fetch body"

    def fake_stream(method, url, **kwargs):
        assert url == "http://files.example.com/docs/spec.pdf"
        return _FakeStreamResponse(
            status_code=200,
            headers={"content-type": "application/pdf; charset=binary"},
            body=payload,
        )

    monkeypatch.setattr(httpx, "stream", fake_stream)

    out = kt._handle_attach_url({"url": "http://files.example.com/docs/spec.pdf"})
    d = json.loads(out)
    assert d.get("ok") is True, out
    assert d["size"] == len(payload)

    conn = kbc.connect()
    try:
        atts = kb.list_attachments(conn, worker_env)
        assert [a.filename for a in atts] == ["spec.pdf"]
        assert atts[0].content_type == "application/pdf"
        assert Path(atts[0].stored_path).read_bytes() == payload
    finally:
        conn.close()
