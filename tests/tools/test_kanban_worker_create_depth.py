"""Bounded worker-spawned kanban child creation (divide-and-conquer Phase 1).

A dispatcher-spawned worker may fan out with ``kanban_create``, but the new card's
parent-chain depth is capped by ``kanban.max_spawn_depth`` (floor 1, default 1 =
flat fan-out, no ceiling). Depth is a store-level fact computed from the
``task_links`` edges: a card with no parents is depth 0; a card's depth is
1 + max(parent depth). Only dispatcher-owned workers are capped; orchestrator
profiles, CLI and humans are not.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def depth_worker(monkeypatch, tmp_path):
    """Factory building a root->child->... chain of ``depth`` edges, claiming the
    leaf, and pinning the dispatcher env to it. Returns ``(leaf_id, ids)`` where
    ``ids`` is the chain in root-first order (ids[i] sits at depth i).
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_MAX_SPAWN_DEPTH", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    def _make(depth: int):
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc

        kb._INITIALIZED_PATHS.clear()
        kb.init_db()
        conn = kbc.connect()
        try:
            ids = [kb.create_task(conn, title="root-card", assignee="orchestrator")]
            for d in range(depth):
                ids.append(kb.create_task(
                    conn, title=f"level-{d + 1}", assignee="test-worker",
                    parents=[ids[-1]],
                ))
            leaf = ids[-1]
            kb.claim_task(conn, leaf)
            run_id = kb._current_run_id(conn, leaf)
        finally:
            conn.close()
        monkeypatch.setenv("HERMES_KANBAN_TASK", leaf)
        # A real dispatcher pins the run id; mirror that for the ownership proof.
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
        return leaf, ids

    return _make


def _create(args: dict) -> dict:
    from tools import kanban_tools as kt
    return json.loads(kt._handle_create(args))


def _task_count() -> int:
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        return conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Store-level depth: the parent-chain fact every cap is measured against.
# ---------------------------------------------------------------------------

def test_task_depth_is_parent_chain_length(depth_worker):
    leaf, ids = depth_worker(2)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        assert kb.task_depth(conn, ids[0]) == 0
        assert kb.task_depth(conn, ids[1]) == 1
        assert kb.task_depth(conn, leaf) == 2
        # Fan-in measures the deepest parent: 1 + max(1, 0) = 2.
        fan = kb.create_task(conn, title="fan-in", assignee="peer",
                             parents=[ids[0], ids[1]])
        assert kb.task_depth(conn, fan) == 2
    finally:
        conn.close()


def test_task_depth_unknown_task_is_none(depth_worker):
    depth_worker(0)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        assert kb.task_depth(conn, "nope123") is None
    finally:
        conn.close()


def test_task_depth_survives_a_raw_cycle(depth_worker):
    """The graph builders reject cycles, so this is a corrupt-board guard: a
    raw cyclic edge must not hang the depth walk (it fails closed at a ceiling).
    """
    _, ids = depth_worker(1)
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
                     (ids[1], ids[0]))
        d = conn.execute("PRAGMA integrity_check").fetchone()[0]
        assert d == "ok"
        from hermes_cli import kanban_db as kb
        depth = kb.task_depth(conn, ids[0])
        assert isinstance(depth, int) and depth <= 64, depth
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Worker-spawned children are depth-capped, default flat.
# ---------------------------------------------------------------------------

def test_root_worker_can_spawn_depth1_child(depth_worker):
    leaf, _ = depth_worker(0)
    out = _create({"title": "child", "assignee": "peer", "parents": [leaf]})
    assert out["ok"] is True, out
    assert out["status"] == "todo"
    assert out["gated"] is True and out["gated_by"] == leaf


def test_depth1_worker_spawn_rejected_by_default(depth_worker):
    leaf, _ = depth_worker(1)
    before = _task_count()
    out = _create({"title": "grandchild", "assignee": "peer", "parents": [leaf]})
    assert "error" in out, out
    assert "kanban.max_spawn_depth" in out["error"], out
    assert _task_count() == before, "rejection must create no card"


def test_worker_spawn_without_parents_is_not_nesting(depth_worker):
    """A deep worker filing a fresh top-level card is not a divide-and-conquer
    spawn; the cap measures the new card's parent chain, not the caller."""
    _, _ = depth_worker(3)
    out = _create({"title": "fresh top-level card", "assignee": "peer"})
    assert out["ok"] is True, out


def test_worker_fan_in_is_measured_from_deepest_parent(depth_worker, monkeypatch):
    _, ids = depth_worker(1)
    # Move the worker to the root card; the new card is gated by the depth-1 child,
    # so it lands at depth 2 and the default cap of 1 refuses it.
    monkeypatch.setenv("HERMES_KANBAN_TASK", ids[0])
    out = _create({"title": "gated deep", "assignee": "peer", "parents": [ids[1]]})
    assert "error" in out, out
    assert "kanban.max_spawn_depth" in out["error"], out


def test_raising_the_knob_allows_deeper_spawn(depth_worker, tmp_path, monkeypatch):
    leaf, _ = depth_worker(1)
    (tmp_path / ".hermes" / "config.yaml").write_text(
        "kanban:\n  max_spawn_depth: 2\n"
    )
    out = _create({"title": "grandchild", "assignee": "peer", "parents": [leaf]})
    assert out["ok"] is True, out
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        assert kb.task_depth(conn, out["task_id"]) == 2
    finally:
        conn.close()


def test_env_override_raises_the_cap(depth_worker, monkeypatch):
    leaf, _ = depth_worker(2)
    monkeypatch.setenv("HERMES_KANBAN_MAX_SPAWN_DEPTH", "3")
    out = _create({"title": "deep grandchild", "assignee": "peer", "parents": [leaf]})
    assert out["ok"] is True, out


def test_invalid_config_falls_back_to_default(depth_worker, tmp_path):
    (tmp_path / ".hermes" / "config.yaml").write_text(
        "kanban:\n  max_spawn_depth: not-a-number\n"
    )
    leaf, _ = depth_worker(1)
    out = _create({"title": "grandchild", "assignee": "peer", "parents": [leaf]})
    assert "error" in out, out


def test_config_below_floor_is_floored(depth_worker, tmp_path):
    """A cap below 1 is not meaningful (the root card itself is depth 0); the
    floor keeps flat fan-out working and depth-2 spawns capped."""
    (tmp_path / ".hermes" / "config.yaml").write_text(
        "kanban:\n  max_spawn_depth: 0\n"
    )
    leaf, _ = depth_worker(0)
    out = _create({"title": "child", "assignee": "peer", "parents": [leaf]})
    assert out["ok"] is True, out
    leaf2, _ = depth_worker(1)
    out2 = _create({"title": "grandchild", "assignee": "peer", "parents": [leaf2]})
    assert "error" in out2, out2


# ---------------------------------------------------------------------------
# Non-worker callers keep the orchestrator contract: no cap.
# ---------------------------------------------------------------------------

def test_orchestrator_is_not_depth_capped(depth_worker, monkeypatch):
    _, ids = depth_worker(3)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    out = _create({"title": "deep fan-out", "assignee": "peer", "parents": [ids[-1]]})
    assert out["ok"] is True, out


def test_schema_states_the_cap():
    """The model-facing schema must say the worker spawn is bounded, so the
    worker plans its fan-out within the cap instead of hitting the wall."""
    from tools import kanban_tools_schemas as kts
    desc = kts.KANBAN_CREATE_SCHEMA["parameters"]["properties"]["parents"]["description"]
    assert "max_spawn_depth" in desc
