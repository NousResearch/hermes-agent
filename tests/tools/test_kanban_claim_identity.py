"""Claim-scoped lifecycle tools require run identity, not just profile identity.

Incident class (fleet finding "profile-identity lifecycle collision"): a
self-DM (``hermes -p forge chat -c "Bot Chat" --create-if-missing``) spawned a
SECOND full agent context of the same profile with NO dispatcher run record.
That fork's kanban_block passed the worker-scope guard (no HERMES_KANBAN_TASK
looks like an "orchestrator"), fell through the ``expected_run_id=None`` CAS
as unconditional, and TERMINATED the live claimant's run mid-flight while the
claimant's process kept executing.

The fix: ``_enforce_claim_run_identity`` refuses claim-scoped lifecycle calls
(kanban_complete / kanban_block / kanban_request_review /
kanban_request_changes) when the task is claimed by an active run and the
caller holds no run identity for it. Fail-open on board/probe errors only —
the guard must never strand a live worker.
"""
from __future__ import annotations

import json
import os

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def claimed_board(monkeypatch, tmp_path):
    """Board with a task claimed by a live run (status=running,
    current_run_id set), the way ``claim_task`` leaves it. Returns the task id
    and pins the board DB via HERMES_KANBAN_DB."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="claimed", assignee="forge")
        kb.claim_task(conn, tid)
    finally:
        conn.close()

    # The FORK's env: a full agent session with NO dispatcher task/run env.
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_CLAIM_LOCK"):
        monkeypatch.delenv(var, raising=False)
    return tid


def _caller_env(monkeypatch, profile="forge"):
    monkeypatch.setenv("HERMES_PROFILE", profile)
    monkeypatch.setenv("HERMES_AGENT", "true")
    monkeypatch.setenv("HERMES_SESSION_ID", "20260830_144815_forksession")


# ---------------------------------------------------------------------------
# The defect (RED against base): a no-run caller must not terminate a live claim
# ---------------------------------------------------------------------------

def test_no_run_caller_cannot_block_claimed_task(claimed_board, monkeypatch):
    """kanban_block from a second agent context of the same profile (no run
    env) must be refused while a live run holds the claim."""
    tid = claimed_board
    _caller_env(monkeypatch)
    from tools import kanban_tools as kt

    out = kt._handle_block({"task_id": tid, "kind": "needs_input",
                            "reason": "fork trying to close someone else's run"})
    d = json.loads(out)
    assert d.get("ok") is not True, f"fork's kanban_block terminated the live claim: {out}"
    assert "run #" in out or "no run identity" in out


def test_no_run_caller_cannot_complete_claimed_task(claimed_board, monkeypatch):
    tid = claimed_board
    _caller_env(monkeypatch)
    from tools import kanban_tools as kt

    out = kt._handle_complete({"task_id": tid, "summary": "fork completion"})
    assert json.loads(out).get("ok") is not True, (
        f"fork's kanban_complete terminated the live claim: {out}")


def test_no_run_caller_cannot_request_review_on_claimed_task(
        claimed_board, monkeypatch):
    tid = claimed_board
    _caller_env(monkeypatch)
    from tools import kanban_tools as kt

    out = kt._handle_request_review({"task_id": tid, "summary": "fork review"})
    assert json.loads(out).get("ok") is not True, (
        f"fork's kanban_request_review cleared the live claim: {out}")


def test_no_run_caller_cannot_request_changes_on_claimed_task(
        claimed_board, monkeypatch):
    tid = claimed_board
    _caller_env(monkeypatch)
    from tools import kanban_tools as kt

    out = kt._handle_request_changes({"task_id": tid, "reason": "fork gate"})
    assert json.loads(out).get("ok") is not True, (
        f"fork's kanban_request_changes cleared the live claim: {out}")


def test_claim_still_alive_after_fork_attempts(claimed_board, monkeypatch):
    """After every refused fork call, the claimant's run must still be the
    active run — the fork must not have perturbed the task at all."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = claimed_board
    _caller_env(monkeypatch)
    from tools import kanban_tools as kt

    kt._handle_block({"task_id": tid, "kind": "needs_input", "reason": "x"})
    kt._handle_complete({"task_id": tid, "summary": "x"})

    conn = kbc.connect()
    try:
        row = conn.execute(
            "SELECT status, current_run_id FROM tasks WHERE id = ?",
            (tid,)).fetchone()
    finally:
        conn.close()
    assert row["status"] == "running"
    assert row["current_run_id"]


# ---------------------------------------------------------------------------
# The claimant itself is never blocked (fail-open where it matters)
# ---------------------------------------------------------------------------

def test_claimant_with_own_run_id_still_succeeds(claimed_board, monkeypatch):
    """The worker holding the claim passes its run id (env) and proceeds
    normally — the CAS stays the exact identity check."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = claimed_board
    _caller_env(monkeypatch)
    conn = kbc.connect()
    try:
        run_id = int(conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?",
            (tid,)).fetchone()["current_run_id"])
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    from tools import kanban_tools as kt

    out = kt._handle_block({"task_id": tid, "kind": "needs_input",
                            "reason": "legit claimant blocking"})
    assert json.loads(out).get("ok") is True, out


def test_guard_fails_open_when_board_probe_errors(claimed_board, monkeypatch):
    """A broken board probe must strand nobody: the guard fails open instead
    of raising (same rule as the goal-judge gate)."""
    tid = claimed_board
    from tools import kanban_tools as kt

    class _Boom:
        @staticmethod
        def _current_run_id(conn, task_id):
            raise RuntimeError("disk on fire")

    # Direct call: fail-open means the probe neither raises nor refuses even
    # when the board probe dies — and the sink still binds a no-run-env
    # caller's write to current_run_id IS NULL (covered by the sink tests).
    assert kt._claim_run_identity_probe(_Boom(), None, tid) is True


def test_unclaimed_task_stays_orchestral(claimed_board, monkeypatch):
    """With no active run, an orchestrator-shaped caller (no run env) may
    still mutate the task — legit routing must keep working."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = claimed_board
    # Release the claim so current_run_id is NULL (task stays running-shaped
    # via _retry_status_for_run; force the row as orchestrators find it).
    conn = kbc.connect()
    try:
        conn.execute(
            "UPDATE tasks SET current_run_id = NULL, status = 'ready' "
            "WHERE id = ?", (tid,))
        conn.commit()
    finally:
        conn.close()

    _caller_env(monkeypatch)
    from tools import kanban_tools as kt

    out = kt._handle_block({"task_id": tid, "kind": "needs_input",
                            "reason": "orchestrator routing a ready task"})
    assert json.loads(out).get("ok") is True, out


# ---------------------------------------------------------------------------
# The sink defect (RED against base): no run identity must mean IS NULL,
# never "no predicate" — the invariant must hold at the write itself
# ---------------------------------------------------------------------------

def test_interleave_probe_read_then_second_claim_survives_complete(
        claimed_board, monkeypatch):
    """Adversarial interleave: the no-run guard reads current_run_id = NULL,
    a dispatcher claims the task on a SECOND connection, the no-run caller's
    lifecycle mutation then resumes. The new claim must survive.

    On the probe-before-mutation head this fails: the guard saw NULL, the
    caller proceeds, and ``expected_run_id=None`` compiles to a predicate-free
    UPDATE that transitions the freshly claimed task."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = claimed_board
    _caller_env(monkeypatch)
    from tools import kanban_tools as kt

    # Start from the UNCLAIMED board the guard's NULL read requires (release
    # the fixture's claim; the dispatcher claim lands later, in the window).
    conn = kbc.connect()
    try:
        conn.execute(
            "UPDATE tasks SET current_run_id = NULL, status = 'ready', "
            "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ?", (tid,))
        conn.commit()
        # Prove the board really is unclaimed at probe time (guard reads NULL).
        assert kb._current_run_id(conn, tid) is None
    finally:
        conn.close()

    def _claim_on_second_connection(task_id, kb_mod, conn):
        """Runs inside the guard, after the no-run read, before the mutation:
        the dispatcher claims the task on its own connection."""
        # connect() via the defining module: the kanban_db pointer is a
        # plugin-compat shim, removed 2026-09-14 (see COMPAT_MANIFEST.md).
        c2 = kbc.connect()
        try:
            claimed = kb_mod.claim_task(c2, task_id)
            assert claimed is not None, "interleave setup: second claim failed"
        finally:
            c2.close()

    original_hook = kt._PRE_MUTATION_OBSERVER
    kt._PRE_MUTATION_OBSERVER = _claim_on_second_connection
    try:
        out = kt._handle_complete({"task_id": tid, "summary": "late no-run completion"})
    finally:
        kt._PRE_MUTATION_OBSERVER = original_hook

    assert json.loads(out).get("ok") is not True, (
        f"no-run caller's complete_task transitioned the newly claimed task: {out}")

    conn = kbc.connect()
    try:
        row = conn.execute(
            "SELECT status, current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()
    finally:
        conn.close()
    # The dispatcher's claim is intact: still running under ITS run.
    c2 = kbc.connect()
    try:
        claim_row = c2.execute(
            "SELECT id FROM task_runs WHERE task_id = ? AND ended_at IS NULL "
            "ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
    finally:
        c2.close()
    assert row["status"] == "running", f"claim lost: status={row['status']}"
    assert row["current_run_id"] and claim_row and (
        int(row["current_run_id"]) == int(claim_row["id"])), (
        f"the late no-run mutation ended the new claim: {dict(row)} vs run {claim_row}")


def test_sink_no_run_id_requires_unclaimed_row_at_the_write(claimed_board):
    """DB-layer statement of the invariant: with no expected run id, a
    lifecycle mutation may only land while current_run_id IS NULL — the
    claimant's row must not be writable by the None caller."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = claimed_board
    conn = kbc.connect()
    try:
        run_id = int(conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?",
            (tid,)).fetchone()["current_run_id"])
        # Claimed row + no run proof -> refused at the sink.
        assert kb.complete_task(conn, tid, summary="sink probe") is False
        assert kb.block_task(conn, tid, reason="sink probe") is False
        assert kb.request_review(conn, tid, summary="sink probe") is False
        ok, reason = kb.request_changes(conn, tid, reason="sink probe")
        assert ok is False
        assert conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?",
            (tid,)).fetchone()["current_run_id"] == run_id
    finally:
        conn.close()


def test_sink_no_run_id_still_allows_genuinely_unclaimed_writes(claimed_board):
    """The None caller is not locked out: an UNCLAIMED task (the orchestrator
    routing shape) still mutates — None means IS NULL, not 'never'."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = claimed_board
    conn = kbc.connect()
    try:
        # Drop the claim exactly as the reclaim invariant would.
        conn.execute("UPDATE tasks SET current_run_id = NULL WHERE id = ?", (tid,))
        conn.commit()
        assert kb.block_task(conn, tid, reason="orchestrator, unclaimed") is True
    finally:
        conn.close()


def test_claimant_run_id_still_binds_the_write_exactly(claimed_board):
    """With an expected run id the CAS is unchanged: right id lands, a wrong
    id refuses (identity check stays exact)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = claimed_board
    conn = kbc.connect()
    try:
        run_id = int(conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?",
            (tid,)).fetchone()["current_run_id"])
        assert kb.block_task(conn, tid, reason="claimant",
                             expected_run_id=run_id) is True
        # Reset to running for the wrong-id leg.
        conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (tid,))
        conn.commit()
        assert kb.block_task(conn, tid, reason="wrong id",
                             expected_run_id=run_id + 10_000) is False
    finally:
        conn.close()
