"""Tool-surface integration for access units (default-off flags).

Covers the model-facing seams:

* ``kanban_create`` with ``access_outcome_key`` / ``access_continuation_of``
  (flag-gated; converge on the active unit instead of duplicate lanes);
* ``kanban_block`` with ``semantic_fingerprint`` (flag-gated; progress vs
  loop);
* the historical regression patterns named in the accepted design:
  duplicate-review lanes, active-PR event storm, false block-loop on
  guided-session progress, and serial connector starvation (independent
  units must never grow dependency edges).

All calls go through the REAL registry dispatch (tools.kanban_tools
handlers), not bare kernel functions — the contract is what the model
surface returns. Flags are injected via the kernel seam so each test is
deterministic; with a flag off, the tool arguments are inert (legacy
behaviour).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_access_units as kau
from hermes_cli import kanban_db_connect as kbc
from tools.kanban_tools import _handle_create, _handle_block
from tools.registry import registry


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def dispatcher_ctx(monkeypatch: pytest.MonkeyPatch):
    """Make the tools behave as an orchestrator (no pinned worker task)."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "nadia")


@pytest.fixture
def flags(monkeypatch: pytest.MonkeyPatch):
    state = {name: False for name in (
        "outcome_keys", "review_keys", "continuation_keys",
        "semantic_recurrence", "narrow_pr_guards", "stale_reconciliation",
    )}
    monkeypatch.setattr(kb, "_access_unit_flags", lambda: state)
    return state


def test_kanban_create_converges_on_active_outcome_unit(
    kanban_home, dispatcher_ctx, flags
):
    flags["outcome_keys"] = True
    key = kau.build_outcome_key("ruth", "ga4", "properties/5", "read_only")
    first = json.loads(_handle_create({
        "title": "GA4 unit", "assignee": "vladamir", "access_outcome_key": key}))
    second = json.loads(_handle_create({
        "title": "GA4 duplicate", "assignee": "ruth", "access_outcome_key": key}))
    assert first["ok"] is True and second["ok"] is True
    assert first["task_id"] == second["task_id"]  # converged, no second lane


def test_kanban_create_outcome_args_inert_when_flag_off(
    kanban_home, dispatcher_ctx, flags
):
    key = kau.build_outcome_key("ruth", "ga4", "properties/5", "read_only")
    first = json.loads(_handle_create({
        "title": "a", "assignee": "v", "access_outcome_key": key}))
    second = json.loads(_handle_create({
        "title": "b", "assignee": "v", "access_outcome_key": key}))
    assert first["task_id"] != second["task_id"]  # legacy duplicates allowed


def test_kanban_block_semantic_fingerprint_via_tools(kanban_home, dispatcher_ctx, flags):
    """The tool surface accepts the semantic fingerprint and the kernel uses
    it: masked_entry after consent stays 'blocked' (progress), the third
    IDENTICAL fingerprint escalates to triage."""
    flags["semantic_recurrence"] = True
    created = json.loads(_handle_create({"title": "guided", "assignee": "vladamir"}))
    tid = created["task_id"]

    def block(action):
        # Drive to running so block_task can act (worker-guard satisfied by env).
        with kbc.connect_closing() as conn:
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
            assert kb.claim_task(conn, tid, claimer="w") is not None
            fp = kau.recurrence_fingerprint(
                outcome="ruth|ga4|properties/5|read_only", unit="sha256:u",
                actor="chris", action=action, provider="ga4", reason="guided",
            )
            out = json.loads(_handle_block({
                "task_id": tid, "reason": f"need {action}", "kind": "needs_input",
                "semantic_fingerprint": fp}))
            assert out["ok"] is True, out
            kb.unblock_task(conn, tid)

    block("provider_login")
    block("provider_consent")
    with kbc.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="w") is not None
        fp = kau.recurrence_fingerprint(
            outcome="ruth|ga4|properties/5|read_only", unit="sha256:u",
            actor="chris", action="masked_entry", provider="ga4", reason="guided",
        )
        out = json.loads(_handle_block({
            "task_id": tid, "reason": "need masked entry", "kind": "needs_input",
            "semantic_fingerprint": fp}))
        assert out["ok"] is True, out
        t = kb.get_task(conn, tid)
        assert t.status == "blocked"  # progress did not escalate
        assert t.block_recurrences == 1  # reset by the fingerprint change


def test_serial_connector_starvation_regression(kanban_home, dispatcher_ctx, flags):
    """The historical ProfitMetrics->QuickFile->GoogleAds->GA4->Meta serial
    chain: five independent connector units created in sequence must have NO
    dependency edges between them and all sit ready in parallel."""
    flags["outcome_keys"] = True
    providers = ["profitmetrics", "quickfile", "googleads", "ga4", "meta"]
    ids = []
    for provider in providers:
        key = kau.build_outcome_key("ruth", provider, "reports", "read_only")
        out = json.loads(_handle_create({
            "title": f"{provider} unit", "assignee": "vladamir",
            "access_outcome_key": key}))
        ids.append(out["task_id"])
    with kbc.connect_closing() as conn:
        for tid in ids:
            assert kb.parent_ids(conn, tid) == []
            assert kb.child_ids(conn, tid) == []
            assert kb.get_task(conn, tid).status == "ready"  # all parallel


def test_duplicate_review_lane_regression(kanban_home, dispatcher_ctx, flags):
    """The historical duplicate-review pattern: two watchdogs requesting a
    review of the SAME (artifact, rubric, class) tuple converge on one review
    card — the second gets the first's task id."""
    flags["review_keys"] = True
    with kbc.connect_closing() as conn:
        first = kb.create_access_review_task(
            conn,
            artifact_digest="sha256:aaa", rubric_digest="sha256:bbb",
            review_class="independent", assignee="vladamir", title="review 1",
            flags=flags,
        )
        second = kb.create_access_review_task(
            conn,
            artifact_digest="sha256:aaa", rubric_digest="sha256:bbb",
            review_class="independent", assignee="ruth", title="review 2 dup",
            flags=flags,
        )
    assert first == second


def test_active_pr_event_storm_regression(kanban_home, dispatcher_ctx, flags):
    """The historical active-PR event storm: a guarded task re-checked by the
    dispatcher every tick must accumulate exactly ONE wait event."""
    from hermes_cli import kanban_db_dispatch as kbd
    flags["narrow_pr_guards"] = True
    created = json.loads(_handle_create({"title": "pr owner", "assignee": "nadia"}))
    tid = created["task_id"]
    kb_close = kbc.connect_closing
    with kb_close() as conn:
        kb.add_comment(
            conn, tid, author="nadia",
            body="Opened https://github.com/acme/erp/pull/7")
        resources = kau.parse_collision_resources(["repo:acme/erp"])
        for _ in range(5):  # five dispatcher ticks
            kbd.record_pr_collision_wait(conn, tid, resources, reason="active_pr")
        waits = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='pr_collision_wait'",
            (tid,),
        ).fetchone()[0]
    assert waits == 1
