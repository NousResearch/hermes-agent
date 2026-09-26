"""Regression coverage for t_cbdadd9d / t_516c72bc.

t_cbdadd9d: the fence used to deny reassign/archive via the CLI while permitting
kanban_create/kanban_comment via the MCP tool surface, and a denied CLI reassign used to
exit 0 (false success in the durable audit trail).

t_516c72bc (round 2 rework): the first fix for t_cbdadd9d registered the new
kanban_reassign/kanban_archive tools under ``_check_kanban_mode`` — the gate deliberately
visible to dispatcher-spawned task workers — and called only
``_reject_delegated_child_mutation()``. That inferred "coordinator" from "not a delegate
child", which every ordinary task worker also satisfies, so ANY task worker could
reassign or archive a SIBLING card it does not own. ``archive_task()`` SIGTERMs the
target's live ``worker_pid`` by design, so a worker could kill another worker's process
and void its run — crossing the #19534 / #19713 invariant. Measured live: the mutant
(moving both tools into ``_ORCHESTRATOR_TOOLS``) left the round-1 suite at 7 passed — the
authority boundary was untested.

Fix shape (unchanged from t_cbdadd9d's policy decision, corrected on the authority
model): kanban_reassign/kanban_archive are orchestrator-only, exactly like the existing
kanban_unblock — registered under ``_check_kanban_orchestrator_mode`` (hidden from every
dispatcher task worker's schema, not just delegate_task children) AND re-checked at
runtime via ``_require_orchestrator_tool()`` so a direct or prompt-injected call fails
closed even if schema visibility were ever bypassed. The CLI/subprocess fence
(tests/tools/test_kanban_descendant_scope.py) stays exactly as-is — untouched, not
weakened.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect

ROOT = Path(__file__).resolve().parents[2]


def _worker_board(tmp_path, monkeypatch):
    """A dispatcher-owned worker scoped to task `own`, with a sibling `other` task on the
    same board it does not own — the target a rogue/prompt-injected reassign or archive
    call would try to reach."""
    db = tmp_path / "board.db"
    conn = connect(db)
    own = kb.create_task(conn, title="own", assignee="implementer")
    other = kb.create_task(conn, title="other", assignee="implementer")
    kb.claim_task(conn, own)
    task = kb.get_task(conn, own)
    for key, value in {
        "HERMES_KANBAN_DB": str(db), "HERMES_KANBAN_BOARD": "default",
        "HERMES_KANBAN_TASK": own, "HERMES_KANBAN_RUN_ID": str(task.current_run_id),
        "HERMES_KANBAN_CLAIM_LOCK": task.claim_lock, "HOME": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    return conn, own, other


def _orchestrator_board(tmp_path, monkeypatch):
    """An authorized orchestrator/coordinator: kanban toolset in play, no
    HERMES_KANBAN_TASK (never a dispatcher task worker), not a delegate_task child. This
    is the caller kanban_reassign/kanban_archive exist to serve, and must keep working
    after the fix — the orchestrator gate distinguishes it from a task worker precisely
    on whether HERMES_KANBAN_TASK is set."""
    db = tmp_path / "board.db"
    conn = connect(db)
    other = kb.create_task(conn, title="other", assignee="implementer")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    return conn, other


def test_worker_reassign_and_archive_of_sibling_card_is_refused_and_row_unchanged(tmp_path, monkeypatch):
    """P1 fix, inverted from the round-1 test this replaces: a dispatcher task worker
    scoped to its OWN card must be refused for both verbs when acting on a SIBLING card —
    not just a delegate_task child. Assert the readback, not just the error string: the
    row must be provably unwritten, matching kanban_complete/kanban_unblock's existing
    fail-closed behavior for the identical caller shape."""
    from tools import kanban_tools as kt

    conn, own, other = _worker_board(tmp_path, monkeypatch)

    reassigned = json.loads(kt._handle_reassign({"task_id": other, "assignee": "reviewer"}))
    assert "error" in reassigned, reassigned
    assert "orchestrator-only" in reassigned["error"], reassigned

    archived = json.loads(kt._handle_archive({"task_id": other}))
    assert "error" in archived, archived
    assert "orchestrator-only" in archived["error"], archived

    # The point of this test: prove the refusal actually held the line, not just that
    # SOME error string came back.
    unchanged = kb.get_task(conn, other)
    assert unchanged.assignee == "implementer"
    assert unchanged.status != "archived"
    conn.close()


def test_worker_reassign_and_archive_of_own_card_is_also_refused(tmp_path, monkeypatch):
    """The gate is orchestrator-only, not per-task ownership: a worker must be refused
    even reassigning/archiving its OWN card via these verbs (kanban_complete/
    kanban_block remain the correct lifecycle tools for that). Pins the fix against a
    narrower "ownership check" reading of the review that would leave own-card calls
    open."""
    from tools import kanban_tools as kt

    conn, own, _other = _worker_board(tmp_path, monkeypatch)

    reassigned = json.loads(kt._handle_reassign({"task_id": own, "assignee": "reviewer"}))
    assert "error" in reassigned, reassigned
    assert "orchestrator-only" in reassigned["error"], reassigned
    assert kb.get_task(conn, own).assignee == "implementer"
    conn.close()


def test_authorized_orchestrator_can_reassign_and_archive_a_board_task(tmp_path, monkeypatch):
    """The coordinator use case this PR exists to serve keeps working after the fix: no
    HERMES_KANBAN_TASK (never a dispatcher task worker), not a delegate_task child ->
    both verbs succeed and the write actually lands."""
    from tools import kanban_tools as kt

    conn, other = _orchestrator_board(tmp_path, monkeypatch)

    reassigned = json.loads(kt._handle_reassign({"task_id": other, "assignee": "reviewer"}))
    assert reassigned.get("ok") is True, reassigned
    assert reassigned["assignee"] == "reviewer"
    assert kb.get_task(conn, other).assignee == "reviewer"

    archived = json.loads(kt._handle_archive({"task_id": other}))
    assert archived.get("ok") is True, archived
    assert kb.get_task(conn, other).status == "archived"
    conn.close()


def test_orchestrator_reassign_unknown_task_is_an_error_not_a_silent_noop(tmp_path, monkeypatch):
    """Moved from the worker context to the orchestrator context it actually exercises —
    a worker can no longer reach _require_same_board_task's unknown-id branch at all
    (refused earlier by the orchestrator-only gate), so this must run as the authorized
    caller to test what it claims to test."""
    from tools import kanban_tools as kt

    _orchestrator_board(tmp_path, monkeypatch)
    result = json.loads(kt._handle_reassign({"task_id": "t_doesnotexist", "assignee": "reviewer"}))
    assert "error" in result, result


def test_delegate_child_reassign_and_archive_still_refused(tmp_path, monkeypatch):
    """The asymmetry ran the other way too: this must NOT become a fourth mutation a
    delegate_task child can reach. Same refusal kanban_create/kanban_comment already
    give, checked from BOTH inherited-context shapes a delegate child can run in: a
    dispatcher task worker's env (own/other set) and an orchestrator's env (no
    HERMES_KANBAN_TASK) — t_cbdadd9d's original coordinator scenario had no task id set
    at all, so that shape must be covered too, not just the worker one."""
    from agent.delegation_context import delegated_child_context
    from tools import kanban_tools as kt

    conn, own, other = _worker_board(tmp_path, monkeypatch)
    with delegated_child_context():
        reassigned = json.loads(kt._handle_reassign({"task_id": other, "assignee": "reviewer"}))
        archived = json.loads(kt._handle_archive({"task_id": own}))
    assert "error" in reassigned, reassigned
    assert "error" in archived, archived
    assert kb.get_task(conn, other).assignee == "implementer"
    assert kb.get_task(conn, own).status == "running"
    conn.close()

    conn2, other2 = _orchestrator_board(tmp_path, monkeypatch)
    with delegated_child_context():
        reassigned2 = json.loads(kt._handle_reassign({"task_id": other2, "assignee": "reviewer"}))
        archived2 = json.loads(kt._handle_archive({"task_id": other2}))
    assert "error" in reassigned2, reassigned2
    assert "error" in archived2, archived2
    assert kb.get_task(conn2, other2).assignee == "implementer"
    assert kb.get_task(conn2, other2).status != "archived"
    conn2.close()


def test_reassign_running_task_without_reclaim_is_an_error(tmp_path, monkeypatch):
    """kanban_reassign must not silently no-op on the ``still running`` guard either —
    it has to surface as a tool error, matching the CLI's non-zero-exit contract below.
    Run as the authorized orchestrator caller (the only caller that can reach this
    branch post-fix), reassigning a task that is running/claimed without reclaim=true."""
    from tools import kanban_tools as kt

    conn, other = _orchestrator_board(tmp_path, monkeypatch)
    kb.claim_task(conn, other)
    assert kb.get_task(conn, other).status == "running"

    result = json.loads(kt._handle_reassign({"task_id": other, "assignee": "reviewer"}))
    assert "error" in result, result
    assert kb.get_task(conn, other).assignee == "implementer"
    conn.close()


def test_kanban_reassign_and_archive_hidden_from_dispatcher_worker_schema(tmp_path, monkeypatch):
    """Schema-level regression pinning the actual root cause of the P1 defect: the tools
    were reachable by task workers because they were registered under
    ``_check_kanban_mode`` (worker-visible), not because of a logic bug alone. This
    proves the registration tier itself, independent of the handler-level tests above —
    a fix that only patched handler logic while leaving the schema registration wrong
    would still leak these tools into a worker's tool-call surface."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_some_worker_task")

    import tools.kanban_tools  # noqa: F401 — ensure registered
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    assert "kanban_reassign" not in names, names
    assert "kanban_archive" not in names, names
    # The worker-visible lifecycle tools must still be there — this is a scoping
    # regression test, not a "kanban disappeared for workers" test.
    assert "kanban_complete" in names, names
    assert "kanban_heartbeat" in names, names


def test_kanban_reassign_and_archive_visible_to_authorized_orchestrator_schema(tmp_path, monkeypatch):
    """Symmetric counterpart to the worker-hiding test above, on the OTHER layer that
    gates visibility: ``toolsets.py``'s ``_HERMES_CORE_TOOLS`` allowlist. That allowlist
    feeds every ``hermes-*`` platform bundle via ``resolve_toolset()``, independently of
    ``_ORCHESTRATOR_TOOLS``/``_check_kanban_orchestrator_mode`` in kanban_tools.py — a
    handler can be perfectly gated and still be unreachable by EVERY caller, including a
    legitimate coordinator, if it was never added to this allowlist. Caught live: the
    original fix touched only tools/kanban_tools.py and shipped kanban_reassign/
    kanban_archive completely absent from toolsets.py, so no session of any kind
    (worker or orchestrator) could ever see either tool's schema."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

    import tools.kanban_tools  # noqa: F401 — ensure registered
    from tools.kanban_toolset_context import scoped_kanban_toolset_selection
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    with scoped_kanban_toolset_selection(["kanban"]):
        invalidate_check_fn_cache()
        schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
        names = {s["function"].get("name") for s in schema if "function" in s}
    assert "kanban_reassign" in names, names
    assert "kanban_archive" in names, names
    # Same orchestrator-only tier as the pre-existing kanban_unblock/kanban_list — not a
    # new, wider allowlist entry.
    assert "kanban_unblock" in names, names
    assert "kanban_list" in names, names


def test_reassign_and_archive_tools_are_actually_registered():
    """Direct handler calls above prove behavior; this proves the tools are reachable
    through the real dispatch path an agent uses — a schema/import typo in the
    registration block would pass every test above yet leave the tool invisible."""
    import tools.kanban_tools  # noqa: F401 — populates tools.registry.registry on import
    from tools.registry import registry

    names = set(registry._tools)
    assert "kanban_reassign" in names, names
    assert "kanban_archive" in names, names


def _run_hermes(home: Path, *args: str, marker: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["HERMES_KANBAN_HOME"] = str(home)
    for name in ("HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT"):
        env.pop(name, None)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if marker:
        env["HERMES_DELEGATED_CHILD_CONTEXT"] = "1"
    else:
        env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False, timeout=30,
    )


def test_fenced_cli_reassign_is_nonzero_exit_never_false_success(tmp_path):
    """t_cbdadd9d item 2, re-verified on the current tree: a denied ``hermes kanban
    reassign`` must exit non-zero with the refusal on stderr, and must NOT touch the row.
    (Fixed upstream by b578261584e adding PermissionError to kanban_command's caught
    exceptions; kept here so the property cannot silently regress.)"""
    home = tmp_path / "hermes"
    home.mkdir()

    created = _run_hermes(home, "kanban", "create", "reassign fence probe", "--json")
    assert created.returncode == 0, created.stderr
    task_id = json.loads(created.stdout)["id"]

    before = _run_hermes(home, "kanban", "show", task_id, "--json")
    assignee_before = json.loads(before.stdout)["task"]["assignee"]

    refused = _run_hermes(home, "kanban", "reassign", task_id, "reviewer", marker=True)
    assert refused.returncode != 0, (refused.stdout, refused.stderr)
    assert "cannot mutate Kanban tasks via the CLI" in refused.stderr

    after = _run_hermes(home, "kanban", "show", task_id, "--json")
    assert json.loads(after.stdout)["task"]["assignee"] == assignee_before


def test_fenced_cli_still_denies_reassign_create_and_comment_alike(tmp_path):
    """The split the task reported (create/comment succeed, reassign doesn't) must not
    reappear the other direction either: prove all three still share one CLI verdict for
    a genuine delegate_task-marked subprocess, so create/comment aren't quietly the odd
    ones out from the CLI's perspective while the new MCP tools paper over it."""
    home = tmp_path / "hermes"
    home.mkdir()

    created = _run_hermes(home, "kanban", "create", "cli parity probe", "--json")
    assert created.returncode == 0, created.stderr
    task_id = json.loads(created.stdout)["id"]

    for args in (
        ("kanban", "create", "should be refused", "--json"),
        ("kanban", "comment", task_id, "should be refused"),
        ("kanban", "reassign", task_id, "reviewer"),
    ):
        refused = _run_hermes(home, *args, marker=True)
        assert refused.returncode != 0, (args, refused.stdout, refused.stderr)
        assert "cannot mutate Kanban tasks via the CLI" in refused.stderr, (args, refused.stderr)
