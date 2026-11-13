"""Durable resumable subagents, review round 2: a running continuation pins where it is now, a tree the child made
stays pinned after it leaves that repository, a late record write never recreates a removed profile, and boot
recovery names the continuable children on the ledger row that reports them (independent completions on or off).

Real SQLite, real git, real sweep, real dispatch and boot recovery against a temp HERMES_HOME; only the model is
scripted.
"""

import json
import os
import sqlite3
import stat
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.tools.test_delegate_continue import _parent, _ScriptedModel
from tests.tools.test_delegate_resumable import _git
from tests.waiting import wait_until

_DEAD_PID = 2 ** 22 + 7


def _repo(path):
    path.mkdir()
    _git("init", "-q", "-b", "main", cwd=path)
    (path / "a").write_text("a")
    _git("add", "a", cwd=path)
    _git("commit", "-qm", "base", cwd=path)
    return path


def _run(parent, sid):
    child = SimpleNamespace(session_id="child", model="m", _parent_session_id=parent.session_id,
                            _delegate_resume={"goal": "g", "route": SimpleNamespace(request={})})
    return SimpleNamespace(child=child, parent_agent=parent, subagent_id=sid, worktree_info=None, child_task_id="t")


def _completions(delegation_ids):
    """Wait for every listed unit's completion event (one drain: waiting per id would drop a sibling's event)."""
    from hermes_agent.tools.process_registry import process_registry
    pending = set(delegation_ids)

    def drained():
        while not process_registry.completion_queue.empty():
            evt = process_registry.completion_queue.get_nowait()
            if evt.get("type") == "async_delegation":
                pending.discard(evt.get("delegation_id"))
        return not pending
    wait_until(drained, timeout=20, what=f"completions of {sorted(pending)}")


def _backdate(*paths):
    old = time.time() - 30 * 3600
    for p in paths:
        os.utime(p, (old, old))


def test_a_running_continuation_pins_the_tree_it_moved_to_not_only_where_it_last_stopped(tmp_path):
    from hermes_cli.worktree_sweep import run_sweep, sweep_settings
    from hermes_agent.tools import delegate_tool_resumable as r
    parent = _parent()
    repo = _repo(tmp_path / "repo")
    a, b = repo / ".worktrees" / "subagent-old", repo / ".worktrees" / "subagent-new"
    _git("worktree", "add", "-qb", "old", str(a), cwd=repo)
    _git("worktree", "add", "-qb", "new", str(b), cwd=repo)
    with sqlite3.connect(parent._session_db.db_path) as conn:  # health: allow HX015 -- temp-home store
        r._upsert(conn, "sa-pins", {"owners": [parent.session_id], "session_id": "child", "cwd": str(a),
                                    "pin_paths": [str(a)], "worktrees_at_start": [str(repo), str(a), str(b)]},
                  state="resumable", pinned=True, expires_at=time.time() + 3600)
    run = _run(parent, "sa-pins")
    with patch.object(r, "_child_cwd", return_value=str(a)):
        r.record_start(run)  # the continue resumes in A ...
    with patch.object(r, "_child_cwd", return_value=str(b)):
        r.checkpoint(run)  # ... and its terminal moves to B while it runs
    meta = repo / ".git" / "worktrees" / "subagent-new"
    _backdate(b, meta / "HEAD", meta / "logs" / "HEAD")

    run_sweep(str(repo), 24, sweep_settings())

    assert b.is_dir() and a.is_dir()


def test_a_tree_the_child_made_stays_pinned_after_it_leaves_that_repository(tmp_path):
    from hermes_agent.tools import delegate_tool_resumable as r
    parent = _parent()
    repo, other = _repo(tmp_path / "repo"), _repo(tmp_path / "other")
    run = _run(parent, "sa-left")
    with patch.object(r, "_child_cwd", return_value=str(repo)):
        r.record_start(run)
    made = repo / ".worktrees" / "subagent-made"
    _git("worktree", "add", "-qb", "made", str(made), cwd=repo)  # made without cd-ing into it
    for cwd in (other, tmp_path):  # then into another repository, then outside any
        with patch.object(r, "_child_cwd", return_value=str(cwd)):
            r.checkpoint(run)
    other_made = other / ".worktrees" / "subagent-other"
    _git("worktree", "add", "-qb", "o", str(other_made), cwd=other)

    record = r._write(r._db_path_for(parent), lambda conn: r._row(conn, "sa-left"))
    pins = r._pin_paths(record)

    assert os.path.realpath(made) in pins and os.path.realpath(other_made) in pins
    assert os.path.realpath(repo) not in pins and os.path.realpath(other) not in pins  # main trees were there before


def test_a_late_record_write_never_recreates_a_removed_profile_and_stays_owner_only(tmp_path):
    from hermes_constants import get_hermes_home, profile_tombstone_path
    from hermes_agent.tools import delegate_tool_resumable as r
    removed = get_hermes_home() / "profiles" / "removed"
    tomb = profile_tombstone_path(removed)
    tomb.parent.mkdir(parents=True, exist_ok=True)
    tomb.write_text("deleted")

    r._write(removed / "state.db", lambda conn: conn.execute("SELECT 1").fetchone())
    assert not removed.exists()

    live = get_hermes_home() / "state.db"
    r._write(live, lambda conn: conn.execute("SELECT 1").fetchone())
    if os.name != "nt":
        assert stat.S_IMODE(live.stat().st_mode) == 0o600


@pytest.mark.parametrize("child", ["crashed", "stopped-by-shutdown"])
@pytest.mark.parametrize("independent", [True, False], ids=["independent-units", "one-unit"])
def test_boot_recovery_names_the_continuable_children_on_the_row_that_reports_them(independent, child):
    """A crash leaves the children's records running; a gateway restart (SIGTERM) lets each child record its
    resumable stop while the owner exits before the unit's row turns terminal. Either way the recovered notice
    names every child the parent can continue."""
    from hermes_constants import get_hermes_home
    from hermes_agent.tools import async_delegation as ad
    from hermes_agent.tools.delegate_tool import delegate_task
    (get_hermes_home() / "config.yaml").write_text(json.dumps({"delegation": {"independent_completions": independent}}))
    parent, model = _parent(), _ScriptedModel()
    with patch("run_agent.AIAgent", side_effect=model):
        out = json.loads(delegate_task(tasks=[{"goal": "write the first parser"}, {"goal": "write the second parser"}],
                                       background=True, parent_agent=parent))
        units = [u["delegation_id"] for u in out["units"]] if independent else [out["delegation_id"]]
        assert len(units) == (2 if independent else 1), out
        _completions(units)
    with sqlite3.connect(parent._session_db.db_path) as conn:  # health: allow HX015 -- temp-home store
        # The owner died before either ledger row turned terminal.
        conn.execute("UPDATE async_delegations SET state='running', owner_pid=?, owner_started_at=NULL, "
                     "result_json=NULL", (_DEAD_PID,))
        if child == "crashed":
            conn.execute("UPDATE resumable_subagents SET state='running', owner_pid=?, owner_started_at=NULL",
                         (_DEAD_PID,))
        else:
            conn.execute("UPDATE resumable_subagents SET state='resumable', stop_cause='gateway_shutdown'")

    assert ad.recover_abandoned_delegations() == len(units)

    with sqlite3.connect(parent._session_db.db_path) as conn:  # health: allow HX015 -- temp-home store
        events = {d: json.loads(raw) for d, raw in conn.execute("SELECT delegation_id, event_json FROM async_delegations")}
    named = sorted(sid for evt in events.values() for sid in evt.get("resumable_subagent_ids") or ())
    assert all(events[u].get("resumable_subagent_ids") for u in units), events
    assert named == sorted(out["subagent_ids"])
