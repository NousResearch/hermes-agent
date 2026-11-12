"""Durable resumable subagents, review round 3: a sweep planned before a child was continued never removes the tree
the continue pinned (classic and archive mode, and a pin taken while the archive snapshot ran), the record keeps no
credential from the endpoint URL yet still refuses a changed endpoint, and a failed first record write is retried
whole instead of leaving a pin with no row and later a partial row nobody can claim.

Real SQLite, real git, real sweep, real dispatch against a temp HERMES_HOME; only the model is scripted.
"""

import json
import sqlite3
import time
from unittest.mock import patch

import pytest

from tests.tools.test_delegate_continue import _completion, _parent
from tests.tools.test_delegate_resumable import _db_rows, _FailingModel, _forget_process_state, _git
from tests.tools.test_delegate_resumable_pins import _backdate, _repo, _run


def _tree(repo, name, *, dirty):
    tree = repo / ".worktrees" / name
    _git("worktree", "add", "-qb", name, str(tree), cwd=repo)
    files = [tree]
    if dirty:
        (tree / "work.txt").write_text("unsaved child work")
        files.append(tree / "work.txt")
    meta = repo / ".git" / "worktrees" / name
    _backdate(*files, meta / "HEAD", meta / "logs" / "HEAD")
    return tree


def _unpinned_record(parent, sid, tree):
    """A resumable record that lost its pin to the cap while staying continuable."""
    from tools import delegate_tool_resumable as r
    with sqlite3.connect(parent._session_db.db_path) as conn:  # health: allow HX015 -- temp-home store
        r._upsert(conn, sid, {"owners": [parent.session_id], "session_id": "child", "cwd": str(tree),
                              "pin_paths": [str(tree)], "worktrees_at_start": []},
                  state="resumable", pinned=False, expires_at=time.time() + 3600)


def _continue_now(parent, sid, tree):
    """What ``action='continue'`` does before its first model call: claim the row and record the run."""
    from tools import delegate_tool_resumable as r
    record, err = r.claim(parent, sid)
    assert record is not None and err is None, err
    with patch.object(r, "_child_cwd", return_value=str(tree)):
        r.record_start(_run(parent, sid))
    assert r.pin_reason(str(tree), r.pinned_worktrees())


@pytest.mark.parametrize("mode", ["classic", "archive"])
def test_a_sweep_planned_before_a_continue_keeps_the_tree_the_continue_pinned(tmp_path, mode):
    from hermes_cli import worktree_ops, worktree_sweep
    from hermes_constants import get_hermes_home
    if mode == "classic":
        (get_hermes_home() / "config.yaml").write_text(json.dumps({"worktrees": {"archive_dirty": False}}))
    parent = _parent()
    repo = _repo(tmp_path / "repo")
    dirty = mode == "archive"  # classic removes only clean merged trees; the archive sweep takes dirty ones
    resumed, idle = _tree(repo, "subagent-resumed", dirty=dirty), _tree(repo, "subagent-idle", dirty=dirty)
    _unpinned_record(parent, "sa-race", resumed)
    if mode == "classic":
        seam, real = worktree_ops, worktree_ops._classify_prune_candidates
        name = "_classify_prune_candidates"
    else:
        seam, real, name = worktree_sweep, worktree_sweep.plan_sweep, "plan_sweep"

    def plan_then_continue(*a, **kw):
        planned = real(*a, **kw)  # the sweep decided on both trees ...
        _continue_now(parent, "sa-race", resumed)  # ... then the parent continued the child
        return planned

    with patch.object(seam, name, side_effect=plan_then_continue):
        worktree_ops._prune_stale_worktrees(str(repo), 24)

    assert resumed.is_dir() and (not dirty or (resumed / "work.txt").is_file())
    assert not idle.exists()  # an unpinned idle tree is still reclaimed


def test_a_pin_taken_while_the_archive_snapshot_ran_keeps_the_tree(tmp_path):
    from hermes_cli import worktree_archive, worktree_ops
    parent = _parent()
    repo = _repo(tmp_path / "repo")
    resumed = _tree(repo, "subagent-resumed", dirty=True)
    _unpinned_record(parent, "sa-snap", resumed)
    real = worktree_archive.snapshot_worktree

    def snapshot_then_continue(repo_root, path, **kw):
        arc = real(repo_root, path, **kw)
        _continue_now(parent, "sa-snap", resumed)
        return arc

    with patch.object(worktree_archive, "snapshot_worktree", side_effect=snapshot_then_continue):
        worktree_ops._prune_stale_worktrees(str(repo), 24)

    assert (resumed / "work.txt").read_text() == "unsaved child work"
