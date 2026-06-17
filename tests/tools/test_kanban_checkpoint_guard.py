"""Regression tests for the kanban commit-before-handoff guard.

``_checkpoint_workspace`` auto-commits a dirty worktree to its task branch
before ``kanban_block`` / ``kanban_complete`` hand off, so the branch (the unit
of handoff) carries the work for the reviewer / downstream verify card / PR.
The guard is best-effort and must NEVER block a handoff: it only acts for the
dispatched worker on its own task, on a non-default branch, outside a
merge/rebase, and swallows every error.

See the field failure that motivated it: card ``t_adfbe7f0`` blocked
``review-required`` on a branch with no commits, handing the reviewer an empty
diff.
"""

from __future__ import annotations

import subprocess

import pytest

from tools.kanban_tools import _checkpoint_workspace


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _commits(repo):
    r = _git(["rev-list", "--count", "HEAD"], repo)
    return int(r.stdout.strip()) if r.returncode == 0 else -1


@pytest.fixture
def repo(tmp_path):
    """A git repo on `main`, with origin/HEAD -> origin/main and one commit."""
    d = tmp_path / "repo"
    d.mkdir()
    _git(["init", "-q"], d)
    _git(["config", "user.email", "t@t"], d)
    _git(["config", "user.name", "t"], d)
    _git(["config", "commit.gpgsign", "false"], d)
    _git(["checkout", "-q", "-b", "main"], d)
    (d / "f.txt").write_text("base\n")
    _git(["add", "f.txt"], d)
    _git(["commit", "-q", "-m", "base"], d)
    _git(["update-ref", "refs/remotes/origin/main", "HEAD"], d)
    _git(["symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"], d)
    return d


def _as_worker(monkeypatch, task, ws):
    monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(ws))


def test_feature_branch_dirty_is_committed(repo, monkeypatch):
    _git(["checkout", "-q", "-b", "wt/t_abc"], repo)
    (repo / "f.txt").write_text("base\nfix\n")
    _as_worker(monkeypatch, "t_abc", repo)
    before = _commits(repo)
    _checkpoint_workspace("t_abc")
    assert _commits(repo) == before + 1
    subject = _git(["log", "-1", "--pretty=%s"], repo).stdout
    assert "t_abc" in subject and "wt/t_abc" in subject


def test_never_commits_on_default_branch(repo, monkeypatch):
    (repo / "f.txt").write_text("base\nx\n")  # dirty, but on main
    _as_worker(monkeypatch, "t_main", repo)
    before = _commits(repo)
    _checkpoint_workspace("t_main")
    assert _commits(repo) == before


def test_clean_worktree_is_noop(repo, monkeypatch):
    _git(["checkout", "-q", "-b", "wt/t_clean"], repo)
    _as_worker(monkeypatch, "t_clean", repo)
    before = _commits(repo)
    _checkpoint_workspace("t_clean")
    assert _commits(repo) == before


def test_only_acts_on_own_task(repo, monkeypatch):
    _git(["checkout", "-q", "-b", "wt/t_other"], repo)
    (repo / "f.txt").write_text("base\ny\n")
    _as_worker(monkeypatch, "t_SOMEONE_ELSE", repo)  # env task != tid
    before = _commits(repo)
    _checkpoint_workspace("t_other")
    assert _commits(repo) == before


def test_skips_mid_merge(repo, monkeypatch):
    _git(["checkout", "-q", "-b", "wt/t_merge"], repo)
    (repo / "f.txt").write_text("base\nz\n")
    (repo / ".git" / "MERGE_HEAD").write_text("deadbeef\n")
    _as_worker(monkeypatch, "t_merge", repo)
    before = _commits(repo)
    _checkpoint_workspace("t_merge")
    assert _commits(repo) == before


def test_captures_new_untracked_files(repo, monkeypatch):
    _git(["checkout", "-q", "-b", "wt/t_new"], repo)
    (repo / "newsrc.py").write_text("print('hi')\n")
    _as_worker(monkeypatch, "t_new", repo)
    before = _commits(repo)
    _checkpoint_workspace("t_new")
    assert _commits(repo) == before + 1
    names = _git(["show", "--name-only", "--pretty=format:", "HEAD"], repo).stdout
    assert "newsrc.py" in names


def test_no_workspace_env_is_noop(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_nows")
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
    _checkpoint_workspace("t_nows")  # must not raise


def test_non_git_workspace_is_noop(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    _as_worker(monkeypatch, "t_plain", plain)
    _checkpoint_workspace("t_plain")  # must not raise


# ---------------------------------------------------------------------------
# End-to-end: the full _handle_block / _handle_complete tool path must commit
# the worktree BEFORE transitioning the card, against an isolated kanban DB.
# ---------------------------------------------------------------------------


def _dirty_feature_repo(base):
    d = base / "ws"
    d.mkdir()
    _git(["init", "-q"], d)
    _git(["config", "user.email", "t@t"], d)
    _git(["config", "user.name", "t"], d)
    _git(["config", "commit.gpgsign", "false"], d)
    _git(["checkout", "-q", "-b", "main"], d)
    (d / "f.txt").write_text("base\n")
    _git(["add", "f.txt"], d)
    _git(["commit", "-q", "-m", "base"], d)
    _git(["update-ref", "refs/remotes/origin/main", "HEAD"], d)
    _git(["symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"], d)
    _git(["checkout", "-q", "-b", "wt/feature"], d)
    (d / "f.txt").write_text("base\nthe worker's edit\n")  # uncommitted change
    return d


@pytest.fixture
def worker_task(monkeypatch, tmp_path):
    """Isolated kanban DB with a claimed task; returns (tid, repo_path)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="commit-guard-e2e", assignee="test-worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()

    repo = _dirty_feature_repo(tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(repo))
    return tid, repo


def test_handle_block_commits_worktree_then_blocks(worker_task):
    tid, repo = worker_task
    from tools import kanban_tools as kt
    before = _commits(repo)
    out = kt._handle_block({"reason": "review-required: e2e guard test"})
    assert '"ok": true' in out or '"ok":true' in out, out
    # the dirty worktree was committed to the feature branch by the guard
    assert _commits(repo) == before + 1
    # and the card actually transitioned to blocked
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
    finally:
        conn.close()


def test_handle_complete_commits_worktree_then_completes(worker_task):
    tid, repo = worker_task
    from tools import kanban_tools as kt
    before = _commits(repo)
    out = kt._handle_complete({"summary": "done; e2e guard test", "created_cards": []})
    assert '"ok": true' in out or '"ok":true' in out, out
    assert _commits(repo) == before + 1
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "done"
    finally:
        conn.close()
