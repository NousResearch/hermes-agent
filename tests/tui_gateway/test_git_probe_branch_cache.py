"""Current-branch probing: the tree builder asks once per SESSION row, so a branch probe is cached
per checkout and invalidated by HEAD's stamp — a ``git switch`` must show on the next render."""

from __future__ import annotations

from pathlib import Path

from tui_gateway import git_probe as gp


def _fake_repo(root: Path, branch: str) -> str:
    (root / ".git").mkdir(parents=True)
    _set_head(root, branch)
    return str(root)


def _set_head(root: Path, branch: str) -> None:
    (root / ".git" / "HEAD").write_text(f"ref: refs/heads/{branch}\n", encoding="utf-8")


def _fake_probe(monkeypatch) -> list[str]:
    """Count probes; each one reads HEAD like `git branch --show-current` would."""
    calls: list[str] = []

    def run_git(cwd, *args):
        calls.append(args[0])
        try:
            head = (Path(cwd) / ".git" / "HEAD").read_text(encoding="utf-8")
        except OSError:  # matches run_git: a failed probe is "", never an exception
            return ""
        return head.split("refs/heads/")[-1].strip()

    monkeypatch.setattr(gp, "run_git", run_git)
    monkeypatch.setattr(gp, "_branches", gp._BranchCache())
    return calls


def test_branch_is_probed_once_until_head_changes(tmp_path, monkeypatch):
    calls = _fake_probe(monkeypatch)
    cwd = _fake_repo(tmp_path, "main")

    assert gp.branch(cwd) == "main"
    assert gp.branch(cwd) == "main"  # cache hit: one spawn for a whole sidebar build
    assert len(calls) == 1

    _set_head(tmp_path, "epoxy49")
    assert gp.branch(cwd) == "epoxy49"  # HEAD moved -> stamp changed -> re-probe
    assert len(calls) == 2


def test_invalidate_drops_cached_branches(tmp_path, monkeypatch):
    calls = _fake_probe(monkeypatch)
    cwd = _fake_repo(tmp_path, "main")
    gp.branch(cwd)

    gp.invalidate()

    assert gp.branch(cwd) == "main"
    assert len(calls) == 2


def test_branch_cache_does_not_leak_across_checkouts(tmp_path, monkeypatch):
    _fake_probe(monkeypatch)
    first = _fake_repo(tmp_path / "a", "main")
    second = _fake_repo(tmp_path / "b", "dev")

    assert (gp.branch(first), gp.branch(second)) == ("main", "dev")


def test_head_stamp_follows_a_linked_worktrees_gitdir(tmp_path):
    # A linked worktree's ``.git`` is a FILE pointing at the real gitdir; the stamp (and so the
    # invalidation key) must live on that HEAD, not on the missing ``<wt>/.git/HEAD``.
    gitdir = tmp_path / "main" / ".git" / "worktrees" / "wt"
    gitdir.mkdir(parents=True)
    (gitdir / "HEAD").write_text("ref: refs/heads/feat\n", encoding="utf-8")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")

    stamp = gp._head_stamp(str(worktree))

    assert stamp is not None
    assert Path(stamp[0]) == gitdir


def test_vanished_cwd_is_not_reprobed_on_every_build(tmp_path, monkeypatch):
    # Sidebars hold rows whose folder was deleted; each build must not spawn git for them.
    calls = _fake_probe(monkeypatch)
    gone = str(tmp_path / "deleted-workspace")

    assert gp.branch(gone) == ""
    assert gp.branch(gone) == ""
    assert calls.count("branch") == 1  # one probe (plus its rev-parse fallback), not one per build


def test_head_stamp_is_none_outside_a_checkout(tmp_path):
    assert gp._head_stamp(str(tmp_path)) is None
