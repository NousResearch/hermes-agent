"""Tests for the shared git-root detection helpers in utils.py.

These cover the chokepoint itself. Four call sites route through it
(``agent.coding_context``, ``agent.prompt_builder``, ``agent.lsp.workspace``,
``tools.checkpoint_manager``), and each previously answered "is this a repo?"
on its own with ``(parent / ".git").exists()`` — a check that accepts an empty
directory ``git`` itself rejects.

Note these assert on the helpers directly and stub ``gettempdir`` rather than
going through the callers. The end-to-end failure only reproduces when the
machine's real temp dir happens to hold ``.git`` debris — true on some hosts,
not on CI — so caller-level tests pass with the bug present and prove nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from utils import find_git_root, is_git_root


# ── is_git_root: what counts as a repo ───────────────────────────────────


def test_empty_git_directory_is_not_a_repo(tmp_path: Path):
    """`git rev-parse` exits 128 on this tree; `.exists()` says yes anyway."""
    (tmp_path / ".git").mkdir()
    assert is_git_root(tmp_path) is False


def test_git_directory_with_head_is_a_repo(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    assert is_git_root(tmp_path) is True


def test_git_file_worktree_is_a_repo(tmp_path: Path):
    """Linked worktrees and submodules use a `.git` *file* holding `gitdir:`."""
    (tmp_path / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n")
    assert is_git_root(tmp_path) is True


def test_no_git_entry_at_all(tmp_path: Path):
    assert is_git_root(tmp_path) is False


def test_accepts_str_paths(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    assert is_git_root(str(tmp_path)) is True


# ── find_git_root: the walk ──────────────────────────────────────────────


def _repo(at: Path) -> Path:
    at.mkdir(parents=True, exist_ok=True)
    (at / ".git").mkdir()
    (at / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return at


def test_finds_root_from_a_subdirectory(tmp_path: Path):
    repo = _repo(tmp_path / "repo")
    deep = repo / "src" / "pkg"
    deep.mkdir(parents=True)
    assert find_git_root(deep) == repo.resolve()


def test_returns_none_when_no_repo_above(tmp_path: Path, monkeypatch):
    # Stub the temp root to tmp_path so the walk stops before reaching the
    # machine's real temp dir, which may hold its own .git debris.
    monkeypatch.setattr("utils.tempfile.gettempdir", lambda: str(tmp_path))
    work = tmp_path / "not-a-repo"
    work.mkdir()
    assert find_git_root(work) is None


def test_empty_git_dir_does_not_stop_the_walk(tmp_path: Path):
    """Debris must be skipped over, not merely rejected at that level."""
    repo = _repo(tmp_path / "repo")
    inner = repo / "scratch"
    inner.mkdir()
    (inner / ".git").mkdir()  # debris, no HEAD
    assert find_git_root(inner) == repo.resolve()


def test_shared_temp_root_is_never_a_repo_root(tmp_path: Path, monkeypatch):
    """A `.git` in the world-writable temp dir must not claim everything under it."""
    fake_tmp = _repo(tmp_path / "shared-tmp")
    work = fake_tmp / "work"
    work.mkdir()
    monkeypatch.setattr("utils.tempfile.gettempdir", lambda: str(fake_tmp))
    assert find_git_root(work) != fake_tmp.resolve()


def test_real_repo_under_the_temp_root_still_counts(tmp_path: Path, monkeypatch):
    """Only the temp root itself is skipped, not everything beneath it.

    pytest's own `tmp_path` lives under the temp root, so over-broad skipping
    would blind every workspace test in the suite.
    """
    fake_tmp = tmp_path / "shared-tmp"
    repo = _repo(fake_tmp / "projects" / "app")
    monkeypatch.setattr("utils.tempfile.gettempdir", lambda: str(fake_tmp))
    assert find_git_root(repo) == repo.resolve()


def test_temp_root_guard_is_not_sidesteppable_via_symlink(tmp_path: Path, monkeypatch):
    """The guard compares resolved forms, so a symlink to the temp root is caught."""
    fake_tmp = _repo(tmp_path / "shared-tmp")
    link = tmp_path / "link-to-tmp"
    try:
        link.symlink_to(fake_tmp, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    monkeypatch.setattr("utils.tempfile.gettempdir", lambda: str(fake_tmp))
    assert find_git_root(link) != fake_tmp.resolve()


# ── resolve=False: the LSP contract ──────────────────────────────────────


def test_resolve_false_keeps_the_path_as_given(tmp_path: Path):
    """agent.lsp.workspace needs unfolded symlinks — some language servers key
    workspace identity on the exact path the user opened."""
    real = _repo(tmp_path / "real-repo")
    link = tmp_path / "via-link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")

    assert find_git_root(link, resolve=True) == real.resolve()
    assert find_git_root(link, resolve=False) == link.absolute()


def test_bad_input_returns_none_rather_than_raising():
    """Callers sit on the lint and prompt-build paths; they must not crash."""
    assert find_git_root("\0invalid") is None
