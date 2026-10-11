"""A delegate lease follows a resolved checkout, not the presence of write tools.

Slack sessions live outside a repo. Those children must start. A coding
delegate names a checkout and gets one linked worktree, never a second clone.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from tools import delegate_tool
from tools.delegate_tool_child_run import _create_isolated_worktree
from tools.subagent_worktree import resolve_repo_root


def _git(args, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _make_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    _git(["config", "user.email", "test@test"], repo)
    _git(["config", "user.name", "Test"], repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", "seed"], repo)
    return repo


def test_non_repo_parent_is_not_a_coding_delegate(tmp_path):
    """Slack cwd is not a checkout. That is not an error and not a writer root."""
    root, error = delegate_tool.resolve_delegate_coding_root(
        None, str(tmp_path), isolate=True,
    )
    assert error is None
    assert root is None


def test_explicit_non_repo_is_a_named_error_not_unsupported_path(tmp_path):
    root, error = delegate_tool.resolve_delegate_coding_root(
        str(tmp_path), None, isolate=False,
    )
    assert root is None
    assert error is not None
    assert "no repo root for a coding delegate" in error
    assert "unsupported_path" not in error


def test_explicit_repo_root_resolves_even_when_isolation_flag_is_off(tmp_path):
    repo = _make_repo(tmp_path)
    root, error = delegate_tool.resolve_delegate_coding_root(
        str(repo), str(tmp_path), isolate=False,
    )
    assert error is None
    assert root == str(repo.resolve())


def test_parent_checkout_is_used_only_when_isolation_is_on(tmp_path):
    repo = _make_repo(tmp_path)
    isolated, isolated_error = delegate_tool.resolve_delegate_coding_root(
        None, str(repo), isolate=True,
    )
    plain, plain_error = delegate_tool.resolve_delegate_coding_root(
        None, str(repo), isolate=False,
    )
    assert isolated_error is None and isolated == str(repo.resolve())
    assert plain_error is None and plain is None


def test_write_tools_without_a_coding_root_do_not_require_a_lease():
    """A concrete source writer with no checkout must not force admit_writer."""
    from run_agent import AIAgent

    writer = object.__new__(AIAgent)
    writer.valid_tool_names = {"terminal", "write_file", "patch"}
    assert delegate_tool._is_concrete_source_writer(writer)
    assert not delegate_tool._requires_writer_admission(writer)


def test_explicit_repo_root_creates_a_linked_worktree_not_a_clone(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path)
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    parent = SimpleNamespace()
    info = _create_isolated_worktree(
        parent, "parent-session", "sa-coding", repo_root=str(repo),
    )
    assert info is not None
    wt = Path(info["path"])
    assert wt.is_dir()
    assert resolve_repo_root(str(wt))
    repo_common = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    wt_common = subprocess.run(
        ["git", "-C", str(wt), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    wt_git = subprocess.run(
        ["git", "-C", str(wt), "rev-parse", "--path-format=absolute", "--git-dir"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert Path(wt_common).resolve() == Path(repo_common).resolve()
    assert Path(wt_git).resolve() != Path(wt_common).resolve()
