"""A ``cd`` into another worktree of the chat's repo is called out in the terminal result.

The desktop labels a chat with its workspace, not the shell's cwd, so when the shell steps into a sibling worktree
the result names both trees and says how to move the chat (``desktop_project`` switch with a path) or keep the visit
short (``workdir``). Real git worktrees, the real result finalizer.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from tools import terminal_tool
from tools.terminal_tool_result import finalize_foreground_result

KEY = "chat-note"


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def trees(tmp_path, monkeypatch):
    """A chat whose declared workspace is ``main``'s checkout, a ``feature`` worktree, and an unrelated repo."""
    repo, feature, other = tmp_path / "proj", tmp_path / "proj-feature", tmp_path / "other"
    for root in (repo, other):
        root.mkdir()
        _git(root, "init", "-b", "main")
        _git(root, "config", "user.email", "t@example.com")
        _git(root, "config", "user.name", "t")
        (root / "README.md").write_text("hi\n", encoding="utf-8")
        _git(root, "add", ".")
        _git(root, "commit", "-m", "init")
    _git(repo, "worktree", "add", "-b", "feature", str(feature))
    (repo / "src").mkdir()
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    terminal_tool.register_task_env_overrides(KEY, {"cwd": str(repo), "cwd_source": "session"})
    yield repo, feature, other
    terminal_tool.clear_session_cwd(KEY)


def _cd(landed, *, start, workdir=None, env_type="local") -> dict:
    return json.loads(finalize_foreground_result(
        command=f"cd {landed}", result={"output": "", "returncode": 0, "cwd_observed": True, "cwd": str(landed)},
        env=None, env_type=env_type, effective_task_id=KEY, task_id=KEY, session_id="s", session_key=KEY,
        workdir=workdir, command_cwd=str(start), approval_note=None))


def test_a_cd_into_a_sibling_worktree_names_both_trees(trees):
    repo, feature, _ = trees

    note = _cd(feature, start=repo).get("workspace", "")

    assert f"{feature.resolve()} (branch feature)" in note
    assert f"{repo.resolve()} (branch main)" in note
    assert "desktop_project action=switch" in note


@pytest.mark.parametrize("case", ["same worktree", "unrelated repo", "transient workdir", "remote backend"])
def test_no_note_when_the_chat_has_not_left_its_tree(trees, case):
    repo, feature, other = trees
    landed, kwargs = {
        "same worktree": (repo / "src", {}),
        "unrelated repo": (other, {}),
        "transient workdir": (feature, {"workdir": str(feature)}),
        "remote backend": (feature, {"env_type": "ssh"}),
    }[case]

    assert "workspace" not in _cd(landed, start=repo, **kwargs)
