"""Behavioral coverage for server-owned per-turn RunBinding identity."""

from __future__ import annotations

import subprocess
from pathlib import Path

from gateway.session_context import RunBinding, current_run_binding, reset_run_binding, set_run_binding
from tui_gateway.git_probe import capture_run_binding


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def test_capture_is_server_owned_and_contains_full_checkout_identity(tmp_path):
    repo = _repo(tmp_path)

    binding = capture_run_binding(
        str(repo), session_key="conversation-1", ui_session_id="window-1", profile="default"
    )

    assert binding.cwd == str(repo.resolve())
    assert binding.worktree_root == str(repo.resolve())
    assert binding.repo_root == str(repo.resolve())
    assert binding.git_common_dir == str((repo / ".git").resolve())
    assert binding.branch
    assert binding.ref == "refs/heads/" + binding.branch
    assert len(binding.head) == 40
    assert binding.session_key == "conversation-1"
    assert binding.ui_session_id == "window-1"
    assert binding.profile == "default"
    assert binding.short_head == binding.head[:12]


def test_binding_differences_identify_owner_and_checkout_drift(tmp_path):
    repo = _repo(tmp_path)
    original = capture_run_binding(
        str(repo), session_key="conversation-1", ui_session_id="window-1", profile="default"
    )
    changed = RunBinding(
        **{**original.__dict__, "session_key": "conversation-2", "profile": "coder"}
    )

    assert original.differences(changed) == ("session_key", "profile")
    assert not original.matches(changed)


def test_run_binding_context_is_explicitly_scoped(tmp_path):
    repo = _repo(tmp_path)
    binding = capture_run_binding(str(repo), session_key="conversation-1")
    token = set_run_binding(binding)
    try:
        assert current_run_binding() is binding
    finally:
        reset_run_binding(token)
    assert current_run_binding() is None
