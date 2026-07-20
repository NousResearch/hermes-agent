from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.session_context import clear_session_vars, get_session_env, set_session_vars
from tools.file_tools import patch_tool, write_file_tool
from tools.terminal_tool import terminal_tool
from tools.workspace_safety import check_terminal_side_effect_allowed


@pytest.fixture(autouse=True)
def clean_session_context(monkeypatch):
    for key in (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_WORKSPACE_SLUG",
        "HERMES_WORKSPACE_REPO_PATH",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_CONFIG_GLOBAL",
    ):
        monkeypatch.delenv(key, raising=False)
    tokens = set_session_vars()
    clear_session_vars(tokens)
    yield
    clear_session_vars(tokens)


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / ".git").mkdir()
    return path


def _gateway_session(bound_repo: str | Path | None = None):
    return set_session_vars(
        platform="matrix",
        chat_id="!room:example.org",
        workspace_slug="example" if bound_repo else "",
        workspace_repo_path=str(bound_repo) if bound_repo else "",
    )


def test_write_and_patch_are_blocked_in_wrong_repo(tmp_path):
    bound_repo = _git_repo(tmp_path / "bound")
    other_repo = _git_repo(tmp_path / "other")
    target = other_repo / "file.txt"
    target.write_text("old", encoding="utf-8")
    tokens = _gateway_session(bound_repo)
    try:
        write_result = json.loads(write_file_tool(str(other_repo / "new.txt"), "data"))
        patch_result = json.loads(
            patch_tool(path=str(target), old_string="old", new_string="new")
        )
    finally:
        clear_session_vars(tokens)

    assert "outside authoritative workspace binding" in write_result["error"]
    assert "outside authoritative workspace binding" in patch_result["error"]
    assert not (other_repo / "new.txt").exists()
    assert target.read_text(encoding="utf-8") == "old"


def test_write_allowed_in_bound_repo(tmp_path):
    bound_repo = _git_repo(tmp_path / "bound")
    target = bound_repo / "file.txt"
    tokens = _gateway_session(bound_repo)
    try:
        result = json.loads(write_file_tool(str(target), "data"))
    finally:
        clear_session_vars(tokens)

    assert result.get("error") in (None, "")
    assert target.read_text(encoding="utf-8") == "data"


def test_unbound_gateway_repo_write_blocked(tmp_path):
    repo = _git_repo(tmp_path / "repo")
    tokens = _gateway_session()
    try:
        result = json.loads(write_file_tool(str(repo / "file.txt"), "data"))
    finally:
        clear_session_vars(tokens)

    assert "no authoritative workspace binding" in result["error"]
    assert not (repo / "file.txt").exists()


def test_non_repo_scratch_write_and_read_only_git_remain_allowed(tmp_path):
    scratch = tmp_path / "scratch"
    target = scratch / "file.txt"
    tokens = _gateway_session()
    try:
        write_result = json.loads(write_file_tool(str(target), "data"))
        terminal_error = check_terminal_side_effect_allowed("git status", scratch)
    finally:
        clear_session_vars(tokens)

    assert write_result.get("error") in (None, "")
    assert target.read_text(encoding="utf-8") == "data"
    assert terminal_error is None


def test_channel_identity_and_workspace_contextvars_coexist(tmp_path):
    bound_repo = _git_repo(tmp_path / "bound")
    tokens = set_session_vars(
        platform="matrix",
        chat_id="!room:example.org",
        workspace_slug="example",
        workspace_repo_path=str(bound_repo),
    )
    try:
        assert get_session_env("HERMES_SESSION_CHAT_ID") == "!room:example.org"
        assert get_session_env("HERMES_WORKSPACE_SLUG") == "example"
        assert get_session_env("HERMES_WORKSPACE_REPO_PATH") == str(bound_repo)
    finally:
        clear_session_vars(tokens)


def test_terminal_tool_blocks_mutating_git_outside_bound_repo(tmp_path, monkeypatch):
    bound_repo = _git_repo(tmp_path / "bound")
    other_repo = _git_repo(tmp_path / "other")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(other_repo))
    tokens = _gateway_session(bound_repo)
    try:
        result = json.loads(terminal_tool("git add file.txt", timeout=5))
    finally:
        clear_session_vars(tokens)

    assert result["status"] == "blocked"
    assert "outside authoritative workspace binding" in result["error"]


@pytest.mark.parametrize("backend,path", [("ssh", "/srv/example/file.py"), ("docker", "/workspace/example/file.py")])
def test_file_write_fails_closed_for_nonlocal_backend_paths(backend, path, tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", backend)
    tokens = _gateway_session(path.rsplit("/", 1)[0])
    try:
        result = json.loads(write_file_tool(path, "data"))
    finally:
        clear_session_vars(tokens)

    assert "cannot verify" in result["error"]
    assert backend in result["error"]


@pytest.mark.parametrize("backend,cwd", [("ssh", "/srv/example"), ("docker", "/workspace/example")])
def test_terminal_fails_closed_for_nonlocal_backend_git(backend, cwd, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", backend)
    monkeypatch.setenv("TERMINAL_CWD", cwd)
    tokens = _gateway_session(cwd)
    try:
        result = json.loads(terminal_tool("git commit -m update", timeout=5))
    finally:
        clear_session_vars(tokens)

    assert result["status"] == "blocked"
    assert "cannot verify" in result["error"]
    assert backend in result["error"]


@pytest.mark.parametrize("shell", ["sh", "bash", "zsh", "/bin/bash"])
def test_shell_c_git_payload_fails_closed(shell, tmp_path):
    bound_repo = _git_repo(tmp_path / "bound")
    tokens = _gateway_session(bound_repo)
    try:
        error = check_terminal_side_effect_allowed(
            f"{shell} -c 'git commit -m nested'", bound_repo
        )
    finally:
        clear_session_vars(tokens)

    assert error is not None
    assert "cannot verify" in error


def test_nested_shell_c_git_payload_fails_closed(tmp_path):
    bound_repo = _git_repo(tmp_path / "bound")
    tokens = _gateway_session(bound_repo)
    try:
        error = check_terminal_side_effect_allowed(
            "bash -lc \"echo ok; sh -c 'git status'\"", bound_repo
        )
    finally:
        clear_session_vars(tokens)

    assert error is not None
    assert "cannot verify" in error


def test_shell_c_without_git_remains_allowed(tmp_path):
    bound_repo = _git_repo(tmp_path / "bound")
    tokens = _gateway_session(bound_repo)
    try:
        error = check_terminal_side_effect_allowed("bash -c 'echo git status'", bound_repo)
    finally:
        clear_session_vars(tokens)

    assert error is None


@pytest.mark.parametrize(
    "command",
    [
        "git fetch origin",
        "git remote add origin https://example.org/project.git",
        "git remote set-url origin https://example.org/project.git",
        "git branch --set-upstream-to origin/main",
        "git branch -u origin/main",
    ],
)
def test_mutating_git_forms_are_blocked_in_wrong_repo(command, tmp_path):
    bound_repo = _git_repo(tmp_path / "bound")
    other_repo = _git_repo(tmp_path / "other")
    tokens = _gateway_session(bound_repo)
    try:
        error = check_terminal_side_effect_allowed(command, other_repo)
    finally:
        clear_session_vars(tokens)

    assert error is not None
    assert "outside authoritative workspace binding" in error


def test_git_remote_get_url_is_read_only(tmp_path):
    bound_repo = _git_repo(tmp_path / "bound")
    other_repo = _git_repo(tmp_path / "other")
    tokens = _gateway_session(bound_repo)
    try:
        error = check_terminal_side_effect_allowed("git remote get-url origin", other_repo)
    finally:
        clear_session_vars(tokens)

    assert error is None


def test_git_dash_c_and_cd_chain_wrong_repo_are_blocked(tmp_path):
    bound_repo = _git_repo(tmp_path / "bound")
    other_repo = _git_repo(tmp_path / "other")
    tokens = _gateway_session(bound_repo)
    try:
        dash_c = check_terminal_side_effect_allowed(
            f"git -C {other_repo} commit -m update", bound_repo
        )
        cd_chain = check_terminal_side_effect_allowed(
            f"cd {other_repo} && git commit -m update", bound_repo
        )
    finally:
        clear_session_vars(tokens)

    assert dash_c is not None and "outside authoritative" in dash_c
    assert cd_chain is not None and "outside authoritative" in cd_chain


@pytest.mark.parametrize(
    "command",
    [
        "git --git-dir=/tmp/other/.git commit -m update",
        "git -c core.worktree=/tmp/other commit -m update",
        "git -ccore.worktree=/tmp/other commit -m update",
        "git -c include.path=/tmp/unsafe.gitconfig commit -m update",
        "git -c includeIf.gitdir:/tmp/repo/.path=/tmp/unsafe.gitconfig commit -m update",
        "git -c alias.ci='!git -C /tmp/other commit -m oops' ci",
        "GIT_DIR=/tmp/other/.git GIT_WORK_TREE=/tmp/other git commit -m update",
        "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.worktree GIT_CONFIG_VALUE_0=/tmp/other git commit -m update",
        "export GIT_CONFIG_GLOBAL=/tmp/unsafe.gitconfig; git commit -m update",
    ],
)
def test_git_target_redirects_fail_closed(command, tmp_path):
    bound_repo = _git_repo(tmp_path / "bound")
    tokens = _gateway_session(bound_repo)
    try:
        error = check_terminal_side_effect_allowed(command, bound_repo)
    finally:
        clear_session_vars(tokens)

    assert error is not None
    assert "cannot verify" in error


def test_ambient_git_config_redirect_fails_closed(tmp_path, monkeypatch):
    bound_repo = _git_repo(tmp_path / "bound")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/tmp/unsafe.gitconfig")
    tokens = _gateway_session(bound_repo)
    try:
        error = check_terminal_side_effect_allowed("git commit -m update", bound_repo)
    finally:
        clear_session_vars(tokens)

    assert error is not None
    assert "cannot verify" in error


def test_unknown_git_subcommand_fails_closed_from_bound_repo(tmp_path):
    bound_repo = _git_repo(tmp_path / "bound")
    tokens = _gateway_session(bound_repo)
    try:
        error = check_terminal_side_effect_allowed("git ci", bound_repo)
    finally:
        clear_session_vars(tokens)

    assert error is not None
    assert "cannot verify" in error


def test_all_git_invocations_are_scanned(tmp_path):
    bound_repo = _git_repo(tmp_path / "bound")
    other_repo = _git_repo(tmp_path / "other")
    tokens = _gateway_session(bound_repo)
    try:
        error = check_terminal_side_effect_allowed(
            f"git add README.md; git -C {other_repo} commit -m oops", bound_repo
        )
    finally:
        clear_session_vars(tokens)

    assert error is not None
    assert "outside authoritative workspace binding" in error


def test_non_repo_invocation_does_not_hide_later_mutation(tmp_path):
    bound_repo = _git_repo(tmp_path / "bound")
    other_repo = _git_repo(tmp_path / "other")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    tokens = _gateway_session(bound_repo)
    try:
        error = check_terminal_side_effect_allowed(
            f"cd {scratch} && git add README.md; git -C {other_repo} commit -m oops",
            bound_repo,
        )
    finally:
        clear_session_vars(tokens)

    assert error is not None
    assert "outside authoritative workspace binding" in error

