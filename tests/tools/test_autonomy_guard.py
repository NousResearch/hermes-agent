"""Autonomy guard tests for protected branches, approvals, and tool wiring."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import autonomy_guard
from tools.autonomy_guard import enforce_write_policy, evaluate_terminal_command, run_bootstrap_preflight


def _init_repo(tmp_path: Path, branch: str = "main") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", branch], cwd=repo, check=True, capture_output=True, text=True)
    (repo / "tracked.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return repo


@pytest.fixture(autouse=True)
def _clear_policy_cache():
    autonomy_guard.load_autonomy_policy.cache_clear()
    yield
    autonomy_guard.load_autonomy_policy.cache_clear()


def test_enforce_write_policy_blocks_protected_branch(tmp_path):
    repo = _init_repo(tmp_path, branch="main")
    decision = enforce_write_policy("write_file", str(repo / "tracked.txt"))
    assert decision["allowed"] is False
    assert decision["status"] == "blocked"
    assert decision["branch"] == "main"


def test_enforce_write_policy_allows_feature_branch(tmp_path):
    repo = _init_repo(tmp_path, branch="codex/test-branch")
    decision = enforce_write_policy("write_file", str(repo / "tracked.txt"))
    assert decision["allowed"] is True
    assert decision["branch"] == "codex/test-branch"


def test_terminal_policy_requires_approval_for_push(tmp_path):
    repo = _init_repo(tmp_path, branch="codex/push-test")
    decision = evaluate_terminal_command("git push origin HEAD", workdir=str(repo))
    assert decision["allowed"] is False
    assert decision["status"] == "approval_required"
    assert "push commits" in decision["description"]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git tag v1.2.3", "git tag"),
        ("gh pr merge 123 --squash", "pull request"),
        ("npm publish", "publish"),
    ],
)
def test_terminal_policy_requires_approval_for_release_commands(tmp_path, command, expected):
    repo = _init_repo(tmp_path, branch="codex/release-test")
    decision = evaluate_terminal_command(command, workdir=str(repo))
    assert decision["allowed"] is False
    assert decision["status"] == "approval_required"
    assert expected in decision["message"].lower() or expected in decision["description"].lower()


def test_terminal_policy_blocks_forbidden_bypass(tmp_path):
    repo = _init_repo(tmp_path, branch="codex/safe")
    decision = evaluate_terminal_command("claude --dangerously-skip-permissions", workdir=str(repo))
    assert decision["allowed"] is False
    assert decision["status"] == "blocked"


def test_terminal_policy_blocks_mutating_command_on_main(tmp_path):
    repo = _init_repo(tmp_path, branch="main")
    decision = evaluate_terminal_command("git add tracked.txt", workdir=str(repo))
    assert decision["allowed"] is False
    assert decision["status"] == "blocked"
    assert "protected branch" in decision["message"]


@patch("tools.file_tools._get_file_ops")
def test_write_file_tool_returns_blocked_when_policy_denies(mock_get):
    from tools.file_tools import write_file_tool

    with patch(
        "tools.file_tools.enforce_write_policy",
        return_value={"allowed": False, "status": "blocked", "branch": "main", "repo_root": "/tmp/repo", "message": "blocked"},
    ):
        result = json.loads(write_file_tool("/tmp/out.txt", "data"))

    assert result["status"] == "blocked"
    assert result["error"] == "blocked"
    mock_get.assert_not_called()


def test_terminal_tool_returns_approval_required_before_exec():
    from tools.terminal_tool import terminal_tool

    with patch("tools.terminal_tool._get_env_config", return_value={
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }), patch("tools.terminal_tool._start_cleanup_thread"), patch(
        "tools.terminal_tool.evaluate_terminal_command",
        return_value={
            "allowed": False,
            "status": "approval_required",
            "description": "push commits to a remote",
            "message": "approval required",
        },
    ):
        result = json.loads(terminal_tool("git push origin HEAD"))

    assert result["status"] == "approval_required"
    assert "approval required" in result["error"]


def test_bootstrap_preflight_accepts_explicit_credentials(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = run_bootstrap_preflight(
        explicit_api_key="test-key",
        explicit_base_url="https://example.com/v1",
        requested_provider="openrouter",
    )

    assert result["ok"] is True


def test_terminal_policy_fails_closed_when_policy_is_malformed(tmp_path, monkeypatch):
    malformed = tmp_path / "autonomy_policy.yaml"
    malformed.write_text("approval: [\n", encoding="utf-8")
    monkeypatch.setattr(autonomy_guard, "POLICY_PATH", malformed)
    autonomy_guard.load_autonomy_policy.cache_clear()

    decision = evaluate_terminal_command("git push origin HEAD", workdir=str(tmp_path))
    assert decision["allowed"] is False
    assert decision["status"] == "blocked"
    assert "malformed" in decision["message"].lower()


def test_bootstrap_preflight_fails_closed_when_policy_is_missing(tmp_path, monkeypatch):
    missing = tmp_path / "missing-policy.yaml"
    monkeypatch.setattr(autonomy_guard, "POLICY_PATH", missing)
    autonomy_guard.load_autonomy_policy.cache_clear()

    result = run_bootstrap_preflight(
        explicit_api_key="test-key",
        explicit_base_url="https://example.com/v1",
        requested_provider="openrouter",
    )

    assert result["ok"] is False
    assert "missing" in result["message"].lower()


def test_write_file_tool_blocks_when_policy_is_malformed(tmp_path, monkeypatch):
    from tools.file_tools import write_file_tool

    malformed = tmp_path / "autonomy_policy.yaml"
    malformed.write_text("branch_protection: [\n", encoding="utf-8")
    monkeypatch.setattr(autonomy_guard, "POLICY_PATH", malformed)
    autonomy_guard.load_autonomy_policy.cache_clear()
    monkeypatch.setattr("tools.file_tools._check_sensitive_path", lambda _path: None)

    result = json.loads(write_file_tool(str(tmp_path / "out.txt"), "data"))
    assert result["status"] == "blocked"
    assert "malformed" in result["error"].lower()
