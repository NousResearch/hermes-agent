import os

import pytest

from tools.approval import (
    _stash_sweep_block_result,
    check_all_command_guards,
    check_dangerous_command,
    detect_stash_sweep_command,
    detect_workspace_retirement_command,
)


@pytest.mark.parametrize(
    "command",
    [
        "git stash",
        "git stash push -m wip",
        "git stash save quick save",
        "git stash clear",
        "git stash drop",
        "git stash drop stash@{0}",
        "git clean -f",
        "git clean -fd",
        "git clean -fdx",
        "git checkout -- .",
        "git checkout .",
        "git restore .",
        "git restore --staged .",
        "git reset --hard",
        "git reset --hard HEAD~1",
        "git checkout HEAD -- .",
        "git restore --source=HEAD .",
        "git -C /path stash",
        "git -C /tmp/proj reset --hard",
        "git -C /tmp clean -fd",
    ],
)
def test_blocks_stash_sweep_commands(command):
    assert detect_stash_sweep_command(command)[0] is True


@pytest.mark.parametrize(
    "command",
    [
        "git stash list",
        "git stash show",
        "git stash show stash@{0}",
        "git stash pop",
        "git stash apply",
        "git stash branch rescue-branch",
        "git checkout main",
        "git checkout -b feature-x",
        "git checkout -- path/to/single-file.py",
        "git checkout HEAD -- single/file.py",
        "git restore path/to/single-file.py",
        "git restore --source=HEAD single/file.py",
        "git -C /path stash list",
        "git clean -n",
        "git clean --dry-run",
        'echo "git stash"',
        "ls -la",
    ],
)
def test_allows_safe_git_commands(command):
    assert detect_stash_sweep_command(command)[0] is False


def test_stash_sweep_block_result_marks_hardline():
    result = _stash_sweep_block_result("git stash sweep")

    assert result["approved"] is False
    assert result["hardline"] is True
    assert result["stash_sweep"] is True
    assert "BLOCKED" in result["message"]
    assert "JARVIS" in result["message"]


def test_stash_sweep_blocks_before_yolo(monkeypatch):
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    result = check_dangerous_command("git stash", "local")

    assert result["approved"] is False
    assert result.get("stash_sweep") is True


def test_stash_sweep_blocks_in_combined_guard_before_yolo(monkeypatch):
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    result = check_all_command_guards("git reset --hard", "local")

    assert result["approved"] is False
    assert result.get("stash_sweep") is True


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /srv/projects/emailhunter",
    ],
)
def test_workspace_retirement_still_blocks(command):
    assert detect_workspace_retirement_command(command)[0] is True


def test_workspace_retirement_still_allows_tmp_cleanup():
    assert detect_workspace_retirement_command("rm -rf /tmp/whatever")[0] is False