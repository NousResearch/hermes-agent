import pytest

from tools.approval import (
    _workspace_retirement_block_result,
    detect_workspace_retirement_command,
)


@pytest.mark.parametrize(
    "command",
    [
        "git worktree remove /home/u/.worktree/Proj/nat",
        "git worktree remove --force /home/u/.worktree/Proj/nat",
        "git worktree remove -f ../wt/foo",
        "rm -rf /home/linux-nat/.worktree/LottoReward/nat",
        "rm -r /home/u/.worktree/Proj/user",
        "rm --recursive /srv/projects/emailhunter",
        "rm -rf /srv/projects/emailhunter",
        "rm -rf /srv/projects/emailhunter/",
    ],
)
def test_blocks_workspace_retirement(command):
    assert detect_workspace_retirement_command(command)[0] is True


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /tmp/whatever",
        "rm -rf node_modules",
        "rm -rf ./build",
        "rm -rf dist",
        "rm -rf /srv/projects/emailhunter/cache",
        "rm -rf /srv/projects/emailhunter/node_modules",
        "git worktree list",
        "git worktree prune",
        "rm file.txt",
        "ls /srv/projects",
    ],
)
def test_allows_safe_commands(command):
    assert detect_workspace_retirement_command(command)[0] is False


def test_workspace_retirement_block_result_marks_hardline():
    result = _workspace_retirement_block_result("delete git worktree")

    assert result["approved"] is False
    assert result["hardline"] is True
    assert result["workspace_retirement"] is True
    assert "BLOCKED" in result["message"]
