"""Git global options before the subcommand (``git -C <dir> reset --hard``) must not
hide a destructive git command from the approval gate."""

import pytest

from tools.approval import detect_dangerous_command


@pytest.mark.parametrize(
    "cmd",
    [
        "git -C /tmp/x reset --hard",
        "git --work-tree /tmp/x reset --hard",
        "git -c core.pager=cat reset --hard",
        "git --git-dir /tmp/x/.git --work-tree /tmp/x reset --hard",
        "git --no-pager -C /tmp/x reset --hard",
        "git -C /tmp/x push --force origin main",
        "git -C /tmp/x push -f origin main",
        "git -C /tmp/x clean -fdx",
        "git -C /tmp/x branch -D feature",
        "git -C /tmp/x branch --delete --force feature",
        'git -C "/tmp/g space" reset --hard',
        'git -C "/tmp/g space" push --force origin main',
        'git -C "/tmp/g space" clean -fdx',
        'git -C "/tmp/g space" branch -D feature',
        "git --work-tree '/tmp/g space' --git-dir '/tmp/g space/.git' reset --hard",
        'git --git-dir="/tmp/g space/.git" reset --hard',
    ],
)
def test_flagged(cmd):
    assert detect_dangerous_command(cmd)[0], cmd


@pytest.mark.parametrize(
    "cmd",
    [
        "git -C /tmp/x status",
        "git -C /tmp/x push origin main",
        "git -C /tmp/x branch -d feature",
        "git --no-pager log push --force",
        "git status && echo push --force",
        'git -C "/tmp/g space" status',
    ],
)
def test_not_flagged(cmd):
    assert not detect_dangerous_command(cmd)[0], cmd
