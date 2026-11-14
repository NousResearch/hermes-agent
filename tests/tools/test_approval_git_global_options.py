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


@pytest.mark.parametrize(
    "description, legacy_key",
    [
        ("git reset --hard (destroys uncommitted changes)", r"git\s+reset\s+--h(?:a(?:r(?:d)?)?)?"),
        ("git force push (rewrites remote history)", r"git\s+push"),
        ("git force push short flag (rewrites remote history)", r"git\s+push"),
        ("git clean with force (deletes untracked files)", r"git\s+clean\s+-[^\s]*f"),
        ("git branch force delete", r"git\s+branch\s+(?-i:-D)"),
        ("git branch force delete", r"git\s+branch\s+-D"),
        ("git branch force delete (long flags)", r"git\s+branch"),
        ("git branch force delete (long flags, force-first)", r"git\s+branch"),
    ],
)
def test_pre_change_approval_keys_still_resolve(description, legacy_key):
    from tools.approval_detection import _approval_key_aliases

    assert legacy_key in _approval_key_aliases(description)
    assert description in _approval_key_aliases(legacy_key)


@pytest.mark.parametrize("option", ["-c", "-c a=b", "-C '/a b'", "--git-dir x"])
def test_a_run_of_global_options_is_scanned_linearly(option):
    """Overlapping option alternatives used to re-split a long option run exponentially."""
    import time

    start = time.monotonic()
    assert not detect_dangerous_command("git " + (option + " ") * 60 + "x")[0]
    assert detect_dangerous_command("git " + (option + " ") * 60 + "reset --hard")[0]
    assert time.monotonic() - start < 2
