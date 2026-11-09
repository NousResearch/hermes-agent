"""Git's global options sit between the binary and the subcommand, and every git rule anchored on a
literal `git <verb>` — so `git -C <path> push --force` ran with no approval prompt at all (#150744).

The shapes below are the option forms git documents before a subcommand (`-C`, `-c`, `--git-dir`,
`--work-tree`), including the multi-option and quoted-path spellings an agent reaches for when it
avoids `cd x && git …`.
"""
import os
import subprocess
import sys

import pytest

from tools.approval_detection import _approval_key_aliases, detect_dangerous_command

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FORCE_PUSH = "git force push (rewrites remote history)"
FORCE_PUSH_SHORT = "git force push short flag (rewrites remote history)"
RESET_HARD = "git reset --hard (destroys uncommitted changes)"
CLEAN_FORCE = "git clean with force (deletes untracked files)"
BRANCH_FORCE_DELETE = "git branch force delete"
BRANCH_FORCE_DELETE_LONG = "git branch force delete (long flags)"


@pytest.mark.parametrize(
    "command, description",
    [
        ("git -C /tmp/r push --force origin x", FORCE_PUSH),
        ("git -C /tmp/r push -f origin x", FORCE_PUSH_SHORT),
        ('git -C "/tmp/my repo" push --force origin x', FORCE_PUSH),
        # A joined quote is one shell word: the value must not be split at its first space, or the
        # remainder leaves an unmatched fragment between the options and the verb.
        ("git -C'/tmp/my repo with spaces' push --force origin x", FORCE_PUSH),
        ("git -C'/tmp/my repo with spaces' push -f origin x", FORCE_PUSH_SHORT),
        ("git --git-dir='/tmp/my repo with spaces/.git' push --force origin x", FORCE_PUSH),
        ("git -c user.name='a b c' push --force origin x", FORCE_PUSH),
        ("git -c core.pager=\"less -R\" push --force origin x", FORCE_PUSH),
        ("git -C'/tmp/my repo' -c 'user.name=John Doe' push --force origin x", FORCE_PUSH),
        ("git -C'/tmp/my repo with spaces' reset --hard HEAD~1", RESET_HARD),
        ("git -C'/tmp/my repo with spaces' clean -fdx", CLEAN_FORCE),
        ("git --work-tree='/tmp/my repo with spaces' clean -fdx", CLEAN_FORCE),
        ("git -C'/tmp/my repo with spaces' branch -D old", BRANCH_FORCE_DELETE),
        ("git -C /tmp/r -c core.pager=cat push --force origin x", FORCE_PUSH),
        ("git --no-pager push --force", FORCE_PUSH),
        ("git --git-dir=/tmp/r/.git push --force origin x", FORCE_PUSH),
        ("git --git-dir /tmp/r/.git --work-tree /tmp/r push --force origin x", FORCE_PUSH),
        ("git -C /tmp/r reset --hard HEAD~1", RESET_HARD),
        ("git -C /tmp/r clean -fdx", CLEAN_FORCE),
        ("git -C /tmp/r branch -D old", BRANCH_FORCE_DELETE),
        ("git -C /tmp/r branch --delete --force old", BRANCH_FORCE_DELETE_LONG),
        ("sudo git -C /tmp/r push --force origin x", FORCE_PUSH),
        ("cd /tmp && git -C /tmp/r push --force origin x", FORCE_PUSH),
        ("GIT -C /tmp/r PUSH --FORCE origin x", FORCE_PUSH),
    ],
)
def test_git_global_options_do_not_hide_the_subcommand(command, description):
    assert detect_dangerous_command(command) == (True, description, description), command


@pytest.mark.parametrize(
    "command",
    [
        "git -C /tmp/r status",
        "git -C /tmp/r log --oneline -5",
        "git -C /tmp/r push origin main",
        "git -C /tmp/r pull --ff-only",
        "git -C /tmp/r branch -d merged",
        "git -C /tmp/r clean -n",
        "git -C /tmp/r checkout -b new-branch",
        "git --git-dir=/tmp/r/.git commit -m wip",
        "git -C /tmp/r -c core.pager=cat log",
        "git -C'/tmp/my repo with spaces' status",
        "git -C'/tmp/my repo with spaces' push origin main",
        "git -C'/tmp/my repo with spaces' branch -d merged",
        "git -C'/tmp/my repo with spaces' clean -n",
        "git --git-dir='/tmp/my repo with spaces/.git' commit -m wip",
    ],
)
def test_safe_git_global_option_forms_stay_safe(command):
    assert detect_dangerous_command(command)[0] is False, command


@pytest.mark.parametrize(
    "description, legacy_key",
    [
        (RESET_HARD, r"git\s+reset"),
        (FORCE_PUSH, r"git\s+push"),
        (FORCE_PUSH_SHORT, r"git\s+push"),
        (CLEAN_FORCE, r"git\s+clean"),
        (BRANCH_FORCE_DELETE, r"git\s+branch"),
        (BRANCH_FORCE_DELETE_LONG, r"git\s+branch"),
        ("git branch force delete (long flags, force-first)", r"git\s+branch"),
    ],
)
def test_option_aware_prefix_keeps_stored_approval_keys_matching(description, legacy_key):
    """The prefix moves these rules' regex-derived approval key; a permanent allowlist entry written
    under the old key must still satisfy the rule."""
    assert legacy_key in _approval_key_aliases(description)


def test_long_option_run_still_reaches_the_subcommand():
    """The shared prefix repeats, so a long option run must terminate (the #129281 hazard) and still
    expose the verb. Bounded in a subprocess: a regression hangs the forking process, not pytest."""
    code = '''
from tools.approval_detection import detect_dangerous_command
run = "--opt val --flag=x -q - " * 58
command = f"git {run}-C /tmp/r push --force origin x"
assert detect_dangerous_command(command)[0] is True, command[:60]
for prefix in ("git", "git -C /tmp/r", "git --git-dir=/tmp/r/.git"):
    for bulk in ("--exclude " * 58, "--opt val " * 58, "--opt=val " * 58,
                 "-C'/tmp/a b' -c 'k=v' " * 58):
        assert detect_dangerous_command(f"{prefix} {bulk}log --oneline")[0] is False, prefix
'''
    env = {**os.environ, "PYTHONPATH": REPO_ROOT}
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30, cwd=REPO_ROOT, env=env)
