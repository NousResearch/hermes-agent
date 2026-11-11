"""`git restore` and `git checkout -- <path>` destroy uncommitted work too.

`DANGEROUS_PATTERNS` has a block headed *"Git destructive operations that can
lose uncommitted work or rewrite shared history"* (tools/approval.py, just above
the `git reset --hard` entry). It covers `reset --hard`, `push --force`,
`clean -f` and `branch -D` — and neither of the two other commands that
overwrite the working tree from the index.

Nothing recovers from either. The reflog holds commits; it does not hold
unstaged edits.

The gating is deliberately narrow on the `checkout` side. `git checkout <branch>`
refuses rather than discards when it would lose changes, and `git checkout -b`
creates; prompting on those would prompt on the most common git command there
is. Only a pathspec (`--`, or a bare `.`) or `-f`/`--force` is matched.

On the `restore` side the exclusion is `--staged` without `--worktree`, which is
the unstage idiom and touches only the index. Short-flag spellings are treated
as dangerous on purpose: `_normalize_command_for_detection` lowercases before
matching, and git distinguishes `-S` (`--staged`) from `-s` (`--source`) by case
alone, so after folding they are the same token. Failing closed there costs one
prompt; failing open costs the file.
"""

import pytest

from tools.approval import detect_dangerous_command


_DESTROYS_UNCOMMITTED_WORK = [
    # git restore — working tree is the default target
    "git restore .",
    "git restore src/",
    "git restore src/app.py",
    "git restore --source=HEAD~1 app.py",
    "git restore -- .",
    # --staged AND --worktree restores both, so the worktree is in scope
    "git restore --staged --worktree app.py",
    # git checkout with an explicit pathspec
    "git checkout -- .",
    "git checkout -- src/app.py",
    "git checkout HEAD~1 -- app.py",
    "git checkout .",
    # git checkout forced
    "git checkout -f",
    "git checkout --force main",
    # git switch — the modern replacement for the branch half of checkout.
    # `--discard-changes` is aliased `-f` / `--force`.
    "git switch --discard-changes main",
    "git switch -f main",
    "git switch --force main",
    # flags that overwrite the working tree without ever spelling `--`
    "git checkout --pathspec-from-file=paths.txt",
    "git restore --pathspec-from-file=paths.txt",
    "git checkout --ours -- app.py",
    "git checkout --theirs app.py",
    # a path that merely CONTAINS the text of the exemption flag must not
    # suppress the prompt for the whole segment
    "git restore -- ./--staged",
    # still caught when it is not the first command in the line
    "cd /repo && git restore .",
    "git fetch && git checkout -- .",
]

_MUST_NOT_PROMPT = [
    # --staged alone restores the index only: the unstage idiom. Already
    # treated as benign by tests/tools/test_self_repo_guard.py.
    "git restore --staged pyproject.toml",
    "git restore --staged .",
    # switching branches refuses rather than discards
    "git checkout main",
    "git checkout -b feature/x",
    "git checkout --track origin/x",
    "git checkout -t origin/x",
    "git switch main",
    "git switch -c feature/x",
    "git switch --create feature/x",
    "git switch --track origin/x",
    "git restore --staged --pathspec-from-file=paths.txt",
    # ordinary read-only or additive git
    "git status",
    "git log --oneline",
    "git diff",
    "git add -A",
    "git commit -m 'msg'",
    "git stash",
    "git fetch --all",
]


@pytest.mark.parametrize("command", _DESTROYS_UNCOMMITTED_WORK)
def test_commands_that_overwrite_the_working_tree_are_dangerous(command):
    is_dangerous, _key, description = detect_dangerous_command(command)
    assert is_dangerous is True, f"{command!r} discards uncommitted work unprompted"
    assert description, f"{command!r} matched with no description"


@pytest.mark.parametrize("command", _MUST_NOT_PROMPT)
def test_non_destructive_git_is_not_dangerous(command):
    is_dangerous, _key, description = detect_dangerous_command(command)
    assert is_dangerous is False, f"false positive on {command!r}: {description}"


def test_the_siblings_this_matches_the_gating_of_still_fire():
    """Controls: the two entries this block was modelled on.

    If either of these stops matching, the new patterns are not the reason the
    suite is green.
    """
    assert detect_dangerous_command("git reset --hard")[0] is True
    assert detect_dangerous_command("git clean -fdx")[0] is True


def test_restore_and_checkout_report_distinct_descriptions():
    """A shared description would make the approval key collide, and an
    approval keyed on the description is granted per category."""
    _, _, restore_desc = detect_dangerous_command("git restore .")
    _, _, checkout_desc = detect_dangerous_command("git checkout -- .")
    assert restore_desc != checkout_desc
    assert "restore" in restore_desc.lower()
    assert "checkout" in checkout_desc.lower()


def test_glued_pathspec_spellings_are_not_gated_because_git_rejects_them():
    """`git checkout --"$f"` is one shell word, and git errors on it.

    Verified against git itself: in a repo with a modified `f.py`,
    `git checkout --"f.py"` prints `error: unknown option 'f.py'` and leaves the
    file modified. The shell glues `--` to the expansion, so git never sees the
    `--` separator followed by a pathspec. Gating the glued form would add a
    prompt on a command that discards nothing, which is why the pattern
    requires whitespace on both sides of `--`.
    """
    for command in ('git checkout --"$f"', "git checkout --'app.py'", "git checkout --app.py"):
        assert detect_dangerous_command(command)[0] is False, command
    # The spaced form -- the one git actually honours -- is gated.
    assert detect_dangerous_command('git checkout -- "$f"')[0] is True


def test_non_ascii_whitespace_still_reaches_the_patterns():
    """Every fixture above uses plain ASCII spaces.

    If `_normalize_command_for_detection` ever stops folding NBSP and friends,
    every pattern in this file silently stops matching and the suite stays
    green. This pins the normalization contract the patterns depend on.
    """
    assert detect_dangerous_command("git\u00a0restore .")[0] is True      # NBSP after `git`
    assert detect_dangerous_command("git restore\u00a0.")[0] is True      # NBSP before the pathspec


def test_the_residual_this_does_not_cover_is_stated():
    """A file named exactly `--staged`, after the `--` separator, is
    indistinguishable from the flag by regex alone and still suppresses.

    Recorded rather than hidden: if this ever starts returning True, the
    pattern grew a real tokenizer and this test should be deleted, not fixed.
    """
    assert detect_dangerous_command("git restore -- --staged")[0] is False
