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
        "git -c core.pager=cat branch --force --delete feature",
        'git -C "/tmp/g space" reset --hard',
        'git -C "/tmp/g space" push --force origin main',
        'git -C "/tmp/g space" clean -fdx',
        'git -C "/tmp/g space" branch -D feature',
        "git --work-tree '/tmp/g space' --git-dir '/tmp/g space/.git' reset --hard",
        'git --git-dir="/tmp/g space/.git" reset --hard',
        # A backslash-escaped space or quote stays inside its word (what tab completion inserts).
        "git -C my\\ repo push --force",
        "git -C /a\\ b\\ c reset --hard",
        "git -c user.name=a\\ b clean -fd",
        "git -C /tmp/my\\\tdir branch -D old",
        'git -c user.name="John \\"Big John\\" Doe" reset --hard',
        "git -c user.name=O\\'Brien push --force",
        # A backslash is literal inside single quotes, and a single quote is literal inside double quotes.
        "git -C 'a\\' -c user.name=O\\'Brien push --force",
        'git -c user.name="O\'Brien \\"Big Jr\\" O\'Neil" reset --hard',
        # A command substitution is one word even unquoted.
        "git -C $(git rev-parse --show-toplevel) push --force",
        "git -C $(realpath $(pwd)) push --force origin main",
        "git -C $(dirname $(realpath $(pwd))) push --force origin main",
        "git -C $(a $(b $(c $(d)))) reset --hard",
        "git -C `git rev-parse --show-toplevel` reset --hard",
        # An option git doesn't list never takes the verb as its value.
        "git --new-flag push --force",
        "git -q push -f",
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
        "git -C /tmp/x reset --soft HEAD~1",
        "git --no-pager log push --force",
        "git status && echo push --force",
        'git -C "/tmp/g space" status',
        "git -C my\\ repo push origin main",
        "git -C $(git rev-parse --show-toplevel) push origin main",
        "git -C $(realpath $(pwd)) push origin main",
        "git --no-pager log --oneline push --force",
        # A value-less option, or an unknown one with no option after its word, takes no value.
        "git -P log --oneline push --force",
        "git -q log push --force",
        "git -C $(dirname $(realpath $(pwd))) push origin main",
    ],
)
def test_not_flagged(cmd):
    assert not detect_dangerous_command(cmd)[0], cmd


@pytest.mark.parametrize(
    "description, legacy_key",
    [
        ("git reset --hard (destroys uncommitted changes)", r"git\s+reset\s+--h(?:a(?:r(?:d)?)?)?"),
        ("git reset --hard (destroys uncommitted changes)", r"git\s+reset\s+--hard"),
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


def test_options_outside_the_value_list_still_reach_the_subcommand():
    """An option git adds later takes its value as a separate word; a long run of them must still
    expose the verb, and stay linear when no verb follows."""
    import time

    start = time.monotonic()
    run = "--opt val --flag=x -q - " * 58
    assert detect_dangerous_command(f"git {run}-C /tmp/r push --force origin x")[0]
    for prefix in ("git", "git -C /tmp/r", "git --git-dir=/tmp/r/.git"):
        for bulk in ("--exclude " * 58, "--opt val " * 58, "--opt=val " * 58,
                     "-C'/tmp/a b' -c 'k=v' " * 58, "-C /a\\ b\\ c -c k=v\\ w " * 58, "-C " + "$(pwd)" * 58 + " "):
            assert not detect_dangerous_command(f"{prefix} {bulk}log --oneline")[0], prefix
    assert time.monotonic() - start < 2


@pytest.mark.parametrize("value", ["$(" * 30000, "$(a " * 25000, "$(" + "(a)" * 30000, "$(" + "((a)" * 25000])
def test_an_unclosed_nested_substitution_is_scanned_linearly(value):
    """Each `$(` nesting level is atomic: an unclosed run must fail in one pass, not re-split per level.
    The detector's parser budget refuses such a command before the rules run, so time the git rules."""
    import re
    import time

    from tools.approval_detection import DANGEROUS_PATTERNS
    from tools.approval_git_options import _GIT

    rules = [re.compile(p, re.IGNORECASE) for p, _ in DANGEROUS_PATTERNS if isinstance(p, str) and _GIT in p]
    assert rules
    start = time.monotonic()
    for rule in rules:
        assert not rule.search(f"git -C {value} log --oneline")
    assert time.monotonic() - start < 2


@pytest.mark.parametrize("prefix, unit", [("", "git -c "), ("", "git push "), ("git push ", "-c git ")])
def test_a_repeated_git_word_does_not_rescan_the_rest(prefix, unit):
    """Each `git` word starts a new search; an uncapped option run or push scan made this quadratic."""
    import time

    start = time.monotonic()
    assert not detect_dangerous_command(prefix + unit * 3000 + "x;")[0]
    assert time.monotonic() - start < 2
