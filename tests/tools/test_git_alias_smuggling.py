"""Tests for git -c alias.NAME=VALUE smuggling detection (#131563).

Git expands an alias only when its name appears at the subcommand position:
a plain value re-enters git as ``git VALUE <args>``, and a ``!SHELL`` value
runs the shell string directly with the leading ``!`` stripped. Before the
fix, the alias value token sat in a position the hardline/dangerous position
anchors never inspected, so a definition plus an invocation of that alias ran
destructive commands through the approval floor unflagged.
"""

from tools.approval import detect_dangerous_command, detect_hardline_command


# -------------------------------------------------------------------------
# Invoked aliases must surface what git would actually run.
# -------------------------------------------------------------------------

def test_hardline_bang_alias_invoked_is_blocked():
    # "b" is defined as a shell-prefixed alias and then invoked at the
    # subcommand position: git runs `!rm -rf /` through /bin/sh directly.
    cmd = "git -c alias.b='!rm -rf /' b"
    is_hardline, description = detect_hardline_command(cmd)
    assert is_hardline, f"bang alias payload must hit the hardline floor, got {description!r}"


def test_hardline_bang_alias_shutdown_is_blocked():
    cmd = "git -c alias.b='!shutdown now' b"
    is_hardline, description = detect_hardline_command(cmd)
    assert is_hardline, f"bang alias payload must hit the hardline floor, got {description!r}"


def test_dangerous_plain_alias_invoked_is_flagged():
    # Plain alias values re-enter git as `git VALUE <args>`; the dangerous
    # layer must see the real "git push --force" spelling.
    cmd = "git -c alias.p='push' p --force origin main"
    is_dangerous, key, _ = detect_dangerous_command(cmd)
    assert is_dangerous and key == "git force push (rewrites remote history)"


def test_dangerous_unquoted_single_word_alias_invoked_is_flagged():
    # Unquoted single-word values need no shell quoting at all.
    cmd = "git -c alias.p=push p --force origin main"
    is_dangerous, key, _ = detect_dangerous_command(cmd)
    assert is_dangerous and key == "git force push (rewrites remote history)"


def test_second_config_def_invoked_is_flagged():
    cmd = "git -c alias.a='log' -c alias.b='!rm -rf /' b"
    is_hardline, _ = detect_hardline_command(cmd)
    assert is_hardline


# -------------------------------------------------------------------------
# Inert definitions and benign invocations must stay clean.
# -------------------------------------------------------------------------

def test_defined_but_not_invoked_stays_clean():
    # git runs `status`; the alias is never expanded.
    cmd = "git -c alias.p='push' status"
    assert detect_hardline_command(cmd)[0] is False
    assert detect_dangerous_command(cmd)[0] is False


def test_bang_defined_but_not_invoked_stays_clean():
    cmd = "git -c alias.b='!shutdown now' status"
    assert detect_hardline_command(cmd)[0] is False
    assert detect_dangerous_command(cmd)[0] is False


def test_invoked_benign_alias_stays_clean():
    cmd = "git -c alias.lg='log --oneline' lg"
    assert detect_hardline_command(cmd)[0] is False
    assert detect_dangerous_command(cmd)[0] is False


def test_different_subcommand_invoked_stays_clean():
    cmd = "git -c alias.p='push' commit -m x"
    assert detect_hardline_command(cmd)[0] is False
    assert detect_dangerous_command(cmd)[0] is False


def test_non_alias_config_value_stays_clean():
    cmd = "git -c color.ui=auto status"
    assert detect_hardline_command(cmd)[0] is False
    assert detect_dangerous_command(cmd)[0] is False
