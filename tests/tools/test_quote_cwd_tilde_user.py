"""``cd`` targets of the form ``~other`` must stay expandable by the remote shell.

Preserving ``~other`` through the config bridge is only half the path. The
bootstrap then runs it through ``_quote_cwd_for_cd``, which fell through to
``shlex.quote`` for the named form, emitting ``cd '~other/x'``. Tilde expansion
does not happen inside quotes, so the ``cd`` fails and the session silently
starts in the login directory: the same quiet failure one layer down.

Verified against a real shell:

    cd ~grimm/'Desktop'   ->  /Users/grimm/Desktop
    cd '~grimm'           ->  bash: cd: ~grimm: No such file or directory
"""

from __future__ import annotations

import subprocess

import pytest

from tools.environments.base import BaseEnvironment

_q = BaseEnvironment._quote_cwd_for_cd


def test_named_tilde_is_not_swallowed_by_quotes():
    """The bug: the whole thing was quoted, so no shell ever expanded it."""
    out = _q("~alice/work")

    assert out != "'~alice/work'", "the named tilde was quoted out of existence"
    assert out.startswith("~alice/")


def test_named_tilde_without_a_path():
    assert _q("~alice") == "~alice"
    assert _q("~alice/") == "~alice"


def test_remainder_is_still_quoted():
    """Only the tilde prefix may be bare; the rest keeps its quoting."""
    assert _q("~alice/my project") == "~alice/'my project'"
    assert _q("~alice/a;rm -rf b") == "~alice/'a;rm -rf b'"


@pytest.mark.parametrize(
    "cwd",
    ["~$(id)/x", "~`id`/x", "~a b/x", "~-bad/x", "~/../etc", "~*/x"],
)
def test_unsafe_account_names_take_the_quoted_path(cwd):
    """An account name outside the conservative set must never go out bare."""
    out = _q(cwd)

    assert out.startswith("'") or out.startswith("$HOME"), out


def test_existing_forms_are_unchanged():
    """Guards: the three cases the helper already handled keep their output."""
    assert _q("~") == "~"
    assert _q("~/") == "$HOME"
    assert _q("~/work") == "$HOME/work"
    assert _q("/abs/path") == "/abs/path"
    assert _q("/a b") == "'/a b'"


def test_a_real_shell_resolves_the_emitted_form():
    """End to end: what we emit must actually expand in bash.

    Uses the running account, so the home directory genuinely exists.
    """
    import getpass

    user = getpass.getuser()
    emitted = _q(f"~{user}")
    assert emitted == f"~{user}"

    resolved = subprocess.run(
        ["bash", "-c", f"cd {emitted} && pwd"],
        capture_output=True, text=True, timeout=30,
    )
    assert resolved.returncode == 0, resolved.stderr
    assert resolved.stdout.strip()

    quoted = subprocess.run(
        ["bash", "-c", f"cd '~{user}' && pwd"],
        capture_output=True, text=True, timeout=30,
    )
    assert quoted.returncode != 0, (
        "the pre-fix form unexpectedly worked; this test no longer proves anything"
    )
