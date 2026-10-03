"""Public names for two terminal-guard primitives.

A plugin that adds its own command policy on top of the terminal tool (a
``tool_request`` middleware, say) needs to read commands the way the built-in
guards do: quoted and heredoc text blanked before any keyword match, and
``--help`` / ``--version`` invocations never blocked. Each public name is the
SAME object as the private spelling, so the built-in guards are unchanged.
"""

import pytest

from tools import terminal_tool_guards as guards


@pytest.mark.parametrize(
    ("public", "private"),
    [
        ("strip_quotes", "_strip_quotes"),
        ("looks_like_help_or_version_command", "_looks_like_help_or_version_command"),
    ],
)
def test_public_name_is_the_private_helper(public, private):
    assert getattr(guards, public) is getattr(guards, private)


def test_strip_quotes_blanks_quoted_text():
    command = "echo 'nohup &' \"x\" `y`"
    assert guards.strip_quotes(command) == "echo '' \"\" ``"


def test_looks_like_help_or_version_command():
    assert guards.looks_like_help_or_version_command("npm run dev --help")
    assert guards.looks_like_help_or_version_command("node -v")
    assert not guards.looks_like_help_or_version_command("npm run dev")
