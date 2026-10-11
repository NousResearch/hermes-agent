"""Public names for two terminal-guard primitives.

A plugin that adds its own command policy on top of the terminal tool (a
``tool_request`` middleware, say) needs to read commands the way the built-in
guards do: quoted and heredoc text blanked before any keyword match, and
``--help`` / ``--version`` invocations never blocked. The private spellings
stay as aliases bound to the same objects; the built-in guard reads the public
names, so overriding a public name is what takes effect.
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


def test_overriding_strip_quotes_reaches_the_foreground_guard(monkeypatch):
    seen = []

    def fake_strip(command):
        seen.append(command)
        return "npm run dev"

    monkeypatch.setattr(guards, "strip_quotes", fake_strip)
    assert guards._foreground_background_guidance("echo 'npm run dev'") is not None
    assert seen == ["echo 'npm run dev'"]


def test_overriding_looks_like_help_reaches_the_foreground_guard(monkeypatch):
    assert guards._foreground_background_guidance("npm run dev") is not None
    monkeypatch.setattr(guards, "looks_like_help_or_version_command", lambda command: True)
    assert guards._foreground_background_guidance("npm run dev") is None
