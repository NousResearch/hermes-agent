"""Regression tests for PowerShell's encoded-command approval rule."""

from tools.approval_detection import _iter_dangerous_descriptions


def test_grep_extended_regex_is_not_powerShell_encoded_command():
    """Nested Unix `grep -E` is not PowerShell's `-EncodedCommand`/`-enc` switch."""
    command = 'pwsh -NoProfile -Command "grep -E pattern input.txt"'
    descriptions = _iter_dangerous_descriptions(
        command, only=frozenset({"PowerShell encoded command execution"})
    )
    assert "PowerShell encoded command execution" not in set(descriptions)
