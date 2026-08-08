"""Regression tests for cua_backend._parse_key_combo.

A trailing '+'/'-' is the key being pressed, not a combo separator. re.split
on '[+\\-]' would otherwise consume it, leaving key=None so key() reports
"Could not parse key" and the keypress never happens (e.g. zoom-out 'cmd+-').
"""

from tools.computer_use.cua_backend import _parse_key_combo


def test_minus_key_with_modifier_is_preserved():
    # macOS/Chrome zoom-out; '-' is the key, not a separator.
    assert _parse_key_combo("cmd+-") == ("-", ["cmd"])
    assert _parse_key_combo("ctrl+-") == ("-", ["ctrl"])


def test_plus_key_with_modifier_is_preserved():
    # Zoom-in; the trailing '+' is the key.
    assert _parse_key_combo("ctrl++") == ("+", ["ctrl"])


def test_bare_minus_and_plus_are_preserved():
    assert _parse_key_combo("-") == ("-", [])
    assert _parse_key_combo("+") == ("+", [])


def test_existing_combos_unaffected():
    # Normal '+'-combined combos still parse.
    assert _parse_key_combo("cmd+s") == ("s", ["cmd"])
    assert _parse_key_combo("ctrl+alt+t") == ("t", ["ctrl", "option"])
    # Emacs-style '-' separators still resolve to (key, modifiers).
    assert _parse_key_combo("ctrl-c") == ("c", ["ctrl"])
