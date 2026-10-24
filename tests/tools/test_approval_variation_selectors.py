"""Variation selectors are invisible: after an ordinary letter or digit they can hide data in a command.

Emoji presentation (a selector after an emoji-capable code point) and ZWJ emoji sequences are
ordinary text and must not prompt.
"""

import pytest

from tools.approval_detection import detect_dangerous_command

INVISIBLE = "invisible or bidirectional Unicode control character (possible obfuscation)"


@pytest.mark.parametrize("command", [
    "printf 'a\ufe0f'",
    "echo 'x\U000E0100'",
    "echo '7\ufe0f'",
    "echo 'id\ufe00'",
    "cat 'notes\U000E01EF.txt'",
    "\ufe0fls",
    # A second selector after an emoji's own VS16 is not presentation.
    "echo '\u2764\ufe0f\ufe0f'",
])
def test_selector_after_ordinary_character_is_flagged(command):
    assert detect_dangerous_command(command)[:2] == (True, INVISIBLE)


@pytest.mark.parametrize("command", [
    "echo 'I \u2764\ufe0f this'",
    "ls '\U0001F5DE\ufe0f Journal/'",
    "ls '\u2139\ufe0f Info/' '\u25b6\ufe0f Media/' '\u2b05\ufe0f Back/' '\u24c2\ufe0f Metro/'",
    "echo '\u00a9\ufe0f 2026 \u2122\ufe0f'",
    "echo '\u263a\ufe0e'",
    "echo '\U0001F468\u200d\U0001F469\u200d\U0001F467 family'",
    "echo '\U0001F3F3\ufe0f\u200d\U0001F308'",
    "echo '\U0001F1EF\U0001F1F5'",
    "echo '\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8'",
])
def test_emoji_presentation_and_plain_text_pass(command):
    assert detect_dangerous_command(command)[0] is False
