"""Emoji-presentation bases and the hidden variation-selector matcher for :mod:`tools.approval_detection` (moved from there)."""

import re


_EMOJI_RANGE = r'\U0001F000-\U0001FAFF\u2600-\u27BF'
# Code points that take VS15/VS16 for text/emoji presentation: the two emoji blocks above plus the BMP
# singletons with the Unicode Emoji property (© ™ ℹ, arrows, ⌚ ▶ ⬅ ⭐ 〰 ㊗ …), as in #114843. Digits,
# ``#`` and ``*`` are deliberately absent: a selector after a letter or digit is the hiding signal.
_EMOJI_PRESENTATION_BASES = (
    r'\u00a9\u00ae\u203c\u2049\u2122\u2139\u2194-\u2199\u21a9\u21aa\u231a\u231b\u2328\u23cf'
    r'\u23e9-\u23f3\u23f8-\u23fa\u24c2\u25aa\u25ab\u25b6\u25c0\u25fb-\u25fe\u2934\u2935'
    r'\u2b05-\u2b07\u2b1b\u2b1c\u2b50\u2b55\u3030\u303d\u3297\u3299' + _EMOJI_RANGE
)
# Matched on the RAW command: detection variants are NFKC-folded, which turns ℹ + VS16 into ``i`` + VS16.
# A keycap emoji (digit, ``#`` or ``*`` + VS16 + U+20E3) is exempt only as that exact sequence, so a
# selector after a digit with no keycap mark is still flagged.
_HIDDEN_VARIATION_SELECTOR_RE = re.compile(
    rf'(?<![{_EMOJI_PRESENTATION_BASES}])(?!(?<=[0-9#*])\ufe0f\u20e3)[\ufe00-\ufe0f\U000e0100-\U000e01ef]')
