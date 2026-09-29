"""Rewrite dotted identifiers, emails, long flags and shell operators for speech.

``smooth_whitespace_for_tts`` puts a space after any dot followed by a letter, so
``api.anthropic.com`` leaves the pipeline as ``api. anthropic. com`` and the engine
reads three sentences. ``normalize_symbols_for_tts`` maps ``&`` to " and ", so ``&&``
becomes "and and". Spelling these tokens out first leaves no dot-then-letter pair and
no bare ``&`` for those rules to act on.

Matches are stashed behind placeholders so a later rule here cannot re-match the
output of an earlier one.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

_EMAIL_RE = re.compile(r"(?<![\w.])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Domain, config key, module path: two or more alphanumeric segments joined by dots.
_DOTTED_RE = re.compile(r"(?<![\w./@-])[A-Za-z_][\w-]*(?:\.[A-Za-z_][\w-]*)+(?![\w./@-])")
# A dotted quad matches the version and dotted-identifier shapes too, so it is
# claimed first and kept verbatim.
_IP_RE = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?![\w.])")
_VERSION_RE = re.compile(r"(?<![\w.])v?\d+\.\d+(?:\.\d+)*(?![\w.])")
_FLAG_RE = re.compile(r"(?<![\w-])--([A-Za-z][\w-]*)")

# Only doubled forms are shell operators. A single "&" stays with the generic
# rule so "R&D" and "Tom & Jerry" keep reading as prose.
_SHELL_OPERATORS: Tuple[Tuple[str, str], ...] = (
    ("2>&1", " redirecting errors to output "),
    ("&&", " and then "),
    ("||", " or else "),
    (">>", " appending to "),
)
_SHELL_OPERATOR_RE = re.compile(
    "|".join(re.escape(operator) for operator, _ in _SHELL_OPERATORS))
_SHELL_OPERATOR_WORDS: Dict[str, str] = dict(_SHELL_OPERATORS)

_PLACEHOLDER = "\x01%d\x02"
_CHAR_WORDS = {".": "dot", "-": "dash", "_": "underscore", "+": "plus"}


def _speak_characters(token: str) -> str:
    """Replace token punctuation with words, keeping alphanumeric runs together."""
    pieces = [_CHAR_WORDS.get(character, character) for character in token]
    spoken: List[str] = []
    word = ""
    for piece in pieces:
        if len(piece) == 1 and piece.isalnum():
            word += piece
            continue
        if word:
            spoken.append(word)
            word = ""
        spoken.append(piece)
    if word:
        spoken.append(word)
    return " ".join(spoken)


def speak_technical_tokens(text: str) -> str:
    """Return ``text`` with dotted identifiers, emails, flags and operators spelled out."""
    if not text:
        return ""
    stash: List[str] = []

    def protect(value: str) -> str:
        stash.append(value)
        return _PLACEHOLDER % (len(stash) - 1)

    # Addresses and versions are spoken correctly by the engines we tested, so
    # they are protected rather than respelled.
    text = _IP_RE.sub(lambda match: protect(match.group(0)), text)
    text = _VERSION_RE.sub(lambda match: protect(match.group(0)), text)
    text = _SHELL_OPERATOR_RE.sub(
        lambda match: protect(_SHELL_OPERATOR_WORDS[match.group(0)]), text)
    text = _EMAIL_RE.sub(
        lambda match: protect(_speak_characters(match.group(0).replace("@", " at "))), text)
    text = _DOTTED_RE.sub(lambda match: protect(_speak_characters(match.group(0))), text)
    text = _FLAG_RE.sub(
        lambda match: protect("dash dash " + match.group(1).replace("-", " dash ")), text)

    for index, value in enumerate(stash):
        text = text.replace(_PLACEHOLDER % index, value)
    return re.sub(r"[ \t]{2,}", " ", text)
