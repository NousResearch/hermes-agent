"""Conservative correction of escape drift before shared fuzzy matching.

Corrections require the corrected ``old_string`` to appear verbatim in the file.
That evidence cannot establish the intended escaping of newly added code: deliberate
double backslashes look the same as a serialization artifact. Backslash correction
therefore halves only ``old_string`` and requires a backslash-free ``new_string``.
An otherwise verified doubled anchor with replacement backslashes raises an explicit
error; callers must reject it before fuzzy matching can apply the ambiguous text.

Quote correction retains its separate consistency checks. Successful corrections
return a note for the tool result; already-exact anchors are left unchanged.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_BACKSLASH_RUN_RE = re.compile(r"\\+")


class AmbiguousEscapeDriftError(ValueError):
    """The replacement's intentional backslashes cannot be distinguished from drift."""


def _halve_backslash_runs(s: str) -> str:
    """Replace every maximal run of backslashes with half its length."""
    return _BACKSLASH_RUN_RE.sub(lambda m: "\\" * (len(m.group(0)) // 2), s)


def _all_backslash_runs_even(s: str) -> bool:
    return all(len(run) % 2 == 0 for run in _BACKSLASH_RUN_RE.findall(s))


def maybe_correct_backslash_doubling(old_string: str, new_string: str, content: str) -> Tuple[str, str]:
    """Halve a verified doubled anchor only when the replacement has no backslashes.

    Every old_string run must be even and its halved form must occur in the file.
    Replacement backslashes are ambiguous regardless of run length, so raise
    ``AmbiguousEscapeDriftError`` rather than let fuzzy matching apply them.
    """
    if "\\" not in old_string or old_string in content:
        return old_string, new_string
    if not _all_backslash_runs_even(old_string):
        return old_string, new_string

    halved_old = _halve_backslash_runs(old_string)
    if halved_old not in content:
        return old_string, new_string

    if "\\" in new_string:
        raise AmbiguousEscapeDriftError(
            "Escape-drift detected: halving old_string's backslashes matches the file, "
            "but correcting new_string could remove intentional backslashes. "
            "Re-read the file and resend old_string/new_string with their intended escaping."
        )
    return halved_old, new_string


_QUOTE_WITH_BACKSLASHES_RE = {q: re.compile(r"(\\*)" + q) for q in ("'", '"')}


def _strip_spurious_quote_escapes(s: str) -> str:
    """Un-escape a spurious backslash before a quote character: ``\\'`` -> ``'``, ``\\"`` -> ``"``."""
    return s.replace("\\'", "'").replace('\\"', '"')


def _quote_escapes_all_spurious(*strings: str) -> bool:
    """True when every escaped quote in ``strings`` is attributable to the wire artifact.

    The artifact escapes quotes mechanically, so a quote kind it touched is escaped at EVERY
    occurrence across the whole call, by exactly one backslash. A backslash the model meant
    (``"say \\"hi\\""`` in new code) went through the same extra escape and therefore arrives
    as two or more backslashes before the quote, never one. So for each kind that appears as
    ``\\q`` anywhere: every ``q`` in every string must be preceded by exactly one backslash.
    A bare ``q`` beside a ``\\q`` (the kind was not drifted, so the ``\\q`` is real) or a longer
    run (an intended escape) cannot be told apart from the artifact, so it refuses.
    """
    for quote, pattern in _QUOTE_WITH_BACKSLASHES_RE.items():
        if not any("\\" + quote in s for s in strings):
            continue
        if any(len(m.group(1)) != 1 for s in strings for m in pattern.finditer(s)):
            return False
    return True


def maybe_correct_quote_escaping(old_string: str, new_string: str, content: str) -> Tuple[str, str]:
    """Return a corrected ``(old_string, new_string)`` if a spurious backslash-before-quote wire
    artifact is detected AND verified against the file's actual content; otherwise unchanged.

    Mirrors ``_detect_escape_drift``'s quote-suspect check: ``\\'`` or ``\\"`` present in BOTH
    old_string and new_string is the signature of an apostrophe/quote that picked up a spurious
    backslash in transit, not a literal escape the model intended. The stripped old_string must
    appear verbatim in content, which verifies old_string (and rules out a file that really
    contains ``\\'``/``\\"``). new_string's added code is not in the file, so it is only
    stripped when ``_quote_escapes_all_spurious`` holds for the whole call; otherwise the call
    is left unchanged for the existing reject-and-ask guard.
    """
    has_quote_suspect = (
        ("\\'" in old_string and "\\'" in new_string)
        or ('\\"' in old_string and '\\"' in new_string)
    )
    if not has_quote_suspect or old_string in content:
        return old_string, new_string
    if not _quote_escapes_all_spurious(old_string, new_string):
        return old_string, new_string

    stripped_old = _strip_spurious_quote_escapes(old_string)
    if stripped_old not in content:
        return old_string, new_string

    return stripped_old, _strip_spurious_quote_escapes(new_string)


def maybe_correct_escape_drift(old_string: str, new_string: str, content: str) -> Tuple[str, str, Optional[str]]:
    """Apply whichever escape-drift auto-correction (if any) is verified against the file's
    actual content and report which one fired.

    Tries quote escaping, then backslash correction of the anchor only. Raises
    ``AmbiguousEscapeDriftError`` for a verified doubled anchor whose replacement
    contains backslashes; the caller must return an error without attempting a match.

    Returns ``(corrected_old, corrected_new, note)``: ``note`` is a human-readable description
    of the correction that was applied, or ``None`` if neither fired. The caller attaches
    ``note`` to the successful result (e.g. ``PatchResult.note``) so a corrected call stays
    distinguishable from one whose arguments never needed correction.
    """
    corrected_old, corrected_new = maybe_correct_quote_escaping(old_string, new_string, content)
    if corrected_old != old_string:
        return corrected_old, corrected_new, (
            "escape-drift auto-corrected: stripped a spurious backslash before a quote/apostrophe "
            "in old_string/new_string (a JSON-double-escaping wire artifact) before matching."
        )

    corrected_old, corrected_new = maybe_correct_backslash_doubling(old_string, new_string, content)
    if corrected_old != old_string:
        return corrected_old, corrected_new, (
            "escape-drift auto-corrected: halved doubled backslash runs in old_string "
            "before matching; new_string has no backslashes and was left unchanged."
        )

    return old_string, new_string, None
