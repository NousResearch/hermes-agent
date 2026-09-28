"""Auto-correction for JSON-double-escaping wire artifacts on ``patch`` calls.

``tools/fuzzy_match.py::_detect_escape_drift``/``_detect_backslash_doubling`` already diagnose,
with certainty, two shapes of the same underlying artifact -- a spurious backslash before a
quote/apostrophe (``\\'``/``\\"``) that the file does not have, or every backslash run in
``old_string`` being exactly twice as long as the matched region's -- but only ever reject the
call and ask the model to resend, costing a full round trip.

Applied unconditionally, not gated to any particular model: each correction is safe by
construction rather than by knowing which model is talking. Both only ever fire when the
corrected ``old_string`` is verified to appear verbatim in the file's actual content -- a model
that deliberately sent genuinely-escaped content (a correctly backslash-doubled string in a
regex/shell string, or a source file that legitimately contains the two-character sequence
``\\'``/``\\"``) would need the corrected form to coincidentally already exist at the same spot
in the file for either correction to apply, which the checks below rule out; a wrong guess
simply fails verification and falls through to the existing reject-and-ask guard unchanged.
That verification covers ``old_string`` only: the lines ``new_string`` adds are not in the file,
so each correction also requires ``new_string`` to be uniformly escaped (every backslash run
even; every escaped quote escaped by exactly one backslash) and otherwise makes no correction,
since un-escaping a backslash the model meant in new code would corrupt the file silently. An
earlier version of the backslash-doubling correction gated it to a model-name substring match;
that gate was removed because the substring check turned out to be unreliable in practice (the
resolved "model" value at the check site did not reliably match the serving model's real name),
and because the verification-against-file-content step already provides the safety property the
gate was meant to add. Per-model gating can be reintroduced later if a model is found where a
verified correction itself proves unsafe -- nothing observed so far suggests that.

Called once from ``tools/file_operations.py::patch_replace``, before fuzzy matching runs, so a
verified correction produces a clean exact match instead of ever reaching the guard. The applied
correction (if any) is surfaced back to the caller as a note so it stays observable and
auditable -- a bare `success: true` does not by itself distinguish a call that was silently
auto-corrected from one whose arguments never needed correction at all.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_BACKSLASH_RUN_RE = re.compile(r"\\+")


def _halve_backslash_runs(s: str) -> str:
    """Replace every maximal run of backslashes with half its length."""
    return _BACKSLASH_RUN_RE.sub(lambda m: "\\" * (len(m.group(0)) // 2), s)


def _all_backslash_runs_even(s: str) -> bool:
    return all(len(run) % 2 == 0 for run in _BACKSLASH_RUN_RE.findall(s))


def maybe_correct_backslash_doubling(old_string: str, new_string: str, content: str) -> Tuple[str, str]:
    """Return a corrected ``(old_string, new_string)`` if a doubled-backslash wire artifact is
    detected AND verified against the file's actual content; otherwise return them unchanged.

    Two requirements, one per string. The halved ``old_string`` must appear verbatim in
    ``content`` -- that verifies old_string. Nothing in the file can verify new_string (its
    new lines are, by definition, not in the file yet), so it is checked structurally: an
    extra escape doubles EVERY backslash, so every run in both strings must have even length.
    An odd run (``print("done\\n")`` added alongside a doubled old_string) means that string
    was not double-escaped; halving it would silently delete a real backslash, so the call is
    left unchanged for the existing reject-and-ask guard.
    """
    if "\\" not in old_string or old_string in content:
        return old_string, new_string
    if not (_all_backslash_runs_even(old_string) and _all_backslash_runs_even(new_string)):
        return old_string, new_string

    halved_old = _halve_backslash_runs(old_string)
    if halved_old not in content:
        return old_string, new_string

    return halved_old, _halve_backslash_runs(new_string)


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

    Tries quote-escaping then backslash-doubling; both are independently gated on the corrected
    old_string being confirmed present in content, so at most one meaningfully changes anything
    for a given call -- trying both in sequence just avoids an artificial ordering dependency,
    not a real ambiguity between them.

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
            "escape-drift auto-corrected: halved doubled backslash runs in old_string/new_string "
            "(a JSON-double-escaping wire artifact) before matching."
        )

    return old_string, new_string, None
