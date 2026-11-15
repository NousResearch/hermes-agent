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
from typing import Optional

_BACKSLASH_RUN_RE = re.compile(r"\\+")


class AmbiguousEscapeDriftError(ValueError):
    """The replacement's intentional backslashes cannot be distinguished from drift."""


def _halve_backslash_runs(s: str) -> str:
    """Replace every maximal run of backslashes with half its length."""
    return _BACKSLASH_RUN_RE.sub(lambda m: "\\" * (len(m.group(0)) // 2), s)


def _all_backslash_runs_even(s: str) -> bool:
    return all(len(run) % 2 == 0 for run in _BACKSLASH_RUN_RE.findall(s))


def maybe_correct_backslash_doubling(old_string: str, new_string: str, content: str) -> tuple[str, str]:
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


def maybe_correct_quote_escaping(old_string: str, new_string: str, content: str) -> tuple[str, str]:
    """Return a corrected ``(old_string, new_string)`` if a spurious backslash-before-quote wire
    artifact is detected AND verified against the file's actual content; otherwise unchanged.

    Mirrors ``_detect_escape_drift``'s quote-suspect check: ``\\'`` or ``\\"`` present in BOTH
    old_string and new_string is the signature of an apostrophe/quote that picked up a spurious
    backslash in transit, not a literal escape the model intended. The stripped old_string must
    appear verbatim in content, which verifies old_string (and rules out a file that really
    contains ``\\'``/``\\"``). new_string's added code is not in the file, so it is only
    stripped when ``_quote_escapes_all_spurious`` holds for the whole call; otherwise the call
    is left unchanged for the existing reject-and-ask guard. Raises ``AmbiguousEscapeDriftError``
    when new_string escapes a quote kind that old_string does not.
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
    # Only old_string is checked against the file, so only the quote kinds it escapes are
    # proven drift. A kind escaped only in new_string may be an escape the model meant.
    unproven = [q for q in ("'", '"') if "\\" + q in new_string and "\\" + q not in old_string]
    if unproven:
        shown = " and ".join("\\" + q for q in unproven)
        raise AmbiguousEscapeDriftError(
            "Escape-drift detected: removing the spurious quote escapes in old_string matches "
            f"the file, but new_string also contains {shown}, which old_string gives no evidence "
            "for. Re-read the file and resend old_string/new_string with their intended escaping."
        )

    return stripped_old, _strip_spurious_quote_escapes(new_string)


# A doubled backslash is one token so `\\u4f60` (a literal backslash + "u4f60") never decodes.
_UNICODE_ESCAPE_RE = re.compile(r"\\\\|\\u[0-9a-fA-F]{4}")
_UNICODE_ESCAPE_TOKEN_RE = re.compile(r"\\u[0-9a-fA-F]{4}")
_ODD_BACKSLASH_BEFORE_NON_ASCII_RE = re.compile(r"(?:^|[^\\])(?:\\\\)*\\[^\x00-\x7f]")
_ODD_TRAILING_BACKSLASH_RE = re.compile(r"(?:^|[^\\])(?:\\\\)*\\$")


def _decode_unicode_escapes(text: str) -> Optional[str]:
    """``text`` with its non-ASCII ``\\uXXXX`` escapes decoded (surrogate pairs joined);
    None when that leaves a lone surrogate. ASCII escapes stay spelled out: they are
    not the drift this repairs and ``\\u0022`` may be deliberate."""
    def decode(match: re.Match) -> str:
        token = match.group(0)
        return token if len(token) == 2 or int(token[-4:], 16) < 0x80 else chr(int(token[-4:], 16))
    decoded = _UNICODE_ESCAPE_RE.sub(decode, text)
    try:
        decoded = decoded.encode("utf-16", "surrogatepass").decode("utf-16")
    except UnicodeDecodeError:
        return None
    return decoded


def _splits_escape(content: str, offset: int) -> bool:
    """True when a slice boundary at ``offset`` falls inside an escape token."""
    if offset > 0 and content[offset - 1] == "\\" and content[offset:offset + 1] in set("\\'\"/bfnrtv0u"):
        return True
    return any(content[start] == "\\" and _UNICODE_ESCAPE_TOKEN_RE.fullmatch(content, start, start + 6)
               for start in range(max(0, offset - 5), offset))


def _unicode_recovery_pattern(decoded: str) -> re.Pattern:
    """Each non-ASCII char matches itself or its ``\\uXXXX`` spelling (either hex case)."""
    parts = []
    for char in decoded:
        if ord(char) < 0x80:
            parts.append(re.escape(char))
            continue
        units = char.encode("utf-16-be")
        hexes = [units[i:i + 2].hex() for i in range(0, len(units), 2)]
        escaped = "".join(r"(?<!\\)\\u" + "".join(f"[{d}{d.upper()}]" if d.isalpha() else d for d in h)
                          for h in hexes)
        parts.append(f"(?:{re.escape(char)}|{escaped})")
    return re.compile("".join(parts))


def _canonical_unicode_slice(content: str, decoded: str, new_string: str) -> Optional[str]:
    """The single file spelling ``decoded`` occurs as, or None when that is ambiguous.

    Refuses differently spelled candidates, slices splitting an escape token, a trailing
    replacement backslash that would fuse with a following one, and any literal occurrence
    the escape-aware pattern did not accept (str replacement would edit it too).
    """
    spans = [m.span() for m in _unicode_recovery_pattern(decoded).finditer(content)]
    if not spans:
        return None
    canonical = content[spans[0][0]:spans[0][1]]
    fuses = _ODD_TRAILING_BACKSLASH_RE.search(new_string) is not None
    for start, end in spans:
        if (content[start:end] != canonical or _splits_escape(content, start) or _splits_escape(content, end)
                or (fuses and content[end:end + 1] == "\\")):
            return None
    literal = [m.start() for m in re.finditer(re.escape(canonical), content)]
    return canonical if literal == [start for start, _ in spans] else None


def _escape_like(new_string: str, canonical: str) -> Optional[str]:
    """``new_string`` with non-ASCII chars spelled as ``\\uXXXX``, reusing the file's hex case;
    None for a backslash that would fuse with a generated escape."""
    if _ODD_BACKSLASH_BEFORE_NON_ASCII_RE.search(new_string):
        return None
    digits = "".join(t[2:] for t in _UNICODE_ESCAPE_TOKEN_RE.findall(canonical))
    upper = sum(d.isupper() for d in digits) > sum(d.islower() for d in digits)
    out = []
    for char in new_string:
        if ord(char) < 0x80:
            out.append(char)
            continue
        units = char.encode("utf-16-be")
        for i in range(0, len(units), 2):
            h = units[i:i + 2].hex()
            out.append("\\u" + (h.upper() if upper else h))
    return "".join(out)


def maybe_correct_unicode_escapes(old_string: str, new_string: str, content: str) -> tuple[str, str]:
    """Match an anchor whose only difference from the file is how non-ASCII text is spelled:
    ``\\u4f60\\u597d`` in old_string against ``你好`` on disk, or the reverse.

    The recovered anchor must be one unambiguous file slice; new_string is rewritten into
    the file's spelling (decoded for a plain file, escaped with the file's hex case for an
    escaped one). A slice mixing both spellings, or any unsafe boundary, is left unchanged
    so the normal no-match error fires.
    """
    if old_string in content:
        return old_string, new_string
    decoded = _decode_unicode_escapes(old_string)
    # The cap bounds the per-char alternation regex, run before every fuzzy patch with non-ASCII.
    if decoded is None or len(decoded) > 256 or not any(ord(c) >= 0x80 for c in decoded):
        return old_string, new_string
    if decoded == old_string and "\\u" not in content:
        return old_string, new_string
    canonical = _canonical_unicode_slice(content, decoded, new_string)
    if canonical is None:
        return old_string, new_string
    has_escapes = any(int(t[-4:], 16) >= 0x80 for t in _UNICODE_ESCAPE_TOKEN_RE.findall(canonical))
    if not has_escapes:
        return canonical, _decode_unicode_escapes(new_string) or new_string
    if any(ord(c) >= 0x80 for c in canonical):
        return old_string, new_string
    escaped_new = _escape_like(new_string, canonical)
    return (canonical, escaped_new) if escaped_new is not None else (old_string, new_string)


def maybe_correct_escape_drift(old_string: str, new_string: str, content: str) -> tuple[str, str, Optional[str]]:
    """Apply whichever escape-drift auto-correction (if any) is verified against the file's
    actual content and report which one fired.

    Tries quote escaping, then backslash correction of the anchor only. Raises
    ``AmbiguousEscapeDriftError`` for a verified doubled anchor whose replacement
    contains backslashes, or a verified quote correction whose replacement escapes a
    quote kind the anchor does not; the caller must return an error without matching.

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

    corrected_old, corrected_new = maybe_correct_unicode_escapes(old_string, new_string, content)
    if corrected_old != old_string:
        return corrected_old, corrected_new, (
            "escape-drift auto-corrected: old_string spelled non-ASCII text differently from the "
            "file (\\uXXXX escapes vs literal characters); matched the file's spelling and wrote "
            "new_string in that same spelling."
        )

    return old_string, new_string, None
