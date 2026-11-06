"""Small independent V4A targeting oracle and hand-computed self-tests.

Does not import the production parser. ``evaluate`` returns WRITE or REFUSE
under docs/v4a-targeting-contract.md rules 1-6, or raises ``UnsupportedInput``
when a hint's effect depends on an unmodeled window. That exception is not a
contract REFUSE. Fuzz must catch it and exclude the input.

``LIMITATIONS`` is the exact list. Draft commits that modeled boundaries stay
in history; this module does not.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pytest

_TIER_EXACT = 1
_TIER_NORMALIZED = 2
_TIER_RAW = 3
_TOKEN_TAIL = re.compile(r"[0-9A-Za-z_]")
_BEGIN = re.compile(r"^\*\*\*\s*Begin\s+Patch\s*$")
_END = re.compile(r"^\*\*\*\s*End\s+Patch\s*$")
_UPDATE = re.compile(r"^\*\*\*\s*Update\s+File:\s*(.+)$")
_OTHER = re.compile(r"^\*\*\*\s*(?:Add|Delete|Move)\s+File:")
_HINT = re.compile(r"^@@\s*(.*?)\s*@@\s*$")

LIMITATIONS: tuple[str, ...] = (
    "Hint windows are not modeled. A search hint is a resolved WRITE only when "
    "every raw occurrence lies inside exactly one strict-tier span. It is a "
    "resolved REFUSE when those occurrences lie inside more than one "
    "strict-tier span (ambiguous-hint), when the hint text is absent "
    "(hint-miss), or when the hint lies only inside a lower-tier candidate "
    "while exactly one strict-tier site exists (no fallback; ambiguous-hint). "
    "Every other search hint raises UnsupportedInput('unresolved-hint'): a "
    "neighboring line, the same line outside the matched span, a repeated "
    "hint that is not contained in the strict-tier spans, and any "
    "distance-dependent proximity. That exception is not a contract REFUSE "
    "and must be excluded from fuzz. The production 500-before/2000-after "
    "window is not read and is not an oracle outcome.",
    "Resolved inputs return only WRITE or REFUSE. There is no UNSPECIFIED "
    "result and no boundary-variant list. UnsupportedInput is not an outcome.",
    "Addition-only placement is the established insertion rule, not a search "
    "window: no hint appends at EOF; a hint that occurs once inserts after "
    "the end of that line (a file or hint line with no trailing newline gets "
    "one separator); zero occurrences are hint-miss; more than one are "
    "ambiguous-hint. That repeated-hint refusal is a resolved contract "
    "REFUSE, not an excluded input. Occurrence is a raw substring, so head "
    "inside forehead counts.",
    "There is no fuzzy-similarity tier. A normalized fragment such as "
    "value = 10 for search value = 1 is not a tier-2 or tier-3 site.",
    "Normalization is NFKC, then every Unicode whitespace run (tab, NBSP "
    "U+00A0, U+3000, and the rest of str.split) collapses to one space. No "
    "other equivalence is used. A hint is a raw substring, not a normalized "
    "string. Bytes outside the replaced span are kept, including the file "
    "newline.",
    "Each file is CRLF or LF. A mix of CRLF and bare LF is refused as "
    "mixed-newlines and not converted. Bare CR is not a newline. Replacements "
    "use that file's newline.",
    "Diagnostics are reason codes only. They do not include source or hint text.",
    "Refusal is selection atomicity: any hunk or file refusal returns zero "
    "writes and does not mutate the input mapping. The oracle performs no "
    "I/O and does not claim rollback after an independent write failure.",
    "A tier-3 span is consumed and replaced as a whole, so inserted bytes "
    "inside that span cannot be edited later. Whole-line tiers keep "
    "context-line provenance; written context may anchor a later hunk only "
    "as read-only search text.",
    "Only Update File hunks are planned. Add, Delete, and Move refuse as "
    "unsupported-op. A backslash no-newline marker is skipped, not modeled. "
    "A missing prefix is treated as a context line of the whole line.",
    "Cursor position and hunk order never choose among several eligible edit "
    "sites. A context-only hunk whose strict-tier span is unique is an anchor. "
    "It narrows later hunks of that file to sites of the strictest eligible "
    "tier that start at or after the anchor end, or whose span overlaps the "
    "anchor. Exactly one remaining site is selected. Zero remaining sites, "
    "while a strict-tier site exists elsewhere, refuse as ambiguous-hint with "
    "no lower-tier fallback. Many remaining sites refuse as ambiguous. A "
    "non-unique context-only match does not bind and does not cursor-pick an "
    "occurrence; it clears any previous anchor. The latest unique anchor "
    "replaces the previous one. A hint on a context-only hunk that is not "
    "inside a matched span raises UnsupportedInput('unresolved-hint'). It is "
    "not a contract REFUSE.",
    "A hunk whose search is absent, and whose removed lines were never "
    "consumed by an earlier hunk of this patch, is a skipped no-op when its "
    "replacement block already exists as consecutive whole lines. A substring "
    "or token overlap is not presence. There is no minimum replacement "
    "length. The skip emits no site, does not block later hunks, and does not "
    "weaken atomicity: any later refusal still returns zero writes. Rule 4 "
    "still refuses when the original was consumed. A remaining eligible "
    "source site is edited even if the replacement text also exists.",
    "The consumed set is original byte indexes of removed content (end "
    "greater than start). Separator bytes dropped only because they sat "
    "inside a spliced window are not added to that set; they are absent from "
    "the planned text. An empty-only deletion, whose content range is empty "
    "and which has no context line, is refused as no-site rather than "
    "claimed. An EOF insertion reports origin_start equal to the original "
    "length and origin_line None.",
    "A pure deletion of one or more non-empty whole lines removes that "
    "content and the separator that followed the window, so no blank line "
    "remains. The site names the removed content, not that separator. A "
    "tier-3 substring deletion replaces only the matched span and does not "
    "consume the following separator.",
)


def normalize(value: str) -> str:
    """Canonical tier-2 form. Fragment refusal uses this same function."""
    return " ".join(unicodedata.normalize("NFKC", value).split())


def normalized_fragment(search: str, candidate: str) -> bool:
    """True when ``candidate`` only matches ``search`` as a longer token fragment.

    ``value = 10`` is a fragment of ``value = 1``. Equal lines are not.
    ``call(value = 1)`` is not: neither normalized form prefixes the other.
    A trailing-space partial such as ``prefix `` is not a fragment.
    """
    left = normalize(search)
    right = normalize(candidate)
    if not left or not right or left == right:
        return False
    if right.startswith(left):
        return bool(_TOKEN_TAIL.match(right[len(left) :]))
    if left.startswith(right):
        return bool(_TOKEN_TAIL.match(left[len(right) :]))
    return False


class Outcome:
    """Decision outcome. String values so reports stay stable."""

    WRITE = "WRITE"
    REFUSE = "REFUSE"


class UnsupportedInput(Exception):
    """A hint whose effect depends on an unmodeled window.

    Not a contract refusal. Fuzz must catch this and exclude the input.
    ``reason`` is a code; it never includes source or hint text.
    """

    def __init__(self, reason: str = "unresolved-hint") -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class Site:
    """A chosen edit or insertion.

    Offsets ``start``/``end`` are into the content at selection time.
    ``origin_start``/``origin_end`` are original-file byte provenance, end
    exclusive. ``origin_line`` is the 1-based original line of ``origin_start``.
    ``line`` is the 1-based line at selection time. ``tier`` is 1 exact,
    2 normalized, 3 raw substring, 0 insertion.
    """

    path: str
    start: int
    end: int
    origin_start: int | None
    origin_end: int | None
    tier: int
    line: int
    origin_line: int | None


@dataclass(frozen=True)
class Decision:
    """Plan for the whole patch. ``writes`` is empty unless every hunk can apply."""

    outcome: str
    site: Site | None
    sites: tuple[Site, ...]
    writes: tuple[tuple[str, str], ...]
    reason: str


@dataclass(frozen=True)
class _HunkLine:
    prefix: str
    content: str


@dataclass(frozen=True)
class _Hunk:
    context_hint: str | None
    lines: tuple[_HunkLine, ...]


@dataclass(frozen=True)
class _Update:
    path: str
    hunks: tuple[_Hunk, ...]


@dataclass(frozen=True)
class _Hit:
    start: int
    end: int
    tier: int
    line_start: int
    line_end: int
    raw: bool


@dataclass
class _Doc:
    text: str
    origin: list[int | None]
    newline: str
    origin_lines: list[int]


def _newline_style(text: str) -> str | None:
    """Per-file CRLF or LF. Mixed endings are unresolved, so the caller refuses."""
    has_crlf = "\r\n" in text
    has_bare_lf = "\n" in text.replace("\r\n", "")
    if has_crlf and has_bare_lf:
        return None
    return "\r\n" if has_crlf else "\n"


def _byte_lines(text: str, newline: str) -> list[int]:
    out: list[int] = []
    line = 1
    index = 0
    while index < len(text):
        if text.startswith(newline, index):
            out.extend([line] * len(newline))
            index += len(newline)
            line += 1
        else:
            out.append(line)
            index += 1
    return out


def _split_lines(text: str, newline: str) -> list[tuple[int, int]]:
    """Content spans, excluding separators. A trailing separator adds an empty line."""
    if text == "":
        return [(0, 0)]
    spans: list[tuple[int, int]] = []
    index = 0
    while True:
        found = text.find(newline, index)
        if found < 0:
            spans.append((index, len(text)))
            break
        spans.append((index, found))
        index = found + len(newline)
        if index == len(text):
            spans.append((index, index))
            break
    return spans


def _line_number(text: str, newline: str, offset: int) -> int:
    spans = _split_lines(text, newline)
    if offset >= len(text):
        return len(spans)
    for index, (start, _end) in enumerate(spans):
        next_start = spans[index + 1][0] if index + 1 < len(spans) else len(text) + 1
        if start <= offset < next_start:
            return index + 1
    return len(spans)


def _origin_line(doc: _Doc, origin_start: int | None) -> int | None:
    if origin_start is None or not 0 <= origin_start < len(doc.origin_lines):
        return None
    return doc.origin_lines[origin_start]


def _occurrences(text: str, hint: str) -> list[tuple[int, int]]:
    if not hint:
        return []
    found: list[tuple[int, int]] = []
    start = 0
    while True:
        at = text.find(hint, start)
        if at < 0:
            return found
        found.append((at, at + len(hint)))
        start = at + 1


def _parse(patch: str) -> tuple[tuple[_Update, ...] | None, str]:
    lines = [line.removesuffix("\r") for line in patch.split("\n")]
    begin = next((i for i, line in enumerate(lines) if _BEGIN.match(line)), -1)
    end = next((i for i, line in enumerate(lines) if _END.match(line)), -1)
    if begin < 0 or end <= begin:
        return None, "parse-error"
    updates: list[_Update] = []
    current: list[_Hunk] | None = None
    path = ""
    hunk: list[_HunkLine] | None = None
    hint: str | None = None

    def flush_hunk() -> None:
        nonlocal hunk, hint
        if current is not None and hunk:
            current.append(_Hunk(hint, tuple(hunk)))
        hunk = None
        hint = None

    def flush_op() -> str | None:
        nonlocal current, path
        flush_hunk()
        if current is None:
            return None
        if not path or not current:
            return "parse-error"
        updates.append(_Update(path, tuple(current)))
        current = None
        path = ""
        return None

    for line in lines[begin + 1 : end]:
        if _OTHER.match(line):
            return None, "unsupported-op"
        update = _UPDATE.match(line)
        if update:
            error = flush_op()
            if error:
                return None, error
            path = update.group(1).strip()
            current = []
            continue
        if line.startswith("@@"):
            if current is None:
                return None, "parse-error"
            flush_hunk()
            if line.strip() == "@@":
                hint = None
            else:
                matched = _HINT.match(line)
                if not matched or not matched.group(1).strip():
                    return None, "parse-error"
                hint = matched.group(1).strip()
            hunk = []
            continue
        if not line or line.startswith("\\"):
            continue
        if current is None:
            return None, "parse-error"
        if hunk is None:
            hunk = []
        marked = line[0] in "+- "
        hunk.append(_HunkLine(line[0] if marked else " ", line[1:] if marked else line))
    error = flush_op()
    if error:
        return None, error
    if not updates:
        return None, "empty-patch"
    return tuple(updates), ""


def _split(
    hunk: _Hunk,
) -> tuple[tuple[tuple[str, bool], ...], tuple[tuple[str, bool], ...]]:
    search = tuple(
        (line.content, line.prefix == "-") for line in hunk.lines if line.prefix != "+"
    )
    replace = tuple(
        (line.content, line.prefix == "+") for line in hunk.lines if line.prefix != "-"
    )
    return search, replace


def _logical(text: str, newline: str) -> tuple[str, list[int]]:
    if newline == "\n":
        return text, list(range(len(text) + 1))
    chars: list[str] = []
    raw_of: list[int] = []
    index = 0
    while index < len(text):
        if text.startswith(newline, index):
            chars.append("\n")
            raw_of.append(index)
            index += len(newline)
        else:
            chars.append(text[index])
            raw_of.append(index)
            index += 1
    raw_of.append(len(text))
    return "".join(chars), raw_of


def _fragment_span(
    doc: _Doc, spans: list[tuple[int, int]], start: int, end: int, pattern: str
) -> bool:
    if "\n" in pattern or start >= end:
        return False
    line = 0
    for index, (content_start, content_end) in enumerate(spans):
        next_start = (
            spans[index + 1][0] if index + 1 < len(spans) else len(doc.text) + 1
        )
        if content_start <= start < next_start:
            line = index
            break
    content_start, content_end = spans[line]
    if not (content_start <= start and end <= content_end):
        return False
    if normalized_fragment(pattern, doc.text[content_start:content_end]):
        return True
    raw_suffix = doc.text[end:content_end]
    suffix = normalize(raw_suffix)
    return bool(
        pattern
        and not pattern[-1].isspace()
        and raw_suffix
        and not raw_suffix[0].isspace()
        and suffix
        and _TOKEN_TAIL.match(suffix)
    )


def _overlaps(left: int, right: int, other_left: int, other_right: int) -> bool:
    return left < other_right and other_left < right


def _enumerate(
    doc: _Doc,
    spans: list[tuple[int, int]],
    search: tuple[tuple[str, bool], ...],
    mutant: str | None,
) -> list[_Hit]:
    wanted = [text for text, _removed in search]
    count = len(wanted)
    hits: list[_Hit] = []
    for index in range(len(spans) - count + 1):
        window = spans[index : index + count]
        texts = [doc.text[start:end] for start, end in window]
        start, end = window[0][0], window[-1][1]
        line_end = index + count
        if texts == wanted:
            hits.append(_Hit(start, end, _TIER_EXACT, index, line_end, False))
            continue
        same = all(
            normalize(got) == normalize(need) for got, need in zip(texts, wanted)
        )
        fragment = any(
            normalized_fragment(need, got) for got, need in zip(texts, wanted)
        )
        broken = (
            mutant == "fragment-wins"
            and fragment
            and all(
                normalize(got) == normalize(need) or normalized_fragment(need, got)
                for got, need in zip(texts, wanted)
            )
        )
        if (same and not fragment) or broken:
            hits.append(_Hit(start, end, _TIER_NORMALIZED, index, line_end, False))
    pattern = "\n".join(wanted)
    if not pattern:
        return hits
    logical, raw_of = _logical(doc.text, doc.newline)
    cursor = 0
    while True:
        at = logical.find(pattern, cursor)
        if at < 0:
            break
        raw_start = raw_of[at]
        raw_end = raw_of[at + len(pattern)]
        cursor = at + 1
        if _fragment_span(doc, spans, raw_start, raw_end, pattern):
            continue
        if any(
            _overlaps(raw_start, raw_end, hit.start, hit.end) and hit.tier < _TIER_RAW
            for hit in hits
        ):
            continue
        hits.append(_Hit(raw_start, raw_end, _TIER_RAW, 0, 0, True))
    return hits


def _deleted_ranges(
    doc: _Doc,
    spans: list[tuple[int, int]],
    hit: _Hit,
    search: tuple[tuple[str, bool], ...],
) -> list[tuple[int, int]]:
    if hit.raw:
        if any(removed for _text, removed in search) and hit.end > hit.start:
            return [(hit.start, hit.end)]
        return []
    ranges = []
    window = spans[hit.line_start : hit.line_end]
    for (_text, removed), (start, end) in zip(search, window):
        if removed and end > start:
            ranges.append((start, end))
    return ranges


def _editable(doc: _Doc, ranges: list[tuple[int, int]], consumed: set[int]) -> bool:
    for start, end in ranges:
        for index in range(start, end):
            origin = doc.origin[index] if index < len(doc.origin) else None
            if origin is None or origin in consumed:
                return False
    return True


def _origin_bounds(
    doc: _Doc, ranges: list[tuple[int, int]]
) -> tuple[int | None, int | None]:
    origins = [
        value
        for start, end in ranges
        for index in range(start, min(end, len(doc.origin)))
        if isinstance((value := doc.origin[index]), int)
    ]
    if not origins:
        return None, None
    return min(origins), max(origins) + 1


def _point_origin(doc: _Doc, at: int) -> int:
    if 0 <= at < len(doc.origin):
        found = doc.origin[at]
        if isinstance(found, int):
            return found
    index = min(at, len(doc.origin)) - 1
    while index >= 0:
        found = doc.origin[index]
        if isinstance(found, int):
            return found + 1
        index -= 1
    return 0


def _splice(
    doc: _Doc, start: int, end: int, text: str, origins: list[int | None]
) -> None:
    if len(origins) != len(text):
        raise RuntimeError("origin length drifted")
    doc.text = doc.text[:start] + text + doc.text[end:]
    doc.origin = doc.origin[:start] + origins + doc.origin[end:]
    if len(doc.origin) != len(doc.text):
        raise RuntimeError("document provenance drifted")


def _apply_aligned(
    doc: _Doc,
    spans: list[tuple[int, int]],
    hit: _Hit,
    search: tuple[tuple[str, bool], ...],
    replace: tuple[tuple[str, bool], ...],
) -> list[tuple[int, int]]:
    window = spans[hit.line_start : hit.line_end]
    start, end = window[0][0], window[-1][1]
    ranges = _deleted_ranges(doc, spans, hit, search)
    pieces: list[str] = []
    origins: list[int | None] = []
    search_index = 0
    for text, added in replace:
        if pieces:
            pieces.append(doc.newline)
            origins.extend([None] * len(doc.newline))
        if added:
            pieces.append(text)
            origins.extend([None] * len(text))
            continue
        while search_index < len(search) and search[search_index][1]:
            search_index += 1
        src_start, src_end = window[search_index]
        pieces.append(doc.text[src_start:src_end])
        origins.extend(doc.origin[src_start:src_end])
        search_index += 1
    text = "".join(pieces)
    splice_end = end
    # A pure deletion's match ends at the content. The separator after that
    # window is the deleted line's terminator; leaving it would invent a blank.
    if not replace and doc.text.startswith(doc.newline, end):
        splice_end = end + len(doc.newline)
    _splice(doc, start, splice_end, text, origins)
    return ranges


def _apply_raw(
    doc: _Doc, hit: _Hit, replace: tuple[tuple[str, bool], ...]
) -> list[tuple[int, int]]:
    text = doc.newline.join(part for part, _added in replace)
    ranges = [(hit.start, hit.end)] if hit.end > hit.start else []
    _splice(doc, hit.start, hit.end, text, [None] * len(text))
    return ranges


def _remember(
    consumed: set[int], prior: list[int | None], ranges: list[tuple[int, int]]
) -> None:
    for start, end in ranges:
        for index in range(start, end):
            if index < len(prior):
                found = prior[index]
                if isinstance(found, int):
                    consumed.add(found)


def _matches_consumed(
    search: tuple[tuple[str, bool], ...], consumed_text: list[str]
) -> bool:
    removed = [text for text, gone in search if gone and text]
    if not removed or not consumed_text:
        return False
    exact = set(consumed_text)
    folded = {normalize(item) for item in consumed_text}
    return any(line in exact or normalize(line) in folded for line in removed)


def _contains_any(hit: _Hit, found: list[tuple[int, int]]) -> bool:
    return any(hit.start <= start and end <= hit.end for start, end in found)


def _contains_all(hit: _Hit, found: list[tuple[int, int]]) -> bool:
    return all(hit.start <= start and end <= hit.end for start, end in found)


def _select(
    hits: list[_Hit], hint: str | None, doc: _Doc, mutant: str | None
) -> _Hit | str:
    if mutant == "fragment-wins":
        return min(hits, key=lambda hit: (hit.start, hit.tier))
    best = min(hit.tier for hit in hits)
    pool = [hit for hit in hits if hit.tier == best]
    seen: list[_Hit] = []
    for hit in pool:
        if not any((hit.start, hit.end) == (old.start, old.end) for old in seen):
            seen.append(hit)
    if not hint:
        return seen[0] if len(seen) == 1 else "ambiguous"
    found = _occurrences(doc.text, hint)
    if not found:
        return "hint-miss"
    inside = [hit for hit in seen if _contains_any(hit, found)]
    if len(inside) == 1 and _contains_all(inside[0], found):
        return inside[0]
    if len(inside) > 1:
        return "ambiguous-hint"
    # One strict span holds some, but not every, occurrence: a window around
    # the outside occurrence could select a different site.
    if len(inside) == 1:
        raise UnsupportedInput()
    lower = [hit for hit in hits if hit.tier > best]
    if len(seen) == 1 and any(_contains_any(hit, found) for hit in lower):
        return "ambiguous-hint"
    raise UnsupportedInput()


def _site_of(
    path: str,
    doc: _Doc,
    hit: _Hit,
    ranges: list[tuple[int, int]],
    newline: str,
) -> Site:
    origin_start, origin_end = _origin_bounds(doc, ranges)
    if origin_start is None:
        point = _point_origin(doc, hit.start)
        origin_start, origin_end = point, point
    return Site(
        path,
        hit.start,
        hit.end,
        origin_start,
        origin_end,
        hit.tier,
        _line_number(doc.text, newline, hit.start),
        _origin_line(doc, origin_start),
    )


def _insert(
    doc: _Doc, hunk: _Hunk, replace: tuple[tuple[str, bool], ...], path: str
) -> Site | str:
    body = doc.newline.join(part for part, _added in replace)
    if hunk.context_hint:
        places = _occurrences(doc.text, hunk.context_hint)
        if not places:
            return "hint-miss"
        if len(places) != 1:
            return "ambiguous-hint"
        line_end = doc.text.find(doc.newline, places[0][0])
        if line_end < 0:
            at = len(doc.text)
            separator = (
                "" if not doc.text or doc.text.endswith(doc.newline) else doc.newline
            )
            piece = separator + body + doc.newline
        else:
            at = line_end + len(doc.newline)
            piece = body + doc.newline
    else:
        at = len(doc.text)
        separator = (
            "" if not doc.text or doc.text.endswith(doc.newline) else doc.newline
        )
        piece = separator + body + doc.newline
    origin = _point_origin(doc, min(at, len(doc.origin)))
    site = Site(
        path,
        at,
        at,
        origin,
        origin,
        0,
        _line_number(doc.text, doc.newline, min(at, len(doc.text))),
        _origin_line(doc, origin),
    )
    _splice(doc, at, at, piece, [None] * len(piece))
    return site


@dataclass
class _Scope:
    """Latest unique context-only anchor in the planned file, or none."""

    text: str | None = None
    start: int | None = None
    end: int | None = None


def _strict_hits(hits: list[_Hit]) -> list[_Hit]:
    if not hits:
        return []
    best = min(hit.tier for hit in hits)
    seen: list[_Hit] = []
    for hit in hits:
        if hit.tier != best:
            continue
        if any((hit.start, hit.end) == (old.start, old.end) for old in seen):
            continue
        seen.append(hit)
    return seen


def _set_scope(scope: _Scope, doc: _Doc, hit: _Hit) -> None:
    scope.start = hit.start
    scope.end = hit.end
    scope.text = doc.text[hit.start : hit.end]


def _clear_scope(scope: _Scope) -> None:
    scope.text = None
    scope.start = None
    scope.end = None


def _in_anchor(hit: _Hit, span: tuple[int, int]) -> bool:
    """True when ``hit`` is after the anchor end or overlaps the anchor span."""
    start, end = span
    return hit.start >= end or (hit.start < end and hit.end > start)


def _anchor_span(doc: _Doc, scope: _Scope) -> tuple[int, int] | str | None:
    if scope.text is None:
        return None
    start, end = scope.start, scope.end
    if (
        start is not None
        and end is not None
        and 0 <= start <= end <= len(doc.text)
        and doc.text[start:end] == scope.text
    ):
        return start, end
    places = _occurrences(doc.text, scope.text)
    if len(places) == 1:
        scope.start, scope.end = places[0]
        return places[0]
    return "ambiguous"


def _narrow(
    hits: list[_Hit], span: tuple[int, int] | None, hint: str | None, doc: _Doc
) -> list[_Hit] | str:
    if span is None:
        return hits
    narrowed = [hit for hit in _strict_hits(hits) if _in_anchor(hit, span)]
    if not narrowed:
        return "ambiguous-hint"
    chosen = narrowed[0]
    if len(narrowed) == 1 and hint and hint not in doc.text[chosen.start : chosen.end]:
        return "hint-miss" if hint not in doc.text else "ambiguous-hint"
    if len(narrowed) > 1 and not hint:
        return "ambiguous"
    return narrowed


def _bind_anchor(
    doc: _Doc, hits: list[_Hit], hint: str | None, scope: _Scope
) -> str | None:
    """Bind a unique context-only span. Many matches clear; they never cursor-pick."""
    if not hits:
        return "no-site"
    if hint:
        chosen = _select(hits, hint, doc, None)
        if isinstance(chosen, str):
            return chosen
        _set_scope(scope, doc, chosen)
        return None
    strict = _strict_hits(hits)
    if len(strict) == 1:
        _set_scope(scope, doc, strict[0])
        return None
    _clear_scope(scope)
    return None


def _choose(
    hits: list[_Hit],
    hint: str | None,
    doc: _Doc,
    mutant: str | None,
    scope: _Scope,
) -> _Hit | str:
    if mutant == "fragment-wins":
        return _select(hits, hint, doc, mutant)
    located = _anchor_span(doc, scope)
    if isinstance(located, str):
        return located
    narrowed = _narrow(hits, located, hint, doc)
    if isinstance(narrowed, str):
        return narrowed
    return _select(narrowed, hint, doc, mutant)


def _replacement_block_present(
    doc: _Doc, replace: tuple[tuple[str, bool], ...]
) -> bool:
    """True when added lines already exist as consecutive whole lines, not a fragment."""
    lines = [text for text, _added in replace]
    if not lines or not any(added for _text, added in replace):
        return False
    texts = [doc.text[start:end] for start, end in _split_lines(doc.text, doc.newline)]
    width = len(lines)
    if width > len(texts):
        return False
    return any(
        texts[index : index + width] == lines for index in range(len(texts) - width + 1)
    )


def _absent(
    doc: _Doc,
    search: tuple[tuple[str, bool], ...],
    replace: tuple[tuple[str, bool], ...],
    blocked: bool,
    consumed_text: list[str],
    mutant: str | None,
) -> str | None:
    if blocked or _matches_consumed(search, consumed_text):
        return "consumed"
    if mutant != "fragment-wins" and _replacement_block_present(doc, replace):
        return None
    return "no-site"


def _commit(
    doc: _Doc,
    spans: list[tuple[int, int]],
    hit: _Hit,
    search: tuple[tuple[str, bool], ...],
    replace: tuple[tuple[str, bool], ...],
    consumed: set[int],
    consumed_text: list[str],
) -> None:
    ranges = _deleted_ranges(doc, spans, hit, search)
    prior = list(doc.origin)
    consumed_text.extend(doc.text[start:end] for start, end in ranges if end > start)
    edited = (
        _apply_raw(doc, hit, replace)
        if hit.raw
        else _apply_aligned(doc, spans, hit, search, replace)
    )
    _remember(consumed, prior, edited)


def _plan_hunk(
    doc: _Doc,
    hunk: _Hunk,
    mutant: str | None,
    consumed: set[int],
    consumed_text: list[str],
    path: str,
    scope: _Scope,
) -> Site | str | None:
    search, replace = _split(hunk)
    if not search:
        return _insert(doc, hunk, replace, path) if replace else "no-site"
    spans = _split_lines(doc.text, doc.newline)
    hits = _enumerate(doc, spans, search, mutant)
    editing = any(removed for _text, removed in search) or any(
        added for _text, added in replace
    )
    if not editing:
        return _bind_anchor(doc, hits, hunk.context_hint, scope)
    eligible = []
    blocked = False
    removing = any(removed for _text, removed in search)
    anchored = any(not removed for _text, removed in search)
    for hit in hits:
        ranges = _deleted_ranges(doc, spans, hit, search)
        # An empty content range is not an edit site unless context anchors it.
        if removing and not ranges and not anchored:
            continue
        if ranges and not _editable(doc, ranges, consumed):
            blocked = True
            continue
        eligible.append(hit)
    if not eligible:
        return _absent(doc, search, replace, blocked, consumed_text, mutant)
    chosen = _choose(eligible, hunk.context_hint, doc, mutant, scope)
    if isinstance(chosen, str):
        return chosen
    ranges = _deleted_ranges(doc, spans, chosen, search)
    site = _site_of(path, doc, chosen, ranges, doc.newline)
    _commit(doc, spans, chosen, search, replace, consumed, consumed_text)
    return site


def _plan_file(
    path: str, text: str, hunks: tuple[_Hunk, ...], mutant: str | None
) -> Decision:
    newline = _newline_style(text)
    if newline is None:
        return Decision(Outcome.REFUSE, None, (), (), "mixed-newlines")
    doc = _Doc(text, list(range(len(text))), newline, _byte_lines(text, newline))
    consumed: set[int] = set()
    consumed_text: list[str] = []
    sites: list[Site] = []
    scope = _Scope()
    for hunk in hunks:
        result = _plan_hunk(doc, hunk, mutant, consumed, consumed_text, path, scope)
        if isinstance(result, str):
            return Decision(Outcome.REFUSE, None, (), (), result)
        if isinstance(result, Site):
            sites.append(result)
    changed = ((path, doc.text),) if doc.text != text else ()
    site = sites[0] if sites else None
    return Decision(Outcome.WRITE, site, tuple(sites), changed, "ok")


def _group(ops: tuple[_Update, ...]) -> dict[str, tuple[_Hunk, ...]]:
    grouped: dict[str, list[_Hunk]] = {}
    for op in ops:
        grouped.setdefault(op.path, []).extend(op.hunks)
    return {path: tuple(hunks) for path, hunks in grouped.items()}


def evaluate(
    files: dict[str, str], patch: str, *, mutant: str | None = None
) -> Decision:
    """Plan rules 1-6. ``mutant='fragment-wins'`` is a deliberately broken selector.

    The mutant treats a normalized fragment as a tier-2 site and writes the
    earliest eligible site, ignoring uniqueness, hints, context-only anchors,
    and already-applied skips. Hand-computed tests must fail against it.

    Raises ``UnsupportedInput`` when a hint cannot be resolved without a
    window. That is not a ``REFUSE`` decision; callers must exclude the input.
    """
    if mutant not in (None, "fragment-wins"):
        raise ValueError("unknown mutant")
    ops, error = _parse(patch)
    if error or ops is None:
        return Decision(Outcome.REFUSE, None, (), (), error or "parse-error")
    writes: list[tuple[str, str]] = []
    sites: list[Site] = []
    for path, hunks in _group(ops).items():
        if path not in files:
            return Decision(Outcome.REFUSE, None, (), (), "missing-file")
        planned = _plan_file(path, files[path], hunks, mutant)
        if planned.outcome != Outcome.WRITE:
            return Decision(Outcome.REFUSE, None, (), (), planned.reason)
        writes.extend(planned.writes)
        sites.extend(planned.sites)
    site = sites[0] if sites else None
    return Decision(Outcome.WRITE, site, tuple(sites), tuple(writes), "ok")


def _patch(path: str, body: str) -> str:
    return f"*** Begin Patch\n*** Update File: {path}\n{body}\n*** End Patch\n"


def _files_patch(parts: tuple[tuple[str, str], ...]) -> str:
    chunks = ["*** Begin Patch"]
    for path, body in parts:
        chunks.append(f"*** Update File: {path}")
        chunks.append(body)
    chunks.append("*** End Patch")
    return "\n".join(chunks) + "\n"


def _one(source: str, body: str, *, mutant: str | None = None) -> Decision:
    return evaluate({"f.py": source}, _patch("f.py", body), mutant=mutant)


def _excluded(files: dict[str, str], patch: str, *secrets: str) -> UnsupportedInput:
    """Boundary-dependent hint: exception, unchanged inputs, no leaked text."""
    before = dict(files)
    with pytest.raises(UnsupportedInput) as caught:
        evaluate(files, patch)
    assert caught.value.reason == "unresolved-hint"
    assert str(caught.value) == "unresolved-hint"
    for secret in secrets:
        assert secret not in str(caught.value)
        assert secret not in caught.value.reason
    assert files == before
    return caught.value


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("value\t=\t1", "value = 1"),
        ("value\u00a0=\u00a01", "value = 1"),
        ("value\u3000=\u30001", "value = 1"),
        ("\u212b\t  x", "\u00c5 x"),
    ],
)
def test_normalize(source: str, expected: str) -> None:
    assert normalize(source) == expected


def test_normalized_fragment_refuses_value_10_not_a_partial() -> None:
    assert normalized_fragment("value = 1", "value = 10")
    assert normalized_fragment("value = 1", "value\u00a0=\u00a010")
    assert normalized_fragment("value = 1", "value\u3000=\u300010")
    assert not normalized_fragment("value = 1", "value = 1")
    assert not normalized_fragment("value = 1", "value\t=\t1")
    assert not normalized_fragment("value = 1", "call(value = 1)")
    assert not normalized_fragment("prefix ", "prefix SECRET")


def test_rule1_inserted_bytes_are_not_reedited() -> None:
    decision = _one("a = 1\n", "@@\n-a = 1\n+a = 1\n+b = 2\n@@\n-b = 2\n+b = 9")
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.sites == ()
    assert decision.reason == "consumed"
    assert "b = 2" not in decision.reason


def test_rule1_written_context_in_the_span_may_anchor() -> None:
    # Hint "b = 2" is inside the second hunk's matched span, not a window.
    # Original c = 3 is bytes [6, 11) on original line 2. The match window
    # starts on the written context line, which is current line 2.
    source = "a = 1\nc = 3\n"
    body = "@@\n-a = 1\n+a = 1\n+b = 2\n@@ b = 2 @@\n b = 2\n-c = 3\n+c = 4"
    decision = _one(source, body)
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "a = 1\nb = 2\nc = 4\n"),)
    assert decision.sites[0].origin_start == 0
    assert decision.sites[0].origin_end == 5
    assert decision.sites[0].origin_line == 1
    assert decision.sites[0].tier == _TIER_EXACT
    assert decision.sites[1].origin_start == 6
    assert decision.sites[1].origin_end == 11
    assert decision.sites[1].origin_line == 2
    assert decision.sites[1].line == 2
    assert decision.sites[1].tier == _TIER_EXACT


def test_consumed_provenance_refuses_reedit_of_inserted_span() -> None:
    decision = _one(
        "call(value = 1)\n", "@@\n-value = 1\n+CHANGED\n@@\n-call(CHANGED)\n+NO"
    )
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.reason == "consumed"


def test_rule2_value_1_beats_value_10_and_keeps_surrounding_bytes() -> None:
    # value = 10\\n is 11 bytes; the exact line is [11, 20) on line 2.
    source = "value = 10\nvalue = 1\n"
    files = {"f.py": source}
    before = dict(files)
    decision = evaluate(files, _patch("f.py", "@@\n-value = 1\n+CHANGED"))
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "value = 10\nCHANGED\n"),)
    assert decision.site is not None
    assert decision.site.start == 11
    assert decision.site.end == 20
    assert decision.site.origin_start == 11
    assert decision.site.origin_end == 20
    assert decision.site.origin_line == 2
    assert decision.site.tier == _TIER_EXACT
    assert decision.site.line == 2
    assert files == before


@pytest.mark.parametrize(
    ("line", "tier"),
    [
        ("value\t=\t1", _TIER_NORMALIZED),
        ("value  =  1", _TIER_NORMALIZED),
        ("value\u00a0=\u00a01", _TIER_NORMALIZED),
        ("value\u3000=\u30001", _TIER_NORMALIZED),
    ],
)
def test_rule2_canonical_normalize_is_tier2(line: str, tier: int) -> None:
    source = f"value = 10\n{line}\nfooter = old\n"
    decision = _one(source, "@@\n-value = 1\n+CHANGED")
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "value = 10\nCHANGED\nfooter = old\n"),)
    assert decision.site is not None
    assert decision.site.tier == tier
    assert decision.site.origin_start == len("value = 10\n")
    assert decision.site.origin_line == 2
    assert source[decision.site.origin_start : decision.site.origin_end] == line


def test_rule2_angstrom_uses_nfkc_normalize() -> None:
    source = "\u212b = 1\n"
    decision = _one(source, "@@\n-\u00c5 = 1\n+CHANGED")
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "CHANGED\n"),)
    assert decision.site is not None
    assert decision.site.tier == _TIER_NORMALIZED
    assert decision.site.origin_start == 0
    assert decision.site.origin_end == 5
    assert decision.site.origin_line == 1


@pytest.mark.parametrize(
    "line", ["value = 10", "value\u00a0=\u00a010", "value\u3000=\u300010"]
)
def test_rule2_fragment_alone_is_refused(line: str) -> None:
    decision = _one(f"{line}\n", "@@\n-value = 1\n+CHANGED")
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.reason == "no-site"


def test_fragment_inside_call_is_not_an_edit_site() -> None:
    decision = _one("call(value = 10)\n", "@@\n-value = 1\n+changed")
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.reason == "no-site"


def test_hint_inside_lower_tier_cannot_bypass_exact_site() -> None:
    # Exact U+00C5 line is [0, 5). U+212B normalizes to it and is the only
    # place the hint occurs. That lower-tier site must not be written.
    source = "\u00c5 = 1\n\u212b = 1\n"
    assert source[6:].startswith("\u212b")
    decision = _one(source, "@@ \u212b @@\n-\u00c5 = 1\n+CHANGED")
    assert decision.outcome == Outcome.REFUSE
    assert decision.reason == "ambiguous-hint"
    assert decision.writes == ()
    assert decision.sites == ()
    assert "\u212b" not in decision.reason


def test_hint_on_a_non_site_lookalike_is_excluded() -> None:
    # The comment line is not a normalized site. A window might still see it.
    source = "value = 1\nvalue\t=\t1 # marker\n"
    assert source[10:].startswith("value\t=\t1")
    _excluded(
        {"f.py": source},
        _patch("f.py", "@@ marker @@\n-value = 1\n+CHANGED"),
        "marker",
        "value = 1",
    )


def test_rule3_identical_sites_without_hint_refuse() -> None:
    source = "value = 1\nvalue = 1\nfooter = old\n"
    body = "@@\n-value = 1\n+CHANGED\n@@\n-footer = old\n+footer = new"
    decision = _one(source, body)
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.reason == "ambiguous"
    assert source[0:9] == "value = 1"
    assert source[10:19] == "value = 1"


def test_content_hint_selects_the_only_span_that_contains_it() -> None:
    # keep SECRET is [0, 11). The adjacent copy does not contain the hint.
    source = "keep SECRET\nkeep OTHER\n"
    decision = _one(source, "@@ SECRET @@\n-keep SECRET\n+CHANGED")
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "CHANGED\nkeep OTHER\n"),)
    assert decision.site is not None
    assert decision.site.start == 0
    assert decision.site.end == 11
    assert decision.site.origin_start == 0
    assert decision.site.origin_end == 11
    assert decision.site.origin_line == 1
    assert decision.site.tier == _TIER_EXACT


def test_content_hint_in_both_copies_is_ambiguous_hint() -> None:
    source = "keep SECRET\nkeep SECRET\n"
    assert source[12:23] == "keep SECRET"
    decision = _one(source, "@@ SECRET @@\n-keep SECRET\n+CHANGED")
    assert decision.outcome == Outcome.REFUSE
    assert decision.reason == "ambiguous-hint"
    assert decision.writes == ()
    assert "SECRET" not in decision.reason


def test_absent_hint_is_hint_miss_not_a_fallback() -> None:
    decision = _one("value = 1\n", "@@ missing @@\n-value = 1\n+CHANGED")
    assert decision.outcome == Outcome.REFUSE
    assert decision.reason == "hint-miss"
    assert decision.writes == ()
    assert "missing" not in decision.reason
    assert "value = 1" not in decision.reason


def test_rule4_replacement_already_present_is_not_success() -> None:
    source = 'keep = HOSTISH_SECRET\nhost = "a"\nhost = "b"\n'
    body = '@@\n-host = "a"\n+host = "c"\n@@\n-host = "a"\n+keep = HOSTISH_SECRET'
    decision = _one(source, body)
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.reason == "consumed"
    assert "HOSTISH_SECRET" not in decision.reason


def test_normalized_consumed_source_is_not_a_different_success() -> None:
    source = 'keep = HOSTISH_SECRET\nhost  = "a"\nhost = "b"\n'
    body = '@@\n-host = "a"\n+host = "c"\n@@\n-host = "a"\n+keep = HOSTISH_SECRET'
    decision = _one(source, body)
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.reason == "consumed"
    assert "HOSTISH_SECRET" not in decision.reason


def test_multiline_consumed_source_is_not_hidden_by_replacement() -> None:
    source = "foo = 1\nbar = 2\n"
    body = "@@\n-foo = 1\n+keep = HOSTISH_SECRET\n@@\n-foo = 1\n-bar = 2\n+keep = HOSTISH_SECRET"
    decision = _one(source, body)
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.reason == "consumed"
    assert "HOSTISH_SECRET" not in decision.reason


def test_no_similarity_tier_does_not_edit_a_different_block() -> None:
    # The first hunk's context makes foo = 1 unique, so that line is consumed.
    # foo = 1 / bar = 9 is not a stand-in for foo = 1 / bar = 2.
    source = "keep = HOSTISH_SECRET\nfoo = 1\nbar = 2\nfoo = 1\nbar = 9\n"
    body = (
        "@@\n keep = HOSTISH_SECRET\n-foo = 1\n+keep = HOSTISH_SECRET\n"
        "@@\n-foo = 1\n-bar = 2\n+keep = HOSTISH_SECRET"
    )
    decision = _one(source, body)
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.reason == "consumed"
    assert "bar = 9" not in decision.reason
    assert "HOSTISH_SECRET" not in decision.reason


def test_rule4_remaining_whole_line_copy_is_selected() -> None:
    # Hint "one" is inside the first search span. The second copy is [17, 27).
    source = 'host = "a" # one\nhost = "a"\n'
    body = '@@ one @@\n-host = "a" # one\n+host = "c"\n@@\n-host = "a"\n+host = "d"'
    decision = _one(source, body)
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", 'host = "c"\nhost = "d"\n'),)
    assert decision.sites[1].origin_start == 17
    assert decision.sites[1].origin_end == 27
    assert decision.sites[1].origin_line == 2
    assert decision.sites[1].tier == _TIER_EXACT


def test_rule5_cross_file_refusal_writes_nothing() -> None:
    files = {"a.py": "old\n", "b.py": "other\n"}
    patch = _files_patch(
        (
            ("a.py", "@@\n-old\n+new"),
            ("b.py", "@@\n-absent\n+replacement"),
        )
    )
    before = dict(files)
    decision = evaluate(files, patch)
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.sites == ()
    assert decision.reason == "no-site"
    assert files == before


def test_unresolved_hint_on_later_file_is_not_a_contract_refuse() -> None:
    # The marker is the line before the only site. A window may write it.
    # That is not a contract REFUSE, and the earlier file must not be written.
    files = {"a.py": "old\n", "b.py": "# marker\nold\n"}
    patch = _files_patch(
        (
            ("a.py", "@@\n-old\n+new"),
            ("b.py", "@@ marker @@\n-old\n+new"),
        )
    )
    _excluded(files, patch, "marker", "old")


def test_resolved_hint_collision_on_later_file_still_refuses() -> None:
    # SECRET is inside both strict-tier spans. No window can make that one site.
    files = {"a.py": "old\n", "b.py": "keep SECRET\nkeep SECRET\n"}
    patch = _files_patch(
        (
            ("a.py", "@@\n-old\n+new"),
            ("b.py", "@@ SECRET @@\n-keep SECRET\n+CHANGED"),
        )
    )
    before = dict(files)
    decision = evaluate(files, patch)
    assert decision.outcome == Outcome.REFUSE
    assert decision.reason == "ambiguous-hint"
    assert decision.writes == ()
    assert files == before
    assert "SECRET" not in decision.reason


def test_rule6_unique_partial_preserves_surrounding_bytes() -> None:
    # call( is 5 bytes; value = 1 occupies [5, 14) on line 1.
    source = "call(value = 1)\n"
    decision = _one(source, "@@\n-value = 1\n+CHANGED")
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "call(CHANGED)\n"),)
    assert decision.site is not None
    assert decision.site.start == 5
    assert decision.site.end == 14
    assert decision.site.origin_start == 5
    assert decision.site.origin_line == 1
    assert decision.site.tier == _TIER_RAW


def test_rule6_addition_only_eof_insertion() -> None:
    source = "first\n"
    decision = _one(source, "@@\n+last")
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "first\nlast\n"),)
    assert decision.site is not None
    assert decision.site.start == 6
    assert decision.site.end == 6
    assert decision.site.origin_start == 6
    assert decision.site.origin_line is None
    assert decision.site.tier == 0


def test_rule6_hinted_insertion_is_established_not_a_window() -> None:
    decision = _one("head\nfoot\n", "@@ head @@\n+mid")
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "head\nmid\nfoot\n"),)
    assert decision.site is not None
    assert decision.site.start == 5
    assert decision.site.origin_start == 5
    assert decision.site.tier == 0


def test_hinted_insertion_without_trailing_newline_adds_a_separator() -> None:
    decision = _one("head", "@@ head @@\n+mid")
    assert decision.writes == (("f.py", "head\nmid\n"),)
    assert decision.site is not None
    assert decision.site.origin_start == 4


def test_repeated_insertion_hint_is_a_resolved_refuse() -> None:
    # Established insertion rule, not a search window: more than one occurrence
    # refuses. Fuzz must not treat this as an excluded unresolved input.
    decision = _one("head\nhead\n", "@@ head @@\n+mid")
    assert decision.outcome == Outcome.REFUSE
    assert decision.reason == "ambiguous-hint"
    assert decision.writes == ()


def test_missing_insertion_hint_is_hint_miss() -> None:
    decision = _one("head\n", "@@ missing @@\n+mid")
    assert decision.outcome == Outcome.REFUSE
    assert decision.reason == "hint-miss"
    assert decision.writes == ()
    assert "missing" not in decision.reason


def test_rule6_empty_anchor_preserves_later_edit() -> None:
    # b is original bytes [3, 4) on line 3. The blank anchor does not move it.
    source = "a\n\nb\n"
    decision = _one(source, "@@\n a\n@@\n \n@@\n-b\n+B")
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "a\n\nB\n"),)
    assert decision.site is not None
    assert decision.site.origin_start == 3
    assert decision.site.origin_end == 4
    assert decision.site.origin_line == 3


def test_rule6_distinct_ordered_hunks() -> None:
    # a = old is [0, 7). b = old starts at 8 and ends at 15.
    source = "a = old\nb = old\n"
    decision = _one(source, "@@\n-a = old\n+a = new\n@@\n-b = old\n+b = new")
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "a = new\nb = new\n"),)
    assert decision.sites[0].origin_start == 0
    assert decision.sites[0].origin_end == 7
    assert decision.sites[0].origin_line == 1
    assert decision.sites[1].origin_start == 8
    assert decision.sites[1].origin_end == 15
    assert decision.sites[1].origin_line == 2


def test_crlf_and_lf_are_per_file() -> None:
    files = {"win.py": "value = 1\r\n", "unix.py": "value = 1\n"}
    patch = _files_patch(
        (
            ("win.py", "@@\n-value = 1\n+CHANGED"),
            ("unix.py", "@@\n-value = 1\n+CHANGED"),
        )
    )
    before = dict(files)
    decision = evaluate(files, patch)
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("win.py", "CHANGED\r\n"), ("unix.py", "CHANGED\n"))
    assert decision.sites[0].origin_end == 9
    assert decision.sites[0].origin_line == 1
    assert decision.sites[1].origin_end == 9
    assert files == before


def test_crlf_value_1_beats_value_10() -> None:
    # CRLF: value = 10\\r\\n is 12 bytes, so the exact line is [12, 21).
    source = "value = 10\r\nvalue = 1\r\n"
    decision = _one(source, "@@\n-value = 1\n+CHANGED")
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "value = 10\r\nCHANGED\r\n"),)
    assert decision.site is not None
    assert decision.site.start == 12
    assert decision.site.end == 21
    assert decision.site.origin_line == 2


def test_crlf_insertion_and_mixed_newline_refusal() -> None:
    inserted = _one("first\r\n", "@@\n+last")
    assert inserted.outcome == Outcome.WRITE
    assert inserted.writes == (("f.py", "first\r\nlast\r\n"),)
    assert inserted.site is not None
    assert inserted.site.origin_start == 7
    assert inserted.site.origin_line is None
    mixed = _one("value = 1\r\nvalue = 1\n", "@@\n-value = 1\n+CHANGED")
    assert mixed.outcome == Outcome.REFUSE
    assert mixed.writes == ()
    assert mixed.reason == "mixed-newlines"


def test_crlf_file_is_not_written_when_another_file_refuses() -> None:
    files = {"win.py": "value = 1\r\n", "unix.py": "other\n"}
    patch = _files_patch(
        (
            ("win.py", "@@\n-value = 1\n+CHANGED"),
            ("unix.py", "@@\n-absent\n+replacement"),
        )
    )
    decision = evaluate(files, patch)
    assert decision.outcome == Outcome.REFUSE
    assert decision.reason == "no-site"
    assert decision.writes == ()
    assert files["win.py"] == "value = 1\r\n"


def test_overlapping_distinct_whole_line_candidates_are_ambiguous() -> None:
    decision = _one("a\na\na\n", "@@\n-a\n-a\n+b")
    assert decision.outcome == Outcome.REFUSE
    assert decision.reason == "ambiguous"
    assert decision.writes == ()


def test_eof_insertion_preserves_existing_blank_lines() -> None:
    decision = _one("first\n\n", "@@\n+last")
    assert decision.writes == (("f.py", "first\n\nlast\n"),)


def test_repeated_update_sections_share_planned_content() -> None:
    patch = _files_patch((("f.py", "@@\n-old\n+new"), ("f.py", "@@\n-other\n+changed")))
    decision = evaluate({"f.py": "old\nother\n"}, patch)
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "new\nchanged\n"),)
    assert decision.sites[0].origin_line == 1
    assert decision.sites[1].origin_line == 2


@pytest.mark.parametrize(
    ("source", "body", "expected"),
    [
        ("a\nb\nc\n", "@@\n-b", "a\nc\n"),
        ("a\nb\nc\n", "@@\n-a", "b\nc\n"),
        ("a\nb\nc\n", "@@\n-c", "a\nb\n"),
        ("only\n", "@@\n-only", ""),
        ("only", "@@\n-only", ""),
        ("a\nb", "@@\n-b", "a\n"),
        ("a\nb\nc\nd\n", "@@\n-b\n-c", "a\nd\n"),
        ("a\r\nb\r\nc\r\n", "@@\n-b", "a\r\nc\r\n"),
        ("a\n \nb\n", "@@\n- ", "a\nb\n"),
        ("keep\nvalue\t=\t1\n", "@@\n-value = 1", "keep\n"),
    ],
)
def test_pure_deletion_removes_the_line_and_its_separator(
    source: str, body: str, expected: str
) -> None:
    # A '-' line deletes that line. The separator after the matched content
    # is the line terminator, not a blank line left behind.
    decision = _one(source, body)
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", expected),)
    assert decision.reason == "ok"
    assert decision.site is not None
    # The reported site is the removed content. The consumed separator is not
    # part of that span and is not a leftover blank line in the write.


def test_pure_deletion_then_edit_does_not_keep_the_deleted_separator() -> None:
    decision = _one("a\nb\nc\n", "@@\n-a\n@@\n-b\n+B")
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "B\nc\n"),)


def test_context_deletion_keeps_the_separator_of_a_kept_line() -> None:
    bracketed = _one("a\nb\nc\n", "@@\n a\n-b\n c")
    assert bracketed.writes == (("f.py", "a\nc\n"),)
    prefix = _one("a\nb\nc\n", "@@\n a\n-b")
    assert prefix.writes == (("f.py", "a\nc\n"),)


def test_empty_added_line_is_not_a_pure_deletion() -> None:
    decision = _one("a\nb\nc\n", "@@\n-b\n+")
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "a\n\nc\n"),)


def test_partial_deletion_does_not_consume_the_following_separator() -> None:
    decision = _one("call(value = 1)\n", "@@\n-value = 1")
    assert decision.writes == (("f.py", "call()\n"),)
    assert decision.site is not None
    assert decision.site.tier == _TIER_RAW


def test_empty_only_deletion_is_no_site_not_consumed() -> None:
    decision = _one("a\n\nb\n", "@@\n-")
    assert decision.outcome == Outcome.REFUSE
    assert decision.reason == "no-site"
    assert decision.writes == ()
    assert decision.sites == ()


def test_context_brackets_an_empty_line_deletion() -> None:
    decision = _one("a\n\nb\n", "@@\n a\n-\n b")
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "a\nb\n"),)


def test_spec_open_adjacent_hint_is_excluded() -> None:
    # A proximity window might write this. The hint is not inside the span.
    source = "# marker\nvalue = old\n"
    assert source[9:20] == "value = old"
    plain = _one(source, "@@\n-value = old\n+value = new")
    assert plain.writes == (("f.py", "# marker\nvalue = new\n"),)
    _excluded(
        {"f.py": source},
        _patch("f.py", "@@ marker @@\n-value = old\n+value = new"),
        "marker",
        "value = old",
    )


def test_spec_open_hint_two_lines_away_is_excluded() -> None:
    # marker is not inside value = 1. value = 10 is a fragment, not a site.
    source = "value = 1\nxxxx\n# marker\nvalue = 10\n"
    plain = _one(source, "@@\n-value = 1\n+CHANGED")
    assert plain.writes == (("f.py", "CHANGED\nxxxx\n# marker\nvalue = 10\n"),)
    _excluded(
        {"f.py": source},
        _patch("f.py", "@@ marker @@\n-value = 1\n+CHANGED"),
        "marker",
        "value = 10",
    )


def test_spec_open_repeated_adjacent_hints_are_excluded() -> None:
    source = "# target\nvalue = old\n# target\nvalue = old\n"
    assert source[9:20] == "value = old"
    assert source[30:41] == "value = old"
    _excluded(
        {"f.py": source},
        _patch("f.py", "@@ target @@\n-value = old\n+value = new"),
        "target",
        "value = old",
    )


def test_spec_open_secret_hint_does_not_leak() -> None:
    source = "only = 1\n\n# ZZSECRETZZ\n"
    _excluded(
        {"f.py": source},
        _patch("f.py", "@@ ZZSECRETZZ @@\n-only = 1\n+CHANGED"),
        "ZZSECRETZZ",
        "only = 1",
    )


def test_context_only_hint_outside_span_is_excluded() -> None:
    source = "# marker\nold\n"
    _excluded(
        {"f.py": source},
        _patch("f.py", "@@ marker @@\n old"),
        "marker",
        "old",
    )


@pytest.mark.parametrize("gap", [100, 3000])
def test_spec_open_same_line_hint_outside_span_is_excluded(gap: int) -> None:
    # SECRET_A_EDIT is on the same line as one prefix span and not inside it.
    # Distance changes a 500/2000 window; it must not change this oracle into
    # a WRITE or a contract REFUSE.
    source = "prefix AAA_UNRELATED\n" + ("x" * gap) + "\nprefix SECRET_A_EDIT\n"
    near = len("prefix AAA_UNRELATED\n") + gap + 1
    assert source[0:7] == "prefix "
    assert source[near : near + 7] == "prefix "
    _excluded(
        {"f.py": source},
        _patch("f.py", "@@ SECRET_A_EDIT @@\n-prefix \n+prefixZ"),
        "SECRET_A_EDIT",
    )


def test_spec_open_equidistant_hint_is_excluded() -> None:
    source = "target\n# hint\ntarget\n"
    assert source[0:6] == "target"
    assert source[14:20] == "target"
    _excluded(
        {"f.py": source},
        _patch("f.py", "@@ hint @@\n-target\n+changed"),
        "target",
    )


def test_spec_open_sites_beyond_fixed_extent_are_excluded() -> None:
    lines = ["# marker", *["xxxx"] * 4, "value = 1", *["xxxx"] * 6, "value = 1"]
    source = "\n".join(lines) + "\n"
    assert source[29:38] == "value = 1"
    assert source[69:78] == "value = 1"
    _excluded(
        {"f.py": source},
        _patch("f.py", "@@ marker @@\n-value = 1\n+CHANGED"),
        "marker",
        "value = 1",
    )


def test_spec_open_long_file_proximity_is_excluded() -> None:
    source = (
        "# marker\n"
        + ("xxxx\n" * 99)
        + "value = 1\n"
        + ("xxxx\n" * 399)
        + "value = 1\n"
    )
    assert len(source) == 2519
    assert source[504:513] == "value = 1"
    assert source[2509:2518] == "value = 1"
    _excluded(
        {"f.py": source},
        _patch("f.py", "@@ marker @@\n-value = 1\n+CHANGED"),
        "marker",
        "value = 1",
    )


def test_mutation_broken_fragment_winner_fails_hand_computed_rule2() -> None:
    source = "value = 10\nvalue = 1\n"
    body = "@@\n-value = 1\n+CHANGED"
    good = _one(source, body)
    bad = _one(source, body, mutant="fragment-wins")
    assert good.outcome == Outcome.WRITE
    assert good.writes == (("f.py", "value = 10\nCHANGED\n"),)
    assert good.site is not None and good.site.origin_start == 11
    assert bad.writes == (("f.py", "CHANGED\nvalue = 1\n"),)
    assert bad.site is not None and bad.site.origin_start == 0
    assert bad.writes != good.writes


def test_mutation_earliest_identical_site_is_not_success() -> None:
    source = "value = 1\nvalue = 1\n"
    body = "@@\n-value = 1\n+CHANGED"
    good = _one(source, body)
    bad = _one(source, body, mutant="fragment-wins")
    assert good.outcome == Outcome.REFUSE
    assert good.writes == ()
    assert good.reason == "ambiguous"
    assert bad.writes == (("f.py", "CHANGED\nvalue = 1\n"),)
    assert bad.site is not None and bad.site.origin_start == 0


def test_absent_original_with_replacement_present_skips_and_keeps_later_hunk() -> None:
    # RETRY_LIMIT = 30\\n is 17 bytes. keep = 1 is original [17, 25) on line 2.
    # The first search was never in the file; its replacement line is already there.
    source = "RETRY_LIMIT = 30\nkeep = 1\n"
    body = "@@\n-TIMEOUT_WINDOW = 9000\n+RETRY_LIMIT = 30\n@@\n-keep = 1\n+keep = 2"
    good = _one(source, body)
    bad = _one(source, body, mutant="fragment-wins")
    assert good.outcome == Outcome.WRITE
    assert good.writes == (("f.py", "RETRY_LIMIT = 30\nkeep = 2\n"),)
    assert len(good.sites) == 1
    assert good.sites[0].origin_start == 17
    assert good.sites[0].origin_end == 25
    assert good.sites[0].origin_line == 2
    assert good.sites[0].tier == _TIER_EXACT
    assert bad.outcome == Outcome.REFUSE
    assert bad.reason == "no-site"
    assert bad.writes == ()


def test_short_replacement_already_present_is_still_a_skip() -> None:
    # xy\\n is 3 bytes. No 8-character floor: the replacement line is present.
    source = "xy\nkeep = 1\n"
    body = "@@\n-zz\n+xy\n@@\n-keep = 1\n+keep = 2"
    decision = _one(source, body)
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "xy\nkeep = 2\n"),)
    assert decision.sites[0].origin_start == 3
    assert decision.sites[0].origin_end == 11
    assert decision.sites[0].origin_line == 2


def test_already_applied_middle_hunk_preserves_both_neighbors() -> None:
    # left = 1 is [0, 8). right = 1 is [26, 35). The middle search is absent.
    source = "left = 1\nRETRY_LIMIT = 30\nright = 1\n"
    body = (
        "@@\n-left = 1\n+left = 2\n"
        "@@\n-TIMEOUT_WINDOW = 9000\n+RETRY_LIMIT = 30\n"
        "@@\n-right = 1\n+right = 2"
    )
    decision = _one(source, body)
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "left = 2\nRETRY_LIMIT = 30\nright = 2\n"),)
    assert len(decision.sites) == 2
    assert decision.sites[0].origin_start == 0
    assert decision.sites[0].origin_end == 8
    assert decision.sites[1].origin_start == 26
    assert decision.sites[1].origin_end == 35
    assert decision.sites[1].origin_line == 3


def test_already_applied_skip_does_not_weaken_atomicity() -> None:
    source = "RETRY_LIMIT = 30\nvalue = 1\nvalue = 1\n"
    body = "@@\n-TIMEOUT_WINDOW = 9000\n+RETRY_LIMIT = 30\n@@\n-value = 1\n+value = 9"
    files = {"f.py": source}
    decision = evaluate(files, _patch("f.py", body))
    assert decision.outcome == Outcome.REFUSE
    assert decision.reason == "ambiguous"
    assert decision.writes == ()
    assert decision.sites == ()
    assert files == {"f.py": source}
    assert "value = 1" not in decision.reason


def test_replacement_overlap_is_not_already_present() -> None:
    # value = new overlaps call(value = new) and is not a whole line. Do not skip.
    source = "call(value = new)\nkeep = 1\n"
    body = "@@\n-value = old\n+value = new\n@@\n-keep = 1\n+keep = 2"
    files = {"f.py": source}
    decision = evaluate(files, _patch("f.py", body))
    assert decision.outcome == Outcome.REFUSE
    assert decision.reason == "no-site"
    assert decision.writes == ()
    assert files == {"f.py": source}
    assert "value = new" not in decision.reason


def test_remaining_source_is_edited_even_if_replacement_exists() -> None:
    # value = old is [0, 11). The replacement line on line 2 must not hide it.
    source = "value = old\nvalue = new\n"
    decision = _one(source, "@@\n-value = old\n+value = new")
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "value = new\nvalue = new\n"),)
    assert decision.site is not None
    assert decision.site.origin_start == 0
    assert decision.site.origin_end == 11
    assert decision.site.origin_line == 1


def test_context_anchor_selects_the_strict_site_after_it() -> None:
    # value = 1\\n is 10 bytes; class Marker:\\n is 14. The later copy is [24, 33).
    source = "value = 1\nclass Marker:\nvalue = 1\n"
    body = "@@\n class Marker:\n@@\n-value = 1\n+value = 9"
    good = _one(source, body)
    bad = _one(source, body, mutant="fragment-wins")
    assert good.outcome == Outcome.WRITE
    assert good.writes == (("f.py", "value = 1\nclass Marker:\nvalue = 9\n"),)
    assert good.site is not None
    assert good.site.origin_start == 24
    assert good.site.origin_end == 33
    assert good.site.origin_line == 3
    assert good.site.tier == _TIER_EXACT
    assert bad.writes == (("f.py", "value = 9\nclass Marker:\nvalue = 1\n"),)
    assert bad.site is not None and bad.site.origin_start == 0


def test_context_anchor_refuses_when_strict_site_is_only_before_it() -> None:
    # Exact value = 1 is before the anchor. The tab line is only tier 2.
    source = "value = 1\nclass Marker:\nvalue\t=\t1\n"
    body = "@@\n class Marker:\n@@\n-value = 1\n+value = 9"
    good = _one(source, body)
    bad = _one(source, body, mutant="fragment-wins")
    assert good.outcome == Outcome.REFUSE
    assert good.reason == "ambiguous-hint"
    assert good.writes == ()
    assert good.sites == ()
    assert "value = 1" not in good.reason
    assert bad.writes == (("f.py", "value = 9\nclass Marker:\nvalue\t=\t1\n"),)
    assert bad.site is not None and bad.site.origin_start == 0


def test_context_anchor_does_not_fall_back_to_a_raw_site_after_it() -> None:
    # call(value = 1) starts at 24. It is tier 3 and must not win over the exact line.
    source = "value = 1\nclass Marker:\ncall(value = 1)\n"
    assert source[24:39] == "call(value = 1)"
    body = "@@\n class Marker:\n@@\n-value = 1\n+CHANGED"
    decision = _one(source, body)
    assert decision.outcome == Outcome.REFUSE
    assert decision.reason == "ambiguous-hint"
    assert decision.writes == ()
    assert "CHANGED" not in decision.reason


def test_context_anchor_many_after_does_not_use_cursor_order() -> None:
    # Two identical sites after the anchor. A second hunk must not walk the cursor.
    source = "class Marker:\nvalue = 1\nvalue = 1\n"
    body = "@@\n class Marker:\n@@\n-value = 1\n+one\n@@\n-value = 1\n+two"
    good = _one(source, body)
    bad = _one(source, body, mutant="fragment-wins")
    assert good.outcome == Outcome.REFUSE
    assert good.reason == "ambiguous"
    assert good.writes == ()
    assert bad.writes == (("f.py", "class Marker:\none\ntwo\n"),)


def test_latest_context_anchor_replaces_the_previous_one() -> None:
    # alpha = 1\\n is 10; class A:\\n is 9, so the second alpha is [19, 28).
    # beta = 1\\n is 9 and class B:\\n is 9, so the last beta is [47, 55).
    source = "alpha = 1\nclass A:\nalpha = 1\nbeta = 1\nclass B:\nbeta = 1\n"
    body = "@@\n class A:\n@@\n-alpha = 1\n+alpha = 9\n@@\n class B:\n@@\n-beta = 1\n+beta = 9"
    decision = _one(source, body)
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (
        ("f.py", "alpha = 1\nclass A:\nalpha = 9\nbeta = 1\nclass B:\nbeta = 9\n"),
    )
    assert decision.sites[0].origin_start == 19
    assert decision.sites[0].origin_end == 28
    assert decision.sites[0].origin_line == 3
    assert decision.sites[1].origin_start == 47
    assert decision.sites[1].origin_end == 55
    assert decision.sites[1].origin_line == 6
    # class B is before class A in the file. A cursor stuck after class A would
    # see one beta and write; the latest anchor still sees both betas.
    ahead = "class B:\nbeta = 1\nalpha = 1\nclass A:\nbeta = 1\nalpha = 1\n"
    narrowed = _one(ahead, "@@\n class A:\n@@\n-alpha = 1\n+alpha = 9")
    assert narrowed.writes == (
        ("f.py", "class B:\nbeta = 1\nalpha = 1\nclass A:\nbeta = 1\nalpha = 9\n"),
    )
    # 9+9+10+9+9 = 46: class B, beta, alpha, class A, beta; the last alpha starts there.
    assert narrowed.site is not None and narrowed.site.origin_start == 46
    both = _one(
        ahead,
        "@@\n class A:\n@@\n-alpha = 1\n+alpha = 9\n@@\n class B:\n@@\n-beta = 1\n+beta = 9",
    )
    assert both.outcome == Outcome.REFUSE
    assert both.reason == "ambiguous"
    assert both.writes == ()


def test_context_anchor_within_overlap_still_selects_that_site() -> None:
    # HEAD is [10, 14), not after its own end. Overlap must keep the site.
    source = "other = 1\nHEAD\nother = 1\n"
    decision = _one(source, "@@\n HEAD\n@@\n-HEAD\n+TAIL")
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "other = 1\nTAIL\nother = 1\n"),)
    assert decision.site is not None
    assert decision.site.origin_start == 10
    assert decision.site.origin_end == 14
    assert decision.site.origin_line == 2
    assert decision.site.tier == _TIER_EXACT


def test_cross_file_already_applied_skip_preserves_the_other_file() -> None:
    files = {"a.py": "xy\n", "b.py": "keep = 1\n"}
    patch = _files_patch(
        (
            ("a.py", "@@\n-zz\n+xy"),
            ("b.py", "@@\n-keep = 1\n+keep = 2"),
        )
    )
    before = dict(files)
    decision = evaluate(files, patch)
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("b.py", "keep = 2\n"),)
    assert decision.sites[0].path == "b.py"
    assert decision.sites[0].origin_start == 0
    assert decision.sites[0].origin_end == 8
    assert files == before


def test_cross_file_already_applied_skip_stays_atomic() -> None:
    files = {"a.py": "xy\n", "b.py": "value = 1\nvalue = 1\n"}
    patch = _files_patch(
        (
            ("a.py", "@@\n-zz\n+xy"),
            ("b.py", "@@\n-value = 1\n+value = 9"),
        )
    )
    before = dict(files)
    decision = evaluate(files, patch)
    assert decision.outcome == Outcome.REFUSE
    assert decision.reason == "ambiguous"
    assert decision.writes == ()
    assert files == before
