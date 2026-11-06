"""Independent brute-force V4A targeting oracle and hand-computed contract tests.

This module does not import the production parser. It enumerates every physical
candidate, then decides WRITE(site), REFUSE, or UNSPECIFIED under rules 1-6 of
docs/v4a-targeting-contract.md.

Hint-window unit and numeric boundaries are not defined by the contract. This
oracle does not guess a width and does not read planner window constants. For
each file it measures the distance from the nearest hint occurrence to every
eligible candidate, in line units and in character units. Each distinct distance,
and that distance ±1, is a class boundary. Selection runs once for the class
below the smallest boundary, once above the largest, and once at each boundary
with both an inclusive and an exclusive endpoint. If every class returns the
same WRITE(site) or REFUSE, that is the decision. Otherwise the result is
UNSPECIFIED and ``variants`` records every class, including any site that class
would write. Pass ``variants`` to force a hypothesis. ``mutant="fixed-width-list"``
replays the rejected fixed ±1 list and must fail the extent counterexample.

Limitations, stated rather than invented:
- Addition-only placement is the established rule (unique hint inserts after that
  line; no hint appends at EOF) and is not varied.
- There is no fuzzy-similarity tier. A normalized fragment is not a site.
- A file that mixes CRLF and bare LF is refused (``mixed-newlines``), not converted.
- Diagnostics are reason codes. They do not include source or hint text.
- Rule 5 is selection atomicity, not rollback after an independent I/O failure.
- A multi-line tier-3 span is treated as entirely edited, so inserted context
  inside that span refuses the site. Whole-line tiers keep context provenance.
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


def normalize(value: str) -> str:
    """Canonical tier-2 form: NFKC, then collapse Unicode whitespace to one space.

    Tier-2 equality and normalized-fragment refusal both call this function.
    No other collapse is authoritative.
    """
    return " ".join(unicodedata.normalize("NFKC", value).split())


def normalized_fragment(search: str, candidate: str) -> bool:
    """True when ``candidate`` only matches ``search`` as a longer token fragment.

    ``value = 10`` is a fragment of ``value = 1``. Equal lines are not fragments.
    ``call(value = 1)`` is not a fragment: neither normalized form prefixes the
    other. A trailing-space partial such as ``prefix `` is not a fragment: the
    remainder starts at a token boundary.
    """
    left = normalize(search)
    right = normalize(candidate)
    if not left or not right or left == right:
        return False
    if right.startswith(left):
        return bool(_TOKEN_TAIL.match(right[len(left):]))
    if left.startswith(right):
        return bool(_TOKEN_TAIL.match(left[len(right):]))
    return False


class Outcome:
    """Decision outcome. String values so reports stay stable."""

    WRITE = "WRITE"
    REFUSE = "REFUSE"
    UNSPECIFIED = "UNSPECIFIED"


@dataclass(frozen=True)
class HintVariant:
    """One hint-window class, or one forced hypothesis.

    ``unit`` is ``line`` or ``char``. ``endpoint`` is ``inclusive`` (distance
    <= boundary), ``exclusive`` or ``below`` (distance < boundary), or ``above``
    (a radius larger than every observed distance). ``line_delta`` and
    ``legacy_inclusive`` are used only by the fixed-width mutant.
    """

    name: str
    unit: str
    boundary: int
    endpoint: str
    line_delta: int = 0
    legacy_inclusive: bool = False
    right_boundary: int | None = None
    right_endpoint: str | None = None


def _boundary_edges(distances: set[int]) -> list[int]:
    return sorted({dist + shift for dist in distances for shift in (-1, 0, 1)})


def hint_boundary_classes(distances: dict[str, set[int]]) -> tuple[HintVariant, ...]:
    """Class algorithm. Keys are ``line`` and ``char``. Not a width list."""
    found: list[HintVariant] = []
    for unit in ("line", "char"):
        edges = _boundary_edges(distances.get(unit, set()))
        if not edges:
            continue
        found.append(HintVariant(f"{unit}:below:{edges[0]}", unit, edges[0], "below"))
        for edge in edges:
            found.append(HintVariant(f"{unit}:in:{edge}", unit, edge, "inclusive"))
            found.append(HintVariant(f"{unit}:ex:{edge}", unit, edge, "exclusive"))
        last = edges[-1]
        found.append(HintVariant(f"{unit}:above:{last}", unit, last, "above"))
        singles = [v for v in found if v.unit == unit]
        for left in singles:
            for right in singles:
                if left.boundary == right.boundary and left.endpoint == right.endpoint:
                    continue
                found.append(HintVariant(
                    f"{left.name}|{right.name}", unit, left.boundary, left.endpoint,
                    right_boundary=right.boundary, right_endpoint=right.endpoint,
                ))
    return tuple(found)


def _fixed_width_variants() -> tuple[HintVariant, ...]:
    """The rejected list: inclusive/exclusive at line deltas -1, 0, and +1."""
    variants = []
    for delta in (-1, 0, 1):
        for inclusive in (True, False):
            kind = "inclusive" if inclusive else "exclusive"
            variants.append(
                HintVariant(f"{kind}{delta:+d}", "line", delta, "legacy", delta, inclusive)
            )
    return tuple(variants)


@dataclass(frozen=True)
class Site:
    """A physical candidate. Offsets are into the content at selection time.

    ``origin_start``/``origin_end`` are provenance into the original file bytes
    (end exclusive). ``line`` is 1-based. ``tier`` is 1 exact, 2 normalized,
    3 raw substring, 0 insertion.
    """

    path: str
    start: int
    end: int
    origin_start: int | None
    origin_end: int | None
    tier: int
    line: int


@dataclass(frozen=True)
class VariantDecision:
    """What one hint-boundary hypothesis would do. ``reason`` is a code."""

    variant: HintVariant
    outcome: str
    site: Site | None
    writes: tuple[tuple[str, str], ...]
    reason: str
    sites: tuple[Site, ...] = ()


@dataclass(frozen=True)
class Decision:
    """Aggregate decision. ``writes`` is empty unless every hypothesis agrees.

    ``site`` is the first edit or insertion when the classes agree on a write;
    ``sites`` is the full sequence. When the outcome is UNSPECIFIED, aggregate
    ``writes`` stay empty and ``variants`` records every class, including the
    sites and writes that class would have produced.
    """

    outcome: str
    site: Site | None
    sites: tuple[Site, ...]
    writes: tuple[tuple[str, str], ...]
    variants: tuple[VariantDecision, ...]
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
    window: int
    raw: bool


@dataclass
class _Doc:
    text: str
    origin: list[int | None]
    newline: str


@dataclass(frozen=True)
class _Plan:
    outcome: str
    site: Site | None
    sites: tuple[Site, ...]
    writes: tuple[tuple[str, str], ...]
    reason: str


def _newline_style(text: str) -> str | None:
    """Per-file CRLF or LF. Mixed endings are unresolved, so the caller refuses."""
    has_crlf = "\r\n" in text
    has_bare_lf = "\n" in text.replace("\r\n", "")
    if has_crlf and has_bare_lf:
        return None
    return "\r\n" if has_crlf else "\n"


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


def _line_at(spans: list[tuple[int, int]], offset: int) -> int:
    for index, (start, end) in enumerate(spans):
        next_start = spans[index + 1][0] if index + 1 < len(spans) else end
        if start <= offset < end or (start == end and offset == start):
            return index
        if end <= offset < next_start:
            return index
    return max(0, len(spans) - 1)


def _span_lines(spans: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    if start == end:
        line = _line_at(spans, start)
        return line, line
    return _line_at(spans, start), _line_at(spans, end - 1) + 1


def _distance(line_start: int, line_end: int, hint_line: int) -> int:
    if line_end == line_start:
        return abs(hint_line - line_start)
    if line_start <= hint_line < line_end:
        return 0
    if hint_line < line_start:
        return line_start - hint_line
    return hint_line - (line_end - 1)


def _char_distance(site_start: int, site_end: int, hint_start: int, hint_end: int) -> int:
    """Gap between two spans. Overlap and touching are both distance 0."""
    if site_start < hint_end and hint_start < site_end:
        return 0
    if site_start >= hint_end:
        return site_start - hint_end
    return hint_start - site_end


def _hint_spans(text: str, hint: str) -> list[tuple[int, int]]:
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


def _nearest(
    hit: _Hit,
    hint: str,
    doc: _Doc,
    spans: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """Minimum line distance and minimum character distance, independently."""
    lines: list[int] = []
    chars: list[int] = []
    for start, end in _hint_spans(doc.text, hint):
        lines.append(_distance(hit.line_start, hit.line_end, _line_at(spans, start)))
        chars.append(_char_distance(hit.start, hit.end, start, end))
    if not lines:
        return None
    return min(lines), min(chars)


def _within(distance: int, variant: HintVariant) -> bool:
    if variant.endpoint == "above":
        return True
    if variant.endpoint == "inclusive":
        return distance <= variant.boundary
    return distance < variant.boundary


def _legacy_covers(
    hit: _Hit,
    hint: str,
    doc: _Doc,
    spans: list[tuple[int, int]],
    variant: HintVariant,
) -> bool:
    """Rejected model: the hint line is always in, plus one optional neighbor."""
    start = 0
    while True:
        at = doc.text.find(hint, start)
        if at < 0:
            return False
        distance = _distance(hit.line_start, hit.line_end, _line_at(spans, at))
        neighbor = distance == 1 and variant.line_delta >= 1 and variant.legacy_inclusive
        if distance == 0 or neighbor:
            return True
        start = at + 1


def _covers(
    hit: _Hit,
    hint: str,
    doc: _Doc,
    spans: list[tuple[int, int]],
    variant: HintVariant,
    mutant: str | None,
) -> bool:
    if mutant == "fixed-width-list":
        return _legacy_covers(hit, hint, doc, spans, variant)
    for start, end in _hint_spans(doc.text, hint):
        if variant.unit == "line":
            distance = _distance(hit.line_start, hit.line_end, _line_at(spans, start))
        else:
            distance = _char_distance(hit.start, hit.end, start, end)
        effective = variant
        if hit.start >= end and variant.right_boundary is not None:
            effective = HintVariant(variant.name, variant.unit, variant.right_boundary, variant.right_endpoint or "exclusive")
        if _within(distance, effective):
            return True
    return False


def _parse(patch: str) -> tuple[tuple[_Update, ...] | None, str]:
    lines = [line.removesuffix("\r") for line in patch.split("\n")]
    begin = next((i for i, line in enumerate(lines) if re.match(r"^\*\*\*\s*Begin\s+Patch\s*$", line)), -1)
    end = next((i for i, line in enumerate(lines) if re.match(r"^\*\*\*\s*End\s+Patch\s*$", line)), len(lines))
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

    for line in lines[begin + 1:end]:
        update = re.match(r"^\*\*\*\s*Update\s+File:\s*(.+)$", line)
        other = re.match(r"^\*\*\*\s*(Add|Delete|Move)\s+File:", line)
        if update or other:
            error = flush_op()
            if error:
                return None, error
            if other or update is None:
                return None, "unsupported-op"
            path = update.group(1).strip()
            current = []
        elif line.startswith("@@"):
            if current is None:
                return None, "parse-error"
            flush_hunk()
            matched = re.match(r"^@@\s*(.+?)\s*@@\s*$", line)
            hint = matched.group(1) if matched else None
            hunk = []
        elif current is not None and line:
            if hunk is None:
                hunk = []
            prefix = line[0] if line[0] in "+- " else " "
            content = line[1:] if line[0] in "+- " else line
            if line[0] != "\\":
                hunk.append(_HunkLine(prefix, content))
    error = flush_op()
    if error:
        return None, error
    if not updates:
        return None, "empty-patch"
    return tuple(updates), ""


def _split(hunk: _Hunk) -> tuple[tuple[tuple[str, bool], ...], tuple[tuple[str, bool], ...]]:
    search = tuple((line.content, line.prefix == "-") for line in hunk.lines if line.prefix != "+")
    replace = tuple((line.content, line.prefix == "+") for line in hunk.lines if line.prefix != "-")
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


def _fragment_span(doc: _Doc, spans: list[tuple[int, int]], start: int, end: int, pattern: str) -> bool:
    if "\n" in pattern or start >= end:
        return False
    line = _line_at(spans, start)
    content_start, content_end = spans[line]
    if not (content_start <= start and end <= content_end):
        return False
    if normalized_fragment(pattern, doc.text[content_start:content_end]):
        return True
    raw_suffix = doc.text[end:content_end]
    suffix = normalize(raw_suffix)
    return bool(pattern and not pattern[-1].isspace() and raw_suffix and not raw_suffix[0].isspace()
                and suffix and _TOKEN_TAIL.match(suffix))


def _enumerate(doc: _Doc, spans: list[tuple[int, int]], search: tuple[tuple[str, bool], ...], mutant: str | None) -> list[_Hit]:
    wanted = [text for text, _removed in search]
    count = len(wanted)
    hits: list[_Hit] = []
    for index in range(len(spans) - count + 1):
        window = spans[index:index + count]
        texts = [doc.text[start:end] for start, end in window]
        start, end = window[0][0], window[-1][1]
        line_end = index + count
        if texts == wanted:
            hits.append(_Hit(start, end, _TIER_EXACT, index, line_end, index, False))
            continue
        same = all(normalize(got) == normalize(need) for got, need in zip(texts, wanted))
        fragment = any(normalized_fragment(need, got) for got, need in zip(texts, wanted))
        broken = mutant == "fragment-wins" and fragment and all(
            normalize(got) == normalize(need) or normalized_fragment(need, got)
            for got, need in zip(texts, wanted)
        )
        if (same and not fragment) or broken:
            hits.append(_Hit(start, end, _TIER_NORMALIZED, index, line_end, index, False))
    pattern = "\n".join(wanted)
    if not pattern:
        return _distinct(hits)
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
        if any(_overlaps(raw_start, raw_end, hit.start, hit.end) and hit.tier < _TIER_RAW for hit in hits):
            continue
        line_start, line_end = _span_lines(spans, raw_start, raw_end)
        hits.append(_Hit(raw_start, raw_end, _TIER_RAW, line_start, line_end, -1, True))
    return _distinct(hits)


def _overlaps(left: int, right: int, other_left: int, other_right: int) -> bool:
    return left < other_right and other_left < right


def _distinct(hits: list[_Hit]) -> list[_Hit]:
    kept: list[_Hit] = []
    for hit in sorted(hits, key=lambda item: (item.tier, item.start, item.end)):
        if any((hit.start, hit.end) == (old.start, old.end) for old in kept):
            continue
        kept.append(hit)
    return kept


def _strictest(hits: list[_Hit]) -> list[_Hit]:
    best = min(hit.tier for hit in hits)
    return [hit for hit in hits if hit.tier == best]


def _deleted_ranges(doc: _Doc, spans: list[tuple[int, int]], hit: _Hit, search: tuple[tuple[str, bool], ...]) -> list[tuple[int, int]]:
    if hit.raw:
        if any(removed for _text, removed in search) and hit.end > hit.start:
            return [(hit.start, hit.end)]
        return []
    ranges = []
    window = spans[hit.window:hit.window + len(search)]
    for (_text, removed), (start, end) in zip(search, window):
        if not removed:
            continue
        if end > start:
            ranges.append((start, end))
        elif start < len(doc.text):
            ranges.append((start, start + 1))
    return ranges


def _editable(doc: _Doc, ranges: list[tuple[int, int]], consumed: set[int]) -> bool:
    for start, end in ranges:
        for index in range(start, end):
            origin = doc.origin[index] if index < len(doc.origin) else None
            if origin is None or origin in consumed:
                return False
    return True


def _origin_at(origin: list[int | None], index: int) -> int | None:
    if 0 <= index < len(origin):
        value = origin[index]
        if isinstance(value, int):
            return value
    return None


def _origin_bounds(doc: _Doc, ranges: list[tuple[int, int]]) -> tuple[int | None, int | None]:
    origins = [
        value
        for start, end in ranges
        for index in range(start, min(end, len(doc.origin)))
        if (value := _origin_at(doc.origin, index)) is not None
    ]
    if not origins:
        return None, None
    return min(origins), max(origins) + 1


def _point_origin(doc: _Doc, at: int) -> int:
    found = _origin_at(doc.origin, at)
    if found is not None:
        return found
    index = min(at, len(doc.origin)) - 1
    while index >= 0:
        found = _origin_at(doc.origin, index)
        if found is not None:
            return found + 1
        index -= 1
    return 0


def _apply_aligned(
    doc: _Doc,
    spans: list[tuple[int, int]],
    hit: _Hit,
    search: tuple[tuple[str, bool], ...],
    replace: tuple[tuple[str, bool], ...],
) -> list[tuple[int, int]]:
    window = spans[hit.window:hit.window + len(search)]
    start, end = window[0][0], window[-1][1]
    consumed = _deleted_ranges(doc, spans, hit, search)
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
    _splice(doc, start, end, "".join(pieces), origins)
    return consumed


def _apply_raw(doc: _Doc, hit: _Hit, replace: tuple[tuple[str, bool], ...]) -> list[tuple[int, int]]:
    text = doc.newline.join(part for part, _added in replace)
    _splice(doc, hit.start, hit.end, text, [None] * len(text))
    return [(hit.start, hit.end)] if hit.end > hit.start else []


def _splice(doc: _Doc, start: int, end: int, text: str, origins: list[int | None]) -> None:
    if len(origins) != len(text):
        raise RuntimeError("origin length drifted")
    doc.text = doc.text[:start] + text + doc.text[end:]
    doc.origin = doc.origin[:start] + origins + doc.origin[end:]
    if len(doc.origin) != len(doc.text):
        raise RuntimeError("document provenance drifted")


def _matches_consumed(search: tuple[tuple[str, bool], ...], consumed_text: list[str]) -> bool:
    removed = [text for text, gone in search if gone and text]
    if not removed or not consumed_text:
        return False
    exact = set(consumed_text)
    folded = {normalize(item) for item in consumed_text}
    return any(line in exact or normalize(line) in folded for line in removed)


def _remember(consumed: set[int], doc_origin: list[int | None], ranges: list[tuple[int, int]]) -> None:
    for start, end in ranges:
        for index in range(start, end):
            found = _origin_at(doc_origin, index)
            if found is not None:
                consumed.add(found)


def _select(
    hits: list[_Hit],
    hint: str | None,
    doc: _Doc,
    spans: list[tuple[int, int]],
    variant: HintVariant,
    mutant: str | None,
) -> _Hit | str:
    if mutant == "fragment-wins":
        return min(hits, key=lambda hit: (hit.start, hit.tier))
    pool = _distinct(_strictest(hits))
    if not hint:
        return pool[0] if len(pool) == 1 else "ambiguous"
    narrowed = _distinct([hit for hit in pool if _covers(hit, hint, doc, spans, variant, mutant)])
    if len(narrowed) == 1:
        return narrowed[0]
    return "hint-miss" if not narrowed else "ambiguous"


def _insert(doc: _Doc, hunk: _Hunk, replace: tuple[tuple[str, bool], ...], path: str) -> Site | str:
    body = doc.newline.join(part for part, _added in replace)
    if hunk.context_hint:
        places = []
        start = 0
        while True:
            at = doc.text.find(hunk.context_hint, start)
            if at < 0:
                break
            places.append(at)
            start = at + 1
        if len(places) != 1:
            return "ambiguous" if places else "hint-miss"
        line_end = doc.text.find(doc.newline, places[0])
        at = len(doc.text) if line_end < 0 else line_end + len(doc.newline)
        piece = body + doc.newline
    else:
        at = len(doc.text)
        separator = "" if not doc.text or doc.text.endswith(doc.newline) else doc.newline
        piece = separator + body + doc.newline
    origin = _point_origin(doc, min(at, len(doc.origin)))
    site = Site(path, at, at, origin, origin, 0, _line_at(_split_lines(doc.text, doc.newline), min(at, len(doc.text))) + 1)
    _splice(doc, at, at, piece, [None] * len(piece))
    return site


def _site_of(path: str, doc: _Doc, hit: _Hit, ranges: list[tuple[int, int]], spans: list[tuple[int, int]]) -> Site:
    origin_start, origin_end = _origin_bounds(doc, ranges)
    if origin_start is None:
        point = _point_origin(doc, hit.start)
        origin_start, origin_end = point, point
    return Site(path, hit.start, hit.end, origin_start, origin_end, hit.tier, _line_at(spans, hit.start) + 1)


def _plan_hunk(
    doc: _Doc,
    hunk: _Hunk,
    variant: HintVariant,
    mutant: str | None,
    consumed: set[int],
    consumed_text: list[str],
    path: str,
) -> Site | str | None:
    search, replace = _split(hunk)
    if not search:
        return _insert(doc, hunk, replace, path) if replace else "no-site"
    spans = _split_lines(doc.text, doc.newline)
    hits = _enumerate(doc, spans, search, mutant)
    if not any(removed for _text, removed in search) and not any(added for _text, added in replace):
        return "no-site" if not hits else None
    eligible = []
    for hit in hits:
        ranges = _deleted_ranges(doc, spans, hit, search)
        if ranges and not _editable(doc, ranges, consumed):
            continue
        eligible.append(hit)
    if not eligible:
        return "consumed" if hits or _matches_consumed(search, consumed_text) else "no-site"
    chosen = _select(eligible, hunk.context_hint, doc, spans, variant, mutant)
    if isinstance(chosen, str):
        return chosen
    ranges = _deleted_ranges(doc, spans, chosen, search)
    site = _site_of(path, doc, chosen, ranges, spans)
    prior = list(doc.origin)
    consumed_text.extend(doc.text[start:end] for start, end in ranges if end > start)
    if chosen.raw:
        edited = _apply_raw(doc, chosen, replace)
    else:
        edited = _apply_aligned(doc, spans, chosen, search, replace)
    _remember(consumed, prior, edited)
    return site


def _plan_file(path: str, text: str, hunks: tuple[_Hunk, ...], variant: HintVariant, mutant: str | None) -> _Plan:
    newline = _newline_style(text)
    if newline is None:
        return _Plan(Outcome.REFUSE, None, (), (), "mixed-newlines")
    doc = _Doc(text, list(range(len(text))), newline)
    consumed: set[int] = set()
    consumed_text: list[str] = []
    sites: list[Site] = []
    for hunk in hunks:
        result = _plan_hunk(doc, hunk, variant, mutant, consumed, consumed_text, path)
        if isinstance(result, str):
            return _Plan(Outcome.REFUSE, None, (), (), result)
        if isinstance(result, Site):
            sites.append(result)
    writes = ((path, doc.text),) if doc.text != text else ()
    site = sites[0] if sites else None
    return _Plan(Outcome.WRITE, site, tuple(sites), writes, "ok")


def _plan_all(files: dict[str, str], ops: tuple[_Update, ...], variant: HintVariant, mutant: str | None) -> _Plan:
    writes: list[tuple[str, str]] = []
    sites: list[Site] = []
    grouped: dict[str, list[_Hunk]] = {}
    for op in ops:
        grouped.setdefault(op.path, []).extend(op.hunks)
    for path, hunks in grouped.items():
        if path not in files:
            return _Plan(Outcome.REFUSE, None, (), (), "missing-file")
        planned = _plan_file(path, files[path], tuple(hunks), variant, mutant)
        if planned.outcome != Outcome.WRITE:
            return _Plan(Outcome.REFUSE, None, (), (), planned.reason)
        writes.extend(planned.writes)
        sites.extend(planned.sites)
    site = sites[0] if sites else None
    return _Plan(Outcome.WRITE, site, tuple(sites), tuple(writes), "ok")


def _collapse(plans: list[_Plan], decisions: tuple[VariantDecision, ...]) -> Decision:
    outcomes = {plan.outcome for plan in plans}
    if outcomes == {Outcome.REFUSE}:
        reasons = {plan.reason for plan in plans}
        reason = plans[0].reason if len(reasons) == 1 else "refused"
        return Decision(Outcome.REFUSE, None, (), (), decisions, reason)
    agreed = all(plan.writes == plans[0].writes and plan.sites == plans[0].sites for plan in plans)
    if outcomes == {Outcome.WRITE} and agreed:
        return Decision(Outcome.WRITE, plans[0].site, plans[0].sites, plans[0].writes, decisions, "ok")
    return Decision(Outcome.UNSPECIFIED, None, (), (), decisions, "variants-differ")


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
    if hit.raw:
        edited = _apply_raw(doc, hit, replace)
    else:
        edited = _apply_aligned(doc, spans, hit, search, replace)
    _remember(consumed, prior, edited)


def _copy_doc(doc: _Doc) -> _Doc:
    return _Doc(doc.text, list(doc.origin), doc.newline)


def _record_distances(
    hits: list[_Hit],
    hint: str,
    doc: _Doc,
    spans: list[tuple[int, int]],
    line_distances: set[int],
    char_distances: set[int],
) -> None:
    for hit in hits:
        for start, end in _hint_spans(doc.text, hint):
            line_distances.add(_distance(hit.line_start, hit.line_end, _line_at(spans, start)))
            char_distances.add(_char_distance(hit.start, hit.end, start, end))


def _collect_distances(
    text: str,
    hunks: tuple[_Hunk, ...],
    mutant: str | None,
    newline: str,
) -> tuple[set[int], set[int]]:
    """Distances in every content a class can reach. Extra edges are harmless."""
    line_distances: set[int] = set()
    char_distances: set[int] = set()

    def visit(doc: _Doc, index: int, consumed: set[int], consumed_text: list[str]) -> None:
        if index >= len(hunks):
            return
        hunk = hunks[index]
        search, replace = _split(hunk)
        if not search:
            _visit_insert(doc, hunk, replace, index, consumed, consumed_text)
            return
        spans = _split_lines(doc.text, doc.newline)
        hits = _enumerate(doc, spans, search, mutant)
        editing = any(removed for _text, removed in search) or any(added for _text, added in replace)
        if not editing:
            if hits:
                visit(doc, index + 1, consumed, consumed_text)
            return
        eligible = [
            hit for hit in hits
            if not (ranges := _deleted_ranges(doc, spans, hit, search)) or _editable(doc, ranges, consumed)
        ]
        if not eligible:
            return
        pool = eligible if mutant == "fragment-wins" else _distinct(_strictest(eligible))
        if hunk.context_hint:
            _record_distances(pool, hunk.context_hint, doc, spans, line_distances, char_distances)
            targets = pool
        elif len(pool) == 1:
            targets = pool
        else:
            return
        for hit in targets:
            branch = _copy_doc(doc)
            branch_consumed = set(consumed)
            branch_text = list(consumed_text)
            _commit(branch, spans, hit, search, replace, branch_consumed, branch_text)
            visit(branch, index + 1, branch_consumed, branch_text)

    def _visit_insert(
        doc: _Doc,
        hunk: _Hunk,
        replace: tuple[tuple[str, bool], ...],
        index: int,
        consumed: set[int],
        consumed_text: list[str],
    ) -> None:
        if not replace:
            return
        branch = _copy_doc(doc)
        if isinstance(_insert(branch, hunk, replace, ""), str):
            return
        visit(branch, index + 1, consumed, consumed_text)

    visit(_Doc(text, list(range(len(text))), newline), 0, set(), [])
    return line_distances, char_distances


def _classes_for(text: str, hunks: tuple[_Hunk, ...], mutant: str | None) -> tuple[HintVariant, ...]:
    newline = _newline_style(text)
    if newline is None:
        return (HintVariant("unhinted", "line", 0, "above"),)
    line_distances, char_distances = _collect_distances(text, hunks, mutant, newline)
    classes = hint_boundary_classes({"line": line_distances, "char": char_distances})
    if classes:
        return classes
    return (HintVariant("unhinted", "line", 0, "above"),)


def _cartesian(
    groups: list[list[tuple[HintVariant, _Plan]]],
) -> list[list[tuple[HintVariant, _Plan]]]:
    combos: list[list[tuple[HintVariant, _Plan]]] = [[]]
    for group in groups:
        combos = [prev + [item] for prev in combos for item in group]
    return combos


def _merge_combo(combo: list[tuple[HintVariant, _Plan]]) -> _Plan:
    writes: list[tuple[str, str]] = []
    sites: list[Site] = []
    for _item, plan in combo:
        if plan.outcome != Outcome.WRITE:
            return _Plan(Outcome.REFUSE, None, (), (), plan.reason)
        writes.extend(plan.writes)
        sites.extend(plan.sites)
    site = sites[0] if sites else None
    return _Plan(Outcome.WRITE, site, tuple(sites), tuple(writes), "ok")


def _combo_variant(combo: list[tuple[HintVariant, _Plan]]) -> HintVariant:
    if len(combo) == 1:
        return combo[0][0]
    name = "|".join(item.name for item, _plan in combo)
    return HintVariant(name, "mixed", 0, "above")


def _decisions_for(
    chosen: tuple[HintVariant, ...] | list[HintVariant],
    plans: list[_Plan],
) -> tuple[VariantDecision, ...]:
    return tuple(
        VariantDecision(item, plan.outcome, plan.site, plan.writes, plan.reason, plan.sites)
        for item, plan in zip(chosen, plans)
    )


def _collapse_global(
    files: dict[str, str],
    ops: tuple[_Update, ...],
    chosen: tuple[HintVariant, ...],
    mutant: str | None,
) -> Decision:
    if not chosen:
        return Decision(Outcome.UNSPECIFIED, None, (), (), (), "no-variant")
    plans = [_plan_all(files, ops, item, mutant) for item in chosen]
    return _collapse(plans, _decisions_for(chosen, plans))


def _evaluate_dynamic(
    files: dict[str, str],
    ops: tuple[_Update, ...],
    mutant: str | None,
) -> Decision:
    grouped: dict[str, list[_Hunk]] = {}
    for op in ops:
        grouped.setdefault(op.path, []).extend(op.hunks)
    per_file: list[list[tuple[HintVariant, _Plan]]] = []
    for path, hunks in grouped.items():
        if path not in files:
            missing = HintVariant("missing", "line", 0, "below")
            plan = _Plan(Outcome.REFUSE, None, (), (), "missing-file")
            return _collapse([plan], _decisions_for((missing,), [plan]))
        classes = _classes_for(files[path], tuple(hunks), mutant)
        planned = [
            (item, _plan_file(path, files[path], tuple(hunks), item, mutant))
            for item in classes
        ]
        per_file.append(planned)
    combos = _cartesian(per_file)
    plans = [_merge_combo(combo) for combo in combos]
    decisions = tuple(
        VariantDecision(
            _combo_variant(combo),
            plan.outcome,
            plan.site,
            plan.writes,
            plan.reason,
            plan.sites,
        )
        for combo, plan in zip(combos, plans)
    )
    return _collapse(plans, decisions)


def evaluate(
    files: dict[str, str],
    patch: str,
    *,
    variants: tuple[HintVariant, ...] | None = None,
    mutant: str | None = None,
) -> Decision:
    """Brute-force rules 1-6. UNSPECIFIED when hint-boundary classes differ.

    ``mutant="fragment-wins"`` is a deliberately broken selector: a normalized
    fragment becomes a tier-2 site and the earliest site wins. Hand-computed
    tests must fail against it.

    ``mutant="fixed-width-list"`` replays the rejected ±1 line list. The extent
    counterexample must fail against it.
    """
    if mutant not in (None, "fragment-wins", "fixed-width-list"):
        raise ValueError("unknown mutant")
    ops, error = _parse(patch)
    if error or ops is None:
        chosen = variants or (HintVariant("parse", "line", 0, "below"),)
        blank = _decisions_for(
            chosen,
            [_Plan(Outcome.REFUSE, None, (), (), error or "parse-error") for _item in chosen],
        )
        return Decision(Outcome.REFUSE, None, (), (), blank, error or "parse-error")
    if variants is not None:
        return _collapse_global(files, ops, variants, mutant)
    if mutant == "fixed-width-list":
        return _collapse_global(files, ops, _fixed_width_variants(), mutant)
    return _evaluate_dynamic(files, ops, mutant)


def _patch(path: str, body: str) -> str:
    return f"*** Begin Patch\n*** Update File: {path}\n{body}\n*** End Patch\n"


def _files_patch(parts: tuple[tuple[str, str], ...]) -> str:
    chunks = ["*** Begin Patch"]
    for path, body in parts:
        chunks.append(f"*** Update File: {path}")
        chunks.append(body)
    chunks.append("*** End Patch")
    return "\n".join(chunks) + "\n"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("value\t=\t1", "value = 1"),
        ("value\u00a0=\u00a01", "value = 1"),
        ("value\u3000=\u30001", "value = 1"),
        ("\u212b\t  x", "Å x"),
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
    # Hand-computed: the second hunk's only site is bytes the first hunk inserted.
    source = "a = 1\n"
    body = "@@\n-a = 1\n+a = 1\n+b = 2\n@@\n-b = 2\n+b = 9"
    decision = evaluate({"f.py": source}, _patch("f.py", body))
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.reason == "consumed"
    assert "b = 2" not in decision.reason


def test_rule1_written_context_may_anchor() -> None:
    # a = 1\\n is 6 bytes; c = 3 occupies original offsets 6..11.
    # The written anchor overlaps the site (distance 0). An extent below every
    # boundary misses it, so the classes do not agree. Every writing class uses
    # that original span and does not re-edit the inserted line.
    source = "a = 1\nc = 3\n"
    body = "@@\n-a = 1\n+a = 1\n+b = 2\n@@ b = 2 @@\n b = 2\n-c = 3\n+c = 4"
    decision = evaluate({"f.py": source}, _patch("f.py", body))
    assert decision.outcome == Outcome.UNSPECIFIED
    assert decision.writes == ()
    writers = [item for item in decision.variants if item.outcome == Outcome.WRITE]
    assert writers
    expected = (("f.py", "a = 1\nb = 2\nc = 4\n"),)
    assert {item.writes for item in writers} == {expected}
    assert all(item.sites[1].origin_start == 6 for item in writers)
    assert all(item.sites[1].origin_end == 11 for item in writers)
    assert all(item.sites[1].tier == _TIER_EXACT for item in writers)
    assert all(not item.writes for item in decision.variants if item.outcome != Outcome.WRITE)


def test_rule2_value_1_beats_value_10_and_keeps_surrounding_bytes() -> None:
    # value = 10\\n is 11 bytes; the exact line starts at 11 and ends at 20.
    source = "value = 10\nvalue = 1\n"
    decision = evaluate({"f.py": source}, _patch("f.py", "@@\n-value = 1\n+CHANGED"))
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "value = 10\nCHANGED\n"),)
    assert decision.site is not None
    assert decision.site.start == 11
    assert decision.site.end == 20
    assert decision.site.origin_start == 11
    assert decision.site.origin_end == 20
    assert decision.site.tier == _TIER_EXACT
    assert decision.site.line == 2


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
    decision = evaluate({"f.py": source}, _patch("f.py", "@@\n-value = 1\n+CHANGED"))
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "value = 10\nCHANGED\nfooter = old\n"),)
    assert decision.site is not None
    assert decision.site.tier == tier
    assert decision.site.origin_start == len("value = 10\n")
    assert source[decision.site.origin_start:decision.site.origin_end] == line


def test_rule2_angstrom_uses_nfkc_normalize() -> None:
    source = "\u212b = 1\n"
    decision = evaluate({"f.py": source}, _patch("f.py", "@@\n-Å = 1\n+CHANGED"))
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "CHANGED\n"),)
    assert decision.site is not None
    assert decision.site.tier == _TIER_NORMALIZED
    assert decision.site.origin_start == 0
    assert decision.site.origin_end == len(source.rstrip("\n"))


@pytest.mark.parametrize("line", ["value = 10", "value\u00a0=\u00a010", "value\u3000=\u300010"])
def test_rule2_fragment_alone_is_refused(line: str) -> None:
    decision = evaluate({"f.py": f"{line}\n"}, _patch("f.py", "@@\n-value = 1\n+CHANGED"))
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.reason == "no-site"


def test_rule3_identical_sites_without_hint_refuse() -> None:
    source = "value = 1\nvalue = 1\nfooter = old\n"
    body = "@@\n-value = 1\n+CHANGED\n@@\n-footer = old\n+footer = new"
    decision = evaluate({"f.py": source}, _patch("f.py", body))
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.reason == "ambiguous"


def test_rule3_hint_two_lines_away_cannot_fall_back() -> None:
    # Sole tier-1 site is line distance 2 and char distance 8 from "marker".
    # Extents disagree. value = 10 is a fragment and is never the written site.
    source = "value = 1\nxxxx\n# marker\nvalue = 10\n"
    decision = evaluate({"f.py": source}, _patch("f.py", "@@ marker @@\n-value = 1\n+CHANGED"))
    assert decision.outcome == Outcome.UNSPECIFIED
    assert decision.writes == ()
    writers = [item for item in decision.variants if item.outcome == Outcome.WRITE]
    assert writers
    expected = (("f.py", "CHANGED\nxxxx\n# marker\nvalue = 10\n"),)
    assert {item.writes for item in writers} == {expected}
    assert all("value = 10" not in item.reason for item in decision.variants)


def test_rule3_adjacent_hint_boundary_is_unspecified() -> None:
    # # marker\\n is 9 bytes. "value = old" occupies 9..20. Line and char
    # distance are both 1, so inclusive 1 writes and exclusive 1 refuses.
    source = "# marker\nvalue = old\n"
    decision = evaluate({"f.py": source}, _patch("f.py", "@@ marker @@\n-value = old\n+value = new"))
    assert decision.outcome == Outcome.UNSPECIFIED
    assert decision.writes == ()
    assert decision.reason == "variants-differ"
    by_name = {item.variant.name: item for item in decision.variants}
    chosen = by_name["line:in:1"]
    assert chosen.outcome == Outcome.WRITE
    assert chosen.site is not None
    assert chosen.site.start == 9
    assert chosen.site.end == 20
    assert chosen.writes == (("f.py", "# marker\nvalue = new\n"),)
    assert by_name["line:ex:1"].outcome == Outcome.REFUSE
    assert by_name["char:ex:1"].outcome == Outcome.REFUSE
    assert by_name["line:below:0"].outcome == Outcome.REFUSE


def test_rule3_repeated_adjacent_hints_can_be_unspecified() -> None:
    source = "# target\nvalue = old\n# target\nvalue = old\n"
    body = "@@ target @@\n-value = old\n+value = new"
    decision = evaluate({"f.py": source}, _patch("f.py", body))
    assert decision.outcome == Outcome.UNSPECIFIED
    assert decision.writes == ()
    assert {item.outcome for item in decision.variants} == {Outcome.REFUSE, Outcome.WRITE}


def test_rule4_replacement_already_present_is_not_success() -> None:
    source = 'keep = HOSTISH_SECRET\nhost = "a"\nhost = "b"\n'
    body = '@@\n-host = "a"\n+host = "c"\n@@\n-host = "a"\n+keep = HOSTISH_SECRET'
    decision = evaluate({"f.py": source}, _patch("f.py", body))
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.reason == "consumed"
    assert "HOSTISH_SECRET" not in decision.reason
    assert all("HOSTISH_SECRET" not in item.reason for item in decision.variants)


def test_rule4_remaining_whole_line_copy_is_selected() -> None:
    # host = "a" is 10 bytes. The second copy starts at original offset 17 and
    # ends at 27. The hint overlaps the first line (distance 0), so an empty
    # extent refuses while a covering extent writes both hunks.
    source = 'host = "a" # one\nhost = "a"\n'
    body = '@@ one @@\n-host = "a" # one\n+host = "c"\n@@\n-host = "a"\n+host = "d"'
    decision = evaluate({"f.py": source}, _patch("f.py", body))
    assert decision.outcome == Outcome.UNSPECIFIED
    assert decision.writes == ()
    writers = [item for item in decision.variants if item.outcome == Outcome.WRITE]
    assert writers
    assert {item.writes for item in writers} == {(("f.py", 'host = "c"\nhost = "d"\n'),)}
    assert all(item.sites[1].origin_start == 17 for item in writers)
    assert all(item.sites[1].origin_end == 27 for item in writers)
    assert all(item.sites[1].tier == _TIER_EXACT for item in writers)
    assert all(not item.writes for item in decision.variants if item.outcome != Outcome.WRITE)


def test_rule5_cross_file_refusal_writes_nothing() -> None:
    files = {"a.py": "old\n", "b.py": "other\n"}
    patch = _files_patch((
        ("a.py", "@@\n-old\n+new"),
        ("b.py", "@@\n-absent\n+replacement"),
    ))
    before = dict(files)
    decision = evaluate(files, patch)
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert decision.reason == "no-site"
    assert files == before


def test_rule5_reason_does_not_leak_hint_or_source() -> None:
    # The site is two lines from the hint, so extents disagree. Reasons stay codes.
    source = "only = 1\n\n# ZZSECRETZZ\n"
    decision = evaluate({"f.py": source}, _patch("f.py", "@@ ZZSECRETZZ @@\n-only = 1\n+CHANGED"))
    assert decision.outcome == Outcome.UNSPECIFIED
    assert decision.writes == ()
    assert "ZZSECRETZZ" not in decision.reason
    assert "only = 1" not in decision.reason
    assert all("ZZSECRETZZ" not in item.reason for item in decision.variants)
    assert all("only = 1" not in item.reason for item in decision.variants)


def test_rule6_unique_partial_preserves_surrounding_bytes() -> None:
    # call( is 5 bytes; value = 1 occupies 5..14.
    source = "call(value = 1)\n"
    decision = evaluate({"f.py": source}, _patch("f.py", "@@\n-value = 1\n+CHANGED"))
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "call(CHANGED)\n"),)
    assert decision.site is not None
    assert decision.site.start == 5
    assert decision.site.end == 14
    assert decision.site.tier == _TIER_RAW
    assert decision.site.origin_start == 5


def test_rule6_hint_on_the_site_selects_partial_and_ignores_far_decoy() -> None:
    # Near "prefix " touches "SECRET" (distance 0). The decoy is line distance 2
    # and char distance 16. A covering extent writes only the near site; an
    # extent that also reaches the decoy refuses instead of writing it.
    source = "prefix AAA\nxxxx\nprefix SECRET\n"
    decision = evaluate({"f.py": source}, _patch("f.py", "@@ SECRET @@\n-prefix \n+prefixZ"))
    assert decision.outcome == Outcome.UNSPECIFIED
    assert decision.writes == ()
    near = (("f.py", "prefix AAA\nxxxx\nprefixZSECRET\n"),)
    far = (("f.py", "prefixZAAA\nxxxx\nprefix SECRET\n"),)
    writers = [item for item in decision.variants if item.outcome == Outcome.WRITE]
    assert writers
    assert {item.writes for item in writers} == {near}
    assert far not in {item.writes for item in decision.variants}
    assert all(item.site is not None and item.site.start == 16 for item in writers)
    assert all(item.site is not None and item.site.tier == _TIER_RAW for item in writers)


def test_rule6_addition_only_eof_insertion() -> None:
    decision = evaluate({"f.py": "first\n"}, _patch("f.py", "@@\n+last"))
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "first\nlast\n"),)
    assert decision.site is not None
    assert decision.site.start == 6
    assert decision.site.end == 6
    assert decision.site.origin_start == 6
    assert decision.site.tier == 0


def test_rule6_hinted_insertion_is_established_not_a_window() -> None:
    decision = evaluate({"f.py": "head\nfoot\n"}, _patch("f.py", "@@ head @@\n+mid"))
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "head\nmid\nfoot\n"),)
    assert {item.outcome for item in decision.variants} == {Outcome.WRITE}


def test_rule6_empty_anchor_preserves_later_edit() -> None:
    # b starts at original offset 3.
    source = "a\n\nb\n"
    body = "@@\n a\n@@\n \n@@\n-b\n+B"
    decision = evaluate({"f.py": source}, _patch("f.py", body))
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "a\n\nB\n"),)
    assert decision.site is not None
    assert decision.site.origin_start == 3
    assert decision.site.origin_end == 4


def test_rule6_distinct_ordered_hunks() -> None:
    # a = old and b = old are 7 bytes. The second line starts at original offset 8.
    source = "a = old\nb = old\n"
    body = "@@\n-a = old\n+a = new\n@@\n-b = old\n+b = new"
    decision = evaluate({"f.py": source}, _patch("f.py", body))
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "a = new\nb = new\n"),)
    assert decision.sites[0].origin_start == 0
    assert decision.sites[0].origin_end == 7
    assert decision.sites[1].origin_start == 8
    assert decision.sites[1].origin_end == 15


def test_crlf_and_lf_are_per_file() -> None:
    files = {"win.py": "value = 1\r\n", "unix.py": "value = 1\n"}
    patch = _files_patch((
        ("win.py", "@@\n-value = 1\n+CHANGED"),
        ("unix.py", "@@\n-value = 1\n+CHANGED"),
    ))
    decision = evaluate(files, patch)
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("win.py", "CHANGED\r\n"), ("unix.py", "CHANGED\n"))
    assert decision.sites[0].origin_end == 9
    assert decision.sites[1].origin_end == 9


def test_crlf_value_1_beats_value_10() -> None:
    # CRLF: value = 10\\r\\n is 12 bytes, so the exact line starts at 12.
    source = "value = 10\r\nvalue = 1\r\n"
    decision = evaluate({"f.py": source}, _patch("f.py", "@@\n-value = 1\n+CHANGED"))
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "value = 10\r\nCHANGED\r\n"),)
    assert decision.site is not None
    assert decision.site.start == 12
    assert decision.site.end == 21


def test_crlf_insertion_and_mixed_newline_refusal() -> None:
    inserted = evaluate({"f.py": "first\r\n"}, _patch("f.py", "@@\n+last"))
    assert inserted.outcome == Outcome.WRITE
    assert inserted.writes == (("f.py", "first\r\nlast\r\n"),)
    assert inserted.site is not None
    assert inserted.site.origin_start == 7
    mixed = evaluate({"f.py": "value = 1\r\nvalue = 1\n"}, _patch("f.py", "@@\n-value = 1\n+CHANGED"))
    assert mixed.outcome == Outcome.REFUSE
    assert mixed.writes == ()
    assert mixed.reason == "mixed-newlines"


def test_crlf_file_is_not_written_when_another_file_refuses() -> None:
    files = {"win.py": "value = 1\r\n", "unix.py": "other\n"}
    patch = _files_patch((
        ("win.py", "@@\n-value = 1\n+CHANGED"),
        ("unix.py", "@@\n-absent\n+replacement"),
    ))
    decision = evaluate(files, patch)
    assert decision.outcome == Outcome.REFUSE
    assert decision.writes == ()
    assert files["win.py"] == "value = 1\r\n"


def test_hint_boundary_classes_are_configurable_for_the_fuzz_card() -> None:
    classes = hint_boundary_classes({"line": {5, 12}})
    expected = ["line:below:4"]
    for edge in (4, 5, 6, 11, 12, 13):
        expected.append(f"line:in:{edge}")
        expected.append(f"line:ex:{edge}")
    expected.append("line:above:13")
    assert [item.name for item in classes[:len(expected)]] == expected
    assert "line:in:1" not in expected
    source = "# marker\nvalue = old\n"
    patch = _patch("f.py", "@@ marker @@\n-value = old\n+value = new")
    narrow = HintVariant("forced-narrow", "line", 0, "inclusive")
    wide = HintVariant("forced-wide", "line", 0, "above")
    refused = evaluate({"f.py": source}, patch, variants=(narrow,))
    assert refused.outcome == Outcome.REFUSE
    assert refused.writes == ()
    written = evaluate({"f.py": source}, patch, variants=(wide,))
    assert written.outcome == Outcome.WRITE
    assert written.writes == (("f.py", "# marker\nvalue = new\n"),)


def test_mutation_broken_fragment_winner_fails_hand_computed_rule2() -> None:
    source = "value = 10\nvalue = 1\n"
    patch = _patch("f.py", "@@\n-value = 1\n+CHANGED")
    good = evaluate({"f.py": source}, patch)
    bad = evaluate({"f.py": source}, patch, mutant="fragment-wins")
    assert good.outcome == Outcome.WRITE
    assert good.writes == (("f.py", "value = 10\nCHANGED\n"),)
    assert good.site is not None and good.site.origin_start == 11
    assert bad.writes != good.writes
    assert not (
        bad.outcome == Outcome.WRITE
        and bad.writes == good.writes
        and bad.site is not None
        and bad.site.origin_start == 11
    )


def test_fragment_inside_call_is_not_an_edit_site() -> None:
    decision = evaluate({"f.py": "call(value = 10)\n"}, _patch("f.py", "@@\n-value = 1\n+changed"))
    assert decision.outcome == Outcome.REFUSE
    assert not decision.writes


def test_asymmetric_hint_window_can_select_one_equidistant_candidate() -> None:
    source = "target\n# hint\ntarget\n"
    decision = evaluate({"f.py": source}, _patch("f.py", "@@ hint @@\n-target\n+changed"))
    assert decision.outcome == Outcome.UNSPECIFIED
    assert {site.origin_start for item in decision.variants for site in item.sites} == {0, 14}


def test_overlapping_distinct_whole_line_candidates_are_ambiguous() -> None:
    decision = evaluate({"f.py": "a\na\na\n"}, _patch("f.py", "@@\n-a\n-a\n+b"))
    assert decision.outcome == Outcome.REFUSE
    assert not decision.writes


def test_eof_insertion_preserves_existing_blank_lines() -> None:
    decision = evaluate({"f.py": "first\n\n"}, _patch("f.py", "@@\n+last"))
    assert decision.writes == (("f.py", "first\n\nlast\n"),)


def test_repeated_update_sections_share_planned_content() -> None:
    patch = _files_patch((("f.py", "@@\n-old\n+new"), ("f.py", "@@\n-other\n+changed")))
    decision = evaluate({"f.py": "old\nother\n"}, patch)
    assert decision.outcome == Outcome.WRITE
    assert decision.writes == (("f.py", "new\nchanged\n"),)


def _sites_beyond_fixed_extent() -> tuple[str, str]:
    """Identical sites at line distances 5 and 12, both outside a ±1 window.

    First site is [29, 38). Second is [69, 78). Char distances are 21 and 61.
    """
    lines = ["# marker", *["xxxx"] * 4, "value = 1", *["xxxx"] * 6, "value = 1"]
    source = "\n".join(lines) + "\n"
    nearer = source[:29] + "CHANGED" + source[38:]
    return source, nearer


def test_two_candidates_beyond_fixed_extent_are_unspecified() -> None:
    source, nearer = _sites_beyond_fixed_extent()
    assert source[29:38] == "value = 1"
    assert source[69:78] == "value = 1"
    decision = evaluate({"f.py": source}, _patch("f.py", "@@ marker @@\n-value = 1\n+CHANGED"))
    assert decision.outcome == Outcome.UNSPECIFIED
    assert decision.writes == ()
    by_name = {item.variant.name: item for item in decision.variants}
    assert by_name["line:in:5"].writes == (("f.py", nearer),)
    assert by_name["line:in:5"].site is not None
    assert by_name["line:in:5"].site.line == 6
    assert by_name["char:in:21"].writes == (("f.py", nearer),)
    assert by_name["line:in:12"].outcome == Outcome.REFUSE
    assert by_name["char:in:61"].outcome == Outcome.REFUSE
    assert by_name["line:below:4"].outcome == Outcome.REFUSE
    assert "inclusive+1" not in by_name
    writers = [item for item in decision.variants if item.outcome == Outcome.WRITE]
    assert writers
    assert {item.writes for item in writers} == {(("f.py", nearer),)}


def test_equidistant_hinted_sites_allow_asymmetric_windows() -> None:
    source = "# target\nvalue = old\n# target\nvalue = old\n"
    decision = evaluate({"f.py": source}, _patch("f.py", "@@ target @@\n-value = old\n+value = new"))
    assert decision.outcome == Outcome.UNSPECIFIED
    assert decision.writes == ()
    assert {item.outcome for item in decision.variants} == {Outcome.REFUSE, Outcome.WRITE}
    names = {item.variant.name for item in decision.variants}
    assert {"line:below:0", "line:in:1", "char:in:1"} <= names


def test_long_file_extent_classes_are_not_a_fixed_window() -> None:
    # Hint on line 0. Sites on lines 100 and 500.
    # First site [504, 513), char distance 496. Second [2509, 2518), char 2501.
    source = "# marker\n" + ("xxxx\n" * 99) + "value = 1\n" + ("xxxx\n" * 399) + "value = 1\n"
    assert len(source) == 2519
    assert source[504:513] == "value = 1"
    assert source[2509:2518] == "value = 1"
    nearer = source[:504] + "CHANGED" + source[513:]
    decision = evaluate({"f.py": source}, _patch("f.py", "@@ marker @@\n-value = 1\n+CHANGED"))
    assert decision.outcome == Outcome.UNSPECIFIED
    assert decision.writes == ()
    by_name = {item.variant.name: item for item in decision.variants}
    assert by_name["line:in:100"].writes == (("f.py", nearer),)
    assert by_name["char:in:496"].writes == (("f.py", nearer),)
    assert by_name["line:in:500"].outcome == Outcome.REFUSE
    assert by_name["char:in:2501"].outcome == Outcome.REFUSE
    writers = [item for item in decision.variants if item.outcome == Outcome.WRITE]
    assert {item.writes for item in writers} == {(("f.py", nearer),)}


def test_mutation_fixed_width_list_fails_extent_counterexample() -> None:
    source, nearer = _sites_beyond_fixed_extent()
    patch = _patch("f.py", "@@ marker @@\n-value = 1\n+CHANGED")
    good = evaluate({"f.py": source}, patch)
    bad = evaluate({"f.py": source}, patch, mutant="fixed-width-list")
    assert good.outcome == Outcome.UNSPECIFIED
    assert any(item.writes == (("f.py", nearer),) for item in good.variants)
    assert bad.outcome == Outcome.REFUSE
    assert bad.writes == ()
    assert not any(item.writes == (("f.py", nearer),) for item in bad.variants)
    assert {item.variant.name for item in bad.variants} == {
        "inclusive-1",
        "exclusive-1",
        "inclusive+0",
        "exclusive+0",
        "inclusive+1",
        "exclusive+1",
    }
