"""Provenance planner for one V4A Update.

The original file is an immutable list of ``(text, eol)`` lines. Every later
decision names original line indexes, never a fresh copy of the text. A hunk
is ``Resolved``, ``AlreadyApplied``, or ``Refused``; a refusal is never turned
into a no-op. Replay evidence is checked before a not-found refusal is built.
A search consumed by an earlier hunk of this patch is never a no-op.
This module does not write.
"""

from __future__ import annotations

import bisect
import itertools
import re
import unicodedata
from dataclasses import dataclass

_WS = re.compile(r"\s+")
_ADVICE_HINTED = "include unique context lines in its search text"
_ADVICE_BARE = (
    "add a unique @@ hint @@ to this hunk or include unique context lines "
    "in its search text"
)
_NOT_FOUND = "Could not find a match for old_string in the file"
_WRITTEN = "its search text is only found inside text an earlier hunk of this patch wrote"


@dataclass(frozen=True)
class Piece:
    """One planned line. ``origin`` is the original index; insertions have none."""

    kind: str
    text: str
    eol: str
    origin: int | None
    ins_id: int | None


@dataclass(frozen=True)
class Site:
    """A candidate in planned-piece coordinates. ``partial`` is a char span."""

    start: int
    end: int
    edit_piece: int
    tier: str
    partial: tuple[int, int] | None = None


@dataclass(frozen=True)
class Resolved:
    site: Site
    tier: str


@dataclass(frozen=True)
class AlreadyApplied:
    """Explicit replay proof. Never constructed from a refusal."""

    proof: str


@dataclass(frozen=True)
class _Unproven:
    """No site yet. Not a refusal — whole-file replay evidence is still open."""

    index: int
    hint: str | None
    search: tuple[str, ...]
    added: tuple[str, ...]


@dataclass(frozen=True)
class Refused:
    code: str
    message: str


def canon(line: str) -> str:
    """The one normalizer: NFKC, then Unicode whitespace collapsed and stripped."""

    return _WS.sub(" ", unicodedata.normalize("NFKC", line)).strip()


def _windows() -> tuple[int, int]:
    """Keep the live hint window. Do not invent another boundary."""

    from tools.patch_parser import _HINT_WINDOW_AFTER, _HINT_WINDOW_BEFORE

    return _HINT_WINDOW_BEFORE, _HINT_WINDOW_AFTER


def split_pieces(content: str) -> list[Piece]:
    """Decode physical lines. A terminator closes one line; no phantom empty tail."""

    if content == "":
        return []
    pieces: list[Piece] = []
    i = origin = 0
    n = len(content)
    while i < n:
        crlf = content.find("\r\n", i)
        lf = content.find("\n", i)
        if crlf == -1 and lf == -1:
            pieces.append(Piece("orig", content[i:], "", origin, None))
            break
        if lf != -1 and (crlf == -1 or lf < crlf):
            pieces.append(Piece("orig", content[i:lf], "\n", origin, None))
            i = lf + 1
        else:
            pieces.append(Piece("orig", content[i:crlf], "\r\n", origin, None))
            i = crlf + 2
        origin += 1
        if i == n:
            break
    return pieces


def render(pieces: list[Piece]) -> str:
    return "".join(piece.text + piece.eol for piece in pieces)


def file_eol(pieces: list[Piece]) -> str:
    crlf = sum(piece.eol == "\r\n" for piece in pieces)
    lf = sum(piece.eol == "\n" for piece in pieces)
    if crlf > lf:
        return "\r\n"
    return "\n" if lf or not crlf else "\r\n"


def char_spans(pieces: list[Piece]) -> list[tuple[int, int]]:
    spans = []
    pos = 0
    for piece in pieces:
        end = pos + len(piece.text) + len(piece.eol)
        spans.append((pos, end))
        pos = end
    return spans


def _label(index: int, hint: str | None) -> str:
    return f"hunk {index} " + ("(hinted)" if hint else "(no hint)")


def _advice(hint: str | None) -> str:
    return _ADVICE_HINTED if hint else _ADVICE_BARE


def _not_found(index: int, hint: str | None) -> str:
    return f"{_label(index, hint)} not found — {_NOT_FOUND}"


def _ambiguous(index: int, hint: str | None, count: int, tail: str = "") -> str:
    extra = f" {tail}" if tail else ""
    return (
        f"{_label(index, hint)} is ambiguous ({count} matches{extra}) — {_advice(hint)}"
    )


def _split(hunk) -> tuple[list[str], list[bool], list[str]]:
    search: list[str] = []
    removed: list[bool] = []
    added: list[str] = []
    for line in hunk.lines:
        if line.prefix != "+":
            search.append(line.content)
            removed.append(line.prefix == "-")
        if line.prefix == "+":
            added.append(line.content)
    return search, removed, added


def _positions(content: str, needle: str) -> list[int]:
    if not needle:
        return []
    return [match.start() for match in re.finditer(f"(?={re.escape(needle)})", content)]


def _hint_windows(content: str, hint: str | None) -> list[tuple[int, int]]:
    before, after = _windows()
    return [
        (max(0, pos - before), min(len(content), pos + after))
        for pos in _positions(content, hint or "")
    ]


def _lines_equal(window: list[Piece], wanted: list[str], mode: str) -> bool:
    if len(window) != len(wanted):
        return False
    if mode == "exact":
        return all(piece.text == line for piece, line in zip(window, wanted))
    return all(canon(piece.text) == canon(line) for piece, line in zip(window, wanted))


def _eligible(window: list[Piece], removed: list[bool], consumed: set[int]) -> bool:
    for piece, gone in zip(window, removed):
        if not gone:
            continue
        if piece.kind != "orig" or piece.origin is None or piece.origin in consumed:
            return False
    return True


def _edit_piece(start: int, removed: list[bool]) -> int:
    for offset, gone in enumerate(removed):
        if gone:
            return start + offset
    return start


def whole_line_sites(
    pieces: list[Piece],
    search: list[str],
    removed: list[bool],
    consumed: set[int],
    mode: str,
) -> list[Site]:
    if not search:
        return []
    n = len(search)
    found = []
    for start in range(len(pieces) - n + 1):
        window = pieces[start:start + n]
        if not _lines_equal(window, search, mode) or not _eligible(window, removed, consumed):
            continue
        found.append(Site(start, start + n, _edit_piece(start, removed), mode))
    return found


def _word(char: str) -> bool:
    return char.isalnum() or char == "_"


def _joins_before(line: str, start: int, first: str) -> bool:
    """The source character before the match continues the token the needle starts."""

    if start == 0:
        return False
    prev = line[start - 1]
    if _word(first) and _word(prev):
        return True
    return prev == "." and first.isdigit() and start >= 2 and line[start - 2].isdigit()


def _joins_after(line: str, end: int, last: str) -> bool:
    """The source character after the match continues the token the needle ends."""

    if end >= len(line):
        return False
    nxt = line[end]
    if _word(last) and _word(nxt):
        return True
    return nxt == "." and last.isdigit() and end + 1 < len(line) and line[end + 1].isdigit()


def _edge_ok(line: str, start: int, end: int, search: str, *, check_start: bool, check_end: bool) -> bool:
    """A raw fragment of a different token is not a partial site.

    Tokens are identifiers (alphanumerics and ``_``) and decimal numbers. A boundary is
    required only where the needle's own edge character is part of a token, so
    ``.append(x)`` may follow ``items`` while ``name`` may not sit inside ``rename`` or
    ``my_name``, ``value = 1`` not inside ``value = 10`` or ``value = 1.5``. A search that
    ends in whitespace is a deliberate mid-line edit (``prefix ``): its END edge is exempt.
    """

    open_end = search.endswith((" ", "\t", "\n"))
    if check_start and search and _joins_before(line, start, search[0]):
        return False
    return not (check_end and not open_end and search and _joins_after(line, end, search[-1]))


def partial_sites(
    pieces: list[Piece],
    search: list[str],
    consumed: set[int],
    eol: str,
    removed: list[bool],
) -> list[Site]:
    if not search or not any(search):
        return []
    if (removed[0] and not canon(search[0])) or (removed[-1] and not canon(search[-1])):
        # A removed blank (empty or whitespace-only) edge line is a whole
        # physical blank line (whole tier). As a substring edge it would match a
        # neighbour's terminator or indentation across the line break.
        return []
    content = render(pieces)
    pattern = "\n".join(search)
    needles = [pattern] if eol == "\n" else [pattern, pattern.replace("\n", eol)]
    spans = char_spans(pieces)
    starts = [lo for lo, _hi in spans]
    found: list[Site] = []
    seen: set[tuple[int, int]] = set()
    for needle in needles:
        start = 0
        while needle and (at := content.find(needle, start)) >= 0:
            end = at + len(needle)
            start = at + 1
            if (at, end) in seen:
                continue
            seen.add((at, end))
            first_idx = max(0, bisect.bisect_right(starts, at) - 1)
            covered = []
            for i in range(first_idx, len(spans)):
                lo, hi = spans[i]
                if lo >= end:
                    break
                if lo < end and at < hi:
                    covered.append(i)
            if not covered:
                continue
            if any(
                pieces[i].kind != "orig" or pieces[i].origin in consumed
                for i in covered
            ):
                continue
            first = pieces[covered[0]]
            rel = at - spans[covered[0]][0]
            rel_end = min(end, spans[covered[0]][0] + len(first.text)) - spans[covered[0]][0]
            last = pieces[covered[-1]]
            last_lo = spans[covered[-1]][0]
            single = len(covered) == 1
            edges_ok = _edge_ok(
                first.text, rel, rel_end, needle, check_start=True, check_end=single
            ) and (
                single
                or _edge_ok(last.text, 0, end - last_lo, needle, check_start=False, check_end=True)
            )
            if not edges_ok:
                continue
            found.append(Site(covered[0], covered[-1] + 1, covered[0], "partial", (at, end)))
    return found


def _in_window(site: Site, spans: list[tuple[int, int]], lo: int, hi: int, pos: int, hint: str) -> bool:
    if not spans:
        return False
    edit = min(site.edit_piece, len(spans) - 1)
    edit_at = spans[edit][0]
    site_end = spans[min(site.end, len(spans)) - 1][1]
    contains = spans[site.start][0] <= pos < site_end and site.start < site.end
    return (lo <= edit_at and site_end <= hi) or contains or (
        site.start <= edit and spans[site.start][0] <= pos + len(hint) <= site_end
    )


def _contains_pos(site: Site, spans: list[tuple[int, int]], pos: int) -> bool:
    if not spans or site.start >= site.end:
        return False
    return spans[site.start][0] <= pos < spans[min(site.end, len(spans)) - 1][1]


def _hint_stale(pieces: list[Piece], spans: list[tuple[int, int]], pos: int, consumed: set[int]) -> bool:
    """A hint inside text this patch already edited cannot authorize a different line."""

    for index, (lo, hi) in enumerate(spans):
        if lo <= pos < hi:
            piece = pieces[index]
            return piece.kind == "ins" or piece.origin in consumed
    return False


def hint_filter(
    sites: list[Site], content: str, hint: str | None, pieces: list[Piece], consumed: set[int],
) -> list[Site] | None:
    """None when the hint text does not occur. Otherwise the sites it names."""

    if not hint:
        return None
    positions = _positions(content, hint)
    if not positions:
        return None
    windows = _hint_windows(content, hint)
    spans = char_spans(pieces)
    groups = []
    for (lo, hi), pos in zip(windows, positions):
        if _hint_stale(pieces, spans, pos, consumed):
            groups.append([site for site in sites if _contains_pos(site, spans, pos)])
        else:
            groups.append([site for site in sites if _in_window(site, spans, lo, hi, pos, hint)])
    if all(len(group) <= 1 for group in groups):
        agreed: list[Site] = []
        for group in groups:
            if group and group[0] not in agreed:
                agreed.append(group[0])
        return agreed
    union: list[Site] = []
    for group in groups:
        for site in group:
            if site not in union:
                union.append(site)
    return union


def _after(sites: list[Site], floor: int | None) -> list[Site]:
    if floor is None:
        return sites
    return [site for site in sites if site.edit_piece >= floor]


def choose_tier(
    sites: list[Site],
    pieces: list[Piece],
    search: list[str],
    removed: list[bool],
    consumed: set[int],
    hint: str | None,
    floor: int | None,
    index: int,
    content: str,
) -> Resolved | Refused:
    hinted = hint_filter(sites, content, hint, pieces, consumed)
    if hinted is not None:
        if len(hinted) == 1 and floor is not None and hinted[0].edit_piece < floor:
            later = _after(sites, floor)
            if any(site.start != hinted[0].start for site in later):
                return Refused(
                    "behind",
                    f"{_label(index, hint)} is ambiguous: its @@ hint targets a site "
                    f"before the previous hunk while another site follows it — {_advice(hint)}",
                )
        if len(hinted) == 1:
            return Resolved(hinted[0], hinted[0].tier)
        if len(hinted) > 1:
            return Refused("ambiguous", _ambiguous(index, hint, len(hinted)))
        if sites:
            return Refused("inconsistent", _ambiguous(index, hint, len(sites), "outside the hint"))
    narrowed = _after(sites, floor)
    if len(narrowed) == 1:
        return Resolved(narrowed[0], narrowed[0].tier)
    if len(narrowed) > 1:
        tail = "after the previous hunk" if floor is not None else ""
        return Refused("ambiguous", _ambiguous(index, hint, len(narrowed), tail))
    if floor is not None and sites:
        # Every site lies behind the previous anchor or hunk: never guess backwards.
        return Refused("inconsistent", _ambiguous(index, hint, len(sites)))
    if len(sites) == 1:
        return Resolved(sites[0], sites[0].tier)
    if sites:
        return Refused("ambiguous", _ambiguous(index, hint, len(sites)))
    return Refused("not_found", _not_found(index, hint))


def _search_consumed(search: list[str], removed: list[bool],
                     original: tuple[Piece, ...], consumed: set[int]) -> bool:
    """Determine consumed searches solely from original line provenance."""
    lines = list(original)
    sites = (whole_line_sites(lines, search, removed, set(), "exact")
             or whole_line_sites(lines, search, removed, set(), "canon"))
    if sites:
        return any(
            any(gone and lines[site.start + offset].origin in consumed
                for offset, gone in enumerate(removed))
            for site in sites
        )
    return any(
        any(lines[index].origin in consumed for index in range(site.start, site.end))
        for site in partial_sites(lines, search, set(), file_eol(lines), removed)
    )


def _written_only(pieces, search, removed, consumed) -> bool:
    if not search:
        return False
    n = len(search)
    for start in range(len(pieces) - n + 1):
        window = pieces[start:start + n]
        if not _lines_equal(window, search, "exact") and not _lines_equal(window, search, "canon"):
            continue
        if _eligible(window, removed, consumed):
            continue
        if any(gone and piece.kind == "ins" for piece, gone in zip(window, removed)):
            return True
    return False


def _prefix_decoy(pieces: list[Piece], search: list[str]) -> bool:
    if len(search) != 1 or not search[0].strip():
        return False
    wanted = canon(search[0])
    return any(canon(piece.text) != wanted and canon(piece.text).startswith(wanted) for piece in pieces)


# Identifiers, numbers, quoted strings, and operators. A character suffix that
# splits a token (``d`` in ``old`` / ``hostish_d``) is not a token.
_TOKEN = re.compile(
    r"""
    '(?:\\.|[^'\\])*' |
    "(?:\\.|[^"\\])*" |
    \d+(?:\.\d+)? |
    [^\W\d]\w* |
    ==|!=|<=|>=|\*\*|//|-> |
    [^\w\s]
    """,
    re.VERBOSE,
)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def _token_edge_run(search: list[str], added: list[str]) -> bool:
    """Nonempty token-aligned common leading or trailing run.

    The run is whole tokens only. ``missing = old`` and ``keep = hostish_d``
    share the character ``d`` and a middle ``=``, and neither is an edge run.
    """

    left = _tokens("\n".join(search))
    right = _tokens("\n".join(added))
    if not left or not right:
        return False

    def _run(pairs) -> int:
        count = 0
        for one, two in pairs:
            if one != two:
                break
            count += 1
        return count

    if _run(zip(left, right)):
        return True
    return bool(_run(zip(reversed(left), reversed(right))))


def _exact_block_starts(pieces: list[Piece], block: list[str]) -> list[int]:
    if not block:
        return []
    width = len(block)
    return [
        start
        for start in range(len(pieces) - width + 1)
        if all(piece.text == line for piece, line in zip(pieces[start:start + width], block))
    ]


def _search_present(pieces: list[Piece], search: list[str]) -> bool:
    if not search:
        return False
    return "\n".join(search) in render(pieces)


def _hint_proves_unique(pieces: list[Piece], block: list[str], hint: str) -> bool:
    """The hint overlaps exactly one exact whole-line replacement block."""

    starts = _exact_block_starts(pieces, block)
    if not hint or not starts:
        return False
    content = render(pieces)
    spans = char_spans(pieces)
    positions = _positions(content, hint)
    width = len(block)
    proven: list[int] = []
    for start in starts:
        lo = spans[start][0]
        hi = spans[start + width - 1][1]
        if any(lo <= pos < hi or (lo <= pos and pos + len(hint) <= hi) for pos in positions):
            proven.append(start)
    return len(proven) == 1


def _hint_replay(pieces, search, added, hint) -> bool:
    """Explicit hint proof. Independent of other hunks; never a consumed search."""

    if not hint or not added or _search_present(pieces, search):
        return False
    return _hint_proves_unique(pieces, added, hint)


def _unhinted_replay(search: list[str], added: list[str], pieces: list[Piece]) -> bool:
    """Whole-file replay proof for one hunk that found no site.

    Requires a unique exact whole-line replacement and a token-aligned common
    leading or trailing run. Caller must already know no hunk resolved live.
    """

    if not added or not search or search == added:
        return False
    old_text, new_text = "\n".join(search), "\n".join(added)
    if old_text in new_text or new_text in old_text:
        return False
    if _search_present(pieces, search) or _prefix_decoy(pieces, search):
        return False
    if len(_exact_block_starts(pieces, added)) != 1:
        return False
    return _token_edge_run(search, added)


def _refuse_unproven(deferred: list[tuple[_Unproven, list[Piece]]], live: bool) -> list[str]:
    """Build not-found refusals only after whole-file replay evidence is checked."""

    errors = []
    for item, snapshot in deferred:
        if not live and _unhinted_replay(list(item.search), list(item.added), snapshot):
            continue
        errors.append(_not_found(item.index, item.hint))
    return errors


def _correct(pattern: str, replacement: str, content: str):
    from tools.escape_drift_autocorrect import AmbiguousEscapeDriftError, maybe_correct_escape_drift

    try:
        old, new, note = maybe_correct_escape_drift(pattern, replacement, content)
    except AmbiguousEscapeDriftError:
        return pattern, replacement, None, "ESCAPE_DRIFT_AMBIGUOUS"
    return old, new, note, None



def _carry_indent(pieces: list[Piece], site: Site, search: list[str], added: list[str]) -> list[str]:
    """Normalized whole-line match: copy the matched source indent onto the added lines.

    Only when no search line carries leading whitespace and every nonblank matched line
    has the same nonempty indent. Anything else writes the added lines verbatim.
    """

    if any(line[:1].isspace() for line in search):
        return added
    indents = {
        piece.text[: len(piece.text) - len(piece.text.lstrip())]
        for piece in pieces[site.start:site.end]
        if piece.text.strip()
    }
    if len(indents) != 1 or not (indent := next(iter(indents))):
        return added
    return [indent + line if line.strip() else line for line in added]


def apply_whole(pieces, site, hunk, added, eol, consumed, ids):
    out = list(pieces[:site.start])
    idx = site.start
    added_at = 0
    plus_count = sum(line.prefix == "+" for line in hunk.lines)
    minus_count = sum(line.prefix == "-" for line in hunk.lines)
    removed_eols = [
        pieces[site.start + offset].eol
        for offset, line in enumerate(line for line in hunk.lines if line.prefix != "+")
        if line.prefix == "-"
    ]
    for line in hunk.lines:
        if line.prefix == " ":
            out.append(pieces[idx])
            idx += 1
        elif line.prefix == "-":
            piece = pieces[idx]
            if piece.origin is not None:
                consumed.add(piece.origin)
            idx += 1
        else:
            text = added[added_at] if added_at < len(added) else line.content
            # A one-for-one replacement keeps the consumed line's ending, including
            # a missing final newline. Extra inserted lines use the file ending.
            this_eol = removed_eols[0] if plus_count == minus_count == 1 and removed_eols else eol
            added_at += 1
            if out and out[-1].eol == "":
                last = out[-1]
                out[-1] = Piece(last.kind, last.text, eol, last.origin, last.ins_id)
            out.append(Piece("ins", text, this_eol, None, next(ids)))
    if removed_eols and removed_eols[-1] == "" and idx == len(pieces) and out[-1].kind == "ins":
        # The removed run ended an unterminated file: the last added line ends it too.
        last = out[-1]
        out[-1] = Piece(last.kind, last.text, "", last.origin, last.ins_id)
    out.extend(pieces[idx:])
    return out


def apply_span(pieces, span, replacement, eol, consumed, ids):
    start, end = span
    spans = char_spans(pieces)
    covered = [i for i, (lo, hi) in enumerate(spans) if lo < end and start < hi]
    if not covered:
        return pieces
    first, last = covered[0], covered[-1]
    prefix = pieces[first].text[:max(0, start - spans[first][0])]
    last_text_end = spans[last][0] + len(pieces[last].text)
    if end <= last_text_end:
        suffix = pieces[last].text[end - spans[last][0]:]
        suffix_eol = pieces[last].eol
    else:
        suffix = ""
        suffix_eol = "" if end >= spans[last][1] else pieces[last].eol
    for index in covered:
        if pieces[index].origin is not None:
            consumed.add(pieces[index].origin)

    lines = replacement.split("\n")
    made = []
    if len(lines) == 1:
        made.append(Piece("ins", prefix + lines[0] + suffix, suffix_eol, None, next(ids)))
    else:
        made.append(Piece("ins", prefix + lines[0], eol, None, next(ids)))
        made.extend(Piece("ins", line, eol, None, next(ids)) for line in lines[1:-1])
        if lines[-1] == "" and not suffix and not suffix_eol:
            # The replacement ended with the line break the match already consumed.
            made[-1] = Piece("ins", made[-1].text, suffix_eol or eol, None, made[-1].ins_id)
        else:
            made.append(Piece("ins", lines[-1] + suffix, suffix_eol or eol, None, next(ids)))
    return pieces[:first] + made + pieces[last + 1:]


def _insert_at(pieces, at, added, eol, ids):
    made = [Piece("ins", line, eol, None, next(ids)) for line in added]
    return pieces[:at] + made + pieces[at:]


def _append(pieces, added, eol, ids):
    out = list(pieces)
    if out and out[-1].eol == "":
        last = out[-1]
        out[-1] = Piece(last.kind, last.text, eol, last.origin, last.ins_id)
    return out + [Piece("ins", line, eol, None, next(ids)) for line in added]


def _damage(before, after, hints, damaged):
    """A later hunk's hint that occurs less often after this edit was changed by it."""
    for hint in hints:
        if hint and hint not in damaged and len(_positions(after, hint)) < len(_positions(before, hint)):
            damaged.add(hint)


def _resolve(pieces, hunk, consumed, original, floor, index):
    hint = hunk.context_hint or None
    label = _label(index, hint)
    search, removed, added = _split(hunk)
    content = render(pieces)
    pattern = "\n".join(search)
    replacement = "\n".join(
        [line.content for line in hunk.lines if line.prefix != "-"]
    )
    pattern, replacement, note, drift = _correct(pattern, replacement, content)
    if drift:
        return Refused("drift", f"{label}: {drift}"), note, search, added, replacement
    corrected_search = pattern.split("\n")
    if len(corrected_search) == len(search):
        search = corrected_search
    corrected_added = replacement.split("\n")
    replace_lines = [line.content for line in hunk.lines if line.prefix != "-"]
    if len(corrected_added) == len(replace_lines):
        added = [
            text for text, line in zip(
                corrected_added, (ln for ln in hunk.lines if ln.prefix != "-"),
            )
            if line.prefix == "+"
        ]
    exact = whole_line_sites(pieces, search, removed, consumed, "exact")
    normalized = [] if exact else whole_line_sites(pieces, search, removed, consumed, "canon")
    tier = exact or normalized
    if tier:
        decision = choose_tier(
            tier, pieces, search, removed, consumed, hint, floor, index, content,
        )
        return decision, note, search, added, replacement
    if _search_consumed(search, removed, original, consumed):
        if _written_only(pieces, search, removed, consumed):
            return Refused("written", f"{label} not found — {_WRITTEN} — {_advice(hint)}"), note, search, added, replacement
        return Refused("not_found", _not_found(index, hint)), note, search, added, replacement
    parts = partial_sites(pieces, search, consumed, file_eol(pieces), removed)
    if parts:
        decision = choose_tier(
            parts, pieces, search, removed, consumed, hint, floor, index, content,
        )
        return decision, note, search, added, replacement
    if _written_only(pieces, search, removed, consumed):
        return Refused("written", f"{label} not found — {_WRITTEN} — {_advice(hint)}"), note, search, added, replacement
    if _hint_replay(pieces, search, added, hint):
        return AlreadyApplied("hint"), note, search, added, replacement
    return _Unproven(index, hint, tuple(search), tuple(added)), note, search, added, replacement


def _resolve_addition(pieces, hunk, hint, index, added, content):
    if hint:
        positions = _positions(content, hint)
        if not positions:
            return Refused("not_found", f"addition-only hunk {index} context hint not found")
        if len(positions) > 1:
            return Refused(
                "ambiguous",
                f"addition-only hunk {index} context hint is ambiguous ({len(positions)} occurrences)",
            )
        spans = char_spans(pieces)
        at = next((i for i, (lo, hi) in enumerate(spans) if lo <= positions[0] < hi), len(pieces) - 1)
        return Resolved(Site(at + 1, at + 1, at + 1, "addition"), "addition")
    return Resolved(Site(len(pieces), len(pieces), len(pieces), "addition"), "addition")


def _anchor_floor(pieces, hunk, consumed):
    search, removed, _added = _split(hunk)
    if not search or any(removed):
        return None
    exact = whole_line_sites(pieces, search, removed, consumed, "exact")
    sites = exact or whole_line_sites(pieces, search, removed, consumed, "canon")
    if len(sites) != 1:
        return None
    return sites[0].start


def _remap_floor(before: list[Piece], after: list[Piece], floor: int | None) -> int | None:
    """Follow the anchor cursor across an edit by piece identity, never by raw index."""

    if floor is None:
        return None

    def key(piece: Piece):
        return ("o", piece.origin) if piece.kind == "orig" else ("i", piece.ins_id)

    where = {key(piece): at for at, piece in enumerate(after)}
    for piece in before[floor:]:
        if key(piece) in where:
            return where[key(piece)]
    return len(after)


@dataclass
class PlannerState:
    """One logical file's original line provenance across Update operations."""

    original: tuple[Piece, ...]
    pieces: list[Piece]
    consumed: set[int]

    damaged: set[str]
    floor: int | None
    ids: object
    live: bool
    deferred: list


    @classmethod
    def from_content(cls, content: str) -> PlannerState:
        original = tuple(split_pieces(content))
        return cls(original, list(original), set(), set(), None,
                   itertools.count(1), False, [])


def finish_state(state: PlannerState) -> list[str]:
    """Finalize deferred replay proofs after all operations for this file."""
    errors = _refuse_unproven(state.deferred, state.live)
    state.deferred.clear()
    return errors


def plan_hunks(content: str, hunks: list, state: PlannerState | None = None,
               *, finalize: bool = True) -> object:
    """Plan hunks using one provenance state for the lifetime of a logical file."""

    if state is None:
        state = PlannerState.from_content(content)
    state.floor = None  # an anchor scopes the hunks of its own Update block only
    pieces = state.pieces
    consumed = state.consumed

    errors: list[str] = []
    notes: list[str] = []
    strategies: list[str] = []
    damaged = state.damaged
    floor = state.floor
    ids = state.ids
    changes = 0
    live = state.live
    deferred = state.deferred
    for index, hunk in enumerate(hunks, start=1):
        if any(line.prefix in "-+" for line in hunk.lines):
            changes += 1
        hint = hunk.context_hint or None

        if hint in damaged:
            errors.append(
                f"{_label(index, hint)} is ambiguous: its @@ hint was changed by an earlier hunk"
            )
            continue
        search, _removed, added = _split(hunk)
        prefixes = {line.prefix for line in hunk.lines}
        if prefixes <= {" "}:
            if search:
                moved = _anchor_floor(pieces, hunk, consumed)
                if moved is not None:
                    floor = moved
            continue
        before = render(pieces)
        prior = pieces
        if not search:
            decision = _resolve_addition(pieces, hunk, hint, index, added, before)
            if isinstance(decision, Refused):
                errors.append(decision.message)
                continue
            at, lines = decision.site.start, added
            eol = file_eol(pieces)
            pieces = _append(pieces, lines, eol, ids) if at >= len(pieces) else _insert_at(
                pieces, at, lines, eol, ids,
            )

            floor = _remap_floor(prior, pieces, floor)
            live = True
            continue
        resolved = _resolve(
            pieces, hunk, consumed, state.original, floor, index,
        )
        decision, note, search, added, replacement = resolved
        if isinstance(decision, _Unproven):
            deferred.append((decision, list(pieces)))
            continue
        if isinstance(decision, Refused):
            errors.append(decision.message)
            continue
        if isinstance(decision, AlreadyApplied):
            continue
        eol = file_eol(pieces)
        if decision.site.partial is not None and decision.tier == "partial":
            pieces = apply_span(
                pieces, decision.site.partial, replacement, eol, consumed, ids,
            )
            strategies.append("exact")  # the legacy chain's "exact" is a raw-substring match
        else:
            if decision.tier != "exact":
                added = _carry_indent(pieces, decision.site, search, added)
            pieces = apply_whole(pieces, decision.site, hunk, added, eol, consumed, ids)
            strategies.append("exact" if decision.tier == "exact" else "whitespace_normalized")
        floor = _remap_floor(prior, pieces, floor)
        live = True
        if note:
            notes.append(note)
        after = render(pieces)

        later = [later.context_hint for later in hunks[index:] if later.context_hint]
        _damage(before, after, later, damaged)
    state.pieces = pieces
    state.floor = floor
    state.live = live
    if finalize and deferred:
        errors.extend(finish_state(state))
    from tools.patch_parser import _Plan

    return _Plan(render(pieces), errors, changes, notes, strategies)
