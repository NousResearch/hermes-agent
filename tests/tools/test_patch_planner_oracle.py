"""Unhinted V4A targeting oracle. No production imports.

``evaluate(files, patch)`` returns WRITE or REFUSE. A hint raises
``UnsupportedInput`` and is not a REFUSE. ``LIMITATIONS`` is exact.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

_BEGIN = re.compile(r"^\*\*\*\s*Begin\s+Patch\s*$")
_END = re.compile(r"^\*\*\*\s*End\s+Patch\s*$")
_UPDATE = re.compile(r"^\*\*\*\s*Update\s+File:\s*(.*)$")
_OTHER = re.compile(r"^\*\*\*\s*(?:Add|Delete|Move)\s+File:")
_WS = re.compile(r"\s+")
_TAIL = re.compile(r"[0-9A-Za-z_]")
_mutant: str | None = None

LIMITATIONS: tuple[str, ...] = (
    "Only update operations are supported; no hint or cursor is modeled.",
    "Files use LF or CRLF; mixed endings refuse.",
    "Original provenance is preserved for read-only context.",
    "Failures return original input and reason codes, with no I/O.",
)

class UnsupportedInput(Exception):
    """Input this oracle cannot represent. Not a contract REFUSE."""

class _ParseError(Exception):
    pass

class _Refuse(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code

@dataclass(frozen=True)
class _Line:
    content: str
    prov: tuple[int | None, ...]
    start: int
    end: int

@dataclass(frozen=True)
class Outcome:
    status: str
    files: Mapping[str, str]
    reason: str

def _norm(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", text))

def _fragment(search: str, candidate: str) -> bool:
    left, right = _norm(search).strip(), _norm(candidate).strip()
    longer, shorter = (right, left) if right.startswith(left) else (left, right)
    rest = longer[len(shorter):]
    return bool(left and right and left != right and longer.startswith(shorter) and _TAIL.match(rest))

def _raw_fragment(text: str, start: int, end: int, pattern: str, eol: str) -> bool:
    if _mutant == "fragment-wins" or not pattern or eol in pattern or start >= end:
        return False
    cut = text.rfind(eol, 0, start)
    ls = cut + len(eol) if cut >= 0 else 0
    le = text.find(eol, end)
    line = text[ls:len(text) if le < 0 else le]
    suffix = _norm(text[end:ls + len(line)])
    tail = not pattern[-1].isspace() and bool(suffix) and suffix[0] != " " and bool(_TAIL.match(suffix))
    return _fragment(pattern, line) or tail

def _eol(text: str) -> str:
    if "\r\n" in text and "\n" in text.replace("\r\n", ""):
        raise _Refuse("mixed-newlines")
    return "\r\n" if "\r\n" in text else "\n"

def _lines(text: str, prov: list[int | None], eol: str) -> list[_Line]:
    if not text:
        return [_Line("", (), 0, 0)]
    rows: list[_Line] = []
    i = 0
    el = len(eol)
    n = len(text)
    while i < n:
        j = text.find(eol, i)
        if j < 0:
            rows.append(_Line(text[i:], tuple(prov[i:]), i, n))
            break
        rows.append(_Line(text[i:j], tuple(prov[i:j]), i, j))
        i = j + el
    return rows

def _ok(prov: list[int | None], consumed: set[int], a: int, b: int) -> bool:
    return all(
        prov[i] is not None and prov[i] not in consumed for i in range(a, b)
    )

def _eligible(
    lines: list[_Line],
    start: int,
    search: tuple[tuple[str, str], ...],
    prov: list[int | None],
    consumed: set[int],
) -> bool:
    for offset, (prefix, content) in enumerate(search):
        if prefix != "-":
            continue
        if content == "":
            return False
        line = lines[start + offset]
        if not _ok(prov, consumed, line.start, line.end):
            return False
    return True

def _choose(
    text: str,
    prov: list[int | None],
    consumed: set[int],
    search: tuple[tuple[str, str], ...],
    eol: str,
) -> list[tuple]:
    lines = _lines(text, prov, eol)
    wanted = [c for _, c in search]
    n = len(wanted)
    exact: list[tuple] = []
    norm: list[tuple] = []
    if n <= len(lines):
        norms = [_norm(w) for w in wanted]
        for i in range(len(lines) - n + 1):
            if not _eligible(lines, i, search, prov, consumed):
                continue
            block = [lines[i + j].content for j in range(n)]
            if block == wanted:
                exact.append((i, "line"))
            elif [_norm(c) for c in block] == norms or (
                _mutant == "fragment-wins"
                and any(_fragment(w, c) for w, c in zip(wanted, block))
            ):
                norm.append((i, "line"))
    # Strictest nonempty eligible tier. Lower tiers cannot override it.
    if exact:
        return exact
    if norm:
        return norm
    pattern = eol.join(wanted)
    raw: list[tuple] = []
    if not pattern:
        return raw
    start = 0
    while True:
        j = text.find(pattern, start)
        if j < 0:
            break
        end = j + len(pattern)
        pos = j
        eligible = True
        for prefix, content in search:
            stop = pos + len(content)
            if prefix == "-" and (
                not _ok(prov, consumed, pos, stop)
                or _raw_fragment(text, pos, stop, content, eol)
            ):
                eligible = False
            pos = stop + len(eol)
        if not _raw_fragment(text, j, end, pattern, eol) and eligible:
            raw.append((j, end, "raw"))
        start = j + 1
    return raw

def _line_repl(
    lines: list[_Line],
    start_i: int,
    search: tuple[tuple[str, str], ...],
    replace: tuple[tuple[str, str], ...],
    eol: str,
) -> tuple[int, int, str, list[int | None]]:
    matched = lines[start_i:start_i + len(search)]
    parts_t: list[str] = []
    parts_p: list[list[int | None]] = []
    si = 0
    for prefix, content in replace:
        if parts_t:
            parts_t.append(eol)
            parts_p.append([None] * len(eol))
        if prefix == "+":
            parts_t.append(content)
            parts_p.append([None] * len(content))
            continue
        while search[si][0] == "-":
            si += 1
        line = matched[si]
        parts_t.append(line.content)
        parts_p.append(list(line.prov))
        si += 1
    new_text = "".join(parts_t)
    new_prov = [p for part in parts_p for p in part]
    return matched[0].start, matched[-1].end, new_text, new_prov

def _rewrite(
    text: str,
    prov: list[int | None],
    consumed: set[int],
    start: int,
    end: int,
    new_text: str,
    new_prov: list[int | None],
) -> tuple[str, list[int | None]]:
    copied = {p for p in new_prov if p is not None}
    for i in range(start, end):
        p = prov[i]
        if p is not None and p not in copied:
            consumed.add(p)
    out_p = prov[:start] + new_prov + prov[end:]
    out_t = text[:start] + new_text + text[end:]
    if len(out_t) != len(out_p):
        raise _Refuse("parse-error")
    return out_t, out_p

def _append(
    text: str,
    prov: list[int | None],
    replace: tuple[tuple[str, str], ...],
    eol: str,
) -> tuple[str, list[int | None]]:
    insert = eol.join(c for _, c in replace)
    if not insert.endswith(eol):
        insert += eol
    if text and not text.endswith(eol):
        text += eol
        prov = [*prov, *([None] * len(eol))]
    added: list[int | None] = [None] * len(insert)
    if not text:
        return insert, added
    return text + insert, [*prov, *added]

def _plan_hunk(
    text: str,
    prov: list[int | None],
    consumed: set[int],
    hunk: tuple[tuple[str, str], ...],
    eol: str,
) -> tuple[str, list[int | None]]:
    search = tuple(x for x in hunk if x[0] != "+")
    replace = tuple(x for x in hunk if x[0] != "-")
    if not search:
        return _append(text, prov, replace, eol)
    sites = _choose(text, prov, consumed, search, eol)
    if tuple(c for _, c in search) == tuple(c for _, c in replace):
        if len(sites) != 1:
            raise _Refuse("ambiguous" if sites else "not-found")
        return text, prov
    if any(p == "-" and c == "" for p, c in search):
        raise _Refuse("empty-search")
    if len(sites) != 1:
        raise _Refuse("ambiguous" if len(sites) > 1 else "not-found")
    site = sites[0]
    if site[-1] == "line":
        lines = _lines(text, prov, eol)
        start, end, new_text, new_prov = _line_repl(
            lines, site[0], search, replace, eol
        )
    else:
        pos = site[0]
        fragments = []
        for _, content in search:
            stop = pos + len(content)
            fragments.append(_Line(content, tuple(prov[pos:stop]), pos, stop))
            pos = stop + len(eol)
        start, end, new_text, new_prov = _line_repl(fragments, 0, search, replace, eol)
    return _rewrite(text, prov, consumed, start, end, new_text, new_prov)

def _plan(text: str, prov: list[int | None], consumed: set[int], hunks: tuple, eol: str) -> tuple[str, list[int | None]]:
    for hunk in hunks:
        text, prov = _plan_hunk(text, prov, consumed, hunk, eol)
    return text, prov

def _parse(patch: str) -> list[tuple[str, tuple[tuple[tuple[str, str], ...], ...]]]:
    lines = [ln.removesuffix("\r") for ln in patch.split("\n")]
    try:
        start = next(i for i, ln in enumerate(lines) if _BEGIN.match(ln))
        end = next(
            i for i, ln in enumerate(lines) if i > start and _END.match(ln)
        )
    except StopIteration:
        raise _ParseError from None
    ops: list[tuple[str, tuple]] = []
    path: str | None = None
    hunk: list[tuple[str, str]] | None = None
    hunks: list[tuple[tuple[str, str], ...]] = []

    def flush_hunk() -> None:
        nonlocal hunk
        if hunk:
            hunks.append(tuple(hunk))
        hunk = None

    def flush_op() -> None:
        nonlocal path, hunks
        flush_hunk()
        if path is None:
            return
        if not hunks:
            raise _ParseError
        ops.append((path, tuple(hunks)))
        path, hunks = None, []

    for line in lines[start + 1:end]:
        if _OTHER.match(line):
            raise _Refuse("unsupported-op")
        m = _UPDATE.match(line)
        if m:
            flush_op()
            path = m.group(1).strip()
            if not path:
                raise _ParseError
            continue
        if line.startswith("@@"):
            if path is None:
                raise _ParseError
            if line.strip() != "@@" and not re.fullmatch(r"@@\s*@@", line):
                raise UnsupportedInput("hint")
            flush_hunk()
            hunk = []
            continue
        if path is None or not line or line.startswith("\\"):
            continue
        if hunk is None:
            hunk = []
        if line[0] in "+- ":
            hunk.append((line[0], line[1:]))
        else:
            hunk.append((" ", line))
    flush_op()
    if not ops:
        raise _ParseError
    return ops

def evaluate(files: Mapping[str, str], patch: str) -> Outcome:
    """Plan an unhinted V4A update. Does not mutate ``files``."""
    if not isinstance(patch, str):
        raise TypeError("patch must be str")
    original = dict(files)
    if any(
        not isinstance(k, str) or not isinstance(v, str)
        for k, v in original.items()
    ):
        raise TypeError("files must map str to str")
    try:
        ops = _parse(patch)
    except UnsupportedInput:
        raise
    except _Refuse as exc:
        return Outcome("REFUSE", MappingProxyType(dict(original)), exc.code)
    except _ParseError:
        return Outcome("REFUSE", MappingProxyType(dict(original)), "parse-error")
    planned = dict(original)
    live: dict[str, tuple] = {}
    try:
        for path, hunks in ops:
            if path not in planned:
                raise _Refuse("missing-file")
            if path not in live:
                base = planned[path]
                live[path] = (base, list(range(len(base))), set(), _eol(base))
            text, prov, consumed, eol = live[path]
            text, prov = _plan(text, prov, consumed, hunks, eol)
            planned[path], live[path] = text, (text, prov, consumed, eol)
    except _Refuse as exc:
        return Outcome("REFUSE", MappingProxyType(dict(original)), exc.code)
    return Outcome("WRITE", MappingProxyType(planned), "")

# --- self-tests ---

import inspect

import pytest

def _up(body: str, path: str = "f.py") -> str:
    return (
        "*** Begin Patch\n*** Update File: "
        + path
        + "\n"
        + body
        + "\n*** End Patch\n"
    )

def _unchanged(files: dict[str, str], before: dict[str, str], ids: dict[str, int]) -> None:
    assert files == before
    assert {k: id(v) for k, v in files.items()} == ids

def test_interface_cannot_represent_hints() -> None:
    assert list(inspect.signature(evaluate).parameters) == ["files", "patch"]
    assert "hint" not in Outcome.__dataclass_fields__
    src = {"f.py": "a\n"}
    before, ids = dict(src), {k: id(v) for k, v in src.items()}
    with pytest.raises(UnsupportedInput, match="^hint$"):
        evaluate(src, _up("@@ secret hint @@\n-a\n+b"))
    _unchanged(src, before, ids)

def test_rule1_inserted_bytes_are_not_reedited() -> None:
    src = {"f.py": "a = 1\n"}
    body = "@@\n-a = 1\n+b = 2\n@@\n-b = 2\n+b = 3"
    out = evaluate(src, _up(body))
    assert out.status == "REFUSE"
    assert out.reason == "not-found"
    assert out.files["f.py"] == "a = 1\n"

def test_normalized_context_is_preserved_byte_for_byte() -> None:
    source = "a\t=\t1\nb = 2\n"
    result = evaluate({"f.py": source}, _up("@@\n a = 1\n-b = 2\n+b = 9"))
    assert result.status == "WRITE"
    assert result.files["f.py"] == "a\t=\t1\nb = 9\n"


def test_inserted_context_can_anchor_unique_partial_edit() -> None:
    source = "a = 1\nc = 3 tail\n"
    body = "@@\n a = 1\n+b = 2\n@@\n b = 2\n-c = 3\n+c = 4"
    result = evaluate({"f.py": source}, _up(body))
    assert result.status == "WRITE"
    assert result.files["f.py"] == "a = 1\nb = 2\nc = 4 tail\n"


def test_rule1_written_context_may_anchor() -> None:
    body = "@@\n a = 1\n+b = 2\n@@\n b = 2\n-c = 3\n+c = 4"
    out = evaluate({"f.py": "a = 1\nc = 3\n"}, _up(body))
    assert out.status == "WRITE"
    assert out.files["f.py"] == "a = 1\nb = 2\nc = 4\n"

def test_rule1_context_line_stays_editable() -> None:
    body = "@@\n a = 1\n-b = 2\n+b = 9\n@@\n-a = 1\n+a = 0"
    out = evaluate({"f.py": "a = 1\nb = 2\n"}, _up(body))
    assert out.status == "WRITE"
    assert out.files["f.py"] == "a = 0\nb = 9\n"

def test_rule2_exact_beats_fragment_and_keeps_decoy() -> None:
    src = "value = 10\nvalue = 1\nfooter = old\n"
    body = "@@\n-value = 1\n+CHANGED\n@@\n-footer = old\n+footer = new"
    out = evaluate({"f.py": src}, _up(body))
    assert out.status == "WRITE"
    assert out.files["f.py"] == "value = 10\nCHANGED\nfooter = new\n"

def test_rule2_exact_beats_normalized_and_keeps_tabs() -> None:
    src = "value\t=\t1\nvalue = 1\n"
    out = evaluate({"f.py": src}, _up("@@\n-value = 1\n+CHANGED"))
    assert out.files["f.py"] == "value\t=\t1\nCHANGED\n"

@pytest.mark.parametrize(
    "line",
    ["value\t=\t1", "value  =  1", "value\u00a0=\u00a01", "value = \uff11"],
)
def test_rule2_normalized_whole_line(line: str) -> None:
    src = "value = 10\n" + line + "\nkeep\n"
    out = evaluate({"f.py": src}, _up("@@\n-value = 1\n+CHANGED"))
    assert out.status == "WRITE"
    assert out.files["f.py"] == "value = 10\nCHANGED\nkeep\n"

def test_rule2_nfkc_angstrom_is_whole_line() -> None:
    src = "len = \u212b\nkeep\n"
    out = evaluate({"f.py": src}, _up("@@\n-len = \u00c5\n+len = 1"))
    assert out.status == "WRITE"
    assert out.files["f.py"] == "len = 1\nkeep\n"

@pytest.mark.parametrize(
    "line",
    ["value = 10", "value\t=\t10", "value  =  10", "value = \uff11\uff10"],
)
def test_rule2_fragment_decoy_is_not_a_site(line: str) -> None:
    out = evaluate({"f.py": line + "\n"}, _up("@@\n-value = 1\n+CHANGED"))
    assert out.status == "REFUSE"
    assert out.reason == "not-found"
    assert "value" not in out.reason

@pytest.mark.parametrize("tail", ["0", "\uff10"])
def test_multiline_context_never_turns_identifier_fragment_into_site(tail):
    source = "keep\nvalue = 1" + tail + "\n"
    result = evaluate({"f.py": source}, _up("@@\n keep\n-value = 1\n+CHANGED"))
    assert result.status == "REFUSE"
    assert result.files["f.py"] == source


def test_rule2_unique_partial_beside_fragment_decoy() -> None:
    src = "value = 10\ncall(value = 1)\n"
    out = evaluate({"f.py": src}, _up("@@\n-value = 1\n+CHANGED"))
    assert out.status == "WRITE"
    assert out.files["f.py"] == "value = 10\ncall(CHANGED)\n"

def test_rule3_identical_sites_refuse_without_order() -> None:
    src = "value = 1\nvalue = 1\nfooter = old\n"
    body = "@@\n-value = 1\n+CHANGED\n@@\n-footer = old\n+footer = new"
    files = {"f.py": src}
    before, ids = dict(files), {k: id(v) for k, v in files.items()}
    out = evaluate(files, _up(body))
    assert out.status == "REFUSE"
    assert out.reason == "ambiguous"
    assert out.files["f.py"] == src
    _unchanged(files, before, ids)

def test_rule3_overlapping_windows_are_ambiguous() -> None:
    out = evaluate({"f.py": "a\na\na\n"}, _up("@@\n-a\n-a\n+b"))
    assert out.status == "REFUSE"
    assert out.reason == "ambiguous"

def test_rule4_replacement_already_present_is_not_success() -> None:
    src = 'keep = hostish_d\nhost = "a"\nhost = "b"\n'
    body = '@@\n-host = "a"\n+host = "c"\n@@\n-host = "a"\n+keep = hostish_d'
    out = evaluate({"f.py": src}, _up(body))
    assert out.status == "REFUSE"
    assert out.reason == "not-found"
    assert out.files["f.py"] == src

def test_rule4_remaining_whole_line_copy_is_selected() -> None:
    src = 'alpha\nhost = "a"\nbeta\nhost = "a"\n'
    body = (
        '@@\n alpha\n-host = "a"\n+host = "c"\n'
        '@@\n-host = "a"\n+host = "d"'
    )
    out = evaluate({"f.py": src}, _up(body))
    assert out.status == "WRITE"
    assert out.files["f.py"] == 'alpha\nhost = "c"\nbeta\nhost = "d"\n'

def test_rule4_remaining_normalized_copy_is_selected() -> None:
    src = "alpha\nvalue = 1\nvalue\t=\t1\n"
    body = "@@\n alpha\n-value = 1\n+value = 9\n@@\n-value = 1\n+value = 8"
    out = evaluate({"f.py": src}, _up(body))
    assert out.status == "WRITE"
    assert out.files["f.py"] == "alpha\nvalue = 9\nvalue = 8\n"

def test_rule5_cross_file_refusal_is_atomic_and_does_not_mutate() -> None:
    files = {"a.py": "old\n", "b.py": "other\n"}
    before, ids = dict(files), {k: id(v) for k, v in files.items()}
    patch = (
        "*** Begin Patch\n"
        "*** Update File: a.py\n@@\n-old\n+new\n"
        "*** Update File: b.py\n@@\n-absent\n+replacement\n"
        "*** End Patch\n"
    )
    out = evaluate(files, patch)
    assert out.status == "REFUSE"
    assert out.reason == "not-found"
    assert "absent" not in out.reason
    assert out.files == before
    assert out.files is not files
    _unchanged(files, before, ids)

def test_rule5_success_does_not_mutate_input() -> None:
    files = {"f.py": "a = old\n", "g.py": "stay\n"}
    before, ids = dict(files), {k: id(v) for k, v in files.items()}
    out = evaluate(files, _up("@@\n-a = old\n+a = new"))
    assert out.status == "WRITE"
    assert out.reason == ""
    assert out.files["f.py"] == "a = new\n"
    assert out.files["g.py"] == "stay\n"
    _unchanged(files, before, ids)

def test_rule5_missing_file_writes_nothing() -> None:
    files = {"a.py": "old\n"}
    patch = (
        "*** Begin Patch\n"
        "*** Update File: a.py\n@@\n-old\n+new\n"
        "*** Update File: missing.py\n@@\n-old\n+new\n"
        "*** End Patch\n"
    )
    out = evaluate(files, patch)
    assert out.status == "REFUSE"
    assert out.reason == "missing-file"
    assert files == {"a.py": "old\n"}

def test_rule6_unique_partial_preserves_surrounding_bytes() -> None:
    src = "call(value = 1); tail\n"
    out = evaluate({"f.py": src}, _up("@@\n-value = 1\n+CHANGED"))
    assert out.files["f.py"] == "call(CHANGED); tail\n"
    spaced = evaluate({"f.py": "prefix AAA\n"}, _up("@@\n-prefix \n+prefixZ"))
    assert spaced.files["f.py"] == "prefixZAAA\n"

def test_rule6_addition_only_eof_preserves_trailing_blank() -> None:
    out = evaluate({"f.py": "first\n\n"}, _up("@@\n+last"))
    assert out.status == "WRITE"
    assert out.files["f.py"] == "first\n\nlast\n"
    bare = evaluate({"f.py": "first"}, _up("@@\n+last"))
    assert bare.files["f.py"] == "first\nlast\n"

def test_rule6_empty_anchor_and_ordered_hunks() -> None:
    anchor = evaluate({"f.py": "a\n\nb\n"}, _up("@@\n a\n@@\n \n@@\n-b\n+B"))
    assert anchor.status == "WRITE"
    assert anchor.files["f.py"] == "a\n\nB\n"
    ordered = evaluate(
        {"f.py": "a = old\nb = old\n"},
        _up("@@\n-a = old\n+a = new\n@@\n-b = old\n+b = new"),
    )
    assert ordered.files["f.py"] == "a = new\nb = new\n"

def test_rule6_per_file_eol_and_crlf_insertion() -> None:
    patch = (
        "*** Begin Patch\n"
        "*** Update File: a.py\n@@\n-old\n+new\n"
        "*** Update File: b.py\n@@\n-old\n+new\n@@\n+last\n"
        "*** End Patch\n"
    )
    files = {"a.py": "old\n", "b.py": "old\r\n"}
    out = evaluate(files, patch)
    assert out.status == "WRITE"
    assert out.files["a.py"] == "new\n"
    assert out.files["b.py"] == "new\r\nlast\r\n"
    mixed = evaluate({"a.py": "old\n", "b.py": "old\r\nbare\n"}, patch)
    assert mixed.status == "REFUSE"
    assert mixed.reason == "mixed-newlines"
    assert mixed.files["a.py"] == "old\n"

def test_ambiguous_anchor_does_not_cursor_pick() -> None:
    src = "a\na\nb\n"
    out = evaluate({"f.py": src}, _up("@@\n a\n@@\n-b\n+B"))
    assert out.status == "REFUSE"
    assert out.reason == "ambiguous"
    assert out.files["f.py"] == src

def test_repeated_update_sections_share_planned_content() -> None:
    patch = (
        "*** Begin Patch\n"
        "*** Update File: f.py\n@@\n-a = 1\n+a = 2\n"
        "*** Update File: f.py\n@@\n-b = 1\n+b = 2\n"
        "*** End Patch\n"
    )
    out = evaluate({"f.py": "a = 1\nb = 1\n"}, patch)
    assert out.files["f.py"] == "a = 2\nb = 2\n"

def test_pure_deletion_leaves_the_trailing_separator() -> None:
    out = evaluate({"f.py": "value = 1\n"}, _up("@@\n-value = 1"))
    assert out.status == "WRITE"
    assert out.files["f.py"] == "\n"

def test_empty_deletion_is_empty_search() -> None:
    out = evaluate({"f.py": "a\n\nb\n"}, _up("@@\n a\n-\n b"))
    assert out.status == "REFUSE"
    assert out.reason == "empty-search"
    assert out.files["f.py"] == "a\n\nb\n"

def test_unsupported_op_and_parse_error_write_nothing() -> None:
    files = {"f.py": "a\n"}
    add = "*** Begin Patch\n*** Add File: f.py\n+a\n*** End Patch\n"
    out = evaluate(files, add)
    assert out.status == "REFUSE"
    assert out.reason == "unsupported-op"
    assert files == {"f.py": "a\n"}
    bad = evaluate(files, "not a patch")
    assert bad.reason == "parse-error"
    assert bad.files["f.py"] == "a\n"

def _ascii_accepts(text: str, pattern: str) -> bool:
    """Deliberate ASCII-boundary mutant. Fullwidth zero is not a word char."""
    start = text.find(pattern)
    if start < 0:
        return False
    end = start + len(pattern)

    def word(i: int) -> bool:
        ch = text[i]
        return ch.isascii() and (ch.isalnum() or ch == "_")

    left = start and word(start - 1) and word(start)
    right = end < len(text) and word(end - 1) and word(end)
    return not (left or right)

def _fragment_wins(files: dict[str, str], patch: str) -> Outcome:
    global _mutant
    _mutant = "fragment-wins"
    try:
        return evaluate(files, patch)
    finally:
        _mutant = None

def test_fragment_uses_canonical_normalize() -> None:
    assert _norm("value\u00a0=\u00a01") == "value = 1"
    assert _norm("value\u3000=\u30001") == "value = 1"
    assert _norm("value = 1\uff10") == "value = 10"
    assert _norm("value = \uff11\uff10") == "value = 10"
    assert _fragment("value = 1", "value = 10")
    assert _fragment("value = 1", "value\u00a0=\u00a010")
    assert _fragment("value = 1", "value\u3000=\u300010")
    assert _fragment("value = 1", "value = 1\uff10")
    assert _fragment("value = 1", "value = \uff11\uff10")
    assert not _fragment("value = 1", "value = 1")
    assert not _fragment("value = 1", "value\t=\t1")
    assert not _fragment("value = 1", "value\u00a0=\u00a01")
    assert not _fragment("value = 1", "call(value = 1)")
    assert not _fragment("prefix ", "prefix AAA")

@pytest.mark.parametrize(
    "line",
    [
        "value = 10",
        "value\u00a0=\u00a010",
        "value\u3000=\u300010",
        "value = \uff11\uff10",
        "value = 1\uff10",
    ],
)
def test_normalized_fragment_spellings_are_refused(line: str) -> None:
    src = line + "\n"
    out = evaluate({"f.py": src}, _up("@@\n-value = 1\n+CHANGED"))
    assert out.status == "REFUSE"
    assert out.reason == "not-found"
    assert out.files["f.py"] == src
    assert "value" not in out.reason
    assert "\uff10" not in out.reason
    assert "\u00a0" not in out.reason

def test_nbsp_and_ideographic_space_are_boundaries_not_joins() -> None:
    # Hand-computed: "value = 1" is 9 chars; the following space stays.
    nbsp = evaluate({"f.py": "value = 1\u00a0tail\n"}, _up("@@\n-value = 1\n+CHANGED"))
    assert nbsp.status == "WRITE"
    assert nbsp.files["f.py"] == "CHANGED\u00a0tail\n"
    ideo = evaluate({"f.py": "value = 1\u3000tail\n"}, _up("@@\n-value = 1\n+CHANGED"))
    assert ideo.status == "WRITE"
    assert ideo.files["f.py"] == "CHANGED\u3000tail\n"

def test_unique_partial_beside_unicode_fragment() -> None:
    src = "value\u00a0=\u00a010\ncall(value = 1)\n"
    out = evaluate({"f.py": src}, _up("@@\n-value = 1\n+CHANGED"))
    assert out.status == "WRITE"
    assert out.files["f.py"] == "value\u00a0=\u00a010\ncall(CHANGED)\n"
    ff10 = "value = 1\uff10\ncall(value = 1)\n"
    kept = evaluate({"f.py": ff10}, _up("@@\n-value = 1\n+CHANGED"))
    assert kept.status == "WRITE"
    assert kept.files["f.py"] == "value = 1\uff10\ncall(CHANGED)\n"

def _two(first: str, second: str, path: str = "f.py") -> str:
    mark = "*" * 3
    head = mark + " Begin Patch\n" + mark + " Update File: " + path + "\n"
    mid = "\n" + mark + " Update File: " + path + "\n"
    tail = "\n" + mark + " End Patch\n"
    return head + first + mid + second + tail

def test_repeated_sections_do_not_reedit_inserted_source() -> None:
    # Section 2 names only bytes section 1 inserted. No original provenance,
    # so the patch refuses and the 6-byte source stays.
    src = "a = 1\n"
    out = evaluate({"f.py": src}, _two("@@\n-a = 1\n+b = 2", "@@\n-b = 2\n+b = 3"))
    assert out.status == "REFUSE"
    assert out.reason == "not-found"
    assert out.files["f.py"] == src
    assert "b = 2" not in out.reason
    again = evaluate({"f.py": src}, _two("@@\n+b = 2", "@@\n-b = 2\n+b = 3"))
    assert again.status == "REFUSE"
    assert again.reason == "not-found"
    assert again.files["f.py"] == src

def test_repeated_sections_keep_untouched_original_bytes() -> None:
    # "a = 1\\n" is bytes [0, 6). "c = 3\\n" is [6, 12) and stays eligible.
    out = evaluate(
        {"f.py": "a = 1\nc = 3\n"},
        _two("@@\n-a = 1\n+b = 2", "@@\n-c = 3\n+c = 4"),
    )
    assert out.status == "WRITE"
    assert out.files["f.py"] == "b = 2\nc = 4\n"

def test_refusal_reason_omits_source() -> None:
    marker = "marker-keep-line"
    src = "keep = " + marker + "\nvalue = 1\uff10\n"
    out = evaluate({"f.py": src}, _up("@@\n-value = 1\n+CHANGED"))
    assert out.status == "REFUSE"
    assert out.reason == "not-found"
    assert marker not in out.reason
    assert marker not in out.status
    assert out.files["f.py"] == src

def test_ascii_boundary_mutant_is_caught() -> None:
    text = "value = 1\uff10\n"
    assert _ascii_accepts(text, "value = 1")
    assert not _ascii_accepts("value = 10\n", "value = 1")
    out = evaluate({"f.py": text}, _up("@@\n-value = 1\n+CHANGED"))
    assert out.status == "REFUSE"
    assert out.reason == "not-found"
    assert out.files["f.py"] == text
    assert "\uff10" not in out.reason

def test_mutant_fragment_wins_is_caught() -> None:
    body = _up("@@\n-value = 1\n+CHANGED")
    samples = (
        "value\u00a0=\u00a010\n",
        "value\u3000=\u300010\n",
        "value = 1\uff10\n",
        "value = \uff11\uff10\n",
    )
    for src in samples:
        real = evaluate({"f.py": src}, body)
        bad = _fragment_wins({"f.py": src}, body)
        assert real.status == "REFUSE"
        assert real.reason == "not-found"
        assert real.files["f.py"] == src
        assert src.strip() not in real.reason
        assert bad.status == "WRITE"
        assert bad.files["f.py"] == "CHANGED\n"
        assert bad.files != real.files
    assert _mutant is None
