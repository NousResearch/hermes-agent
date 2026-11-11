"""Result dataclasses and pure text helpers shared by ``tools.file_operations``
and its search/lint mixins. Leaf module (imports nothing from ``tools`` at module
scope) so nothing cycles. The ``to_dict`` output of every class here IS tool
behavior — key names, order and omission rules are pinned by tests and read by
the model.
"""

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple


@dataclass
class ReadResult:
    """Result from reading a file."""
    content: str = ""
    total_lines: int = 0
    file_size: int = 0
    truncated: bool = False
    truncated_lines: Optional[bool] = None
    hint: Optional[str] = None
    is_binary: bool = False
    is_image: bool = False
    base64_content: Optional[str] = None
    mime_type: Optional[str] = None
    dimensions: Optional[str] = None  # For images: "WIDTHxHEIGHT"
    error: Optional[str] = None
    #: True only when the path is genuinely absent. An error with this False is a read that
    #: FAILED (transport down, no byte transport installed); callers deciding whether a path
    #: is free must not read that as "absent". See patch_parser._apply_add.
    not_found: bool = False
    similar_files: list[str] = field(default_factory=list)
    _snapshot: Optional[tuple] = None
    # sha256 of the exact on-disk bytes ``read_file_raw`` decoded (BOM included).
    _content_sha256: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and v is not None and v != []}


@dataclass
class WriteResult:
    """Result from writing a file."""
    bytes_written: int = 0
    dirs_created: bool = False
    # True when the on-disk sha256 matched the intended content; None when the
    # backend couldn't verify (no sha256sum). A mismatch is a hard error, never a flag.
    verified: Optional[bool] = None
    _content_sha256: Optional[str] = None
    lint: Optional[dict[str, Any]] = None
    # LSP semantic diagnostics, kept separate from ``lint`` (syntax) so the model
    # reads the two as independent signals. None when LSP is off/inapplicable.
    lsp_diagnostics: Optional[str] = None
    error: Optional[str] = None
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and v is not None}


@dataclass
class PatchResult:
    """Result from patching a file."""
    success: bool = False
    diff: str = ""
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    lint: Optional[dict[str, Any]] = None
    lsp_diagnostics: Optional[str] = None  # see WriteResult.lsp_diagnostics
    error: Optional[str] = None
    # Success-shaped no-op: the edit was already present, nothing written; ``note`` says why.
    no_change: bool = False
    note: Optional[str] = None
    # One ``(path, read_sha256, written_sha256)`` per file write, in order: the sha256 of the
    # bytes the edit read ("" when it created the file, None when unknown) and of the bytes it
    # wrote (None when unknown). patch_tool carries a task's full-file baseline across them.
    _writes: list[tuple[str, Optional[str], Optional[str]]] = field(default_factory=list)

    # Emission order is part of the output contract.
    _DICT_FIELDS: ClassVar[tuple] = (
        "diff", "files_modified", "files_created", "files_deleted",
        "lint", "lsp_diagnostics", "error",
    )

    def to_dict(self) -> dict:
        result: dict[str, Any] = {"success": self.success}
        if self.no_change:
            result["no_change"] = True
        if self.note:
            result["note"] = self.note
        for key in self._DICT_FIELDS:
            value = getattr(self, key)
            if value:
                result[key] = value
        return result


@dataclass
class SearchMatch:
    """A single search match."""
    path: str
    line_number: int
    content: str
    mtime: float = 0.0  # Modification time for sorting


@dataclass
class SearchResult:
    """Result from searching."""
    matches: list[SearchMatch] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    truncated: bool = False
    limit_reason: Optional[str] = None
    warning: Optional[str] = None
    error: Optional[str] = None
    # Unreadable paths behind the "Skipped N unreadable path(s)" sentence in ``warning``,
    # plus the diagnostics dropped past the runner's cap: a multi-root merge recomposes
    # one sentence from this data instead of parsing the text. Never serialized.
    skipped_paths: list[str] = field(default_factory=list)
    skipped_omitted: int = 0

    # Below this many matches the verbose array is already compact enough that
    # a path-grouping header would cost more tokens than it saves.
    _DENSIFY_MIN_MATCHES: ClassVar[int] = 5

    def _densify_matches(self) -> Optional[str]:
        """Lossless path-grouped text block: path once, then ``  <line>: <content>``
        rows. Relies on rg/grep emitting a file's hits consecutively. None when
        too few matches to be worth it."""
        if len(self.matches) < self._DENSIFY_MIN_MATCHES:
            return None
        lines: list[str] = []
        current_path: Optional[str] = None
        for m in self.matches:
            if m.path != current_path:
                lines.append(m.path)
                current_path = m.path
            # rstrip only: leading indentation is meaningful code and kept verbatim.
            lines.append(f"  {m.line_number}: {m.content.rstrip()}")
        return "\n".join(lines)

    def to_dict(self, densify: bool = False) -> dict:
        result: dict[str, object] = {"total_count": self.total_count}
        if self.matches:
            dense = self._densify_matches() if densify else None
            if dense is not None:
                # Self-describing so the model never guesses the block's shape.
                result["matches_format"] = (
                    "path-grouped: each file path on its own line, followed by "
                    "indented '<line>: <content>' rows for matches in that file"
                )
                result["matches_text"] = dense
            else:
                result["matches"] = [
                    {"path": m.path, "line": m.line_number, "content": m.content} for m in self.matches
                ]
        if self.files:
            result["files"] = self.files
        if self.counts:
            result["counts"] = self.counts
        if self.truncated:
            result["truncated"] = True
            result["total_count_is_lower_bound"] = True
        for key in ("limit_reason", "warning", "error"):
            value = getattr(self, key)
            if value:
                result[key] = value
        return result


@dataclass
class LintResult:
    """Result from linting a file."""
    success: bool = True
    skipped: bool = False
    output: str = ""
    message: str = ""

    def to_dict(self) -> dict:
        if self.skipped:
            return {"status": "skipped", "message": self.message}
        result = {"status": "ok" if self.success else "error", "output": self.output}
        if self.message:
            result["message"] = self.message
        return result


@dataclass
class ExecuteResult:
    """Result from executing a shell command."""
    stdout: str = ""
    exit_code: int = 0
    # Set when the backend's own ``builtin cd -- <cwd> || exit 126`` wrapper failed:
    # the command never ran, so the verdict is about the working directory, not
    # the requested path (callers must neither cache it nor blame the path).
    cwd_error: str = ""


# ---------------------------------------------------------------------------
# Pure text helpers (no I/O)
# ---------------------------------------------------------------------------

_OSC_SEQUENCE_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_FENCE_MARKER_RE = re.compile(r"'?\x07?__HERMES_FENCE_[A-Za-z0-9]+__\x07?'?")


_CONFLICT_OPEN = re.compile(r"^\s*\d+\|<<<<<<< ", re.MULTILINE)
_CONFLICT_CLOSE = re.compile(r"^\s*\d+\|>>>>>>> ", re.MULTILINE)


def count_conflict_blocks(formatted_content: str) -> int:
    """Unresolved git merge-conflict blocks in a ``LINE|CONTENT`` read; 0 when the page has no
    balanced ``<<<<<<< `` / ``>>>>>>> `` pair (a lone marker in prose or a test fixture is not a
    conflict). Reported on read so the model resolves the conflict instead of editing around it."""
    opens = len(_CONFLICT_OPEN.findall(formatted_content))
    return min(opens, len(_CONFLICT_CLOSE.findall(formatted_content))) if opens else 0


def _strip_terminal_fence_leaks(text: str) -> str:
    """Strip leaked terminal fence wrappers (OSC sequences, fence markers) from
    command output; drops lines that were nothing but wrapper."""
    if not text:
        return text
    cleaned_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        had_terminal_wrapper = "__HERMES_FENCE_" in line or "\x1b]" in line
        cleaned = _FENCE_MARKER_RE.sub("", _OSC_SEQUENCE_RE.sub("", line)).replace("\x07", "")
        if had_terminal_wrapper and cleaned.strip("'\r\n\t ") == "":
            continue
        cleaned_lines.append(cleaned)
    return "".join(cleaned_lines)


def _detect_line_ending(sample: str) -> Optional[str]:
    """Dominant line ending of ``sample`` (``\\r\\n`` if any CRLF in the first 4KB,
    else ``\\n``), or None for empty/single-line content. Preserves a file's
    endings across write_file/patch: bare-LF tool args would otherwise silently
    normalize CRLF files, and patch would produce mixed endings."""
    head = sample[:4096] if sample else ""
    if "\r\n" in head:
        return "\r\n"
    if "\n" in head:
        return "\n"
    return None


def _normalize_line_endings(text: str, target: str) -> str:
    """Convert every line ending (CRLF, lone CR, LF) in ``text`` to ``target``.
    Idempotent. Collapses to LF first — separate replacements would
    double-convert CRLF → LFLF."""
    lf_normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if target == "\n":
        return lf_normalized
    if target == "\r\n":
        return lf_normalized.replace("\n", "\r\n")
    return text


# UTF-8 BOM (EF BB BF == U+FEFF), prepended by some Windows editors. Stripped on
# read so the model never sees a phantom first character (and patch's first-line
# match works), restored on write when the on-disk file had one.
_UTF8_BOM = "\ufeff"


def _strip_bom(text: str) -> tuple[str, bool]:
    """Return (text-without-leading-BOM, had_bom). Only a leading BOM is
    stripped; mid-content U+FEFF is legitimate data."""
    if _has_bom(text):
        return text[len(_UTF8_BOM):], True
    return text, False


def _has_bom(text: Optional[str]) -> bool:
    """True if ``text`` begins with a UTF-8 BOM."""
    return bool(text) and text.startswith(_UTF8_BOM)


# ---------------------------------------------------------------------------
# Bounded unified diff
# ---------------------------------------------------------------------------

# Port of MiniMax-AI/minimax-code#284.
# difflib's matcher costs roughly (changed lines) x (lines between the first and last
# change). A patch that touches every other line of a 20k-line file measured 10 s of
# CPU after the file was already written. This cap admits a full rewrite of any file up
# to ~4k lines and ~1-2 s of diff work on larger ones; past it the edit still lands and
# the diff becomes a one-line notice.
_MAX_DIFF_WORK = 40_000_000


def _lis(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Longest chain of ``(i, j)`` pairs (already in ``i`` order) with strictly increasing ``j``."""
    from bisect import bisect_left
    tails: list[int] = []
    tail_at: list[int] = []
    prev = [-1] * len(pairs)
    for n, (_i, j) in enumerate(pairs):
        k = bisect_left(tails, j)
        tails[k:k + 1], tail_at[k:k + 1] = [j], [n]
        prev[n] = tail_at[k - 1] if k else -1
    chain, n = [], tail_at[-1] if tail_at else -1
    while n != -1:
        chain.append(pairs[n])
        n = prev[n]
    return chain[::-1]


def _pairs(old: list[str], new: list[str], *, unique: bool) -> list[tuple[int, int]]:
    """Candidate matches in ``old`` order: lines unique on both sides, or (``unique=False``) the
    k-th copy of each line in *old* paired with its k-th copy in *new*."""
    from collections import Counter, defaultdict
    if unique:
        old_count, new_count = Counter(old), Counter(new)
        at = {line: j for j, line in enumerate(new) if new_count[line] == 1}
        return [(i, at[line]) for i, line in enumerate(old) if old_count[line] == 1 and line in at]
    slots: dict[str, list[int]] = defaultdict(list)
    for j, line in enumerate(new):
        slots[line].append(j)
    seen: dict[str, int] = defaultdict(int)
    out = []
    for i, line in enumerate(old):
        k = seen[line]
        seen[line] = k + 1
        if k < len(slots.get(line, ())):
            out.append((i, slots[line][k]))
    return out


def _diff_work_estimate(old: list[str], new: list[str]) -> tuple[int, int]:
    """``(changed lines, span lines)`` after trimming the common head/tail, in O(n log n).

    Matches are a real common subsequence, so ``changed`` never undercounts and a reordering counts
    as change (sorting a 100k-line file costs difflib ~55 s with no line added). Lines unique on both
    sides anchor the alignment (patience diff); between anchors, repeated lines pair k-th copy to
    k-th copy, so a small insert among blank lines keeps its preview."""
    limit = min(len(old), len(new))
    head = 0
    while head < limit and old[head] == new[head]:
        head += 1
    tail = 0
    while tail < limit - head and old[-1 - tail] == new[-1 - tail]:
        tail += 1
    old_mid, new_mid = old[head:len(old) - tail], new[head:len(new) - tail]
    anchors = _lis(_pairs(old_mid, new_mid, unique=True))
    matched, i0, j0 = len(anchors), 0, 0
    for i1, j1 in [*anchors, (len(old_mid), len(new_mid))]:
        matched += len(_lis(_pairs(old_mid[i0:i1], new_mid[j0:j1], unique=False)))
        i0, j0 = i1 + 1, j1 + 1
    return len(old_mid) + len(new_mid) - 2 * matched, max(len(old_mid), len(new_mid))


def bounded_unified_diff(old_text: str, new_text: Optional[str], fromfile: str, tofile: str) -> str:
    """``difflib.unified_diff`` text, or a one-line ``# diff omitted`` notice when computing it
    would be slow. The caller has already written the file; only the preview is dropped."""
    import difflib
    old = old_text.splitlines(keepends=True)
    new = [] if new_text is None else new_text.splitlines(keepends=True)
    changed, span = _diff_work_estimate(old, new)
    if changed * span > _MAX_DIFF_WORK:
        if new_text is None:
            shown = fromfile.removeprefix("a/")
            return f"# diff omitted for {shown}: deleted ({len(old)} lines).\n"
        shown = tofile.removeprefix("b/")
        return (f"# diff omitted for {shown}: about {changed} changed lines across {span} lines is "
                "too large to diff quickly. The edit was applied; use read_file to inspect it.\n")
    return "".join(difflib.unified_diff(old, new, fromfile=fromfile, tofile=tofile))


# ---------------------------------------------------------------------------
# Pagination clamps
# ---------------------------------------------------------------------------

DEFAULT_READ_OFFSET = 1
DEFAULT_READ_LIMIT = 2000
DEFAULT_SEARCH_OFFSET = 0
DEFAULT_SEARCH_LIMIT = 50


def _coerce_int(value: Any, default: int) -> int:
    """Best-effort integer coercion for tool pagination inputs."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_read_pagination(offset: Any = DEFAULT_READ_OFFSET,
                              limit: Any = DEFAULT_READ_LIMIT) -> tuple[int, int]:
    """Clamp read_file pagination so invalid values can never reach a sed range
    like ``0,-1p`` (schemas declare bounds, but not every caller enforces them).
    The ``limit`` ceiling is ``tool_output.max_lines`` from config.yaml."""
    from tools.tool_output_limits import get_max_lines
    normalized_offset = max(1, _coerce_int(offset, DEFAULT_READ_OFFSET))
    normalized_limit = max(1, min(_coerce_int(limit, DEFAULT_READ_LIMIT), get_max_lines()))
    return normalized_offset, normalized_limit


def normalize_search_pagination(offset: Any = DEFAULT_SEARCH_OFFSET,
                                limit: Any = DEFAULT_SEARCH_LIMIT) -> tuple[int, int]:
    """Return safe search pagination bounds for shell head/tail pipelines."""
    return max(0, _coerce_int(offset, DEFAULT_SEARCH_OFFSET)), max(1, _coerce_int(limit, DEFAULT_SEARCH_LIMIT))
