"""Result dataclasses and pure text helpers shared by ``tools.file_operations``
and its search/lint mixins. Leaf module (imports nothing from ``tools`` at module
scope) so nothing cycles. The ``to_dict`` output of every class here IS tool
behavior — key names, order and omission rules are pinned by tests and read by
the model.
"""

import os
import re
import stat
import time
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
    # read_file_raw refused the file for exceeding its size cap: it exists, but no content came back.
    _too_large: bool = False

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


_CONFUSABLE_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'"})  # Finder's curly apostrophes


def filename_match_key(name: str) -> str:
    """Spelling of a file name with its invisible variants collapsed: NFC, every Unicode space
    (``Zs``: macOS puts U+202F before AM/PM in screenshot names) as U+0020, curly single quotes
    as ``'``. Two names with the same key render identically, so a model retyping one can never
    see that it typed the other; read_file and the media resolver both retry on it."""
    import unicodedata
    out = unicodedata.normalize("NFC", name).translate(_CONFUSABLE_QUOTES)
    return "".join(" " if unicodedata.category(ch) == "Zs" else ch for ch in out)


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


# --- read_file views for shell one-liners (tail -n N, ls -la, a cell's parse-ready page) ---

# A listing is a context payload like any read: past this many entries it points to a glob search.
DIR_LISTING_MAX_ENTRIES = 500
_GUTTER = re.compile(r"^\d+\|", re.MULTILINE)
_KINDS = ((stat.S_ISDIR, "dir"), (stat.S_ISLNK, "link"), (stat.S_ISREG, "file"))


def tail_request(offset: Any, limit: int) -> tuple[int | None, int]:
    """``(N, min(N, limit))`` when ``offset`` asks for the last N lines (a negative integer),
    else ``(None, limit)``: the offset places the window, ``limit`` still bounds the page."""
    try:
        value = int(offset)
    except (TypeError, ValueError):
        return None, limit
    return (-value, min(-value, limit)) if value < 0 else (None, limit)


def tail_start(total_lines: int, lines: int | None, offset: int) -> int:
    """First line of the last ``lines`` of a ``total_lines`` file (``offset`` when not a tail read)."""
    return offset if lines is None else max(1, (total_lines or 0) - lines + 1)


def cell_read_fields(content: str, programmatic: bool) -> dict[str, str]:
    """Extra fields for a cell's read: ``text``, the page without its ``N|`` gutter, so code can
    parse or rewrite it without stripping prefixes. The model's own reads get none."""
    return {"text": _GUTTER.sub("", content or "")} if programmatic else {}


def _dir_entry(entry: os.DirEntry) -> dict | None:
    try:
        st = entry.stat(follow_symlinks=False)
    except FileNotFoundError:  # removed between scandir and stat
        return None
    kind = next((name for test, name in _KINDS if test(st.st_mode)), "other")
    # Every kind carries its lstat size, as ``ls -la`` shows it: a "-" for a directory sent the
    # model back to ``stat`` in the live eval when it was asked for every entry's size.
    item = {"name": entry.name, "type": kind, "size": st.st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))}
    if kind == "link":
        item["target"] = os.readlink(entry.path)
    return item


def _dir_row(item: dict) -> str:
    size = str(item["size"])
    name = item["name"] + ("/" if item["type"] == "dir" else "")
    link = f" -> {item['target']}" if item["type"] == "link" else ""
    return f"{size:>10}  {item['mtime']}  {name}{link}"


def directory_listing(path: str, *, structured: bool) -> dict:
    """``ls -la`` for one directory: dirs first, then by name, capped at ``DIR_LISTING_MAX_ENTRIES``.
    ``structured`` (a cell's read) adds the same shown entries as a list of dicts."""
    with os.scandir(path) as it:
        items = [e for e in map(_dir_entry, it) if e is not None]
    items.sort(key=lambda e: (e["type"] != "dir", e["name"]))
    shown = items[:DIR_LISTING_MAX_ENTRIES]
    dirs = sum(1 for e in items if e["type"] == "dir")
    header = f"{path}: {len(items)} entries ({dirs} dirs, {len(items) - dirs} other). size  mtime  name"
    result = {"path": path, "is_directory": True, "total_entries": len(items),
              "content": "\n".join([header, *map(_dir_row, shown)])}
    if len(items) > len(shown):
        result["truncated"] = True
        result["hint"] = (f"Showing {len(shown)} of {len(items)} entries. Use search_files with "
                          "target='files' and a glob to narrow it.")
    if structured:
        result["entries"] = shown
    return result


def tail_offset(file_ops, path: str, lines: int | None, offset: int) -> int:
    """Start line for a read: ``offset``, or for a tail read the first of the file's last ``lines``
    (one count probe; a failed probe gives line 1 and the read reports the failure as usual)."""
    return offset if lines is None else tail_start(file_ops.read_file(path, 1, 1).total_lines, lines, offset)


def settle_tail(file_ops, path: str, lines: int | None, offset: int, limit: int, result):
    """``(offset, result)`` for a tail read whose file changed between the count and the read
    (rotated or appended): the page's own ``total_lines`` names the right start, re-read once there,
    so a nonempty log never comes back as an empty or stale tail."""
    if lines is None or result.error:
        return offset, result
    settled = tail_start(result.total_lines, lines, offset)
    if settled == offset:
        return offset, result
    retry = file_ops.read_file(path, settled, limit)
    # A second change during the retry is reported, never returned as a stale tail.
    if not retry.error and not retry.is_binary and tail_start(retry.total_lines, lines, settled) != settled:
        raise ValueError("File changed during tail read; retry with the same negative offset.")
    return settled, retry
