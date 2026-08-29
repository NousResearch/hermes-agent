#!/usr/bin/env python3
"""
File Operations Module

Provides file manipulation capabilities (read, write, patch, search) that work
across all terminal backends (local, docker, ssh, singularity, modal, daytona, vercel_sandbox).

The key insight is that all file operations can be expressed as shell commands,
so we wrap the terminal backend's execute() interface to provide a unified file API.

Usage:
    from tools.file_operations import ShellFileOperations
    from tools.terminal_tool import _active_environments
    
    # Get file operations for a terminal environment
    file_ops = ShellFileOperations(terminal_env)
    
    # Read a file
    result = file_ops.read_file("/path/to/file.py")
    
    # Write a file
    result = file_ops.write_file("/path/to/new.py", "print('hello')")
    
    # Search for content
    result = file_ops.search("TODO", path=".", file_glob="*.py")
"""

import base64
import binascii
import os
import re
import secrets
import sys
import difflib
import hashlib
import json
import logging
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, ClassVar
from pathlib import Path
from tools.binary_extensions import BINARY_EXTENSIONS

from agent.file_safety import (
    build_write_denied_paths,
    build_write_denied_prefixes,
    get_write_denied_error,
    is_write_denied as _shared_is_write_denied,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Write-path deny list — blocks writes to sensitive system/credential files
# ---------------------------------------------------------------------------

_HOME = str(Path.home())

_MACOS_TCC_PROTECTED_HOME_DIRS = (
    "Desktop",
    "Documents",
    "Downloads",
    "Library",
    "Movies",
    "Music",
    "Pictures",
)


def _macos_protected_search_exclusions(
    path: str,
    *,
    cwd: Optional[str] = None,
    home: Optional[str] = None,
    platform: Optional[str] = None,
) -> List[str]:
    """Return protected home directories below a broad macOS search root.

    Direct searches inside a protected directory remain allowed. Only an
    ancestor search (for example ``$HOME`` or ``/Users``) receives exclusions,
    preventing recursive tools from triggering unattended TCC prompts.
    """
    if (platform or sys.platform) != "darwin":
        return []

    home_path = Path(home or Path.home()).expanduser()
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = Path(cwd or os.getcwd()) / root
    root = Path(os.path.normpath(str(root)))
    home_path = Path(os.path.normpath(str(home_path)))

    exclusions: List[str] = []
    for dirname in _MACOS_TCC_PROTECTED_HOME_DIRS:
        protected = home_path / dirname
        try:
            relative = protected.relative_to(root)
        except ValueError:
            continue
        if relative.parts:
            exclusions.append(relative.as_posix())
    return exclusions


WRITE_DENIED_PATHS = build_write_denied_paths(_HOME)

WRITE_DENIED_PREFIXES = build_write_denied_prefixes(_HOME)


_OSC_SEQUENCE_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_FENCE_MARKER_RE = re.compile(r"'?\x07?__HERMES_FENCE_[A-Za-z0-9]+__\x07?'?")


def _strip_terminal_fence_leaks(text: str) -> str:
    """Strip leaked terminal fence wrappers from file read output."""
    if not text:
        return text

    cleaned_lines: List[str] = []
    for line in text.splitlines(keepends=True):
        had_terminal_wrapper = "__HERMES_FENCE_" in line or "\x1b]" in line
        cleaned = _OSC_SEQUENCE_RE.sub("", line)
        cleaned = _FENCE_MARKER_RE.sub("", cleaned)
        cleaned = cleaned.replace("\x07", "")
        if had_terminal_wrapper and cleaned.strip("'\r\n\t ") == "":
            continue
        cleaned_lines.append(cleaned)
    return "".join(cleaned_lines)


def _detect_line_ending(sample: str) -> Optional[str]:
    """Return the dominant line ending in ``sample`` or None if undetermined.

    Looks at the first few line breaks and picks ``\\r\\n`` if any are
    present (Windows / DOS), otherwise ``\\n`` (Unix).  Returns ``None``
    for empty / single-line content where we can't tell.  Used to
    preserve the file's original line endings across write_file and
    patch operations — without this the agent's bare-LF tool args
    silently normalize Windows-line-ending files, and patch produces
    mixed endings when only a substituted region changes.
    """
    if not sample:
        return None
    # Look at the first chunk — enough to tell, cheap to scan.
    head = sample[:4096]
    if "\r\n" in head:
        return "\r\n"
    if "\n" in head:
        return "\n"
    return None


def _normalize_line_endings(text: str, target: str) -> str:
    """Convert all line endings in ``text`` to ``target`` (``\\n`` or ``\\r\\n``).

    Idempotent: ``_normalize_line_endings(_normalize_line_endings(x, "\\r\\n"), "\\r\\n") == _normalize_line_endings(x, "\\r\\n")``.
    Strips lone ``\\r`` characters as well, so mixed-ending content is
    homogenized in a single pass.
    """
    # First collapse to LF (handle CRLF and lone CR), then expand if target
    # is CRLF.  Order matters: doing the replacements separately would
    # double-convert a CRLF -> LFLF.
    lf_normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if target == "\n":
        return lf_normalized
    if target == "\r\n":
        return lf_normalized.replace("\n", "\r\n")
    return text


# UTF-8 byte order mark. Some Windows editors (Notepad, older Visual Studio,
# some PowerShell redirects) prepend this invisible 3-byte marker
# (EF BB BF == U+FEFF) to UTF-8 text files. It renders as nothing but is a
# real character at the start of the decoded string, so without handling it:
#   - read_file would surface a stray U+FEFF as the first character (the
#     model sees a phantom char before `import ...`), and
#   - patch matches against the true first line would miss, and write_file
#     would silently drop or double the marker on rewrite.
# We strip it on read so the model sees clean content, and restore it on
# write when the original file had one — exactly mirroring the line-ending
# preservation above (detect on disk, preserve across the edit).
_UTF8_BOM = "\ufeff"


def _strip_bom(text: str) -> tuple[str, bool]:
    """Return (text-without-leading-BOM, had_bom).

    Only a single leading BOM is stripped; a BOM appearing mid-content is
    left alone (it's legitimate data there, not a file marker).
    """
    if text and text.startswith(_UTF8_BOM):
        return text[len(_UTF8_BOM):], True
    return text, False


def _has_bom(text: Optional[str]) -> bool:
    """True if ``text`` begins with a UTF-8 BOM."""
    return bool(text) and text.startswith(_UTF8_BOM)


def _is_write_denied(path: str) -> bool:
    """Return True if path is on the write deny list."""
    return _shared_is_write_denied(path)


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class ReadResult:
    """Result from reading a file."""
    content: str = ""
    total_lines: int = 0
    file_size: int = 0
    truncated: bool = False
    hint: Optional[str] = None
    is_binary: bool = False
    is_image: bool = False
    base64_content: Optional[str] = None
    mime_type: Optional[str] = None
    dimensions: Optional[str] = None  # For images: "WIDTHxHEIGHT"
    error: Optional[str] = None
    similar_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != []}


@dataclass
class WriteResult:
    """Result from writing a file."""
    bytes_written: int = 0
    dirs_created: bool = False
    # True when the on-disk sha256 matched the intended content after the
    # write (post-write verification). None when the backend couldn't
    # verify (no sha256sum). A mismatch never reaches the caller as a
    # flag — it becomes a hard error.
    verified: Optional[bool] = None
    lint: Optional[Dict[str, Any]] = None
    # Semantic diagnostics from the LSP layer, when applicable.  Kept in
    # its own field (not folded into ``lint``) so the model and any
    # downstream parsers can read syntax errors and semantic errors as
    # separate signals.  ``None`` when LSP is disabled, when the file
    # isn't in a git workspace, or when no diagnostics were introduced
    # by this edit.
    lsp_diagnostics: Optional[str] = None
    error: Optional[str] = None
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class PatchResult:
    """Result from patching a file."""
    success: bool = False
    diff: str = ""
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    lint: Optional[Dict[str, Any]] = None
    # See :class:`WriteResult.lsp_diagnostics`.
    lsp_diagnostics: Optional[str] = None
    error: Optional[str] = None
    # Set on success-shaped no-ops: the requested edit was already present
    # in the file, so nothing was written. Carries a short note for the
    # model explaining why no diff is included.
    no_change: bool = False
    note: Optional[str] = None

    def to_dict(self) -> dict:
        result: Dict[str, Any] = {"success": self.success}
        if self.no_change:
            result["no_change"] = True
        if self.note:
            result["note"] = self.note
        if self.diff:
            result["diff"] = self.diff
        if self.files_modified:
            result["files_modified"] = self.files_modified
        if self.files_created:
            result["files_created"] = self.files_created
        if self.files_deleted:
            result["files_deleted"] = self.files_deleted
        if self.lint:
            result["lint"] = self.lint
        if self.lsp_diagnostics:
            result["lsp_diagnostics"] = self.lsp_diagnostics
        if self.error:
            result["error"] = self.error
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
    matches: List[SearchMatch] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    truncated: bool = False
    limit_reason: Optional[str] = None
    warning: Optional[str] = None
    error: Optional[str] = None
    
    # Densify content-mode matches into a path-grouped text block above this
    # many matches. Below it, the verbose array is already compact enough that
    # the path-grouping header costs more than it saves.
    _DENSIFY_MIN_MATCHES: ClassVar[int] = 5

    def _densify_matches(self) -> Optional[str]:
        """Render content-mode matches as a compact, path-grouped text block.

        The verbose form repeats the ``{"path","line","content"}`` keys and the
        full path string for every match. This groups consecutive matches by
        path (path printed once, then ``  <line>: <content>`` rows), which is
        lossless — every path, line number, and content byte is preserved — and
        readable by the model without any decode step.

        Returns ``None`` when densification is not worthwhile (too few matches),
        so the caller falls back to the verbose array.
        """
        if len(self.matches) < self._DENSIFY_MIN_MATCHES:
            return None
        # ripgrep emits matches path-ordered (all hits in a file are
        # consecutive), so grouping on path change collapses each file to a
        # single header without reordering results.
        lines: list[str] = []
        current_path: Optional[str] = None
        for m in self.matches:
            if m.path != current_path:
                lines.append(m.path)
                current_path = m.path
            # rstrip trailing whitespace only; leading indentation in code is
            # meaningful and preserved verbatim after the "<line>: " prefix.
            lines.append(f"  {m.line_number}: {m.content.rstrip()}")
        return "\n".join(lines)

    def to_dict(self, densify: bool = False) -> dict:
        result: dict[str, object] = {"total_count": self.total_count}
        if self.matches:
            dense = self._densify_matches() if densify else None
            if dense is not None:
                # Self-describing: the format key tells the model how to read
                # the block so it never has to guess the shape.
                result["matches_format"] = (
                    "path-grouped: each file path on its own line, followed by "
                    "indented '<line>: <content>' rows for matches in that file"
                )
                result["matches_text"] = dense
            else:
                result["matches"] = [
                    {"path": m.path, "line": m.line_number, "content": m.content}
                    for m in self.matches
                ]
        if self.files:
            result["files"] = self.files
        if self.counts:
            result["counts"] = self.counts
        if self.truncated:
            result["truncated"] = True
        if self.limit_reason:
            result["limit_reason"] = self.limit_reason
        if self.warning:
            result["warning"] = self.warning
        if self.error:
            result["error"] = self.error
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


_SEARCH_TIMEOUT_MARKER_RE = re.compile(r"\n?\[Command timed out after \d+s\]\s*$")


def _search_stdout_and_limit(result: ExecuteResult) -> tuple[str, Optional[str]]:
    """Return stdout cleaned for parsing and a limit reason for search timeouts."""
    if result.exit_code == 124:
        return _SEARCH_TIMEOUT_MARKER_RE.sub("", result.stdout), "search_timeout"
    return result.stdout, None


def _split_tool_diagnostics(output: str) -> tuple[str, str]:
    """Separate rg/grep diagnostic lines from real match output.

    ``_exec`` runs commands with ``stderr=subprocess.STDOUT``, so error and
    warning text from ``rg``/``grep`` is interleaved with match lines in a
    single stream. Diagnostics must not be parsed as matches, and on a hard
    failure they are the error message to surface.

    Returns ``(diagnostics, payload)`` where ``payload`` contains only lines
    that look like real search output — a match line (``file:line:content``),
    a files-only path, a count line, or a context line/separator. Everything
    else (tool-prefixed errors, rg's multi-line ``regex parse error`` block
    with its indented carets, blank lines) is folded into ``diagnostics``.

    Classifying by *shape* rather than by error prefix is what lets the
    exit-2 guard distinguish a pure failure (no usable payload → surface the
    error) from a partial failure (some files matched, one was unreadable →
    keep the matches). It also means error text can never be mis-parsed as a
    match, a latent bug that predates the exit-code fix.
    """
    diagnostics: list[str] = []
    payload: list[str] = []
    for line in output.split('\n'):
        if not line.strip():
            continue
        # Tool diagnostics always carry the "<tool>: " prefix (e.g.
        # "rg: <file>: Permission denied", "grep: Invalid regular
        # expression", "rg: regex parse error:"). Check this first: a real
        # match path can legitimately contain "-<digit>" (e.g. a tmp dir like
        # ".../pytest-686/..."), which the shape regex would otherwise treat
        # as a match line.
        stripped = line.lstrip()
        if stripped.startswith("rg: ") or stripped.startswith("grep: "):
            diagnostics.append(line)
            continue
        # Otherwise classify by output shape. rg's regex-parse-error block
        # also emits an indented caret line and a trailing "error: ..." line
        # with no tool prefix; neither matches a search-output shape, so they
        # fall through to diagnostics.
        #   match / count : "<path>:<...>"   (has a colon; rg -c uses path:count)
        #   files_only    : "<path>"         (no whitespace, no leading colon)
        #   context line  : "<path>-<line>-" or the "--" group separator
        if line == "--" or _SEARCH_OUTPUT_RE.match(line):
            payload.append(line)
        else:
            diagnostics.append(line)
    return '\n'.join(diagnostics), '\n'.join(payload)


# A real rg/grep output line starts with a path token and is followed by a
# ``:`` (match/count), a ``-`` (context), or nothing (files_only). Tool
# diagnostics ("rg: ...", "grep: ...", "error: ...", indented carets) never
# match because the path token forbids whitespace and a leading tool prefix
# like "rg" is followed by ": " (space) which the negated class rejects.
_SEARCH_OUTPUT_RE = re.compile(r'^([A-Za-z]:)?[^\s:][^\n]*?[:\-]\d|^[^\s:][^\s]*$')


def _parse_search_context_line(line: str) -> tuple[str, int, str] | None:
    """Parse grep/rg context output in ``path-line-content`` format.

    Context lines are ambiguous because filenames may legitimately contain
    ``-<digits>-`` segments. Prefer the rightmost numeric separator so a path
    like ``dir/file-12-name.py-8-context`` resolves to
    ``dir/file-12-name.py`` line ``8`` instead of truncating at ``file``.
    """
    if not line or line == "--":
        return None

    match = None
    for candidate in re.finditer(r'-(\d+)-', line):
        match = candidate

    if match is None:
        return None

    path = line[:match.start()]
    if not path:
        return None

    return path, int(match.group(1)), line[match.end():]


# =============================================================================
# Abstract Interface
# =============================================================================

_MAGIC_SIGNATURES: tuple = (
    # (prefix bytes, human name) — ordered, first match wins. Longest
    # prefixes for a shared first byte come first.
    (b"\x89PNG\r\n\x1a\n", "PNG image data"),
    (b"\xff\xd8\xff", "JPEG image data"),
    (b"GIF87a", "GIF image data"),
    (b"GIF89a", "GIF image data"),
    (b"RIFF", "RIFF container (WAV/AVI/WebP family)"),
    (b"%PDF-", "PDF document"),
    (b"PK\x03\x04", "ZIP archive (also docx/xlsx/jar/apk)"),
    (b"PK\x05\x06", "ZIP archive (empty)"),
    (b"\x1f\x8b", "gzip compressed data"),
    (b"BZh", "bzip2 compressed data"),
    (b"\xfd7zXZ\x00", "xz compressed data"),
    (b"7z\xbc\xaf\x27\x1c", "7-Zip archive"),
    (b"\x7fELF", "ELF executable"),
    (b"MZ", "Windows PE executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O executable (64-bit)"),
    (b"\xca\xfe\xba\xbe", "Mach-O universal binary / Java class"),
    (b"SQLite format 3\x00", "SQLite database"),
    (b"OggS", "Ogg container"),
    (b"fLaC", "FLAC audio"),
    (b"ID3", "MP3 audio (ID3 tag)"),
    (b"\x00\x00\x00", "ISO media container (MP4/MOV family)"),  # ftyp at +4
    (b"BM", "BMP image data"),
    (b"II*\x00", "TIFF image data (little-endian)"),
    (b"MM\x00*", "TIFF image data (big-endian)"),
)


def identify_binary_bytes(sample: bytes) -> str:
    """Best-effort human name for binary content from its magic bytes.

    Returns e.g. ``"PNG image data"`` or ``"unknown binary"``. Never raises.
    The ISO-media entry additionally checks for ``ftyp`` at offset 4, since
    the leading size field alone (three NULs) is too weak a signature.
    """
    if not sample:
        return "unknown binary"
    for prefix, name in _MAGIC_SIGNATURES:
        if sample.startswith(prefix):
            if name.startswith("ISO media") and sample[4:8] != b"ftyp":
                continue
            return name
    return "unknown binary"


def describe_binary_file(sample: Optional[bytes], file_size: int) -> str:
    """One-line answer for the binary-file refusal.

    Naming the dead end: "Binary file" alone sends the model hunting for
    'appropriate tools' that may not exist in its toolset. Naming the TYPE
    ("PNG image data, 4.1 KB") answers what-is-this in a single read.
    """
    kind = identify_binary_bytes(sample or b"")
    if file_size >= 1024 * 1024:
        size = f"{file_size / (1024 * 1024):.1f} MB"
    elif file_size >= 1024:
        size = f"{file_size / 1024:.1f} KB"
    else:
        size = f"{file_size} bytes"
    return f"Binary file ({kind}, {size}) — cannot display as text."


class FileOperations(ABC):
    """Abstract interface for file operations across terminal backends."""
    
    @abstractmethod
    def read_file(self, path: str, offset: int = 1, limit: int = 2000) -> ReadResult:
        """Read a file with pagination support."""
        ...

    @abstractmethod
    def read_file_raw(self, path: str) -> ReadResult:
        """Read the complete file content as a plain string.

        No pagination, no line-number prefixes, no per-line truncation.
        Returns ReadResult with .content = full file text, .error set on
        failure. Always reads to EOF regardless of file size.
        """
        ...

    def read_file_bytes(self, path: str, max_bytes: Optional[int] = None) -> ReadResult:
        """Read complete binary content as base64 across the backend boundary."""
        return ReadResult(error="Binary reads are not implemented for this backend")

    @abstractmethod
    def write_file(self, path: str, content: str,
                   pre_content: Optional[str] = None) -> WriteResult:
        """Write content to a file, creating directories as needed."""
        ...

    @abstractmethod
    def patch_replace(self, path: str, old_string: str, new_string: str,
                      replace_all: bool = False) -> PatchResult:
        """Replace text in a file using fuzzy matching."""
        ...

    @abstractmethod
    def patch_v4a(self, patch_content: str) -> PatchResult:
        """Apply a V4A format patch."""
        ...

    @abstractmethod
    def delete_file(self, path: str) -> WriteResult:
        """Delete a file. Returns WriteResult with .error set on failure."""
        ...

    def delete_path(self, path: str, recursive: bool = False) -> WriteResult:
        """Cross-platform delete that handles files and (with recursive=True)
        directory trees. Default implementation delegates to ``delete_file``
        for the non-recursive case; backends with native recursive support
        should override.
        """
        if recursive:
            return WriteResult(error="Recursive delete not implemented for this backend")
        return self.delete_file(path)

    @abstractmethod
    def move_file(self, src: str, dst: str) -> WriteResult:
        """Move/rename a file from src to dst. Returns WriteResult with .error set on failure."""
        ...

    @abstractmethod
    def search(self, pattern: str, path: str = ".", target: str = "content",
               file_glob: Optional[str] = None, limit: int = 50, offset: int = 0,
               output_mode: str = "content", context: int = 0,
               order: str = "discovery") -> SearchResult:
        """Search for content or files."""
        ...


# =============================================================================
# Shell-based Implementation
# =============================================================================

# Image extensions (subset of binary that we can return as base64)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico'}

# Shell-based linters by file extension.  Invoked via _exec() with the
# filesystem path.  Cover languages where a compile/type check needs an
# external toolchain (py_compile, node, tsc, go vet, rustfmt).
LINTERS = {
    '.py': 'python -m py_compile {file} 2>&1',
    '.js': 'node --check {file} 2>&1',
    '.ts': 'npx tsc --noEmit {file} 2>&1',
    '.go': 'go vet {file} 2>&1',
    '.rs': 'rustfmt --check {file} 2>&1',
}

# Extensions where the per-file shell linter is structurally weaker than
# a real LSP server AND produces phantom errors on real-world projects:
#
# - ``.ts``: ``tsc --noEmit FILE.ts`` ignores ``tsconfig.json`` and
#   defaults to no-lib / ES5, so every ES2015+ stdlib reference
#   (``Promise``, ``Map``, ``Set``, ``ReadonlySet``, ``Iterable``,
#   ``Math.imul``, ``Number.isFinite``, etc.) reports as missing.  This
#   floods the agent's lint field with 20K+ tokens of false positives on
#   every edit.  No supported tsc flag fixes the single-file invocation;
#   the canonical replacement is ``tsserver`` via LSP, which respects
#   tsconfig and gives true diagnostics.
#
#   ``.tsx`` is intentionally NOT in ``LINTERS`` (and therefore not
#   here): it has no shell linter entry, so it falls through to the
#   ``ext not in LINTERS`` skip case unchanged.  Pre-PR behavior:
#   ``.tsx`` was implicitly ``skipped``.  Keeping it that way means
#   ``.tsx`` edits with LSP disabled get no per-file syntax check
#   (same as before this PR) instead of the broken ``tsc`` invocation
#   that ``.ts`` used to get.  When LSP is enabled, ``.tsx`` is covered
#   by the LSP tier via ``_maybe_lsp_diagnostics`` exactly as ``.ts``.
#
# - ``.go``: ``go vet FILE.go`` fails outside a module / GOPATH with
#   "cannot find package" — already partially handled by
#   ``_LINTER_UNUSABLE_PATTERNS`` but only when the package error is the
#   ONLY output; mixed real+phantom output still leaks through.
#   ``gopls`` is the canonical replacement.
#
# - ``.rs``: ``rustfmt --check FILE.rs`` is style, not type-checking, and
#   rejects non-Cargo project files.  ``rust-analyzer`` is the canonical
#   replacement.
#
# When the LSP service is configured AND ``enabled_for(path)`` for this
# extension's file, ``_check_lint`` skips the shell linter for these
# extensions — the ``lsp_diagnostics`` channel carries the real signal.
# Everything else in ``LINTERS`` (Python ``py_compile``, ``node --check``)
# is fast, file-local, and correct, so it runs unconditionally.
_SHELL_LINTER_LSP_REDUNDANT = frozenset({'.ts', '.go', '.rs'})


# Patterns that indicate the linter base command exists on PATH but
# couldn't actually run — e.g. ``npx tsc`` when tsc isn't installed in
# node_modules, or rustfmt complaining there's no Cargo project.  When
# any of these substrings appears in the linter output, ``_check_lint``
# returns ``skipped`` instead of ``error`` so:
#
# 1. The write isn't flagged for a tooling problem the agent can't fix.
# 2. The LSP semantic tier still runs (it gates on success/skipped).
#
# Patterns are matched case-insensitively against linter stdout.
_LINTER_UNUSABLE_PATTERNS = {
    'npx': (
        # npx prints this banner when the package isn't installed locally
        # AND it can't auto-install (no internet, registry off, etc.) or
        # when the binary it tried to run is the wrong one.
        'this is not the tsc command you are looking for',
        # npx with --no-install resolution failures
        'could not determine executable to run',
        'not found in npm registry',
    ),
    'rustfmt': (
        # rustfmt outside a Cargo project
        'no input filename given',
        'error: not a workspace',
    ),
    'go': (
        # ``go vet`` on a file outside a module / GOPATH
        'cannot find package',
        'go: cannot find main module',
    ),
}


def _looks_like_linter_unusable(base_cmd: str, output: str) -> bool:
    """Return True iff ``output`` from ``base_cmd`` indicates the linter
    itself couldn't run (a tooling gap), as opposed to a real lint error
    in the file being checked.

    ``base_cmd`` is the first word of the linter command line (``npx``,
    ``rustfmt``, ``go``, ...).  ``output`` is the stdout/stderr captured
    from running it.
    """
    patterns = _LINTER_UNUSABLE_PATTERNS.get(base_cmd)
    if not patterns:
        return False
    lower = output.lower()
    return any(p in lower for p in patterns)


def _lint_json_inproc(content: str) -> tuple[bool, str]:
    """In-process JSON syntax check.  Returns (ok, error_message)."""
    import json as _json
    try:
        _json.loads(content)
        return True, ""
    except _json.JSONDecodeError as e:
        return False, f"JSONDecodeError: {e.msg} (line {e.lineno}, column {e.colno})"
    except Exception as e:  # noqa: BLE001 — any parse failure is a lint failure
        return False, f"{type(e).__name__}: {e}"


def _lint_yaml_inproc(content: str) -> tuple[bool, str]:
    """In-process YAML syntax check.  Returns (ok, error_message).

    Skipped gracefully if PyYAML isn't installed — YAML parsing is optional.

    Deliberately a *syntax-only* scan (``yaml.parse``), not ``safe_load``:
    loading rejects perfectly valid YAML that merely isn't a single plain
    document — multi-document streams (``---``-separated Kubernetes
    manifests raise ``ComposerError``) and application-defined tags
    (CloudFormation ``!Sub``/``!Ref``, Ansible ``!vault`` raise
    ``ConstructorError``).  Those are content conventions for whatever
    consumes the file, not syntax errors, and this linter's verdict is
    used as a fail-closed WRITE gate in ``write_file`` — a false positive
    here refuses a legitimate write outright.  ``yaml.parse`` still
    catches real scanner/parser failures (unclosed quotes, bad
    indentation, tab-mangled block maps).
    """
    try:
        import yaml as _yaml
    except ImportError:
        # PyYAML not available — skip silently, caller treats as no linter.
        return True, "__SKIP__"
    try:
        for _event in _yaml.parse(content):
            pass
        return True, ""
    except _yaml.YAMLError as e:
        return False, f"YAMLError: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _lint_toml_inproc(content: str) -> tuple[bool, str]:
    """In-process TOML syntax check (stdlib tomllib, Python 3.11+)."""
    import tomllib as _toml

    try:
        _toml.loads(content)
        return True, ""
    except Exception as e:  # tomllib raises TOMLDecodeError, a ValueError subclass
        return False, f"{type(e).__name__}: {e}"


def _lint_python_inproc(content: str) -> tuple[bool, str]:
    """In-process Python syntax check via ast.parse.

    Catches SyntaxError, IndentationError, and everything else the
    ast module rejects — matching py_compile's scope but with no
    subprocess overhead and no dependency on a ``python`` in PATH.
    """
    import ast as _ast
    try:
        _ast.parse(content)
        return True, ""
    except SyntaxError as e:
        loc = f" (line {e.lineno}, column {e.offset})" if e.lineno else ""
        return False, f"{type(e).__name__}: {e.msg}{loc}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


# In-process linters by file extension.  Preferred over shell linters when
# present — no subprocess overhead, microseconds per call.  Each callable
# takes file content (str) and returns (ok: bool, error: str).  An error
# string of ``"__SKIP__"`` signals the linter isn't available (missing
# dependency) and should be treated as "no linter".
LINTERS_INPROC = {
    '.py': _lint_python_inproc,
    '.json': _lint_json_inproc,
    '.yaml': _lint_yaml_inproc,
    '.yml': _lint_yaml_inproc,
    '.toml': _lint_toml_inproc,
}

# Subset of LINTERS_INPROC that the pre-write fail-closed gate in
# ``write_file`` (see below) refuses on, rather than merely reporting.
# Deliberately excludes ``.py``: unlike JSON/YAML/TOML (atomic structured
# data blobs where "doesn't parse" always means "corrupt"), ``.py`` is
# used throughout this codebase's own test fixtures as a generic
# stand-in extension for arbitrary non-Python text content (e.g.
# ``tests/tools/test_file_operations.py``'s
# ``TestPatchReplacePostWriteVerification`` writes "hello world" /
# "hi world" through a ``*.py`` path purely to exercise write-mechanics,
# not Python validity). Hard-refusing on invalid Python would treat that
# established, exercised pattern as an error and break it. Python source
# keeps the existing (unchanged) post-write lint-delta *report* — still
# visible to the caller, just not a write-blocking refusal.
_FAIL_CLOSED_INPROC_EXTS = frozenset({'.json', '.yaml', '.yml', '.toml'})

# Max limits for read operations
MAX_LINES = 2000
MAX_LINE_LENGTH = 2000
MAX_FILE_SIZE = 50 * 1024  # 50KB
DEFAULT_READ_OFFSET = 1
DEFAULT_READ_LIMIT = 2000
DEFAULT_SEARCH_OFFSET = 0
DEFAULT_SEARCH_LIMIT = 50

# Echoed by the size probe when the path exists but is not a regular file.
# `wc -c` prints only digits, so this can never collide with a real size.
NOT_REGULAR_SENTINEL = "__hermes_not_regular__"

# Echoed by the compound read/write probes when the path does not exist.
# A compound command only reports its *last* exit status, so the missing-file
# signal that ``_size_probe_cmd`` carries in ``exit 1`` has to travel in-band.
MISSING_SENTINEL = "__hermes_missing__"

_READ_SENTINEL_PREFIX = "__HERMES_RF_"
_WRITE_SENTINEL_PREFIX = "__HERMES_WF_"


def _new_sentinel(prefix: str) -> str:
    """Per-call separator line for a compound shell probe.

    128 random bits make a collision with file content negligible, and the
    underscores keep the token outside the base64 alphabet, so a sentinel
    that ever leaked into a sample segment fails base64 validation instead
    of decoding into bytes.
    """
    return f"{prefix}{secrets.token_hex(16)}__"


def _split_segments(output: str, sentinel: str) -> List[str]:
    """Split compound-probe stdout on its sentinel lines.

    Every producer (``wc``, ``base64``, ``cut``) newline-terminates its
    output or prints nothing, so the separator is always ``sentinel + "\\n"``
    on a line of its own. The text after the final sentinel is the status
    segment.
    """
    return output.split(sentinel + "\n")


def _coerce_int(value: Any, default: int) -> int:
    """Best-effort integer coercion for tool pagination inputs."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_read_pagination(offset: Any = DEFAULT_READ_OFFSET,
                              limit: Any = DEFAULT_READ_LIMIT) -> tuple[int, int]:
    """Return safe read_file pagination bounds.

    Tool schemas declare minimum/maximum values, but not every caller or
    provider enforces schemas before dispatch. Clamp here so invalid values
    cannot leak into sed ranges like ``0,-1p``.

    The upper bound on ``limit`` comes from ``tool_output.max_lines`` in
    config.yaml (defaults to the module-level ``MAX_LINES`` constant).
    """
    from tools.tool_output_limits import get_max_lines
    max_lines = get_max_lines()
    normalized_offset = max(1, _coerce_int(offset, DEFAULT_READ_OFFSET))
    normalized_limit = _coerce_int(limit, DEFAULT_READ_LIMIT)
    normalized_limit = max(1, min(normalized_limit, max_lines))
    return normalized_offset, normalized_limit


def normalize_search_pagination(offset: Any = DEFAULT_SEARCH_OFFSET,
                                limit: Any = DEFAULT_SEARCH_LIMIT) -> tuple[int, int]:
    """Return safe search pagination bounds for shell head/tail pipelines."""
    normalized_offset = max(0, _coerce_int(offset, DEFAULT_SEARCH_OFFSET))
    normalized_limit = max(1, _coerce_int(limit, DEFAULT_SEARCH_LIMIT))
    return normalized_offset, normalized_limit


_REGEX_NEWLINE_ESCAPE_RE = re.compile(r"(?<!\\)(?:\\\\)*\\n")


def _pattern_has_regex_newline(pattern: str) -> bool:
    """Return True when a content-search regex tries to match a newline.

    ``search_files`` runs rg/grep in line-oriented mode, not rg
    ``-U``/``--multiline`` mode, so newline regexes cannot match across
    lines.  Detect both a literal newline already decoded into the tool
    argument and a regex ``\n`` escape (odd number of backslashes before
    ``n``).  Even backslashes, e.g. ``\\n``, mean a literal backslash+n
    search and should not warn.
    """
    return "\n" in pattern or bool(_REGEX_NEWLINE_ESCAPE_RE.search(pattern))


def _is_line_oriented_newline_error(error: Optional[str]) -> bool:
    """Return True for rg's hard error when multiline mode is required."""
    if not error:
        return False
    return "literal \"\\n\" is not allowed" in error and "--multiline" in error


def _maybe_warn_line_oriented_newline_pattern(result: SearchResult, pattern: str) -> SearchResult:
    """Attach a newline-regex warning only when search found no usable results."""
    if result.total_count != 0 or not _pattern_has_regex_newline(pattern):
        return result
    if result.error and not _is_line_oriented_newline_error(result.error):
        return result
    result.error = None
    result.warning = (
        "0 results found. Note: search_files content search is line-oriented "
        "and does not run ripgrep with -U/--multiline, so `\\n` in the regex "
        "does not match line breaks. Use context=N to inspect neighboring "
        "lines, or escape as `\\\\n` when searching for a literal backslash+n."
    )
    return result


class ShellFileOperations(FileOperations):
    """
    File operations implemented via shell commands.
    
    Works with ANY terminal backend that has execute(command, cwd) method.
    This includes local, docker, singularity, ssh, modal, and daytona environments.
    """
    
    def __init__(self, terminal_env, cwd: str = None):
        """
        Initialize file operations with a terminal environment.

        Args:
            terminal_env: Any object with execute(command, cwd) method.
                         Returns {"output": str, "returncode": int}
            cwd: Optional explicit fallback cwd when the terminal env has
                 no cwd attribute (rare — most backends track cwd live).

        Note:
            Every _exec() call prefers the LIVE ``terminal_env.cwd`` over
            ``self.cwd`` so ``cd`` commands run via the terminal tool are
            picked up immediately.  ``self.cwd`` is only used as a fallback
            when the env has no cwd at all — it is NOT the authoritative
            cwd, despite being settable at init time.

            Historical bug (fixed): prior versions of this class used the
            init-time cwd for every _exec() call, which caused relative
            paths passed to patch/read/write to target the wrong directory
            after the user ran ``cd`` in the terminal.  Patches would
            claim success and return a plausible diff but land in the
            original directory, producing apparent silent failures.
        """
        self.env = terminal_env
        # Determine cwd from various possible sources.
        # IMPORTANT: do NOT fall back to os.getcwd() -- that's the HOST's local
        # path which doesn't exist inside container/cloud backends (modal, docker).
        # If nothing provides a cwd, use "/" as a safe universal default.
        self.cwd = cwd or getattr(terminal_env, 'cwd', None) or \
                   getattr(getattr(terminal_env, 'config', None), 'cwd', None) or "/"

        # Preserve the historical bool cache for ordinary executables: both
        # hits and misses stay cached. Ripgrep is special because it has an
        # off-PATH resolver and may be installed while this object is alive;
        # only successful rg resolutions are cached.
        self._command_cache: Dict[str, bool] = {}
        self._rg_resolution_cache: Dict[str, str] = {}
        self._rg_modified_capability: Dict[str, Optional[str]] = {}
    
    def _exec(self, command: str, cwd: str = None, timeout: int = None,
              stdin_data: str = None) -> ExecuteResult:
        """Execute command via terminal backend.

        Args:
            stdin_data: If provided, piped to the process's stdin instead of
                        embedding in the command string. Bypasses ARG_MAX.

        Cwd resolution order (critical — see class docstring):
          1. Explicit ``cwd`` arg (if provided)
          2. Live ``self.env.cwd`` (tracks ``cd`` commands run via terminal)
          3. Init-time ``self.cwd`` (fallback when env has no cwd attribute)

        This ordering ensures relative paths in file operations follow the
        terminal's current directory — not the directory this file_ops was
        originally created in.  See test_file_ops_cwd_tracking.py.
        """
        kwargs = {}
        if timeout:
            kwargs['timeout'] = timeout
        if stdin_data is not None:
            kwargs['stdin_data'] = stdin_data

        # Resolve cwd from the live env so `cd` commands are picked up.
        # Fall through to init-time self.cwd only if the env doesn't track cwd.
        effective_cwd = cwd or getattr(self.env, 'cwd', None) or self.cwd
        result = self.env.execute(command, cwd=effective_cwd, **kwargs)
        exit_code = result.get("returncode", 0)
        # A stdin write failure with an otherwise-clean child exit is still
        # a failure: the child never received the intended input. write_file
        # rejects such content up front (Task 3); this mapping is
        # defense-in-depth for any other stdin caller.
        if result.get("stdin_error") and exit_code == 0:
            exit_code = 1
        return ExecuteResult(
            stdout=result.get("output", ""),
            exit_code=exit_code
        )
    
    def _resolve_command(self, cmd: str) -> Optional[str]:
        """Resolve an executable in the command host's namespace.

        Ordinary commands retain the original bool hit/miss cache. Ripgrep
        alone caches successful resolved paths and re-probes misses so a
        mid-session install becomes visible.
        """
        if cmd != "rg":
            return cmd if self._has_command(cmd) else None

        cached = self._rg_resolution_cache.get(cmd)
        if cached:
            return cached

        result = self._exec("command -v rg 2>/dev/null")
        if result.exit_code == 0 and result.stdout.strip():
            resolved = result.stdout.strip().splitlines()[0]
            # Compatibility with old boolean-probe fakes.
            if resolved == "yes":
                resolved = "rg"
            self._rg_resolution_cache[cmd] = resolved
            return resolved

        from tools.environments.local import LocalEnvironment, _IS_WINDOWS

        if _IS_WINDOWS and isinstance(self.env, LocalEnvironment):
            user_profile = os.environ.get("USERPROFILE") or str(Path.home())
            local_app_data = os.environ.get("LOCALAPPDATA")
            scoop = os.environ.get("SCOOP") or os.path.join(user_profile, "scoop")
            candidates = [
                os.path.join(user_profile, ".cargo", "bin", "rg.exe"),
                os.path.join(scoop, "shims", "rg.exe"),
            ]
            if local_app_data:
                candidates.append(
                    os.path.join(local_app_data, "Microsoft", "WinGet", "Links", "rg.exe")
                )
            for candidate in candidates:
                if os.path.isfile(candidate):
                    resolved = candidate.replace("\\", "/")
                    self._rg_resolution_cache[cmd] = resolved
                    return resolved
        return None

    def _has_command(self, cmd: str) -> bool:
        """Check command availability with rg-specific resolution semantics."""
        if cmd == "rg":
            return self._resolve_command(cmd) is not None
        if cmd not in self._command_cache:
            result = self._exec(
                f"command -v {cmd} >/dev/null 2>&1 && echo 'yes'"
            )
            self._command_cache[cmd] = result.stdout.strip() == "yes"
        return self._command_cache[cmd]

    def _modified_rg_capability_error(self, executable: str) -> Optional[str]:
        """Return a cached actionable error unless rg can sort exactly."""
        if executable in self._rg_modified_capability:
            return self._rg_modified_capability[executable]
        quoted = self._quote_executable(executable)
        result = self._exec(f"{quoted} --version", timeout=10)
        match = re.search(r"(?im)^ripgrep\s+(\d+)(?:\.|\s|$)", result.stdout or "")
        if result.exit_code == 0 and match and int(match.group(1)) >= 14:
            error = None
        else:
            error = (
                "Exact modification-time order requires ripgrep 14 or newer; "
                "upgrade ripgrep or use order='discovery'."
            )
        self._rg_modified_capability[executable] = error
        return error

    def _quote_executable(self, executable: str) -> str:
        """Quote an executable without leaking controller path semantics."""
        if re.fullmatch(r"[A-Za-z0-9_.-]+", executable):
            return executable
        from tools.environments.local import LocalEnvironment

        if isinstance(self.env, LocalEnvironment):
            return self._escape_native_tool_arg(executable)
        return "'" + executable.replace("'", "'\"'\"'") + "'"
    
    def _sample_file_bytes(self, path: str, length: int = 1000):
        """Fetch the first ``length`` raw bytes of a file through the terminal.

        File operations run through a terminal backend (possibly remote), so
        raw bytes cannot cross the transport directly — the terminal decodes
        stdout with ``errors="replace"`` and manufactures U+FFFD at every
        byte it cannot decode, including a multibyte character cut in half by
        ``head -c``. Wrapping the sample in base64 lets the original bytes
        survive the transport, so binary detection can happen at the byte
        layer where it is well-defined (#80308 and friends).

        Returns the sample bytes, or ``None`` when the transport could not
        produce clean base64 (exotic shells without ``base64``); callers fall
        back to the legacy text-sample heuristic in that case.
        """
        result = self._exec(
            f"head -c {length} {self._escape_shell_arg(path)} 2>/dev/null | base64"
        )
        if result.exit_code != 0:
            return None
        return self._decode_base64_sample(result.stdout)

    @staticmethod
    def _decode_base64_sample(text: str) -> Optional[bytes]:
        """Decode one base64 sample as emitted by ``head -c N | base64``.

        Whitespace-joins the whole text first (``base64`` wraps at 76
        columns), so callers must hand over exactly one segment; anything
        else in the text fails validation and yields ``None``, which sends
        the caller to the legacy text-sample heuristic.
        """
        encoded = _strip_terminal_fence_leaks(text)
        encoded = "".join(encoded.split())
        if not encoded:
            return b""
        if not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", encoded):
            return None
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return None

    @staticmethod
    def _is_likely_binary_bytes(sample: bytes) -> bool:
        """Byte-layer binary detection (the boundary for the #80308 class).

        Contract: a file is text when its sample is valid UTF-8, allowing one
        incomplete multibyte sequence at the very end (an artifact of cutting
        the sample at a byte boundary, not a property of the file). Anything
        else — NUL bytes, mid-stream invalid UTF-8 such as latin-1 or true
        binaries — stays read-only, preserving the anti-mojibake guarantee
        the old U+FFFD check existed for: a read→edit→write round-trip must
        never rewrite undecodable bytes with replacement characters.

        A file that legitimately *contains* U+FFFD (EF BF BD — e.g. logs of
        lossy output) is valid UTF-8 and reads as text; the old text-layer
        check misclassified it because it could not tell a stored replacement
        character from a transport-manufactured one.
        """
        if not sample:
            return False
        if b"\x00" in sample:
            return True
        try:
            sample.decode("utf-8")
            return False
        except UnicodeDecodeError as exc:
            # UTF-8 sequences are at most 4 bytes: an error starting in the
            # last 3 bytes with a clean prefix is a boundary cut, not binary.
            if exc.start >= len(sample) - 3:
                try:
                    sample[: exc.start].decode("utf-8")
                    return False
                except UnicodeDecodeError:
                    pass
            return True

    def _is_likely_binary(self, path: str, content_sample: str = None) -> bool:
        """
        Check if a file is likely binary.
        
        Uses extension check (fast) + content analysis (fallback).
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in BINARY_EXTENSIONS:
            return True
        
        # Content analysis: >30% non-printable chars = binary
        if content_sample:
            # Undecodable bytes: the terminal env decodes stdout with
            # errors="replace", so any non-UTF-8 byte arrives here already
            # turned into U+FFFD. That char is "printable" (ord 65533), so the
            # non-printable ratio below never catches it — and returning the
            # lossy text would let a read→edit→write round-trip silently
            # overwrite the original bytes with mojibake. Treat a file whose
            # sample carries the replacement char as binary (read-only) so the
            # agent can't corrupt it. Legitimate UTF-8 text effectively never
            # contains U+FFFD.
            if "\ufffd" in content_sample[:1000]:
                return True
            non_printable = sum(1 for c in content_sample[:1000]
                               if ord(c) < 32 and c not in '\n\r\t')
            return non_printable / min(len(content_sample), 1000) > 0.30
        
        return False
    
    def _is_image(self, path: str) -> bool:
        """Check if file is an image we can return as base64."""
        ext = os.path.splitext(path)[1].lower()
        return ext in IMAGE_EXTENSIONS
    
    def _add_line_numbers(self, content: str, start_line: int = 1) -> str:
        """Add line numbers to content in ``LINE_NUM|CONTENT`` format.

        The gutter uses a compact ``<n>|`` prefix (e.g. ``34|foo``) rather
        than a fixed-width zero/space-padded one (``    34|foo``). The
        padding was pure token overhead: on dense source the padded gutter
        cost ~48% more tokens than the bare content and ~16% more than the
        compact form, because the leading spaces + zero-padding tokenize
        into extra tokens on every single line. An A/B (Sonnet 4.6, 2
        passes) showed the compact gutter matches the padded gutter on
        line-reference / patch / value-lookup / structure tasks (4/4 both),
        while dropping line numbers entirely regressed line-referencing
        (the model hand-counted and was off-by-one, 3/4) — so we keep the
        numbers, just not the padding.
        """
        from tools.tool_output_limits import get_max_line_length
        max_line_length = get_max_line_length()
        lines = content.split('\n')
        numbered = []
        for i, line in enumerate(lines, start=start_line):
            # Truncate long lines
            if len(line) > max_line_length:
                line = line[:max_line_length] + "... [truncated]"
            numbered.append(f"{i}|{line}")
        return '\n'.join(numbered)
    
    def _expand_path(self, path: str) -> str:
        """
        Expand shell-style paths like ~ and ~user to absolute paths.
        
        This must be done BEFORE shell escaping, since ~ doesn't expand
        inside single quotes.
        """
        if not path:
            return path
        
        # Handle ~ and ~user
        if path.startswith('~'):
            # Get home directory via the terminal environment
            result = self._exec("echo $HOME")
            if result.exit_code == 0 and result.stdout.strip():
                home = result.stdout.strip()
                if path == '~':
                    return home
                elif path.startswith('~/'):
                    return home + path[1:]  # Replace ~ with home
                # ~username format - extract and validate username before
                # letting shell expand it (prevent shell injection via
                # paths like "~; rm -rf /").
                rest = path[1:]  # strip leading ~
                slash_idx = rest.find('/')
                username = rest[:slash_idx] if slash_idx >= 0 else rest
                if username and re.fullmatch(r'[a-zA-Z0-9._-]+', username):
                    # Only expand ~username (not the full path) to avoid shell
                    # injection via path suffixes like "~user/$(malicious)".
                    expand_result = self._exec(f"echo ~{username}")
                    if expand_result.exit_code == 0 and expand_result.stdout.strip():
                        user_home = expand_result.stdout.strip()
                        suffix = path[1 + len(username):]  # e.g. "/rest/of/path"
                        return user_home + suffix
        
        return path
    
    def _escape_shell_arg(self, arg: str) -> str:
        """Escape a string for safe use in shell commands.

        On Windows native drive paths (``C:\\Users\\x`` / ``C:/Users/x``)
        and mixed MSYS leftovers (``/c/Users\\x``) are rewritten to the
        Git Bash ``/c/Users/x`` form via ``_bash_safe_path``: bash eats
        backslashes and MSYS otherwise mangles drive paths into the
        ``Directory \\drivers\\etc does not exist`` failure class. Reuses
        the env-layer translator so shell file ops and the terminal ``cd``
        agree on the path form. No-op off Windows and for plain POSIX paths.
        """
        from tools.environments.local import _bash_safe_path

        arg = _bash_safe_path(arg)
        # Use single quotes and escape any single quotes in the string
        return "'" + arg.replace("'", "'\"'\"'") + "'"

    def _escape_native_tool_arg(self, arg: str) -> str:
        """Escape a path argument destined for a NATIVE Windows binary.

        ``_escape_shell_arg`` rewrites Windows paths to the Git Bash MSYS
        form (``/c/Users/x``) so bash builtins resolve them. But native
        Windows binaries invoked from that bash (ripgrep installed via
        winget/cargo/choco, native git, etc.) do not understand ``/c/...``
        paths — and Hermes disables MSYS argument conversion for its bash
        subprocesses (``MSYS_NO_PATHCONV=1`` / ``MSYS2_ARG_CONV_EXCL=*``,
        see ``_apply_windows_msys_bash_env_defaults``), so nothing ever
        translates the MSYS form back. The native tool then fails with
        ``The system cannot find the path specified. (os error 3)``.

        The forward-slash native form (``C:/Users/x``) is the one spelling
        every layer accepts: bash passes it through untouched (it is not an
        absolute POSIX path, so no conversion applies even without the
        opt-outs), and Windows APIs treat ``/`` and ``\\`` as equivalent
        separators. MSYS builds of the same tools accept it too, so this is
        safe regardless of which flavor of the binary is installed.

        On non-Windows hosts this is exactly ``_escape_shell_arg``.
        """
        from tools.environments.local import _IS_WINDOWS, _msys_to_windows_path

        if _IS_WINDOWS and arg:
            arg = _msys_to_windows_path(arg).replace("\\", "/")
        return "'" + arg.replace("'", "'\"'\"'") + "'"

    def _atomic_write(self, path: str, content: str) -> "ExecuteResult":
        """Write ``content`` to ``path`` atomically via temp-file + rename.

        Streams ``content`` over stdin into a temp file in the SAME
        directory as ``path`` (so the final ``mv`` is a real rename on the
        same filesystem, not a non-atomic cross-device copy), preserves the
        existing file's mode if it exists, then renames over the target.
        On any failure the temp file is removed so we never leak a partial
        ``.hermes-tmp`` file next to the user's data, and the original file
        is left untouched. Content rides stdin so there is no ARG_MAX limit.

        ``mkdir -p`` for the parent directory is folded into this script
        (one fewer subprocess vs. a separate ``mkdir -p`` call).

        Returns an :class:`ExecuteResult`; ``exit_code == 0`` means the file
        was swapped into place atomically. A non-zero exit means nothing was
        renamed and the original (if any) is intact.
        """
        q_path = self._escape_shell_arg(path)
        parent = os.path.dirname(path) or "."
        q_parent = self._escape_shell_arg(parent)
        # template basename: hidden so it doesn't show up in casual `ls`,
        # carries a marker so an orphaned temp (only possible on a hard
        # crash *between* cat and mv) is identifiable.
        tmpl = self._escape_shell_arg(".hermes-tmp.XXXXXX")

        # One shell script, fully quoted. Notes:
        #  - `mkdir -p "$d"` is folded in here so the parent directory is
        #    created in the same subprocess that writes the temp file —
        #    saves one entire subprocess spawn vs. a separate mkdir call.
        #  - `mktemp` lands the temp in the target's own dir (-p) so `mv` is
        #    same-FS atomic; we fall back to a PID-stamped name if the
        #    backend lacks mktemp (rare; busybox/macOS/Linux all ship it).
        #  - `chmod --reference` is GNU-only, so we read the octal mode with
        #    `stat` (GNU `-c%a` or BSD `-f%Lp`) and `chmod` it explicitly;
        #    silent best-effort — a perms-copy failure must not abort the
        #    write (the file then lands at mktemp's 0600, same as pre-fix).
        #  - brand-new targets get `chmod "=rw"` — the POSIX who-less
        #    symbolic form, which sets rw minus the process umask (e.g.
        #    0644 under umask 022) instead of mktemp's hardcoded 0600
        #    (#70856).  Deliberately NOT shell arithmetic on `$(umask)`:
        #    zsh (reachable via _find_bash's $SHELL fallback) parses
        #    leading-zero constants as decimal and silently computes a
        #    garbage mode, while `chmod "=rw"` is spec-identical in
        #    bash/dash/ash/zsh and degrades to 0600 (pre-fix behavior)
        #    if an exotic chmod rejects it.
        #  - `trap ... EXIT` guarantees the temp is removed on every error
        #    path (cat failure, mv failure, signal) but NOT after a
        #    successful mv (the temp no longer exists by then).
        #  - we `cat >` the temp, then `mv -f` it over the target.
        script = (
            "set -e; "
            f"d={q_parent}; t={q_path}; "
            # Follow a symlink target so we edit the file the link points at,
            # rather than replacing the symlink itself with a plain file (which
            # orphans the real target and destroys the link). Recompute the
            # temp dir from the RESOLVED target so `mv` stays same-filesystem
            # atomic. Best-effort: a broken link or missing readlink/realpath
            # falls back to the original path (pre-fix behavior, no regression).
            'if [ -L "$t" ]; then '
            'rt="$(readlink -f "$t" 2>/dev/null || realpath "$t" 2>/dev/null || true)"; '
            '[ -n "$rt" ] && { t="$rt"; d="$(dirname "$t")"; }; '
            "fi; "
            # Create the parent dir in the SAME subprocess that writes the
            # temp file (one fewer exec vs. a separate mkdir call). Runs
            # AFTER symlink resolution so a resolved target's directory is
            # the one created/confirmed.
            'mkdir -p "$d"; '
            'tmp="$(mktemp -p "$d" ' + tmpl + ' 2>/dev/null '
            '|| mktemp "$d/.hermes-tmp.$$.XXXXXX" 2>/dev/null '
            '|| { tmp="$d/.hermes-tmp.$$"; : > "$tmp" && echo "$tmp"; })"; '
            '[ -n "$tmp" ] || { echo "atomic write: could not create temp file" >&2; exit 1; }; '
            "trap 'rm -f \\\"$tmp\\\"' EXIT; "
            # preserve mode of an existing target (best-effort, never fatal)
            'if [ -e "$t" ]; then '
            'm="$(stat -c%a "$t" 2>/dev/null || stat -f%Lp "$t" 2>/dev/null || true)"; '
            '[ -n "$m" ] && chmod "$m" "$tmp" 2>/dev/null || true; '
            "fi; "
            'cat > "$tmp"; '
            # new file: umask-default perms instead of mktemp's 0600 (#70856).
            # Runs AFTER cat so a write-masking umask can't EACCES the stream;
            # quoted "=rw" so zsh doesn't =word-expand it.
            'if [ ! -e "$t" ]; then chmod "=rw" "$tmp" 2>/dev/null || true; fi; '
            'mv -f "$tmp" "$t"; '
            "trap - EXIT"
        )
        return self._exec(script, stdin_data=content)

    def _detect_file_line_ending(self, path: str, pre_content: Optional[str] = None) -> Optional[str]:
        """Detect the dominant line ending of a file on disk.

        If ``pre_content`` is already available (we just read the file
        for lint/LSP purposes), inspect that — zero extra exec calls.
        Otherwise issue a tiny ``head -c 4096`` to sample the first 4KB.

        Returns ``"\\r\\n"`` for CRLF (Windows), ``"\\n"`` for LF (Unix),
        or ``None`` if undetermined (new file, empty file, single-line
        file with no line break in the first chunk).
        """
        if pre_content:
            return _detect_line_ending(pre_content)
        # File may not exist (new write) — `head` exits 0 with empty
        # stdout in that case which yields None below.  Cheap probe.
        head_cmd = f"head -c 4096 {self._escape_shell_arg(path)} 2>/dev/null"
        head_result = self._exec(head_cmd)
        if head_result.exit_code != 0 or not head_result.stdout:
            return None
        return _detect_line_ending(head_result.stdout)

    def _file_has_bom(self, path: str, pre_content: Optional[str] = None) -> bool:
        """Whether the file on disk starts with a UTF-8 BOM.

        Always probes the first 3 bytes on disk — do NOT trust
        ``pre_content`` for BOM detection because the most common
        provider (``read_file_raw``) deliberately strips BOMs so the
        agent never sees U+FEFF glyphs.  Passing BOM-stripped content
        through ``pre_content`` would cause a false-negative and
        silently remove the marker on rewrite.

        A missing/empty file returns False (new writes get no BOM
        unless the caller explicitly includes one).
        """
        head_cmd = f"head -c 3 {self._escape_shell_arg(path)} 2>/dev/null"
        head_result = self._exec(head_cmd)
        if head_result.exit_code != 0 or not head_result.stdout:
            return False
        return _has_bom(head_result.stdout)


    def _unified_diff(self, old_content: str, new_content: str, filename: str) -> str:
        """Generate unified diff between old and new content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}"
        )
        return ''.join(diff)
    
    # =========================================================================
    # READ Implementation
    # =========================================================================
    
    def _size_probe_cmd(self, path: str) -> str:
        """Byte size of a regular file, without opening one that never ends.

        ``wc -c < path`` opens the path. On a FIFO with no writer, a socket,
        or a character device like /dev/zero that never reaches EOF, that
        read blocks forever — and the read helpers pass no timeout to
        :meth:`_exec`, so the turn wedges until the process is killed. The
        device blocklist in ``tools/file_tools.py`` cannot cover this: it
        matches literal ``/dev/*`` names, while a FIFO is a file *type* and
        can sit at any path.

        ``[ -f ]`` is a stat, not an open — it answers exactly the question
        the size probe needs (regular file, symlinks followed) without
        touching the contents. Non-regular paths that exist report the
        sentinel so callers can say so instead of claiming the file is
        missing; a genuinely absent path still exits non-zero.
        """
        arg = self._escape_shell_arg(path)
        return (
            f"if [ -f {arg} ]; then wc -c < {arg} 2>/dev/null; "
            f"elif [ -e {arg} ]; then echo {NOT_REGULAR_SENTINEL}; "
            f"else exit 1; fi"
        )

    @staticmethod
    def _not_regular_error(path: str) -> ReadResult:
        """Error for a path that exists but would block if read."""
        return ReadResult(
            error=(
                f"Cannot read '{path}': not a regular file (directory, FIFO, "
                "socket, or device). Reading it could block indefinitely."
            )
        )

    # UTF-16 rescue constants (ported from MoonshotAI/kimi-code#2647,
    # detection derived from VS Code's encoding sniffer): sample the leading
    # bytes; trust a BOM first, then a zero-byte parity heuristic — zeros
    # clustering at odd indices mean UTF-16 LE (`0xAA 0x00`), at even indices
    # UTF-16 BE (`0x00 0xAA`). Only the *placement* of zeros is checked, not
    # density, so mixed Latin/CJK content (whose CJK units carry no zero
    # byte) still detects. Zeros at both parities, or a single isolated
    # zero, mean real binary. Legacy 8-bit encodings (GBK, Big5, ...) are
    # never guessed — a wrong silent guess is worse than a clear refusal.
    _UTF16_MAX_BYTES = 10 * 1024 * 1024
    _UTF16_SAMPLE_BYTES = 512

    def _try_read_utf16(self, path: str, offset: int, limit: int,
                        file_size: int) -> "Optional[ReadResult]":
        """Attempt to read ``path`` as UTF-16 text, transcoded to UTF-8.

        Returns a populated ``ReadResult`` when the file is UTF-16 (BOM or
        zero-byte parity heuristic), else ``None`` so the caller falls back
        to the binary-file error. Files over 10 MiB are not rescued.
        ``path`` must already be expanded (caller ran ``_expand_path``).
        """
        # Extensions that are definitively binary (images, archives, ...)
        # never contain UTF-16 text worth rescuing — skip the subprocess.
        ext = os.path.splitext(path)[1].lower()
        if ext in BINARY_EXTENSIONS:
            return None
        if file_size > self._UTF16_MAX_BYTES:
            return None

        snippet = (
            "import sys, json, os\n"
            f"p = {path!r}\n"
            f"offset = {int(offset)}\n"
            f"limit = {int(limit)}\n"
            f"MAX = {self._UTF16_MAX_BYTES}\n"
            f"SAMPLE = {self._UTF16_SAMPLE_BYTES}\n"
            "try:\n"
            "    size = os.path.getsize(p)\n"
            "    if size > MAX:\n"
            "        print('HERMES_UTF16:NO'); sys.exit(0)\n"
            "    with open(p, 'rb') as f:\n"
            "        data = f.read()\n"
            "    sample = data[:SAMPLE]\n"
            "    enc = None\n"
            "    if sample[:2] == b'\\xfe\\xff':\n"
            "        enc = 'utf-16-be'\n"
            "    elif sample[:2] == b'\\xff\\xfe':\n"
            "        enc = 'utf-16-le'\n"
            "    else:\n"
            "        odd = sum(1 for i in range(1, len(sample), 2) if sample[i] == 0)\n"
            "        even = sum(1 for i in range(0, len(sample), 2) if sample[i] == 0)\n"
            "        if even == 0 and odd >= 2:\n"
            "            enc = 'utf-16-le'\n"
            "        elif odd == 0 and even >= 2:\n"
            "            enc = 'utf-16-be'\n"
            "    if enc is None:\n"
            "        print('HERMES_UTF16:NO'); sys.exit(0)\n"
            "    text = data.decode(enc, 'replace')\n"
            "    if text[:1] == '\\ufeff':\n"
            "        text = text[1:]\n"
            "    text = text.replace('\\r\\n', '\\n')\n"
            "    lines = text.split('\\n')\n"
            "    total = len(lines)\n"
            "    sel = lines[offset - 1: offset - 1 + limit]\n"
            "    out = {'total_lines': total, 'encoding': enc,\n"
            "           'content': '\\n'.join(sel)}\n"
            "    print('HERMES_UTF16:OK')\n"
            "    print(json.dumps(out, ensure_ascii=True))\n"
            "except Exception:\n"
            "    print('HERMES_UTF16:NO'); sys.exit(0)\n"
        )

        result = self._exec(f"python3 -c {self._escape_shell_arg(snippet)}")
        if result.exit_code != 0 and "python3" in (result.stdout or ""):
            result = self._exec(f"python -c {self._escape_shell_arg(snippet)}")

        stdout = _strip_terminal_fence_leaks(result.stdout or "")
        marker = stdout.find("HERMES_UTF16:OK")
        if result.exit_code != 0 or marker < 0:
            return None
        payload = stdout[marker + len("HERMES_UTF16:OK"):].strip()
        try:
            data = json.loads(payload.split("\n", 1)[0] if "\n" in payload else payload)
            content = data["content"]
            total_lines = int(data["total_lines"])
            encoding = str(data.get("encoding", "utf-16"))
        except (ValueError, KeyError, TypeError):
            return None

        end_line = offset + limit - 1
        truncated = total_lines > end_line
        hint_parts = [f"Transcoded from {encoding.upper()} to UTF-8 for display. "
                      "Text edits via patch/write_file would re-encode as UTF-8."]
        if truncated:
            hint_parts.append(
                f"Use offset={end_line + 1} to continue reading "
                f"(showing {offset}-{end_line} of {total_lines} lines)"
            )
        return ReadResult(
            content=self._add_line_numbers(content, offset),
            total_lines=total_lines,
            file_size=file_size,
            truncated=truncated,
            hint=" ".join(hint_parts),
        )

    def read_file(self, path: str, offset: int = 1, limit: int = 2000) -> ReadResult:
        """
        Read a file with pagination, binary detection, and line numbers.

        Args:
            path: File path (absolute or relative to cwd)
            offset: Line number to start from (1-indexed, default 1)
            limit: Maximum lines to return (default 500, max 2000)

        Returns:
            ReadResult with content, metadata, or error info

        One shell round-trip answers every question the read needs:
        existence, size, binary sample, the page, line count, trailing
        newline (see ``_read_probe_cmd``). A reply that cannot be parsed
        falls back to ``_read_file_sequential``, the one-probe-per-call
        form, so an exotic shell can never do worse than before. On a local
        POSIX environment the read never touches the shell at all; see
        ``_read_file_native``.
        """
        # Expand ~ and other shell paths
        path = self._expand_path(path)

        offset, limit = normalize_read_pagination(offset, limit)

        if self._native_read_enabled():
            return self._read_file_native(path, offset, limit)

        # Images and known-binary extensions never inline content; the
        # sequential path stops at the probes for them, so nothing is gained
        # by streaming their bytes through the page pipeline.
        if self._is_image(path) or os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS:
            return self._read_file_sequential(path, offset, limit)

        from tools.tool_output_limits import get_max_line_length
        line_clamp_bytes = 4 * get_max_line_length() + 1
        end_line = offset + limit - 1
        sentinel = _new_sentinel(_READ_SENTINEL_PREFIX)
        probe = self._exec(
            self._read_probe_cmd(path, offset, end_line, line_clamp_bytes, sentinel)
        )
        output = probe.stdout or ""

        if sentinel not in output:
            # Single-line replies: the path is missing or not a regular file.
            marker = _strip_terminal_fence_leaks(output).strip()
            if marker == MISSING_SENTINEL:
                return self._read_file_missing(path, offset, limit)
            if marker == NOT_REGULAR_SENTINEL:
                return self._not_regular_error(path)
            logger.debug(
                "read_file: compound probe reply for %s has no sentinel "
                "(exit %s, %d chars); falling back to sequential probes",
                path, probe.exit_code, len(output),
            )
            return self._read_file_sequential(path, offset, limit)

        segments = _split_segments(output, sentinel)
        if probe.exit_code != 0 or len(segments) != 6:
            logger.debug(
                "read_file: compound probe for %s returned exit %s with %d "
                "segments (want 6); falling back to sequential probes",
                path, probe.exit_code, len(segments),
            )
            return self._read_file_sequential(path, offset, limit)
        size_seg, sample_seg, page_seg, wc_seg, tail_seg, status_seg = segments

        status = _strip_terminal_fence_leaks(status_seg).split()
        try:
            sample_rc, read_rc = int(status[0]), int(status[1])
        except (IndexError, ValueError):
            logger.debug(
                "read_file: compound probe for %s has unparseable status %r; "
                "falling back to sequential probes",
                path, status_seg[-40:],
            )
            return self._read_file_sequential(path, offset, limit)

        try:
            file_size = int(_strip_terminal_fence_leaks(size_seg).strip())
        except ValueError:
            file_size = 0

        # Byte-layer binary detection when base64 was available, else the
        # legacy text heuristic over a plain sample: one extra round-trip,
        # paid only on shells without base64.
        sample_bytes = self._decode_base64_sample(sample_seg) if sample_rc == 0 else None
        if sample_bytes is not None:
            is_binary = self._is_likely_binary_bytes(sample_bytes)
        else:
            logger.debug(
                "read_file: no usable base64 sample for %s (base64 exit %s); "
                "paying one extra round-trip for the text heuristic",
                path, sample_rc,
            )
            sample_cmd = f"head -c 1000 {self._escape_shell_arg(path)} 2>/dev/null"
            sample_result = self._exec(sample_cmd)
            sample_output = _strip_terminal_fence_leaks(sample_result.stdout)
            is_binary = self._is_likely_binary(path, sample_output)

        if is_binary:
            return self._read_binary_file(path, offset, limit, file_size, sample_bytes)

        if read_rc != 0:
            return ReadResult(
                error=f"Failed to read file: {_strip_terminal_fence_leaks(page_seg)}"
            )

        read_output = _strip_terminal_fence_leaks(page_seg)
        try:
            total_lines = int(_strip_terminal_fence_leaks(wc_seg).strip())
        except ValueError:
            total_lines = 0
        tail_flag = _strip_terminal_fence_leaks(tail_seg).strip()
        file_ends_with_newline = tail_flag == "1" if tail_flag in ("0", "1") else None

        return self._assemble_read_result(
            read_output,
            offset=offset,
            end_line=end_line,
            total_lines=total_lines,
            file_size=file_size,
            file_ends_with_newline=file_ends_with_newline,
        )

    def _native_read_enabled(self) -> bool:
        """Whether ``read_file`` may bypass the shell and read from this host.

        Only on POSIX with a ``LocalEnvironment``: the file is on this
        machine and the path is already in native form. Windows keeps the
        shell path, since file_operations holds Git-Bash-style paths there.
        ``HERMES_NATIVE_FILE_READ=0`` turns the fast path off.
        """
        flag = os.environ.get("HERMES_NATIVE_FILE_READ", "1").strip().lower()
        if flag in ("0", "false", "no", "off"):
            return False
        # Same "is this env the local host" test the LSP path already uses;
        # ``self.env`` is bound once in __init__ and never rebound, and the
        # isinstance check is microseconds, so there is nothing to memoize.
        return sys.platform != "win32" and self._lsp_local_only()

    def _read_file_native(self, path: str, offset: int, limit: int) -> ReadResult:
        """``read_file`` without a shell: the file lives on this host.

        Same contract as the shell path, byte for byte. ``os.stat`` is the
        ``[ -f ]`` guard (a stat, never an open, so FIFOs and devices are
        refused before anything touches their contents); the first 1000
        bytes drive the byte-layer binary check; the page is produced
        exactly as ``sed -n 'a,bp' | cut -b1-N`` prints it (every line
        clamped to N bytes and newline-terminated), then decoded with
        errors="replace" like the terminal transport. One chunked pass
        counts lines and collects the page, so neither the file nor a
        single pathological line is ever held in memory whole.

        ``path`` is already expanded and ``offset``/``limit`` normalized.
        Anything unexpected from the OS hands over to the shell path.
        """
        import stat as _stat

        full = path if os.path.isabs(path) else os.path.join(
            getattr(self.env, "cwd", None) or self.cwd, path
        )
        try:
            st = os.stat(full)
        except (FileNotFoundError, NotADirectoryError):
            return self._read_file_missing(path, offset, limit)
        except OSError:
            return self._read_file_sequential(path, offset, limit)
        if not _stat.S_ISREG(st.st_mode):
            return self._not_regular_error(path)
        file_size = st.st_size

        # Images are never inlined: redirect to the vision tool
        if self._is_image(path):
            return self._image_redirect_result(file_size)

        from tools.tool_output_limits import get_max_line_length
        clamp = 4 * get_max_line_length() + 1
        end_line = offset + limit - 1

        page: List[bytes] = []
        total_lines = 0
        lineno = 1              # the line currently being scanned
        kept = bytearray()      # first ``clamp`` bytes of that line
        have_partial = False    # that line has bytes but no newline yet
        last_byte = b""
        try:
            with open(full, "rb") as fh:
                sample = fh.read(1000)
                ext_binary = os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS
                if ext_binary or self._is_likely_binary_bytes(sample):
                    return self._read_binary_file(path, offset, limit, file_size, sample)
                fh.seek(0)
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    last_byte = chunk[-1:]
                    if lineno > end_line:
                        # Past the requested window: only the line count and
                        # trailing byte are still needed, so let memchr do the
                        # rest instead of the per-line bookkeeping below.
                        total_lines += chunk.count(b"\n")
                        have_partial = chunk[-1:] != b"\n"
                        continue
                    pos, n = 0, len(chunk)
                    while pos < n:
                        nl = chunk.find(b"\n", pos)
                        in_page = offset <= lineno <= end_line
                        if nl < 0:
                            if in_page and len(kept) < clamp:
                                kept += chunk[pos:pos + (clamp - len(kept))]
                            have_partial = True
                            break
                        if in_page:
                            if len(kept) < clamp:
                                kept += chunk[pos:min(nl, pos + (clamp - len(kept)))]
                            page.append(bytes(kept) + b"\n")
                        kept = bytearray()
                        have_partial = False
                        total_lines += 1
                        lineno += 1
                        pos = nl + 1
        except OSError:
            return self._read_file_sequential(path, offset, limit)
        if have_partial and offset <= lineno <= end_line:
            # ``sed`` prints a final line that lacks a newline; ``cut`` adds one.
            page.append(bytes(kept) + b"\n")

        read_output = _strip_terminal_fence_leaks(
            b"".join(page).decode("utf-8", errors="replace")
        )
        return self._assemble_read_result(
            read_output,
            offset=offset,
            end_line=end_line,
            total_lines=total_lines,
            file_size=file_size,
            file_ends_with_newline=(last_byte == b"\n") if file_size else None,
        )

    @staticmethod
    def _image_redirect_result(file_size: int) -> ReadResult:
        return ReadResult(
            is_image=True,
            is_binary=True,
            file_size=file_size,
            hint=(
                "Image file detected. Automatically redirected to vision_analyze tool. "
                "Use vision_analyze with this file path to inspect the image contents."
            ),
        )

    def _read_probe_cmd(self, path: str, offset: int, end_line: int,
                        line_clamp_bytes: int, sentinel: str) -> str:
        """One shell command answering every question ``read_file`` asks.

        Six segments, each closed by a ``sentinel`` line: byte size, base64
        of the first 1000 bytes, the ``sed | cut`` page, ``wc -l``, whether
        the last byte is a newline, then the base64 and page pipeline
        statuses. The probes run only inside ``[ -f ]``, the same
        stat-not-open guard as ``_size_probe_cmd``, so a FIFO or device
        never reaches ``head``/``sed``. A missing path echoes
        ``MISSING_SENTINEL`` instead of exiting non-zero, because a compound
        command only reports its last status. Every stage silences stderr:
        the local backend merges stderr into stdout and a stray diagnostic
        would otherwise land inside a segment.

        The page clamp is byte-based on purpose; see ``_read_file_sequential``
        for why it is ``4 * max_line_length + 1``.
        """
        arg = self._escape_shell_arg(path)
        mark = f"echo {sentinel}"
        return (
            f"if [ -f {arg} ]; then "
            f"wc -c < {arg} 2>/dev/null; {mark}; "
            f"head -c 1000 {arg} 2>/dev/null | base64 2>/dev/null; __hs=$?; {mark}; "
            f"sed -n '{offset},{end_line}p' {arg} 2>/dev/null"
            f" | cut -b1-{line_clamp_bytes} 2>/dev/null; __hr=$?; {mark}; "
            f"wc -l < {arg} 2>/dev/null; {mark}; "
            f"tail -c 1 {arg} 2>/dev/null | wc -l; {mark}; "
            f'echo "$__hs $__hr"; '
            f"elif [ -e {arg} ]; then echo {NOT_REGULAR_SENTINEL}; "
            f"else echo {MISSING_SENTINEL}; fi"
        )

    def _read_file_missing(self, path: str, offset: int, limit: int) -> ReadResult:
        """Not-found recovery shared by every read path.

        Before failing, try unicode-equivalent spellings: NFC/NFD, narrow
        no-break space, curly quotes render identically in a terminal, so
        the model retyping a visually-correct path can never discover the
        byte mismatch on its own (retrying is the tool's job, not the
        model's). No equivalent spelling → suggest similar files.
        """
        variant = self._unicode_variant_match(path)
        if variant is not None:
            result = self.read_file(variant, offset=offset, limit=limit)
            note = (
                f"Note: '{path}' not found byte-for-byte; resolved to "
                f"the unicode-equivalent file '{variant}' (invisible "
                "encoding difference: NFC/NFD or special space/quote "
                "characters)."
            )
            result.hint = f"{note} {result.hint}" if result.hint else note
            return result
        return self._suggest_similar_files(path)

    def _read_binary_file(self, path: str, offset: int, limit: int,
                          file_size: int, sample_bytes: Optional[bytes]) -> ReadResult:
        """Binary branch shared by every read path.

        UTF-16 rescue (ported from MoonshotAI/kimi-code#2647): the terminal
        env decodes stdout as UTF-8 with errors="replace", so a UTF-16 text
        file (Windows Notepad .txt, PowerShell `>` redirects) arrives
        mangled with U+FFFD and trips the binary guard. Probe the raw bytes
        via the backend's Python and transcode to UTF-8 when a BOM or the
        zero-byte parity heuristic identifies UTF-16.
        """
        utf16_result = self._try_read_utf16(path, offset, limit, file_size)
        if utf16_result is not None:
            return utf16_result
        return ReadResult(
            is_binary=True,
            file_size=file_size,
            error=describe_binary_file(sample_bytes, file_size),
        )

    def _read_file_sequential(self, path: str, offset: int, limit: int) -> ReadResult:
        """One-probe-per-call read: the pre-compound form, kept as fallback.

        ``read_file`` lands here for image / known-binary extensions (only
        the probes matter) and whenever the compound reply cannot be parsed.
        ``path`` is already expanded and ``offset``/``limit`` normalized.
        """
        # Check if file exists and get size (POSIX, works on Linux + macOS)
        stat_result = self._exec(self._size_probe_cmd(path))

        if stat_result.exit_code != 0:
            return self._read_file_missing(path, offset, limit)

        stat_output = _strip_terminal_fence_leaks(stat_result.stdout)
        if stat_output.strip() == NOT_REGULAR_SENTINEL:
            return self._not_regular_error(path)
        try:
            file_size = int(stat_output.strip())
        except ValueError:
            file_size = 0

        # Check if file is too large
        if file_size > MAX_FILE_SIZE:
            # Still try to read, but warn
            pass

        # Images are never inlined — redirect to the vision tool
        if self._is_image(path):
            return self._image_redirect_result(file_size)

        # Read a sample to check for binary content — at the byte layer when
        # the transport allows, falling back to the legacy text heuristic.
        sample_bytes = self._sample_file_bytes(path)
        if sample_bytes is not None:
            ext_binary = os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS
            is_binary = ext_binary or self._is_likely_binary_bytes(sample_bytes)
        else:
            sample_cmd = f"head -c 1000 {self._escape_shell_arg(path)} 2>/dev/null"
            sample_result = self._exec(sample_cmd)
            sample_output = _strip_terminal_fence_leaks(sample_result.stdout)
            is_binary = self._is_likely_binary(path, sample_output)

        if is_binary:
            return self._read_binary_file(path, offset, limit, file_size, sample_bytes)

        # Read with pagination using sed, clamping each line to a byte
        # budget IN THE SHELL so a pathological single-line file (e.g. one
        # 400MB minified line) never crosses the exec transport. The Python
        # clamp in _add_line_numbers still runs afterwards; the shell clamp
        # only bounds what reaches it.
        #
        # Why 4*max_line_length + 1 bytes (not max_line_length + 1):
        # ``cut -c`` on GNU coreutils is byte-based despite its name, and a
        # byte clamp can split a multibyte UTF-8 codepoint at the boundary.
        # The transport decodes with errors="replace", so a split codepoint
        # becomes U+FFFD rather than an exception — but a clamp of
        # max_line_length+1 BYTES yields far fewer CHARS than
        # max_line_length for multibyte text, so the Python clamp would
        # never fire and truncation would be silent (no "... [truncated]"
        # suffix). UTF-8 codepoints are at most 4 bytes, so any line whose
        # first max_line_length chars survive occupies at most
        # 4*max_line_length bytes; keeping one byte more guarantees that
        # every line longer than max_line_length chars still decodes to
        # more than max_line_length chars, which triggers the existing
        # Python-side clamp (len(line) > max_line_length) and its
        # "... [truncated]" suffix. Any U+FFFD from a boundary split lands
        # beyond char max_line_length and is always removed by that clamp,
        # so mojibake is never visible. ``cut -b`` is used explicitly to
        # document the byte semantics.
        from tools.tool_output_limits import get_max_line_length
        line_clamp_bytes = 4 * get_max_line_length() + 1
        end_line = offset + limit - 1
        read_cmd = (
            f"sed -n '{offset},{end_line}p' {self._escape_shell_arg(path)}"
            f" | cut -b1-{line_clamp_bytes}"
        )
        read_result = self._exec(read_cmd)

        if read_result.exit_code != 0:
            return ReadResult(error=f"Failed to read file: {read_result.stdout}")
        read_output = _strip_terminal_fence_leaks(read_result.stdout)

        # Get total line count
        wc_cmd = f"wc -l < {self._escape_shell_arg(path)}"
        wc_result = self._exec(wc_cmd)
        wc_output = _strip_terminal_fence_leaks(wc_result.stdout)
        try:
            total_lines = int(wc_output.strip())
        except ValueError:
            total_lines = 0

        # Only the page that reaches the file's final line can carry the
        # ``cut`` newline artifact (see _assemble_read_result); probe the
        # last byte just for that case, exactly as before.
        file_ends_with_newline: Optional[bool] = None
        if not total_lines > end_line and read_output.endswith('\n'):
            tail_cmd = f"tail -c 1 {self._escape_shell_arg(path)} | wc -l"
            tail_result = self._exec(tail_cmd)
            tail_output = _strip_terminal_fence_leaks(tail_result.stdout)
            if tail_result.exit_code == 0:
                file_ends_with_newline = tail_output.strip() != "0"

        return self._assemble_read_result(
            read_output,
            offset=offset,
            end_line=end_line,
            total_lines=total_lines,
            file_size=file_size,
            file_ends_with_newline=file_ends_with_newline,
        )

    def _assemble_read_result(
        self,
        read_output: str,
        *,
        offset: int,
        end_line: int,
        total_lines: int,
        file_size: int,
        file_ends_with_newline: Optional[bool],
    ) -> ReadResult:
        """Turn a raw ``sed | cut`` page into the final ``ReadResult``.

        Shared by every read path so the BOM strip, pagination hint, the
        ``cut`` newline artifact fix and the ambiguous-silence guards can
        never drift apart. ``file_ends_with_newline`` is ``None`` when the
        caller could not tell (the artifact is then left alone, as before).
        """
        # Strip a leading UTF-8 BOM so the model never sees a phantom U+FEFF
        # before the first real character. Only meaningful on the first
        # chunk (the marker lives at byte 0); later pages can't carry it.
        if offset == 1:
            read_output, _ = _strip_bom(read_output)

        # Check if truncated
        truncated = total_lines > end_line
        hint = None
        if truncated:
            hint = f"Use offset={end_line + 1} to continue reading (showing {offset}-{end_line} of {total_lines} lines)"

        # ``cut`` (unlike sed -n p) always newline-terminates its output,
        # so a file whose final line has no trailing newline would grow a
        # phantom empty last line. Only possible when this page reaches the
        # file's final line; strip the artifact when the last byte says so.
        if (
            not truncated
            and read_output.endswith('\n')
            and file_ends_with_newline is False
        ):
            read_output = read_output[:-1]

        # Ambiguous-silence guards: an empty content string is
        # indistinguishable, from inside the model, from a broken tool —
        # it re-reads, widens the window, tries another path. Name the
        # dead end and its recovery instead.
        if file_size == 0:
            return ReadResult(
                content="",
                total_lines=0,
                file_size=0,
                hint="File is empty (0 bytes).",
            )
        if offset > total_lines > 0:
            return ReadResult(
                content="",
                total_lines=total_lines,
                file_size=file_size,
                hint=(
                    f"Note: offset {offset} is beyond the end of the file "
                    f"({total_lines} lines total). Retry with offset <= "
                    f"{total_lines}."
                ),
            )

        return ReadResult(
            content=self._add_line_numbers(read_output, offset),
            total_lines=total_lines,
            file_size=file_size,
            truncated=truncated,
            hint=hint
        )
    
    def _unicode_variant_match(self, path: str) -> Optional[str]:
        """Find an existing file whose name is unicode-equivalent to ``path``.

        macOS names screenshots with a NARROW NO-BREAK SPACE (U+202F) before
        AM/PM, stores names NFD-decomposed, and Finder renames turn ' into
        \u2019 — all invisible in rendered text. Compare directory entries
        under a normalization that erases exactly those differences and
        return the on-disk spelling when exactly one entry matches.
        """
        dir_path = os.path.dirname(path) or "."
        filename = os.path.basename(path)
        if not filename:
            return None

        def _canon(name: str) -> str:
            # NFC first so composed/decomposed collapse together, then the
            # confusable space/quote characters seen in real filenames.
            out = unicodedata.normalize("NFC", name)
            for src, dst in (
                ("\u202f", " "),  # narrow no-break space
                ("\u00a0", " "),  # no-break space
                ("\u2019", "'"),  # right single quotation mark
                ("\u2018", "'"),  # left single quotation mark
            ):
                out = out.replace(src, dst)
            return out

        target = _canon(filename)
        ls_cmd = f"ls -1 {self._escape_shell_arg(dir_path)} 2>/dev/null"
        ls_result = self._exec(ls_cmd)
        if ls_result.exit_code != 0 or not ls_result.stdout.strip():
            return None
        candidates = [
            entry
            for entry in _strip_terminal_fence_leaks(ls_result.stdout).splitlines()
            if entry and entry != filename and _canon(entry) == target
        ]
        # Exactly one equivalent spelling = unambiguous repair. Zero or
        # several = fall through to suggestions; guessing among homoglyph
        # collisions would silently read the wrong file.
        if len(candidates) == 1:
            return os.path.join(dir_path, candidates[0]) if dir_path != "." or "/" in path else candidates[0]
        return None

    def _suggest_similar_files(self, path: str) -> ReadResult:
        """Suggest similar files when the requested file is not found."""
        dir_path = os.path.dirname(path) or "."
        filename = os.path.basename(path)
        basename_no_ext = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1].lower()
        lower_name = filename.lower()

        # List files in the target directory
        ls_cmd = f"ls -1 {self._escape_shell_arg(dir_path)} 2>/dev/null | head -50"
        ls_result = self._exec(ls_cmd)

        scored: list = []  # (score, filepath) — higher is better
        if ls_result.exit_code == 0 and ls_result.stdout.strip():
            for f in ls_result.stdout.strip().split('\n'):
                if not f:
                    continue
                lf = f.lower()
                score = 0

                # Exact match (shouldn't happen, but guard)
                if lf == lower_name:
                    score = 100
                # Same base name, different extension (e.g. config.yml vs config.yaml)
                elif os.path.splitext(f)[0].lower() == basename_no_ext.lower():
                    score = 90
                # Target is prefix of candidate or vice-versa
                elif lf.startswith(lower_name) or lower_name.startswith(lf):
                    score = 70
                # Substring match (candidate contains query)
                elif lower_name in lf:
                    score = 60
                # Reverse substring (query contains candidate name)
                elif lf in lower_name and len(lf) > 2:
                    score = 40
                # Same extension with some overlap
                elif ext and os.path.splitext(f)[1].lower() == ext:
                    common = set(lower_name) & set(lf)
                    if len(common) >= max(len(lower_name), len(lf)) * 0.4:
                        score = 30
                # Near-miss spelling (AGENT.md -> AGENTS.md): substring
                # checks above find nothing, but a high sequence ratio
                # catches 1-2 edit typos without a homegrown levenshtein.
                if score == 0 and difflib.SequenceMatcher(
                    None, lower_name, lf
                ).ratio() >= 0.8:
                    score = 50

                if score > 0:
                    scored.append((score, os.path.join(dir_path, f)))

        scored.sort(key=lambda x: -x[0])
        similar = [fp for _, fp in scored[:5]]

        return ReadResult(
            error=f"File not found: {path}",
            similar_files=similar
        )
    
    def read_file_raw(self, path: str) -> ReadResult:
        """Read the complete file content as a plain string.

        No pagination, no line-number prefixes, no per-line truncation.
        Uses cat so the full file is returned regardless of size.
        """
        path = self._expand_path(path)
        stat_result = self._exec(self._size_probe_cmd(path))
        if stat_result.exit_code != 0:
            return self._suggest_similar_files(path)
        stat_output = _strip_terminal_fence_leaks(stat_result.stdout)
        if stat_output.strip() == NOT_REGULAR_SENTINEL:
            return self._not_regular_error(path)
        try:
            file_size = int(stat_output.strip())
        except ValueError:
            file_size = 0
        if self._is_image(path):
            return ReadResult(is_image=True, is_binary=True, file_size=file_size)
        sample_bytes = self._sample_file_bytes(path)
        if sample_bytes is not None:
            ext_binary = os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS
            is_binary = ext_binary or self._is_likely_binary_bytes(sample_bytes)
        else:
            sample_result = self._exec(f"head -c 1000 {self._escape_shell_arg(path)} 2>/dev/null")
            sample_output = _strip_terminal_fence_leaks(sample_result.stdout)
            is_binary = self._is_likely_binary(path, sample_output)
        if is_binary:
            return ReadResult(
                is_binary=True, file_size=file_size,
                error=describe_binary_file(sample_bytes, file_size),
            )
        cat_result = self._exec(f"cat {self._escape_shell_arg(path)}")
        if cat_result.exit_code != 0:
            return ReadResult(error=f"Failed to read file: {cat_result.stdout}")
        # Strip a leading UTF-8 BOM so patch's fuzzy matcher operates on
        # clean content (a phantom U+FEFF before line 1 would defeat an
        # exact first-line match). write_file restores the BOM on the way
        # back out — it re-probes the on-disk file, which still has the
        # marker — so the round-trip preserves it.
        raw_content, _ = _strip_bom(_strip_terminal_fence_leaks(cat_result.stdout))
        return ReadResult(
            content=raw_content,
            file_size=file_size,
        )

    def read_file_bytes(self, path: str, max_bytes: Optional[int] = None) -> ReadResult:
        """Read binary-safe bytes from any shell-backed environment."""
        path = self._expand_path(path)
        stat_result = self._exec(self._size_probe_cmd(path))
        if stat_result.exit_code != 0:
            return ReadResult(error=f"File not found: {path}")
        stat_output = _strip_terminal_fence_leaks(stat_result.stdout)
        if stat_output.strip() == NOT_REGULAR_SENTINEL:
            return self._not_regular_error(path)
        try:
            file_size = int(stat_output.strip())
        except ValueError:
            return ReadResult(error=f"Could not determine file size: {path}")
        if max_bytes is not None and file_size > max_bytes:
            return ReadResult(
                file_size=file_size,
                error=f"File is too large ({file_size:,} bytes, limit is {max_bytes:,})",
            )

        encoded = self._exec(f"base64 < {self._escape_shell_arg(path)}")
        if encoded.exit_code != 0:
            return ReadResult(error=f"Failed to read binary file: {encoded.stdout}")
        compact = "".join(_strip_terminal_fence_leaks(encoded.stdout).split())
        try:
            base64.b64decode(compact, validate=True)
        except (ValueError, base64.binascii.Error):
            return ReadResult(error=f"Backend returned invalid binary data for: {path}")
        return ReadResult(
            base64_content=compact,
            file_size=file_size,
            is_binary=True,
        )

    def delete_file(self, path: str) -> WriteResult:
        """Delete a single file.

        Cross-platform: runs via ``python -c`` against the terminal env's
        Python so it works on Windows shells (``cmd.exe``/PowerShell) that
        don't ship ``rm``. Directories are rejected here — use
        ``delete_path(recursive=True)`` for trees.
        """
        return self._python_delete(path, recursive=False)

    def delete_path(self, path: str, recursive: bool = False) -> WriteResult:
        """Cross-platform delete that handles files and (with recursive=True)
        directory trees. Always preferred over emitting ``rm -rf`` /
        ``Remove-Item -Recurse`` directly so the same tool call works on
        every backend (local / docker / ssh / Windows).
        """
        return self._python_delete(path, recursive=recursive)

    def _python_delete(self, path: str, recursive: bool) -> WriteResult:
        path = self._expand_path(path)
        denied = get_write_denied_error(path, verb="Delete")
        if denied:
            return WriteResult(error=denied)

        # We can't shell out to ``rm`` here — it doesn't exist on Windows
        # ``cmd.exe`` or PowerShell, so this code path is what's left when
        # the backend's terminal is a Windows shell. Path is baked into the
        # snippet via ``repr()`` so quoting is correct on every shell.
        snippet = (
            "import shutil, pathlib, sys\n"
            f"p = pathlib.Path({path!r})\n"
            f"recursive = {bool(recursive)!r}\n"
            "try:\n"
            "    if p.is_dir() and not p.is_symlink():\n"
            "        if recursive:\n"
            "            shutil.rmtree(p)\n"
            "        else:\n"
            "            print('is a directory: ' + str(p), file=sys.stderr); sys.exit(2)\n"
            "    else:\n"
            # NOTE: avoid ``unlink(missing_ok=True)`` — that kwarg lands in
            # Python 3.8 and the remote interpreter (docker/ssh) may still
            # be 3.7 on older distros. The FileNotFoundError handler below
            # covers the same case and works back to 3.4.
            "        p.unlink()\n"
            "except FileNotFoundError:\n"
            "    pass\n"
            "except Exception as exc:\n"
            "    print(str(exc), file=sys.stderr); sys.exit(1)\n"
        )

        result = self._exec(f"python3 -c {self._escape_shell_arg(snippet)}")

        # Fall back to ``python`` (Windows / older systems where there's no
        # ``python3`` symlink but a ``python`` binary is on PATH).
        if result.exit_code != 0 and "python3" in (result.stdout or ""):
            result = self._exec(f"python -c {self._escape_shell_arg(snippet)}")

        if result.exit_code != 0:
            return WriteResult(error=f"Failed to delete {path}: {(result.stdout or '').strip() or 'unknown error'}")

        return WriteResult()

    def move_file(self, src: str, dst: str) -> WriteResult:
        """Move a file via mv."""
        src = self._expand_path(src)
        dst = self._expand_path(dst)
        for p in (src, dst):
            denied = get_write_denied_error(p, verb="Move")
            if denied:
                return WriteResult(error=denied)
        result = self._exec(
            f"mv {self._escape_shell_arg(src)} {self._escape_shell_arg(dst)}"
        )
        if result.exit_code != 0:
            return WriteResult(error=f"Failed to move {src} -> {dst}: {result.stdout}")
        return WriteResult()

    # =========================================================================
    # WRITE Implementation
    # =========================================================================

    def _write_probe_cmd(self, path: str, sentinel: str, body: Optional[str]) -> str:
        """One shell command for the on-disk questions ``write_file`` asks.

        Two segments closed by a ``sentinel`` line: base64 of the first three
        bytes (BOM detection at the byte layer, the same on-disk truth as
        ``_file_has_bom``), then ``body``: ``"cat"`` for the full text when
        pre-content is wanted, ``"sample"`` for the 4 KB line-ending sample,
        or ``None`` for nothing. Gated on ``[ -f ]`` so a FIFO or device never
        reaches ``head``/``cat``; a missing path echoes ``MISSING_SENTINEL``.
        """
        arg = self._escape_shell_arg(path)
        if body == "cat":
            body_cmd = f"cat {arg} 2>/dev/null"
        elif body == "sample":
            body_cmd = f"head -c 4096 {arg} 2>/dev/null"
        else:
            body_cmd = ":"
        return (
            f"if [ -f {arg} ]; then "
            f"head -c 3 {arg} 2>/dev/null | base64 2>/dev/null; echo {sentinel}; "
            f"{body_cmd}; "
            f"else echo {MISSING_SENTINEL}; fi"
        )

    def _probe_write_target(
        self, path: str, pre_content: Optional[str], want_pre: bool,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """Return ``(has_bom, pre_content, original_line_ending)`` for ``path``.

        Replaces three probes (``cat`` when pre-content is wanted, a
        ``head -c 4096`` line-ending sample, a ``head -c 3`` BOM check) with
        one round-trip. Semantics are unchanged: pre-content is only read
        when wanted and not supplied; the line ending comes from pre-content
        when there is any, else from the sample; the BOM always comes from
        the bytes on disk. A reply that cannot be parsed falls back to the
        separate probes.
        """
        if want_pre and pre_content is None:
            body_mode: Optional[str] = "cat"
        elif not pre_content:
            body_mode = "sample"
        else:
            body_mode = None

        sentinel = _new_sentinel(_WRITE_SENTINEL_PREFIX)
        probe = self._exec(self._write_probe_cmd(path, sentinel, body_mode))
        output = probe.stdout or ""

        if sentinel not in output:
            if _strip_terminal_fence_leaks(output).strip() == MISSING_SENTINEL:
                ending = _detect_line_ending(pre_content) if pre_content else None
                return False, pre_content, ending
            logger.debug(
                "write_file: pre-write probe reply for %s has no sentinel "
                "(exit %s, %d chars); falling back to sequential probes",
                path, probe.exit_code, len(output),
            )
            return self._probe_write_target_sequential(path, pre_content, want_pre)

        segments = _split_segments(output, sentinel)
        if probe.exit_code != 0 or len(segments) != 2:
            logger.debug(
                "write_file: pre-write probe for %s returned exit %s with %d "
                "segments (want 2); falling back to sequential probes",
                path, probe.exit_code, len(segments),
            )
            return self._probe_write_target_sequential(path, pre_content, want_pre)
        head_seg, body = segments

        head_bytes = self._decode_base64_sample(head_seg)
        if head_bytes is None:
            # No clean base64 on this shell; ask the way we used to.
            logger.debug(
                "write_file: no usable base64 head for %s; paying one extra "
                "round-trip for the BOM probe",
                path,
            )
            has_bom = self._file_has_bom(path, pre_content)
        else:
            has_bom = head_bytes.startswith(_UTF8_BOM.encode("utf-8"))

        if body_mode == "cat" and body:
            pre_content = body

        if pre_content:
            ending = _detect_line_ending(pre_content)
        elif body_mode == "sample" and body:
            ending = _detect_line_ending(body)
        else:
            ending = None
        return has_bom, pre_content, ending

    def _probe_write_target_sequential(
        self, path: str, pre_content: Optional[str], want_pre: bool,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """The pre-compound form of ``_probe_write_target``: one exec per question."""
        if want_pre and pre_content is None:
            # Best-effort read; failure (file missing, permission) leaves
            # pre_content as None which makes both downstream consumers
            # degrade gracefully (lint reports all errors; LSP skips the
            # shift map).
            read_cmd = f"cat {self._escape_shell_arg(path)} 2>/dev/null"
            read_result = self._exec(read_cmd)
            if read_result.exit_code == 0 and read_result.stdout:
                pre_content = read_result.stdout
        ending = self._detect_file_line_ending(path, pre_content)
        has_bom = self._file_has_bom(path, pre_content)
        return has_bom, pre_content, ending

    def write_file(self, path: str, content: str,
                   pre_content: Optional[str] = None) -> WriteResult:
        """
        Write content to a file, creating parent directories as needed.

        Pipes content through stdin to avoid OS ARG_MAX limits on large
        files. The content never appears in the shell command string —
        only the file path does.

        Before anything touches disk, a fail-closed syntax gate runs
        against the CANDIDATE content: if ``path``'s extension is in
        ``_FAIL_CLOSED_INPROC_EXTS`` (JSON/YAML/TOML — structured data
        formats where a parse failure always means corruption) and the
        candidate content doesn't parse, the write is refused outright.
        No temp file, no rename, nothing on disk changes.

        After a write that clears the gate, runs a post-first / pre-lazy
        lint check via ``_check_lint_delta()``.  If the new content is
        clean, the lint call is O(one parse).  If the new content has
        errors the gate didn't already catch (i.e. errors from a linter
        outside ``_FAIL_CLOSED_INPROC_EXTS``, such as Python), the
        pre-write content is linted too and only errors newly introduced
        by this write are surfaced — pre-existing problems are filtered
        out so the agent isn't distracted chasing them.

        Args:
            path: File path to write
            content: Content to write
            pre_content: Pre-edit file content if the caller already has it
                (e.g. patch_replace read the file for fuzzy matching).
                When provided, skips a redundant ``cat`` subprocess to
                re-read the file for lint baseline / line-ending
                detection. BOM detection always probes disk (the most
                common provider — ``read_file_raw`` — strips BOMs, so
                trusting ``pre_content`` for BOM would cause false
                negatives and silent marker loss on rewrite). When
                None, reads from disk as before.

        Returns:
            WriteResult with bytes written, lint summary, or error.
        """
        # Expand ~ and other shell paths
        path = self._expand_path(path)

        # Block writes to sensitive paths
        denied = get_write_denied_error(path)
        if denied:
            return WriteResult(error=denied)

        # Reject lone surrogates up front with a regex scan (no encode, no
        # subprocess). surrogateescape-decoded content (U+DC80–U+DCFF)
        # round-trips through the pipe fine, but surrogates outside that
        # range cannot be encoded at all — and letting them reach the pipe
        # would spawn a child that then hangs, or truncates the target via
        # empty-stdin `cat`. Refuse synchronously before any subprocess.
        m = re.search(r"[\ud800-\udc7f\udd00-\udfff]", content)
        if m:
            return WriteResult(
                error=(
                    f"Refusing to write '{path}': content contains a lone "
                    f"surrogate character ({m.group(0)!r}) that cannot be "
                    "encoded as UTF-8. The file was NOT created or modified."
                )
            )

        # ── Fail-closed pre-write syntax gate ───────────────────────────
        # Validate the CANDIDATE content BEFORE any bytes touch disk —
        # previously this only ran as a post-write lint *report* that the
        # caller could ignore (or that ``files_modified`` gating wouldn't
        # catch, since a lint failure never set the top-level ``error``
        # key). A structured-format write that doesn't even parse (mashed
        # quotes, truncated generation, wrong indentation dialect) is a
        # corrupt write, not a style nit — refuse it outright instead of
        # writing first and reporting the damage afterward.
        #
        # Scope: only extensions in ``_FAIL_CLOSED_INPROC_EXTS`` (JSON/
        # YAML/TOML). ``.py`` deliberately keeps its pre-existing,
        # non-blocking lint-delta *report* instead of a hard refusal — see
        # ``_FAIL_CLOSED_INPROC_EXTS``'s docstring above for why. Extensions
        # with no in-process linter at all (including ones only covered by
        # a shell linter) are completely unaffected — this gate never runs
        # for them, so behavior there is unchanged.
        #
        # Checked against the raw ``content`` argument, before the
        # BOM/CRLF preservation shims below run. Those shims exist purely
        # to match the on-disk file's existing conventions; linting
        # post-shim would false-positive a JSONDecodeError on a
        # legitimately BOM-marked JSON file purely because this method
        # re-adds the marker the read layer strips — see
        # ``_file_has_bom``/``_UTF8_BOM`` below.
        ext = os.path.splitext(path)[1].lower()
        inproc_linter = LINTERS_INPROC.get(ext) if ext in _FAIL_CLOSED_INPROC_EXTS else None
        if inproc_linter is not None:
            _ok, _lint_err = inproc_linter(content)
            if not _ok and _lint_err != "__SKIP__":
                return WriteResult(
                    error=(
                        f"Refusing to write '{path}': candidate content fails "
                        f"{ext} syntax validation ({_lint_err}). The file was "
                        "NOT created or modified. Fix the content and retry."
                    )
                )

        # Capture pre-write content.  Two consumers want it:
        #
        #   1. The lint-delta layer (for in-process linters like ast.parse
        #      and json.loads) needs the previous content to compute the
        #      set of NEW lint errors introduced by this write.
        #   2. The LSP layer needs pre/post content to build a line-shift
        #      map — pre-existing diagnostics below the edit point shift
        #      when lines are added/removed, and the shift map remaps
        #      baseline diagnostics into post-edit coordinates so the
        #      strict (range-aware) delta key matches.
        #
        # The set of extensions we capture pre_content for is therefore
        # the UNION of in-process lint coverage and LSP coverage.  For
        # extensions outside both sets (binaries, opaque formats),
        # skipping the read keeps the hot path fast.
        want_pre = ext in LINTERS_INPROC or self._lsp_handles_extension(ext)
        # One shell round-trip answers every on-disk question the write
        # needs (pre-content when wanted, line endings, BOM); see
        # _probe_write_target. A caller that already has the file content
        # (e.g. patch_replace read it for fuzzy matching) skips the read;
        # the BOM is still taken from disk, never from pre_content.
        has_bom, pre_content, original_ending = self._probe_write_target(
            path, pre_content, want_pre
        )

        # ── Line-ending preservation (Roo Code pattern) ──────────────
        # If the file existed with CRLF endings and the agent's content
        # has bare LFs, convert to CRLF before writing.  Otherwise the
        # write silently normalizes a Windows-line-ending file (and patch
        # produces mixed endings when only a substituted region changes).
        if original_ending == "\r\n":
            content = _normalize_line_endings(content, "\r\n")

        # ── BOM preservation ──────────────────────────────────────────
        # If the file on disk started with a UTF-8 BOM, keep it. read_file
        # strips the BOM so the agent never sees it, which means the
        # content it hands back to write_file / patch has no BOM either —
        # without restoring it here a round-trip would silently strip the
        # marker and change the file's byte signature (some Windows
        # toolchains key on it). Only prepend when the original had a BOM
        # and the new content doesn't already carry one (guards against
        # double-BOM if a caller passed raw bytes).
        if has_bom and not _has_bom(content):
            content = _UTF8_BOM + content

        # Snapshot LSP diagnostics for this file (best-effort) so the
        # post-write LSP layer can return only diagnostics introduced
        # by this specific edit.  Mirrors claude-code's
        # ``beforeFileEdited`` pattern but wired to the local LSP
        # rather than an external IDE.
        self._snapshot_lsp_baseline(path)

        # Write atomically.  ``mkdir -p`` is folded into _atomic_write
        # (one fewer subprocess vs. a separate mkdir call).
        # ``dirs_created`` has always meant "parent dirs ensured" —
        # ``mkdir -p`` exits 0 even when the dirs pre-exist, so the old
        # separate-mkdir code reported True in exactly the same cases.
        # A mkdir failure now surfaces as the atomic-write error return
        # below, before this field is ever emitted.
        parent = os.path.dirname(path)
        dirs_created = bool(parent)

        # Write atomically: stream into a temp file in the SAME directory,
        # then ``mv`` it over the target. The rename is atomic on POSIX
        # (and on every backend FS we run on), so a crash / power loss /
        # truncated pipe mid-write leaves the original file intact instead
        # of a half-written corrupt file. Same-directory is load-bearing —
        # ``mv`` across filesystems degrades to copy+unlink, which is NOT
        # atomic; keeping the temp beside the target guarantees a real
        # rename. Content still rides stdin so there's no ARG_MAX limit.
        #
        # The temp file is created with ``mktemp`` (collision-safe) when the
        # backend has it, falling back to a PID-stamped name otherwise. We
        # then chmod the temp to match the existing file's mode (if any) so
        # the atomic swap doesn't silently widen or narrow permissions, and
        # clean the temp up on any failure so we never leak a ``.hermes-tmp``
        # turd next to the user's file.
        # Encode once for byte count + sha256. surrogateescape is the exact
        # inverse of the decode that may have produced this content, so these
        # are the bytes the pipe transmits and the bytes on disk. The early
        # rejection above guarantees this cannot raise; the try/except is
        # defense for future callers that bypass it.
        try:
            content_bytes = content.encode("utf-8", "surrogateescape")
        except UnicodeEncodeError as exc:
            return WriteResult(
                error=(
                    f"Refusing to write '{path}': content contains a lone "
                    f"surrogate character ({exc}) that cannot be encoded as "
                    "UTF-8. The file was NOT created or modified."
                )
            )
        write_result = self._atomic_write(path, content)

        if write_result.exit_code != 0:
            return WriteResult(error=f"Failed to write file: {write_result.stdout}")

        # Bytes written — computed from the exact bytes we just wrote (len
        # matches wc -c) instead of spawning a ``wc -c`` subprocess. The
        # encode happened up front with surrogateescape — the inverse of the
        # decode that produces surrogate content — so content_bytes == what
        # rode stdin == what is on disk, and the sha256 below compares like
        # with like.
        bytes_written = len(content_bytes)

        # Post-write content verification (cheap, one shell call): compare
        # the on-disk sha256 to the intended content's hash. Production
        # mining shows models re-reading files right after writing them to
        # confirm persistence (154 verify-reads in a 400k-msg window) —
        # an explicit verified flag makes that turn unnecessary, and a
        # mismatch is surfaced as a hard error instead of silent corruption
        # (mirrors patch_replace's post-write verification).
        content_verified: Optional[bool] = None
        try:
            hash_cmd = f"sha256sum {self._escape_shell_arg(path)} 2>/dev/null"
            hash_result = self._exec(hash_cmd)
            if hash_result.exit_code == 0 and hash_result.stdout.strip():
                disk_sha = hash_result.stdout.strip().split()[0]
                expected_sha = hashlib.sha256(content_bytes).hexdigest()
                content_verified = disk_sha == expected_sha
                if not content_verified:
                    return WriteResult(
                        error=(
                            f"Post-write verification failed for {path}: on-disk "
                            "content hash differs from the intended write. The "
                            "write did not persist correctly — re-read the file "
                            "and retry."
                        )
                    )
        except Exception:
            content_verified = None

        # Post-write lint with delta refinement.
        lint_result = self._check_lint_delta(path, pre_content=pre_content, post_content=content)

        # Semantic diagnostics from the LSP layer — separate channel.
        # Only fired when the syntax tier reported clean (no point asking
        # an LSP for a file that won't even parse).  Pass pre/post
        # content so the LSP layer can build a line-shift map and
        # remap baseline diagnostics into post-edit coordinates.
        # Best-effort: ``""`` is returned for any failure path.
        lsp_diagnostics: Optional[str] = None
        if lint_result.success or lint_result.skipped:
            block = self._maybe_lsp_diagnostics(
                path, pre_content=pre_content, post_content=content
            )
            if block:
                lsp_diagnostics = block

        return WriteResult(
            bytes_written=bytes_written,
            dirs_created=dirs_created,
            verified=content_verified,
            lint=lint_result.to_dict() if lint_result else None,
            lsp_diagnostics=lsp_diagnostics,
        )
    
    # =========================================================================
    # PATCH Implementation (Replace Mode)
    # =========================================================================
    
    def patch_replace(self, path: str, old_string: str, new_string: str,
                      replace_all: bool = False) -> PatchResult:
        """
        Replace text in a file using fuzzy matching.

        Args:
            path: File path to modify
            old_string: Text to find (must be unique unless replace_all=True)
            new_string: Replacement text
            replace_all: If True, replace all occurrences

        Returns:
            PatchResult with diff and lint results
        """
        # Expand ~ and other shell paths
        path = self._expand_path(path)

        # Block writes to sensitive paths
        denied = get_write_denied_error(path)
        if denied:
            return PatchResult(error=denied)

        # Read current content
        read_cmd = f"cat {self._escape_shell_arg(path)} 2>/dev/null"
        read_result = self._exec(read_cmd)
        
        if read_result.exit_code != 0:
            return PatchResult(error=f"Failed to read file: {path}")
        
        content = read_result.stdout
        # Preserve raw content (including BOM) for write_file's pre_content
        # so write_file can detect/restore BOM correctly.
        raw_content = content
        # Strip a leading UTF-8 BOM before matching so the fuzzy matcher and
        # the diff operate on clean content (a phantom U+FEFF before line 1
        # defeats an exact first-line match). write_file restores the BOM on
        # the way back out by re-probing the on-disk file, so the round-trip
        # preserves the marker.
        content, _ = _strip_bom(content)

        # Import and use fuzzy matching
        from tools.fuzzy_match import fuzzy_find_and_replace
        
        new_content, match_count, _strategy, error = fuzzy_find_and_replace(
            content, old_string, new_string, replace_all
        )
        
        if error or match_count == 0:
            # Already-applied detection: the most common patch failure in
            # production is a re-send of an edit that has already landed
            # (identical old/new strings, or old_string gone while
            # new_string is present verbatim). Surface that as an explicit
            # success-shaped no-op so the model moves on instead of
            # burning turns on re-reads and re-patches.
            from tools.fuzzy_match import is_already_applied
            if is_already_applied(content, old_string, new_string):
                return PatchResult(
                    success=True,
                    no_change=True,
                    note=(
                        f"File already contains the target text — the edit "
                        f"appears to be already applied to {path}. No write "
                        "performed; do not re-send this patch."
                    ),
                )
            err_msg = error or f"Could not find match for old_string in {path}"
            try:
                from tools.fuzzy_match import format_no_match_hint
                err_msg += format_no_match_hint(err_msg, match_count, old_string, content)
            except Exception:
                pass
            return PatchResult(error=err_msg)

        # ── Line-ending preservation ──────────────────────────────────
        # Models nearly always send old_string/new_string with bare LF
        # in tool args (JSON-encoded), but the file may have CRLF on
        # disk.  After fuzzy_find_and_replace, ``new_content`` is a
        # mixed-ending string: the substituted region is LF, surrounding
        # text keeps the file's CRLF.  Normalize the whole thing to the
        # file's detected line ending so the on-disk file is consistent
        # and the unified diff below reflects the actual change.
        file_ending = _detect_line_ending(content)
        if file_ending:
            new_content = _normalize_line_endings(new_content, file_ending)

        # Write back — pass pre_content (original read, with BOM) to avoid
        # a redundant cat subprocess inside write_file.  Must be the raw
        # content (before _strip_bom) so write_file can detect/restore BOM.
        write_result = self.write_file(path, new_content,
                                       pre_content=raw_content)
        if write_result.error:
            return PatchResult(error=f"Failed to write changes: {write_result.error}")

        # Post-write verification — re-read the file and confirm the bytes we
        # intended to write actually landed. Catches silent persistence
        # failures (backend FS oddities, race with another task, truncated
        # pipe, etc.) that would otherwise return success-with-diff while the
        # file is unchanged on disk.
        verify_cmd = f"cat {self._escape_shell_arg(path)} 2>/dev/null"
        verify_result = self._exec(verify_cmd)
        if verify_result.exit_code != 0:
            return PatchResult(error=f"Post-write verification failed: could not re-read {path}")
        # Normalize line endings before comparing.  On Windows, Python's
        # default text-mode ``open()`` translates ``\n`` → ``\r\n`` on
        # write, so the file on disk legitimately holds CRLFs while our
        # ``new_content`` string has bare LFs.  Without this normalization
        # every patch on Windows returns a bogus "wrote 39, read 42"
        # false-negative even though the edit landed correctly.  POSIX
        # backends don't translate, so this is a no-op there.  We also
        # strip a leading BOM from the re-read: write_file restored the
        # marker on disk but ``new_content`` is the BOM-less string we
        # matched against, so the comparison must drop it to stay
        # apples-to-apples.
        _verify_bomless, _ = _strip_bom(verify_result.stdout)
        _verify_stdout_normalized = _verify_bomless.replace("\r\n", "\n").replace("\r", "\n")
        _new_content_normalized = new_content.replace("\r\n", "\n").replace("\r", "\n")
        if _verify_stdout_normalized != _new_content_normalized:
            return PatchResult(error=(
                f"Post-write verification failed for {path}: on-disk content "
                f"differs from intended write "
                f"(wrote {len(_new_content_normalized)} chars, read back "
                f"{len(_verify_stdout_normalized)} chars after normalizing line endings). "
                "The patch did not persist. Re-read the file and try again."
            ))

        # Generate diff
        diff = self._unified_diff(content, new_content, path)

        # Auto-lint with delta refinement: only surface errors introduced
        # by this patch, filtering out pre-existing lint failures so the
        # agent isn't distracted by problems that were already there.
        lint_result = self._check_lint_delta(path, pre_content=content, post_content=new_content)

        return PatchResult(
            success=True,
            diff=diff,
            files_modified=[path],
            lint=lint_result.to_dict() if lint_result else None,
            # Propagate the LSP diagnostics already captured by the
            # internal ``write_file`` call.  Its baseline was the
            # pre-patch content (taken at the start of write_file via
            # ``_snapshot_lsp_baseline``) so the delta is correct for
            # the patch as a whole.  Keep the field separate from the
            # syntax-check ``lint`` so the agent can read both signals.
            lsp_diagnostics=write_result.lsp_diagnostics,
        )
    
    def patch_v4a(self, patch_content: str) -> PatchResult:
        """
        Apply a V4A format patch.
        
        V4A format:
            *** Begin Patch
            *** Update File: path/to/file.py
            @@ context hint @@
             context line
            -removed line
            +added line
            *** End Patch
        
        Args:
            patch_content: V4A format patch string
        
        Returns:
            PatchResult with changes made
        """
        # Import patch parser
        from tools.patch_parser import parse_v4a_patch, apply_v4a_operations
        
        operations, parse_error = parse_v4a_patch(patch_content)
        if parse_error:
            return PatchResult(error=f"Failed to parse patch: {parse_error}")
        
        # Apply operations
        result = apply_v4a_operations(operations, self)
        return result
    
    def _check_lint(self, path: str, content: Optional[str] = None) -> LintResult:
        """
        Run syntax check on a file after editing.

        Prefers the in-process linter for structured formats (JSON, YAML,
        TOML) when possible — those parse via the Python stdlib in
        microseconds and don't require a subprocess.  Falls back to the
        shell linter table for compiled/type-checked languages
        (py_compile, node --check, tsc, go vet, rustfmt).

        Args:
            path: File path (used to select the linter + for shell invocation).
            content: Optional file content.  If provided AND an in-process
                     linter matches the extension, we lint the content
                     directly without re-reading the file from disk.  Ignored
                     for shell linters.

        Returns:
            LintResult with status and any errors.
        """
        ext = os.path.splitext(path)[1].lower()

        # Prefer in-process linter when available.
        inproc = LINTERS_INPROC.get(ext)
        if inproc is not None:
            # Need content — either passed in or read from disk.
            if content is None:
                read_cmd = f"cat {self._escape_shell_arg(path)} 2>/dev/null"
                read_result = self._exec(read_cmd)
                if read_result.exit_code != 0:
                    return LintResult(skipped=True, message=f"Failed to read {path} for lint")
                content = read_result.stdout
            ok, err = inproc(content)
            if err == "__SKIP__":
                return LintResult(skipped=True, message=f"No linter available for {ext} (missing dependency)")
            return LintResult(success=ok, output="" if ok else err)

        # Fall back to shell linter.
        if ext not in LINTERS:
            return LintResult(skipped=True, message=f"No linter for {ext} files")

        # A per-file `tsc --noEmit <file>` cannot read the project's
        # tsconfig.json, so for any .ts that belongs to a TS project it floods
        # phantom errors — unresolved path aliases (`@/…` → TS2307) and ambient
        # globals (`Window.hermesDesktop` → TS2339) that are defined by the
        # config it never loads. The delta filter then reports the misleading
        # "pre-existing lint errors … the file is still broken", which carries
        # no signal and wastes the caller's turns. When an ancestor
        # tsconfig.json exists, skip the shell tsc entirely; real diagnostics
        # come from the LSP tier (below) or an explicit `tsc -p tsconfig.json`
        # the caller runs deliberately. (.tsx already returns above via the
        # `ext not in LINTERS` branch.)
        if ext == '.ts' and self._has_ancestor_tsconfig(path):
            return LintResult(
                skipped=True,
                message=(
                    "Project tsconfig.json detected — per-file tsc skipped "
                    "(single-file tsc can't resolve project aliases/globals; "
                    "use the LSP tier or `tsc -p tsconfig.json` for real "
                    "diagnostics)."
                ),
            )

        # If a real LSP server is active and claims this file, skip the
        # shell linter for extensions whose per-file shell invocation is
        # structurally weaker / floods phantom errors.  See
        # ``_SHELL_LINTER_LSP_REDUNDANT`` above for the rationale per ext.
        # The LSP tier runs separately via ``_maybe_lsp_diagnostics`` and
        # carries the real diagnostics in ``lsp_diagnostics`` on the
        # WriteResult / PatchResult.
        if ext in _SHELL_LINTER_LSP_REDUNDANT and self._lsp_will_handle(path):
            return LintResult(
                skipped=True,
                message=f"LSP server handles {ext} — shell linter skipped",
            )

        linter_cmd = LINTERS[ext]
        # Extract the base command (first word)
        base_cmd = linter_cmd.split()[0]

        if not self._has_command(base_cmd):
            return LintResult(skipped=True, message=f"{base_cmd} not available")

        # Run linter. Linters (python, node, tsc, go, rustfmt) are native
        # Windows binaries on Windows hosts: they need the C:/... path form.
        # The MSYS /c/... form makes node resolve the file as C:\c\Users\...
        # (double-prefixed) — every .js write then reports a phantom ENOENT
        # lint failure. Native form works for MSYS builds too.
        cmd = linter_cmd.replace("{file}", self._escape_native_tool_arg(path))
        result = self._exec(cmd, timeout=30)

        if result.exit_code != 0 and _looks_like_linter_unusable(base_cmd, result.stdout):
            # The linter command exists on PATH but couldn't actually run
            # (e.g. ``npx tsc`` when tsc isn't in node_modules; ``rustfmt
            # --check`` without a Cargo project).  This is a tooling gap,
            # not a real lint failure — surface it as ``skipped`` so the
            # write doesn't get flagged AND so the LSP tier still runs.
            from tools.ansi_strip import strip_ansi
            cleaned = strip_ansi(result.stdout).strip()
            # Collapse to a single line — the npx banner is multi-line ASCII.
            first_line = next(
                (ln.strip() for ln in cleaned.splitlines() if ln.strip()),
                cleaned[:120],
            )
            return LintResult(
                skipped=True,
                message=f"{base_cmd} not usable: {first_line[:200]}",
            )

        return LintResult(
            success=result.exit_code == 0,
            output=result.stdout.strip() if result.stdout.strip() else ""
        )

    def _check_lint_delta(self, path: str, pre_content: Optional[str],
                          post_content: Optional[str] = None) -> LintResult:
        """
        Run post-write syntax lint with pre-write baseline comparison.

        Two-tier strategy:

        1. **Syntax check** (in-process or shell-based, microseconds).
           Catches the bug class that motivated this layer: corrupt
           writes, mashed quotes, truncated output.  Hot path.

        2. **Delta refinement against pre-write content** when the
           syntax tier reports errors.  Filter out errors that already
           existed pre-edit so the agent isn't distracted by inherited
           state.

        Semantic diagnostics from the LSP layer are fetched separately
        via :meth:`_maybe_lsp_diagnostics` and surfaced in the
        ``lsp_diagnostics`` field on :class:`WriteResult` /
        :class:`PatchResult`.  Keeping the two channels separate lets
        the agent (and any downstream parsers) read syntax errors and
        semantic errors as independent signals.

        Args:
            path: File path (for linter selection).
            pre_content: File content BEFORE the write.  Pass None for new
                         files or when the pre-state isn't available — the
                         delta refinement is skipped and all post errors
                         are returned.
            post_content: File content AFTER the write.  Optional; if None,
                          the shell linter reads from disk (same as
                          _check_lint).

        Returns:
            LintResult.  ``output`` contains either the full post-lint
            errors (no pre-state) or just the new-error lines (delta
            refinement applied).
        """
        post = self._check_lint(path, content=post_content)

        # Hot path: clean post-write syntactically.
        if post.success or post.skipped:
            return post

        # Post-write has syntax errors.  If we have pre-content, run the
        # delta refinement to filter out pre-existing errors.
        if pre_content is None:
            return post

        pre = self._check_lint(path, content=pre_content)
        if pre.success or pre.skipped or not pre.output:
            # Pre-write was clean (or we couldn't lint it) — post errors
            # are all new.  Return the full post output.
            return post

        # Both pre- and post-write had errors.  Compute the set-difference
        # on non-empty stripped lines.  Caveat: single-error parsers
        # (ast.parse, json.loads) stop at the first error and don't report
        # later ones — if the pre-existing error blocks parsing before
        # reaching the edit region, we can't prove the edit is clean.  So
        # if every post error also appeared pre-edit, we report the file
        # as still broken but annotate that this edit introduced nothing
        # new on top — the agent knows it's inherited state, not fresh
        # damage, without silently dropping the error.
        pre_lines = {ln.strip() for ln in pre.output.splitlines() if ln.strip()}
        post_lines = [ln for ln in post.output.splitlines() if ln.strip() and ln.strip() not in pre_lines]

        if not post_lines:
            # Every error in post was also in pre — this edit didn't make
            # anything obviously worse, but the file remains broken and
            # the agent should know.
            return LintResult(
                success=False,
                output=post.output,
                message="Pre-existing lint errors — this edit didn't introduce new ones but the file is still broken.",
            )

        return LintResult(
            success=False,
            output=(
                "New lint errors introduced by this edit "
                "(pre-existing errors filtered out):\n" + "\n".join(post_lines)
            )
        )

    def _lsp_local_only(self) -> bool:
        """Return True iff this FileOperations is wired to a local backend.

        LSP servers run on the host process — they need access to the
        files they're linting.  Remote/sandboxed backends (Docker,
        Modal, SSH, Daytona) keep files inside the sandbox where the
        host-side LSP server can't reach them, so we skip the LSP
        path for those entirely.
        """
        env = getattr(self, "env", None)
        if env is None:
            # Defensive: some tests construct ShellFileOperations via
            # ``__new__`` without going through ``__init__``, so
            # ``self.env`` may be missing.  No env = no LSP path.
            return False
        try:
            from tools.environments.local import LocalEnvironment
        except Exception:  # noqa: BLE001
            return False
        return isinstance(env, LocalEnvironment)

    def _lsp_handles_extension(self, ext: str) -> bool:
        """Return True iff some registered LSP server claims this extension.

        Used to decide whether to capture pre-write content for the
        line-shift map.  Capturing is cheap (one ``cat`` on the host)
        but pointless if no LSP would ever look at the file.

        Safe to call on remote backends — the registry is purely
        in-process metadata; we still gate the actual LSP path on
        :meth:`_lsp_local_only`.
        """
        if not ext:
            return False
        try:
            from agent.lsp.servers import SERVERS
        except Exception:  # noqa: BLE001
            return False
        ext_lower = ext.lower()
        for srv in SERVERS:
            if ext_lower in srv.extensions:
                return True
        return False

    def _has_ancestor_tsconfig(self, path: str) -> bool:
        """True iff a tsconfig.json exists in *path*'s directory or any ancestor.

        A single-file ``tsc`` invocation can't read that config, so its
        diagnostics for such a file are pure noise (unresolved aliases /
        ambient globals). Used by :meth:`_check_lint` to skip the per-file
        shell tsc for project TypeScript files.

        Best-effort and local-host only: a host-side ``os.path`` walk. On a
        remote/sandboxed backend the project tree isn't on this host, so the
        walk returns False and the shell linter runs exactly as before — never
        suppress lint based on a probe that couldn't answer.
        """
        if not self._lsp_local_only():
            return False
        try:
            d = os.path.dirname(os.path.abspath(path))
            while True:
                if os.path.isfile(os.path.join(d, "tsconfig.json")):
                    return True
                parent = os.path.dirname(d)
                if parent == d:
                    return False
                d = parent
        except Exception:  # noqa: BLE001
            return False

    def _lsp_will_handle(self, path: str) -> bool:
        """Return True iff the LSP service is active AND will lint this file.

        Stronger than :meth:`_lsp_handles_extension` — that one only checks
        the static server registry.  This one additionally requires the
        LSP service to be configured/enabled and the file to pass
        :meth:`agent.lsp.manager.LSPService.enabled_for` (which gates on
        workspace detection, disabled-server set, and the broken-pair
        short-circuit).

        Used by :meth:`_check_lint` to decide whether to skip the per-file
        shell linter for extensions in ``_SHELL_LINTER_LSP_REDUNDANT``.

        Best-effort: any failure path returns False so the shell linter
        runs as before — never suppress lint based on an LSP probe that
        couldn't actually answer the question.
        """
        if not self._lsp_local_only():
            return False
        try:
            from agent.lsp import get_service
        except Exception:  # noqa: BLE001
            return False
        try:
            svc = get_service()
        except Exception:  # noqa: BLE001
            return False
        if svc is None:
            return False
        try:
            return bool(svc.enabled_for(path))
        except Exception:  # noqa: BLE001
            return False

    def _snapshot_lsp_baseline(self, path: str) -> None:
        """Capture pre-edit LSP diagnostics so the post-write delta is correct.

        Best-effort.  Silent on every failure path — LSP is an
        enrichment layer and must never break a write.

        Skipped entirely on non-local backends (Docker, Modal, SSH,
        etc.) — the server can't see files inside the sandbox.
        """
        if not self._lsp_local_only():
            return
        try:
            from agent.lsp import get_service
            svc = get_service()
        except Exception:  # noqa: BLE001
            return
        if svc is None:
            return
        try:
            svc.snapshot_baseline(path)
        except Exception:  # noqa: BLE001
            pass

    def _maybe_lsp_diagnostics(
        self,
        path: str,
        *,
        pre_content: Optional[str] = None,
        post_content: Optional[str] = None,
    ) -> str:
        """Best-effort LSP semantic diagnostics for ``path``.

        Returns a formatted ``<diagnostics>`` block, or empty string
        when LSP is unavailable / disabled / produced no errors.

        When both ``pre_content`` and ``post_content`` are provided,
        a line-shift map is built and passed to the LSPService so
        baseline diagnostics are remapped into post-edit coordinates
        before the set-difference.  Without this, edits that delete
        or insert lines surface every pre-existing diagnostic below
        the edit point as "introduced by this edit".

        Wraps everything in a try/except so a misbehaving LSP server
        can't break a write.  This intentionally swallows all errors
        — the calling tier already returned a clean syntax result, so
        ``""`` here just means "no extra info to add".

        Skipped entirely on non-local backends (Docker, Modal, SSH,
        etc.) — same reasoning as ``_snapshot_lsp_baseline``.
        """
        if not self._lsp_local_only():
            return ""
        try:
            from agent.lsp import get_service
        except Exception:  # noqa: BLE001
            return ""
        try:
            svc = get_service()
        except Exception:  # noqa: BLE001
            return ""
        if svc is None or not svc.enabled_for(path):
            return ""

        # Build a line-shift map when we have both pre and post — it
        # remaps baseline diagnostics into post-edit coordinates so
        # the strict (range-aware) delta key matches correctly.
        line_shift = None
        if pre_content is not None and post_content is not None and pre_content != post_content:
            try:
                from agent.lsp.range_shift import build_line_shift
                line_shift = build_line_shift(pre_content, post_content)
            except Exception:  # noqa: BLE001
                line_shift = None

        try:
            diagnostics = svc.get_diagnostics_sync(path, delta=True, line_shift=line_shift)
        except Exception:  # noqa: BLE001
            return ""
        if not diagnostics:
            return ""
        try:
            from agent.lsp.reporter import report_for_file, truncate
            block = report_for_file(path, diagnostics)
            if not block:
                return ""
            return truncate("LSP diagnostics introduced by this edit:\n" + block)
        except Exception:  # noqa: BLE001
            return ""
    
    # =========================================================================
    # SEARCH Implementation
    # =========================================================================
    
    def search(self, pattern: str, path: str = ".", target: str = "content",
               file_glob: Optional[str] = None, limit: int = 50, offset: int = 0,
               output_mode: str = "content", context: int = 0,
               order: str = "discovery") -> SearchResult:
        """
        Search for content or files.
        
        Args:
            pattern: Regex (for content) or glob pattern (for files)
            path: Directory/file to search (default: cwd)
            target: "content" (grep) or "files" (glob)
            file_glob: File pattern filter for content search (e.g., "*.py")
            limit: Max results (default 50)
            offset: Skip first N results
            output_mode: "content", "files_only", or "count"
            context: Lines of context around matches
            order: File-search ordering: fast discovery or exact modified time
        
        Returns:
            SearchResult with matches or file list
        """
        offset, limit = normalize_search_pagination(offset, limit)

        if target == "files" and order not in {"discovery", "modified"}:
            return SearchResult(
                error=(f"Invalid file search order {order!r}; expected "
                       "'discovery' or 'modified'.")
            )

        # Expand ~ and other shell paths
        path = self._expand_path(path)
        
        # Validate that the path exists before searching
        check = self._exec(f"test -e {self._escape_shell_arg(path)} && echo exists || echo not_found")
        if "not_found" in check.stdout:
            # Multi-path recovery: models frequently pass several paths in
            # one string ("dir1 dir2 dir3" or comma-separated). Instead of
            # failing the whole call, split, search every path that exists,
            # merge the results, and report the skipped parts.
            multi = self._try_multi_path_search(
                pattern, path, target, file_glob, limit, offset, output_mode, context,
                order,
            )
            if multi is not None:
                return multi
            # Try to suggest nearby paths
            parent = os.path.dirname(path) or "."
            basename_query = os.path.basename(path)
            hint_parts = [f"Path not found: {path}"]
            # Check if parent directory exists and list similar entries
            parent_check = self._exec(
                f"test -d {self._escape_shell_arg(parent)} && echo yes || echo no"
            )
            if "yes" in parent_check.stdout and basename_query:
                ls_result = self._exec(
                    f"ls -1 {self._escape_shell_arg(parent)} 2>/dev/null | head -20"
                )
                if ls_result.exit_code == 0 and ls_result.stdout.strip():
                    lower_q = basename_query.lower()
                    candidates = []
                    for entry in ls_result.stdout.strip().split('\n'):
                        if not entry:
                            continue
                        le = entry.lower()
                        if lower_q in le or le in lower_q or le.startswith(lower_q[:3]):
                            candidates.append(os.path.join(parent, entry))
                    if candidates:
                        hint_parts.append(
                            "Similar paths: " + ", ".join(candidates[:5])
                        )
            return SearchResult(
                error=". ".join(hint_parts),
                total_count=0
            )
        
        if target == "files":
            result = self._search_files(pattern, path, limit, offset, order)
        else:
            result = self._search_content(pattern, path, file_glob, limit, offset,
                                          output_mode, context)

        exclusions = self._macos_search_exclusions(path)
        if exclusions and not result.error:
            skipped = ", ".join(item.split("/")[-1] for item in exclusions)
            result.warning = (
                "Skipped macOS protected folders during broad search to avoid "
                f"an unattended privacy prompt: {skipped}. Search a protected "
                "folder directly when access is intentional."
            )
        return result

    def _macos_search_exclusions(self, path: str) -> List[str]:
        """Protected descendants to prune for this search root, if any.

        Gated on ``env.is_local``: ``sys.platform``/``Path.home()`` describe
        the CONTROLLER, but search commands execute on ``self.env``'s host — a
        macOS controller driving a Linux container/SSH backend must not prune
        the remote's (unprotected) Downloads, and TCC doesn't exist there
        anyway. A Linux controller driving a macOS SSH host keeps today's
        behavior (no pruning); detecting the remote OS is out of scope here.
        Environments without the flag (test fakes, plugins) default to local
        semantics — pruning is a warning-carrying skip, never data loss.
        """
        env = getattr(self, "env", None)
        if env is not None and getattr(env, "is_local", True) is False:
            return []
        cwd = getattr(self.env, "cwd", None) or self.cwd
        return _macos_protected_search_exclusions(
            path, cwd=cwd, home=_HOME, platform=sys.platform
        )
    
    def _try_multi_path_search(self, pattern: str, path: str, target: str,
                               file_glob: Optional[str], limit: int, offset: int,
                               output_mode: str, context: int,
                               order: str = "discovery") -> Optional[SearchResult]:
        """Recover a not-found ``path`` that is really several paths in one string.

        Production trajectories show models passing "dir1 dir2 dir3" (or
        comma-separated lists) as ``path``. Split on whitespace/commas; when
        at least one candidate exists and at least two candidates were given,
        search every existing path, merge results, and note skipped parts.
        Returns None when this doesn't look like a multi-path string.
        """
        parts = [p for chunk in path.split(",") for p in chunk.split() if p.strip()]
        if len(parts) < 2:
            return None
        existing, missing = [], []
        for p in parts:
            expanded = self._expand_path(p)
            chk = self._exec(
                f"test -e {self._escape_shell_arg(expanded)} && echo exists || echo not_found"
            )
            (existing if "exists" in chk.stdout else missing).append(expanded)
        if not existing:
            return None

        if target == "files":
            # A file search across several roots is one global rg traversal so
            # modified ordering and pagination are exact across the whole set.
            if self._has_command("rg"):
                resolved = self._resolve_command("rg") or "rg"
                merged = self._search_files_rg(
                    pattern.split("/")[-1], existing, limit, offset, order,
                    rg_executable=resolved,
                )
            else:
                # find cannot accept the same cross-platform guarantees as rg;
                # preserve per-root fallback scans, but never swallow an error
                # and apply the caller's page once to the combined result.
                collected: List[str] = []
                any_truncated = False
                fetch = offset + limit + 1
                for root in existing:
                    sub = self._search_files(pattern, root, fetch, 0, order)
                    if sub.error:
                        return sub
                    collected.extend(sub.files)
                    any_truncated = any_truncated or sub.truncated
                merged = SearchResult(
                    files=collected[offset:offset + limit],
                    total_count=len(collected),
                    truncated=any_truncated or len(collected) > offset + limit,
                )
        else:
            merged = SearchResult()
            for root in existing:
                sub = self._search_content(
                    pattern, root, file_glob, limit, offset, output_mode, context
                )
                if sub.error:
                    return sub
                merged.matches.extend(sub.matches)
                merged.files.extend(sub.files)
                merged.counts.update(sub.counts)
                merged.total_count += sub.total_count
                merged.truncated = merged.truncated or sub.truncated
            merged.matches = merged.matches[:limit]
            merged.files = merged.files[:limit]

        note = f"path contained {len(parts)} entries; searched {len(existing)} that exist"
        if missing:
            note += "; skipped missing: " + ", ".join(missing[:3])
            if len(missing) > 3:
                note += f" (+{len(missing) - 3} more)"
        merged.warning = note
        return merged

    def _zero_match_probe(self, pattern: str, path: str,
                          file_glob: Optional[str]) -> Optional[str]:
        """Return a hint for a 0-match content search, or None.

        13.9% of production content searches return zero matches and give
        the model nothing to steer by. Run ONE cheap case-insensitive count
        probe; if it hits, say so. If the pattern contains regex
        metacharacters, also probe it as a fixed string. Bounded: two rg
        invocations max, count-only output.
        """
        rg_executable = self._resolve_command('rg')
        if not rg_executable:
            return None
        rg = self._quote_executable(rg_executable)

        def _tally(stdout: str):
            """Parse ``path:count`` lines from rg --count-matches."""
            total = 0
            per_file = []
            for line in (stdout or "").strip().splitlines():
                p, _sep, n = line.rpartition(":")
                if n.isdigit():
                    total += int(n)
                    per_file.append(p)
            return total, per_file

        def _paths_note(per_file, cap: int = 5) -> str:
            shown = ", ".join(per_file[:cap])
            extra = len(per_file) - cap
            return shown + (f" (+{extra} more)" if extra > 0 else "")

        glob_expr = f" --glob {self._escape_shell_arg(file_glob)}" if file_glob else ""
        probe = self._exec(
            f"{rg} -i --count-matches{glob_expr} "
            f"{self._escape_shell_arg(pattern)} {self._escape_native_tool_arg(path)} "
            f"2>/dev/null | head -50",
            timeout=30,
        )
        ci_total, ci_paths = _tally(probe.stdout)
        if ci_total > 0:
            return (
                f"0 exact matches, but {ci_total} case-insensitive match(es) "
                f"in {len(ci_paths)} file(s): {_paths_note(ci_paths)} — "
                "the pattern's casing may be wrong."
            )
        # Hidden/ignored probe: rg skips dotdirs and .gitignore'd files by
        # default. When the pattern exists only there, say so instead of
        # returning a bare zero (bench case: match in .hidden/ silently
        # missing from results).
        hidden = self._exec(
            f"{rg} --hidden --no-ignore --count-matches{glob_expr} "
            f"{self._escape_shell_arg(pattern)} {self._escape_native_tool_arg(path)} "
            f"2>/dev/null | head -50",
            timeout=30,
        )
        h_total, h_paths = _tally(hidden.stdout)
        if h_total > 0:
            return (
                f"0 matches in visible files, but {h_total} match(es) in "
                f"{len(h_paths)} hidden or gitignored file(s): "
                f"{_paths_note(h_paths)} — these are excluded by default."
            )
        if re.search(r"[.\[\](){}?*+^$\\|]", pattern):
            fixed = self._exec(
                f"{rg} -F --count-matches{glob_expr} "
                f"{self._escape_shell_arg(pattern)} {self._escape_native_tool_arg(path)} "
                f"2>/dev/null | head -50",
                timeout=30,
            )
            f_total, f_paths = _tally(fixed.stdout)
            if f_total > 0:
                return (
                    f"0 regex matches, but {f_total} literal match(es) in "
                    f"{len(f_paths)} file(s): {_paths_note(f_paths)} — the "
                    "pattern contains regex metacharacters that likely need "
                    "escaping (or pass a simpler substring)."
                )
        return None

    def _is_broad_local_search_root(self, path: str) -> bool:
        """Whether a no-rg local root is unsafe for recursive find."""
        from tools.environments.local import (
            LocalEnvironment, _IS_WINDOWS, _msys_to_windows_path,
        )

        if not isinstance(self.env, LocalEnvironment):
            return False

        def normalized(value: str) -> str:
            if _IS_WINDOWS:
                value = _msys_to_windows_path(value).replace("\\", "/")
            if not os.path.isabs(value):
                value = os.path.join(getattr(self.env, "cwd", None) or self.cwd, value)
            return os.path.normcase(os.path.abspath(value))

        root = normalized(path)
        home = normalized(_HOME)
        try:
            common = os.path.commonpath([root, home])
        except ValueError:
            return False
        anchor = os.path.splitdrive(root)[0] + os.sep if os.path.splitdrive(root)[0] else os.path.abspath(os.sep)
        return root == home or common == root or root == os.path.normcase(anchor)

    def _search_files(self, pattern: str, path: str, limit: int, offset: int,
                      order: str = "discovery") -> SearchResult:
        """Search for files by name pattern (glob-like)."""
        # Auto-prepend **/ for recursive search if not already present
        if not pattern.startswith('**/') and '/' not in pattern:
            search_pattern = pattern
        else:
            search_pattern = pattern.split('/')[-1]

        # Prefer ripgrep: bounded parallel traversal with ignore semantics.
        if self._has_command("rg"):
            return self._search_files_rg(
                search_pattern, path, limit, offset, order,
                rg_executable=self._resolve_command("rg") or "rg",
            )

        # A local find traversal rooted at/above the user's home or at a
        # filesystem root can consume minutes and prompt on protected paths.
        # Refuse before invoking find. Controller paths never classify remotes.
        if self._is_broad_local_search_root(path):
            return SearchResult(error=(
                "Broad local file search without ripgrep is disabled because "
                "find cannot keep this traversal safely bounded. Install "
                "ripgrep or search a narrower directory."
            ))

        if not self._has_command("find"):
            return SearchResult(
                error="File search requires 'rg' (ripgrep) or 'find'. "
                      "Install ripgrep for best results: "
                      "https://github.com/BurntSushi/ripgrep#installation"
            )

        # Prune hidden descendant directories while still allowing an
        # explicitly selected hidden root. Hidden files are excluded too,
        # matching rg's default semantics.
        q_path = self._escape_shell_arg(path)
        hidden_prune = (
            f" \\( -type d -name '.*' ! -path {q_path} \\) -prune -o"
        )
        protected_paths = [
            os.path.normpath(os.path.join(path, item))
            for item in self._macos_search_exclusions(path)
        ]
        protected_prune = ""
        if protected_paths:
            terms = " -o ".join(
                f"-path {self._escape_shell_arg(item)}" for item in protected_paths
            )
            protected_prune = f" \\( {terms} \\) -prune -o"

        fetch_limit = offset + limit + 1
        base = (
            f"find {q_path}{protected_prune}{hidden_prune} -type f "
            f"! -name '.*' -name {self._escape_shell_arg(search_pattern)}"
        )
        if order == "modified":
            cmd = (
                "set -o pipefail; " + base
                + f" -printf '%T@ %p\\n' 2>/dev/null | sort -rn | head -n {fetch_limit}"
            )
        else:
            cmd = (
                "set -o pipefail; " + base
                + f" -print 2>/dev/null | head -n {fetch_limit}"
            )

        result = self._exec(cmd, timeout=60)
        stdout, limit_reason = _search_stdout_and_limit(result)
        if order == "modified" and result.exit_code not in {0, 124} and not stdout.strip():
            return SearchResult(error=(
                "Exact modification-time order requires GNU find with "
                "-printf support; install ripgrep 14+ or use order='discovery'."
            ))
        if order == "discovery" and result.exit_code not in {0, 1, 124} and not stdout.strip():
            return SearchResult(error="File search failed while running bounded find traversal.")

        raw_files: List[str] = []
        for line in stdout.splitlines():
            if order == "modified":
                parts = line.split(" ", 1)
                if len(parts) != 2 or not parts[0].replace(".", "", 1).isdigit():
                    continue
                raw_files.append(parts[1])
            elif line:
                raw_files.append(line)

        from tools.environments.local import LocalEnvironment, _IS_WINDOWS, _msys_to_windows_path
        if _IS_WINDOWS and isinstance(self.env, LocalEnvironment):
            raw_files = [_msys_to_windows_path(file_path) for file_path in raw_files]

        page = raw_files[offset:offset + limit]
        return SearchResult(
            files=page,
            total_count=len(raw_files),
            truncated=len(raw_files) > offset + limit or bool(limit_reason),
            limit_reason=limit_reason,
        )

    def _search_files_rg(self, pattern: str, path: str | List[str], limit: int, offset: int,
                         order: str = "discovery",
                         rg_executable: Optional[str] = None) -> SearchResult:
        """Search for files by name using ripgrep's --files mode.

        rg --files respects .gitignore and excludes hidden directories by
        default, and uses parallel directory traversal for ~200x speedup
        over find on wide trees. Discovery order stays bounded and fast;
        exact modification-time ordering is explicit because it scans globally.
        """
        # rg --files -g uses glob patterns; wrap bare names so they match
        # at any depth (equivalent to find -name).
        if '/' not in pattern and not pattern.startswith('*'):
            glob_pattern = f"*{pattern}"
        else:
            glob_pattern = pattern

        roots = [path] if isinstance(path, str) else path
        fetch_limit = limit + offset + 1
        exclusion_terms: List[str] = []
        for root in roots:
            exclusion_terms.extend(
                f"--glob {self._escape_shell_arg(f'!{item}/**')}"
                for item in self._macos_search_exclusions(root)
            )
        exclusion_globs = " ".join(dict.fromkeys(exclusion_terms))
        exclusion_args = f" {exclusion_globs}" if exclusion_globs else ""
        rg_executable = rg_executable or self._resolve_command("rg")
        if not rg_executable:
            return SearchResult(error="File search requires ripgrep (rg).")
        if order == "modified":
            capability_error = self._modified_rg_capability_error(rg_executable)
            if capability_error:
                return SearchResult(error=capability_error)
        rg = self._quote_executable(rg_executable)
        sort_arg = " --sortr=modified" if order == "modified" else ""
        root_args = " ".join(self._escape_native_tool_arg(root) for root in roots)
        cmd = (
            f"set -o pipefail; {rg} --files{sort_arg} -g {self._escape_shell_arg(glob_pattern)}"
            f"{exclusion_args} {root_args} 2>/dev/null | head -n {fetch_limit}"
        )
        result = self._exec(cmd, timeout=60)
        stdout, limit_reason = _search_stdout_and_limit(result)
        all_files = [f for f in stdout.splitlines() if f]

        if result.exit_code not in {0, 1, 124} and not all_files:
            if order == "modified":
                return SearchResult(error=(
                    "Exact modification-time order failed; ripgrep 14+ is "
                    "required. Upgrade ripgrep or use order='discovery'."
                ))
            return SearchResult(error="File search failed while running ripgrep.")

        page = all_files[offset:offset + limit]
        return SearchResult(
            files=page,
            total_count=len(all_files),
            truncated=len(all_files) > offset + limit or bool(limit_reason),
            limit_reason=limit_reason,
        )
    
    def _search_content(self, pattern: str, path: str, file_glob: Optional[str],
                        limit: int, offset: int, output_mode: str, context: int) -> SearchResult:
        """Search for content inside files (grep-like)."""
        # Try ripgrep first (fast), fallback to grep (slower but works)
        used_rg = False
        if self._has_command('rg'):
            used_rg = True
            result = self._search_with_rg(pattern, path, file_glob, limit, offset,
                                          output_mode, context,
                                          rg_executable=self._resolve_command("rg") or "rg")
        elif self._has_command('grep'):
            result = self._search_with_grep(pattern, path, file_glob, limit, offset,
                                            output_mode, context)
        else:
            # Neither rg nor grep available (Windows without Git Bash, etc.)
            return SearchResult(
                error="Content search requires ripgrep (rg) or grep. "
                      "Install ripgrep: https://github.com/BurntSushi/ripgrep#installation"
            )

        # Zero-match steering: a 0-match result with no guidance is a dead
        # turn. Probe cheaply for near-misses (wrong casing, hidden-only
        # matches, unescaped regex metacharacters) and attach the finding
        # as a warning. Runs for BOTH engines.
        if (not result.error and result.total_count == 0
                and not result.matches and not result.files and not result.counts):
            try:
                hint = self._zero_match_probe(pattern, path, file_glob)
            except Exception:
                hint = None
            if hint:
                result.warning = hint if not result.warning else f"{result.warning} {hint}"

        # rg auto-enables --multiline for \n patterns, so the line-oriented
        # explanation only applies to the grep fallback engine.
        if used_rg:
            return result
        return _maybe_warn_line_oriented_newline_pattern(result, pattern)
    
    def _search_with_rg(self, pattern: str, path: str, file_glob: Optional[str],
                        limit: int, offset: int, output_mode: str, context: int,
                        rg_executable: Optional[str] = None) -> SearchResult:
        """Search using ripgrep."""
        rg_executable = rg_executable or self._resolve_command("rg")
        if not rg_executable:
            return SearchResult(error="Content search requires ripgrep (rg).")
        cmd_parts = [self._quote_executable(rg_executable), "--line-number", "--no-heading", "--with-filename"]

        # Giant-single-line containment (ported from cline/cline#13525): a
        # match inside a serialized dump (multi-MB single-line JSON/minified
        # bundle) makes rg emit the ENTIRE line into stdout. `head -n` counts
        # lines, so a 40MB match line sails through untruncated, gets buffered
        # whole into Python, and only THEN hits the per-match [:500] clamp —
        # measured 42MB across the transport / ~180MB peak alloc for one
        # match on main. --max-columns bounds each printed line at the rg
        # layer; --max-columns-preview keeps a truncated prefix (instead of
        # omitting the match) so the model still sees the hit. 2000 cols
        # comfortably exceeds the 500-char content clamp below, so no
        # previously-visible content is lost. Both flags predate rg 11; the
        # engine floor here is already rg 13 (--sortr).
        if output_mode not in ("files_only", "count"):
            cmd_parts.extend(["--max-columns", "2000", "--max-columns-preview"])

        # Auto-multiline: a regex `\n` (or a literal newline in the pattern)
        # cannot match in rg's default line-oriented mode — it used to hard
        # error ("the literal \"\\n\" is not allowed") and burn a turn. When
        # the pattern clearly wants to cross lines, enable -U/--multiline
        # up front and note it in the result.
        multiline = _pattern_has_regex_newline(pattern)
        if multiline:
            cmd_parts.append("--multiline")

        # Add context if requested
        if context > 0:
            cmd_parts.extend(["-C", str(context)])
        
        # Exclude macOS TCC-protected descendants during broad searches.
        for item in self._macos_search_exclusions(path):
            cmd_parts.extend(["--glob", self._escape_shell_arg(f"!{item}/**")])

        # Add file glob filter (must be quoted to prevent shell expansion)
        if file_glob:
            cmd_parts.extend(["--glob", self._escape_shell_arg(file_glob)])
        
        # Output mode handling
        if output_mode == "files_only":
            cmd_parts.append("-l")  # Files only
        elif output_mode == "count":
            cmd_parts.append("-c")  # Count per file
        
        # Add pattern and path
        cmd_parts.append(self._escape_shell_arg(pattern))
        # rg is a native Windows binary when installed via winget/cargo/choco:
        # it needs the C:/... path form, not the MSYS /c/... form (which
        # nothing converts back — Hermes sets MSYS_NO_PATHCONV for its bash).
        cmd_parts.append(self._escape_native_tool_arg(path))
        
        # Fetch extra rows so we can report the true total before slicing.
        # For context mode, rg emits separator lines ("--") between groups,
        # so we grab generously and filter in Python.
        fetch_limit = limit + offset + 200 if context > 0 else limit + offset
        cmd_parts.extend(["|", "head", "-n", str(fetch_limit)])
        
        # `set -o pipefail` so rg's exit status propagates through `| head`.
        # Without it the pipeline reports head's status (0), masking rg's
        # error code (2) and making the guard below unreachable. rg handles a
        # truncating head cleanly (exit 0 on SIGPIPE), so pipefail does not
        # introduce false errors on a successful-but-truncated search.
        cmd = "set -o pipefail; " + " ".join(cmd_parts)
        result = self._exec(cmd, timeout=60)
        stdout, limit_reason = _search_stdout_and_limit(result)

        # _exec merges stderr into stdout (stderr=subprocess.STDOUT), so rg's
        # diagnostic lines ("rg: <file>: <error>", "rg: regex parse error:")
        # are interleaved with match output. Split them out: diagnostics must
        # not be parsed as matches, and on a hard error they ARE the message.
        diagnostics, payload = _split_tool_diagnostics(stdout)

        # rg exit codes: 0=matches found, 1=no matches, 2=error. rg returns 2
        # even on partial errors (e.g. one unreadable file in a tree that
        # otherwise matched), so only surface an error when exit==2 AND no
        # usable match payload remains. Otherwise we keep the real matches.
        if result.exit_code == 2 and not payload.strip():
            error_msg = diagnostics.strip() or result.stdout.strip() or "Search error"
            return SearchResult(error=f"Search failed: {error_msg}", total_count=0)

        # Parse the diagnostic-free payload so error text never becomes a match.
        stdout = payload
        _ml_note = (
            "Pattern contains \\n — multiline mode (-U) was enabled automatically "
            "so the regex can match across line boundaries."
        ) if multiline else None
        # Parse results based on output mode
        if output_mode == "files_only":
            all_files = [f for f in stdout.strip().split('\n') if f]
            total = len(all_files)
            page = all_files[offset:offset + limit]
            return SearchResult(
                files=page,
                total_count=total,
                truncated=bool(limit_reason),
                limit_reason=limit_reason,
                warning=_ml_note,
            )
        
        elif output_mode == "count":
            counts = {}
            for line in stdout.strip().split('\n'):
                if ':' in line:
                    parts = line.rsplit(':', 1)
                    if len(parts) == 2:
                        try:
                            counts[parts[0]] = int(parts[1])
                        except ValueError:
                            pass
            return SearchResult(
                counts=counts,
                total_count=sum(counts.values()),
                truncated=bool(limit_reason),
                limit_reason=limit_reason,
            )
        
        else:
            # Parse content matches and context lines.
            # rg match lines:   "file:lineno:content"  (colon separator)
            # rg context lines: "file-lineno-content"   (dash separator)
            # rg group seps:    "--"
            # Note: on Windows, paths contain drive letters (e.g. C:\path),
            # so naive split(":") breaks. Use regex to handle both platforms.
            _match_re = re.compile(r'^([A-Za-z]:)?(.*?):(\d+):(.*)$')
            matches = []
            for line in stdout.strip().split('\n'):
                if not line or line == "--":
                    continue
                
                # Try match line first (colon-separated: file:line:content)
                m = _match_re.match(line)
                if m:
                    matches.append(SearchMatch(
                        path=(m.group(1) or '') + m.group(2),
                        line_number=int(m.group(3)),
                        content=m.group(4)[:500]
                    ))
                    continue
                
                # Try context line (dash-separated: file-line-content)
                # Only attempt if context was requested to avoid false positives
                if context > 0:
                    parsed = _parse_search_context_line(line)
                    if parsed:
                        matches.append(SearchMatch(
                            path=parsed[0],
                            line_number=parsed[1],
                            content=parsed[2][:500]
                        ))
            
            total = len(matches)
            page = matches[offset:offset + limit]
            return SearchResult(
                matches=page,
                total_count=total,
                truncated=total > offset + limit or bool(limit_reason),
                limit_reason=limit_reason,
                warning=_ml_note,
            )
    
    def _search_with_grep(self, pattern: str, path: str, file_glob: Optional[str],
                          limit: int, offset: int, output_mode: str, context: int) -> SearchResult:
        """Fallback search using grep."""
        cmd_parts = ["grep", "-rnHE"]  # -H forces filenames; -E matches rg regex behavior
        
        # Exclude hidden directories (matching ripgrep's default behavior).
        # This prevents searching inside .hub/index-cache/, .git/, etc.
        cmd_parts.append("--exclude-dir='.*'")

        # Protected-dir pruning CANNOT use --exclude-dir here: grep matches
        # exclude-dir globs against BASENAMES anywhere in the tree, so
        # --exclude-dir=Downloads would silently skip every nested directory
        # named Downloads (a repo's own Downloads/ folder included), not just
        # the protected home child. When exclusions apply (darwin broad-home
        # search on a local backend), route through find's path-scoped -prune
        # instead — same traversal-prevention the find backend uses.
        protected_paths = [
            os.path.normpath(os.path.join(path, item))
            for item in self._macos_search_exclusions(path)
        ]
        if protected_paths:
            return self._search_with_grep_pruned(
                pattern, path, file_glob, limit, offset, output_mode, context,
                protected_paths,
            )
        
        # Add context if requested
        if context > 0:
            cmd_parts.extend(["-C", str(context)])
        
        # Add file pattern filter (must be quoted to prevent shell expansion)
        if file_glob:
            cmd_parts.extend(["--include", self._escape_shell_arg(file_glob)])
        
        # Output mode handling
        if output_mode == "files_only":
            cmd_parts.append("-l")
        elif output_mode == "count":
            cmd_parts.append("-c")
        
        # Add pattern and path. grep applies --exclude-dir to the command-line
        # search root too, so passing the default relative root ``.`` causes
        # ``.*`` to exclude the entire search. Anchor relative paths at the
        # shell's live cwd; quoting $PWD separately keeps user paths escaped
        # while working across local, container, and remote backends.
        cmd_parts.append(self._escape_shell_arg(pattern))
        is_absolute = path.startswith(("/", "\\\\")) or bool(
            re.match(r"^[A-Za-z]:[\\/]", path)
        )
        if is_absolute:
            search_root = self._escape_shell_arg(path)
        else:
            relative_path = path[2:] if path.startswith("./") else path
            search_root = '"$PWD"'
            if relative_path not in {"", "."}:
                search_root += f"/{self._escape_shell_arg(relative_path)}"
        cmd_parts.append(search_root)
        
        # Fetch generously so we can compute total before slicing
        fetch_limit = limit + offset + (200 if context > 0 else 0)
        cmd_parts.extend(["|", "head", "-n", str(fetch_limit)])
        # grep has no --max-columns: bound giant single-line matches (see the
        # rg branch's containment comment) at the pipe layer instead. Safe for
        # the file:line:content parser — truncation only ever drops content
        # tail. Skipped for files_only/count where lines are paths/counts.
        if output_mode not in ("files_only", "count"):
            cmd_parts.extend(["|", "cut", "-c1-2000"])
        
        # `set -o pipefail` so grep's exit status propagates through `| head`
        # (without it the pipeline reports head's 0, masking grep's error 2).
        # A truncating head makes grep exit 141 (SIGPIPE) on an otherwise
        # successful search; the strict `== 2` guard below ignores that, so
        # pipefail does not turn truncated results into false errors.
        cmd = "set -o pipefail; " + " ".join(cmd_parts)
        result = self._exec(cmd, timeout=60)
        return self._parse_grep_search_output(result, output_mode, limit, offset, context)

    def _search_with_grep_pruned(self, pattern: str, path: str, file_glob: Optional[str],
                                 limit: int, offset: int, output_mode: str, context: int,
                                 protected_paths: List[str]) -> SearchResult:
        """grep fallback with PATH-scoped protected-dir pruning.

        Files are enumerated by ``find`` with the same ``-path ... -prune``
        expression the find backend uses (traversal never enters the protected
        dirs, so macOS never sees an access attempt), then handed to grep via
        ``-exec {} +``. This exists because grep's own ``--exclude-dir``
        matches basenames anywhere in the tree — it cannot express "only the
        home-level Downloads". Hidden directories are pruned to mirror the
        plain path's ``--exclude-dir='.*'``. Trade-off: with ``-exec {} +``
        find folds grep's exit code into its own generic non-zero, so a hard
        grep error surfaces as an empty result rather than exit 2 — acceptable
        for this darwin-local-broad-search-only branch.
        """
        grep_parts = ["grep", "-nHE"]
        if context > 0:
            grep_parts.extend(["-C", str(context)])
        if output_mode == "files_only":
            grep_parts.append("-l")
        elif output_mode == "count":
            grep_parts.append("-c")
        grep_parts.append(self._escape_shell_arg(pattern))

        prune_terms = " -o ".join(
            f"-path {self._escape_shell_arg(item)}" for item in protected_paths
        )
        find_parts = [
            "find", self._escape_shell_arg(path or "."),
            f"\\( {prune_terms} \\) -prune", "-o",
            "\\( -type d -name '.*' \\) -prune", "-o",
            "-type f",
        ]
        if file_glob:
            find_parts.extend(["-name", self._escape_shell_arg(file_glob)])
        find_parts.extend(["-exec", *grep_parts, "{}", "+"])
        fetch_limit = limit + offset + (200 if context > 0 else 0)
        # Same giant-single-line bound as the plain grep path (grep lacks
        # --max-columns); see the rg branch's containment comment.
        line_cap = " | cut -c1-2000" if output_mode not in ("files_only", "count") else ""
        cmd = (
            "set -o pipefail; " + " ".join(find_parts)
            + f" 2>/dev/null | head -n {fetch_limit}{line_cap}"
        )
        result = self._exec(cmd, timeout=60)
        return self._parse_grep_search_output(result, output_mode, limit, offset, context)

    def _parse_grep_search_output(self, result, output_mode: str, limit: int,
                                  offset: int, context: int) -> SearchResult:
        """Shared grep output parsing for the plain and pruned variants."""
        stdout, limit_reason = _search_stdout_and_limit(result)

        # _exec merges stderr into stdout, so grep's diagnostic lines
        # ("grep: <file>: <error>") are interleaved with matches. Split them
        # out so they're never parsed as matches and so a hard error has a
        # clean message.
        diagnostics, payload = _split_tool_diagnostics(stdout)

        # grep exit codes: 0=matches found, 1=no matches, 2=error. grep
        # returns 2 on partial errors (e.g. an unreadable file) even when
        # other files matched, so only surface an error when exit==2 AND no
        # usable match payload remains.
        if result.exit_code == 2 and not payload.strip():
            error_msg = diagnostics.strip() or result.stdout.strip() or "Search error"
            return SearchResult(error=f"Search failed: {error_msg}", total_count=0)

        stdout = payload
        if output_mode == "files_only":
            all_files = [f for f in stdout.strip().split('\n') if f]
            total = len(all_files)
            page = all_files[offset:offset + limit]
            return SearchResult(
                files=page,
                total_count=total,
                truncated=bool(limit_reason),
                limit_reason=limit_reason,
            )
        
        elif output_mode == "count":
            counts = {}
            for line in stdout.strip().split('\n'):
                if ':' in line:
                    parts = line.rsplit(':', 1)
                    if len(parts) == 2:
                        try:
                            counts[parts[0]] = int(parts[1])
                        except ValueError:
                            pass
            return SearchResult(
                counts=counts,
                total_count=sum(counts.values()),
                truncated=bool(limit_reason),
                limit_reason=limit_reason,
            )
        
        else:
            # grep match lines:   "file:lineno:content" (colon)
            # grep context lines: "file-lineno-content"  (dash)
            # grep group seps:    "--"
            # Note: on Windows, paths contain drive letters (e.g. C:\path),
            # so naive split(":") breaks. Use regex to handle both platforms.
            _match_re = re.compile(r'^([A-Za-z]:)?(.*?):(\d+):(.*)$')
            matches = []
            for line in stdout.strip().split('\n'):
                if not line or line == "--":
                    continue
                
                m = _match_re.match(line)
                if m:
                    matches.append(SearchMatch(
                        path=(m.group(1) or '') + m.group(2),
                        line_number=int(m.group(3)),
                        content=m.group(4)[:500]
                    ))
                    continue
                
                if context > 0:
                    parsed = _parse_search_context_line(line)
                    if parsed:
                        matches.append(SearchMatch(
                            path=parsed[0],
                            line_number=parsed[1],
                            content=parsed[2][:500]
                        ))

            
            total = len(matches)
            page = matches[offset:offset + limit]
            return SearchResult(
                matches=page,
                total_count=total,
                truncated=total > offset + limit or bool(limit_reason),
                limit_reason=limit_reason,
            )
