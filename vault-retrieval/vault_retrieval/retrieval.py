"""Range-based file extraction with hard large-file rules.

Implements the slice-the-Vault-not-full-read-it part of the protocol:

- Files <= ``large_file_chars``: any range is fine (still need a range,
  not a full read).
- Files > ``large_file_chars``: require a selector, accept at most
  :data:`LARGE_FILE_RANGES_MAX` ranges, each at most
  :data:`LARGE_FILE_LINES_MAX` lines and :data:`LARGE_FILE_CHARS_MAX`
  characters.

Each returned :class:`Extract` carries its source path, range, character
count, and source mtime — so downstream callers can build the
``source_refs`` array the JSONL query log requires.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Extract:
    path: str
    line_start: int
    line_end: int
    chars: int
    content: str
    source_mtime: str
    heading: Optional[str] = None


class ExtractionError(ValueError):
    """A range was malformed or the file could not be read."""


class LargeFileRefused(ExtractionError):
    """A large-file selector was rejected (too many ranges, range too long, etc.)."""


def _read_lines(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ExtractionError(f"cannot read {path}: {exc}") from exc
    # Preserve original line endings; splitlines() strips them.
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    # If file ends without a newline, splitlines(keepends=True) omits a trailing
    # empty string, which is what we want.
    return lines


def extract_range(
    path: Path,
    *,
    line_start: int,
    line_end: int,
    heading: Optional[str] = None,
) -> Extract:
    """Read lines [line_start, line_end] inclusive (1-indexed) from path.

    Raises :class:`ExtractionError` for bad ranges or unreadable files.
    """
    if not isinstance(line_start, int) or not isinstance(line_end, int):
        raise ExtractionError(f"line_start/line_end must be ints: {line_start!r}, {line_end!r}")
    if line_start < 1:
        raise ExtractionError(f"line_start must be >= 1, got {line_start}")
    if line_end < line_start:
        raise ExtractionError(f"line_end {line_end} < line_start {line_start}")

    lines = _read_lines(path)
    if line_start > len(lines):
        raise ExtractionError(
            f"line_start {line_start} exceeds file length {len(lines)}"
        )
    # Clamp line_end to the actual line count so the caller gets a useful
    # error when they overshoot.
    line_end_clamped = min(line_end, len(lines))
    slice_ = lines[line_start - 1:line_end_clamped]
    content = "".join(slice_)
    chars = len(content)
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        mtime = ""
    return Extract(
        path=str(path),
        line_start=line_start,
        line_end=line_end_clamped,
        chars=chars,
        content=content,
        source_mtime=mtime,
        heading=heading,
    )


def extract_from_selectors(
    path: Path,
    selectors: List[Dict[str, Any]],
    *,
    is_large: bool = False,
    max_ranges: int = 2,
    max_range_lines: int = 120,
    max_range_chars: int = 8_000,
) -> List[Extract]:
    """Apply a list of selectors to a single file.

    When ``is_large=True`` the additional hard rules kick in:
      - selectors must be non-empty
      - at most ``max_ranges`` selectors
      - each range at most ``max_range_lines`` lines
      - each range at most ``max_range_chars`` estimated chars

    Raises :class:`LargeFileRefused` (subclass of :class:`ExtractionError`)
    when a large-file selector is rejected.
    """
    if is_large:
        if not selectors:
            raise LargeFileRefused("large file requires a selector")
        if len(selectors) > max_ranges:
            raise LargeFileRefused(
                f"{len(selectors)} selectors exceeds max_ranges {max_ranges}"
            )
        for sel in selectors:
            ls = sel.get("line_start")
            le = sel.get("line_end")
            if isinstance(ls, int) and isinstance(le, int):
                if le - ls + 1 > max_range_lines:
                    raise LargeFileRefused(
                        f"range {ls}-{le} exceeds max_range_lines {max_range_lines}"
                    )
            ec = sel.get("estimated_chars")
            if isinstance(ec, int) and ec > max_range_chars:
                raise LargeFileRefused(
                    f"range estimated_chars {ec} exceeds max_range_chars {max_range_chars}"
                )

    out: List[Extract] = []
    for sel in selectors:
        e = extract_range(
            path,
            line_start=int(sel["line_start"]),
            line_end=int(sel["line_end"]),
            heading=sel.get("heading"),
        )
        out.append(e)
    return out
