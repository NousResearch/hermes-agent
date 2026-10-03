"""Trace storage layer for trace schema v1 (specs/trace-schema.md, version 1.0.0).

Writes exactly one compact-JSON record per line to date-rotated, append-only
JSONL segment files under the trace root (default ``~/.hermes/sessions/traces/``,
which the spec deliberately places *outside* any profile directory so traces
aggregate across profiles).

Writer API
----------

``append_trace(record, trace_root=None, *, now=None, max_segment_bytes=DEFAULT_SEGMENT_CAP_BYTES) -> Path``
    Serialize ``record`` (a dict conforming to ``specs/trace-v1.schema.json``)
    to a single compact JSON line and append it atomically to the segment for
    the UTC date of ``now``. Opens with ``O_APPEND`` (never truncating flags),
    writes the line, then ``fsync``s before returning. Returns the ``Path`` of
    the segment that received the record. Raises :class:`TraceWriteError` when
    the target exists but is not a regular file, or when ``record`` is not a
    dict / contains non-JSON-serializable values. The trace root is created
    with mode ``0o700`` if missing.

``list_segments(trace_root=None) -> list[Path]``
    Sorted list of ``traces-*.jsonl`` segment paths under the trace root
    (``[]`` when the root does not exist).

``iter_records(path, on_torn=None) -> Iterator[tuple[int, dict]]``
    Yield ``(line_number, record_dict)`` for every parseable non-blank line of
    a segment. A torn or malformed final line — e.g. left behind by a kill -9
    mid-append — is skipped with a ``warnings.warn`` never an exception; all
    prior lines are returned intact. ``on_torn`` (optional callable receiving
    the :class:`TornRecordWarning`) additionally fires for every skipped line,
    torn or not.

``TornRecordWarning`` — warning category carried by both ``warnings.warn``
    and the ``on_torn`` callback: ``.path``, ``.line_no``, ``.raw_preview``.

``SEGMENT_CAP_BYTES`` / ``DEFAULT_SEGMENT_CAP_BYTES`` — the 512 MiB per-segment
    cap the spec fixes.

Rotation
--------

Segments are named ``traces-YYYY-MM-DD.jsonl`` using the UTC date of the write
(the date the run *ends*), per the spec. When appending to the primary segment
would exceed ``max_segment_bytes``, the writer rolls to the next available
suffix — ``traces-YYYY-MM-DD.002.jsonl``, ``.003.jsonl``, … (zero-padded 3
digits). Segment files are created and appended to, never rewritten or
truncated.

Crash semantics
---------------

POSIX guarantees an ``O_APPEND`` positional write of up to ``PIPE_BUF`` bytes
is atomic relative to other appenders, so concurrent processes/threads
interleave at line granularity — one JSON line per record, no interleaved
bytes. Records larger than ``PIPE_BUF`` from multiple processes can still
interleave on some filesystems; that is tolerated by design (the spec's read
side already tolerates torn lines), and the fsync-before-return guarantees a
successfully returned append is durable. A kill -9 mid-append can therefore
only damage the very last line — :func:`iter_records` skips it with a warning
and every earlier line is intact.

Related but separate: the ``request_dump_*.json`` debug breadcrumbs under
``~/.hermes/sessions/`` (agent/agent_runtime_helpers.py) are a different
debug-dump mechanism and are not touched by this module.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import sys
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

# Windows opens files in text mode by default: every b"\n" written through a
# text-mode descriptor becomes b"\r\n" on disk, corrupting the JSONL and
# inflating the file by one byte per line. O_BINARY disables the translation;
# on POSIX it does not exist (and is unnecessary — POSIX never translates).
_O_BINARY = getattr(os, "O_BINARY", 0)

# POSIX guarantees O_APPEND writes up to PIPE_BUF bytes are atomic, so
# concurrent threads/processes interleave at line granularity and no
# coordination is needed. Windows makes no such guarantee: two appends can
# overwrite each other's bytes, silently losing records. Serialize appends
# with a process-wide lock there. This module's own tests exercise
# multi-thread and multi-process appends; the lock makes those semantics
# hold on win32 (threads within this process). Cross-process safety on
# Windows remains out of scope, matching the POSIX read side's tolerance:
# iter_records() already skips a torn tail, and no caller of append_trace
# is documented to span processes on Windows.
_win32_append_lock = threading.Lock() if sys.platform == "win32" else None

LOG = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

#: Per-segment size cap fixed by the spec (512 MB, expressed in MiB bytes).
SEGMENT_CAP_BYTES = 512 * 1024 * 1024
DEFAULT_SEGMENT_CAP_BYTES = SEGMENT_CAP_BYTES

_SUFFIX_MIN = 2  # .001 is reserved for the primary segment; rollover starts at .002
_SUFFIX_MAX = 999  # zero-padded 3 digits; beyond this the spec gives no naming


class TraceWriteError(OSError):
    """Raised when an append target refuses append-only semantics.

    Covers: the segment path exists but is not a regular file (symlink,
    directory, FIFO, socket), the trace root cannot be created, or the
    record fails validation/serialization. Inherits :class:`OSError`
    because every underlying failure is an I/O-surface failure.
    """


class TornRecordWarning(Warning):
    """A line in a segment file could not be parsed and was skipped.

    Carries ``path`` (source file), ``line_no`` (1-based line number) and
    ``raw_preview`` (first 120 bytes of the raw line) for diagnostics.
    """

    def __init__(self, message: str, *, path: Path, line_no: int, raw_preview: bytes) -> None:
        super().__init__(message)
        self.path = Path(path)
        self.line_no = line_no
        self.raw_preview = raw_preview


def default_trace_root() -> Path:
    """``~/.hermes/sessions/traces/`` — the trace root named by the spec.

    Deliberately outside any profile directory: traces aggregate across
    profiles, ``profile_slug`` is a record field, not a path component.
    """

    return Path.home() / ".hermes" / "sessions" / "traces"


def _ensure_regular_file(target: Path) -> None:
    """Fail loudly when ``target`` exists but is not a regular file."""

    st = os.lstat(target)
    if not stat.S_ISREG(st.st_mode):
        raise TraceWriteError(
            f"refusing to append to non-regular file: {target} "
            f"(mode {stat.S_IFMT(st.st_mode):#o})"
        )


def _resolve_segment(
    trace_root: Path,
    now: float,
    line_size: int,
    max_segment_bytes: int,
) -> Path:
    """Pick the append target for ``now`` honoring the size-cap rollover.

    Returns the primary ``traces-YYYY-MM-DD.jsonl`` when the line fits, else
    the first free ``.NNN`` suffix. Never picks an existing segment the line
    would push past the cap.
    """

    day = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
    stem = f"traces-{day}"

    def _fits(path: Path) -> bool:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return True
        return size + line_size <= max_segment_bytes

    primary = trace_root / f"{stem}.jsonl"
    if primary.exists():
        _ensure_regular_file(primary)
    if _fits(primary):
        return primary

    suffix = _SUFFIX_MIN
    while suffix <= _SUFFIX_MAX:
        candidate = trace_root / f"{stem}.{suffix:03d}.jsonl"
        if not candidate.exists():
            return candidate
        _ensure_regular_file(candidate)
        if _fits(candidate):
            return candidate
        suffix += 1

    raise TraceWriteError(
        f"all segment files for {day} are at the {max_segment_bytes}-byte cap "
        f"and the .{_SUFFIX_MAX:03d} suffix is exhausted"
    )


def _serialize_record(record: dict[str, Any]) -> bytes:
    """Compact single-line UTF-8 JSON with a trailing newline.

    Raises :class:`TraceWriteError` on non-dict input or unserializable
    values — callers must never let a half-serialized record reach disk.
    """

    if not isinstance(record, dict):
        raise TraceWriteError(
            f"trace record must be a dict per specs/trace-v1.schema.json, "
            f"got {type(record).__name__}"
        )
    try:
        text = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TraceWriteError(f"trace record is not JSON-serializable: {exc}") from exc
    return text.encode("utf-8") + b"\n"


def append_trace(
    record: dict[str, Any],
    trace_root: Path | str | None = None,
    *,
    now: float | None = None,
    max_segment_bytes: int = DEFAULT_SEGMENT_CAP_BYTES,
) -> Path:
    """Append one trace record to today's (UTC) rotated segment file.

    Parameters
    ----------
    record:
        The trace record. Must be a dict matching
        ``specs/trace-v1.schema.json``; this storage layer validates shape
        (dict-ness and serializability) only — schema validation belongs to
        the instrumentation layer that assembled the record.
    trace_root:
        Root directory of traces. Defaults to
        :func:`default_trace_root` (``~/.hermes/sessions/traces/``).
    now:
        Unix timestamp whose UTC date names the segment — the run's *end*
        time per the spec. ``None`` uses the current time.
    max_segment_bytes:
        Size cap per segment file; rolling past it starts the next ``.NNN``
        suffix. Defaults to the spec's 512 MB.

    Returns
    -------
    Path
        The segment that received the record.

    Raises
    ------
    TraceWriteError
        If the target exists but is not a regular appendable file, the root
        cannot be created, or the record is invalid/unserializable. Existing
        files are never overwritten or truncated by this function.

    The append is a single ``os.write`` on an ``O_APPEND`` descriptor followed
    by ``os.fsync`` before returning.
    """

    if now is None:
        now = time.time()
    root = Path(trace_root) if trace_root is not None else default_trace_root()

    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise TraceWriteError(f"cannot create trace root {root}: {exc}") from exc
    if not root.is_dir():
        raise TraceWriteError(f"trace root is not a directory: {root}")

    line = _serialize_record(record)
    segment = _resolve_segment(root, now, len(line), max_segment_bytes)

    # O_APPEND: every write is positioned at end-of-file atomically with the
    # write itself, so concurrent appenders never overwrite each other's
    # bytes — POSIX guarantees this for writes up to PIPE_BUF; Windows makes
    # no such guarantee, so appends there are serialized under a process-wide
    # lock instead (see _win32_append_lock). O_CREAT creates a missing
    # segment; O_TRUNC/O_WRONLY-style truncation flags are never used
    # anywhere in this module. O_BINARY is REQUIRED on Windows: without it
    # os.open() opens in text mode and every b"\n" is translated to b"\r\n",
    # corrupting the JSONL and inflating st_size by one byte per line.
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | _O_BINARY
    if sys.platform == "win32":
        with _win32_append_lock:
            fd = os.open(segment, flags, 0o600)
            try:
                os.write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
    else:
        fd = os.open(segment, flags, 0o600)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)
    LOG.debug("appended trace record (%d bytes) to %s", len(line), segment)
    return segment


def segment_name_for(t: float) -> str:
    """Segment basename (``traces-YYYY-MM-DD.jsonl``) for a Unix timestamp's UTC date."""

    return f"traces-{datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y-%m-%d')}.jsonl"


def _segment_sort_key(path: Path) -> tuple[str, int]:
    """Sort key: (UTC date, suffix index). The primary segment (no ``.NNN``
    suffix) sorts BEFORE its own rollovers, so replaying segments in
    ``list_segments()`` order reads records in write order.

    Lexical sorting cannot do this: ``"traces-2026-08-29.002.jsonl"`` sorts
    before ``"traces-2026-08-29.jsonl"`` because ``"."`` (0x2E) sorts below
    ``"j"`` (0x6A), which would place the primary after its own rollovers.
    """

    name = path.name
    stem = name[: -len(".jsonl")] if name.endswith(".jsonl") else name
    date = stem[len("traces-"):] if stem.startswith("traces-") else stem
    if "." in date:
        date, _, suffix = date.rpartition(".")
        try:
            index = int(suffix)
        except ValueError:
            index = 0
    else:
        index = 0
    return (date, index)


def list_segments(trace_root: Path | str | None = None) -> list[Path]:
    """All ``traces-*.jsonl`` segment files under ``trace_root``.

    Sorted by (UTC date, suffix index) via :func:`_segment_sort_key` — the
    primary segment precedes its own ``.NNN`` rollovers, so iteration order
    is write order. Lexical name order would shuffle them.
    """

    root = Path(trace_root) if trace_root is not None else default_trace_root()
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.glob("traces-*.jsonl")
         if p.is_file() and not p.is_symlink()),
        key=_segment_sort_key,
    )


def iter_records(
    path: Path | str,
    on_torn: Callable[[TornRecordWarning], None] | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Iterate ``(line_number, record)`` over a segment, tolerating torn lines.

    Reading is binary to survive a torn line containing malformed UTF-8.
    Blank lines are skipped silently. A line that fails UTF-8 decoding or
    JSON parsing is skipped with a :class:`TornRecordWarning` via
    ``warnings.warn`` (never raises), and ``on_torn`` is invoked with the
    same warning object when provided. Every other line is yielded intact —
    a corrupt final line never hides prior records.
    """

    seg = Path(path)
    with open(seg, "rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            warning: TornRecordWarning | None = None
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                warning = TornRecordWarning(
                    f"skipping undecodable line {line_no} in {seg}: {exc}",
                    path=seg,
                    line_no=line_no,
                    raw_preview=raw[:120],
                )
            else:
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    warning = TornRecordWarning(
                        f"skipping unparseable line {line_no} in {seg}: {exc}",
                        path=seg,
                        line_no=line_no,
                        raw_preview=raw[:120],
                    )
                else:
                    if not isinstance(record, dict):
                        warning = TornRecordWarning(
                            f"skipping non-object JSON on line {line_no} in {seg}",
                            path=seg,
                            line_no=line_no,
                            raw_preview=raw[:120],
                        )
                    else:
                        yield line_no, record
                        continue
            assert warning is not None
            warnings.warn(warning, stacklevel=2)
            if on_torn is not None:
                on_torn(warning)
