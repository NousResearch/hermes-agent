"""delegate_task ``files``: the parent names ``path`` or ``path:start-end`` ranges and the child's first turn
carries their current bytes, so the child starts with the code it needs instead of re-reading it.

The ranges are read once at dispatch against the workspace the child is told about, capped per file and in total,
and marked as already read in the child's read_file dedup state (``mark_files_read``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

MAX_FILE_LINES = 400
MAX_TOTAL_BYTES = 60_000
FILES_HEADING = "FILES (already read for you, current as of dispatch):"
_RANGE_RE = re.compile(r"^(?P<path>.+?):(?P<start>\d+)-(?P<end>\d+)$")
_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class FileSpec:
    """One requested entry; ``start``/``end`` are 1-based inclusive, ``None`` for the whole file."""
    path: str
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass(frozen=True)
class InlinedFile:
    """What the child was shown: the resolved path, the line span, whether every requested line fit unredacted,
    and the file's metadata stamp taken before the read (``_file_metadata``; None when unstattable)."""
    resolved: str
    start: int
    end: int
    whole_file: bool
    complete: bool
    version: Optional[tuple]


@dataclass(frozen=True)
class TaskFiles:
    """The rendered ``FILES`` block for one task and the spans it holds."""
    block: str
    spans: tuple[InlinedFile, ...]


def parse_file_spec(raw: Any, i: int) -> tuple[Optional[FileSpec], Optional[str]]:
    if not isinstance(raw, str) or not raw.strip():
        return None, f"Task {i} 'files' entries must be non-empty strings ('path' or 'path:start-end')."
    text = raw.strip()
    match = _RANGE_RE.match(text)
    if match is None:
        return FileSpec(text), None
    start, end = int(match["start"]), int(match["end"])
    if start < 1 or end < start:
        return None, f"Task {i} 'files' entry {text!r}: the range must be 1-based with start <= end."
    return FileSpec(match["path"], start, end), None


def coerce_task_files(
    task_list: list[dict[str, Any]], files: Any,
) -> tuple[list[Optional[list[FileSpec]]], Optional[str]]:
    """Per-task parsed ``files``; the legacy top-level ``files`` applies to a single task only, like ``images``."""
    out: list[Optional[list[FileSpec]]] = []
    for i, task in enumerate(task_list):
        raw = task.get("files")
        if raw is None and len(task_list) == 1:
            raw = files
        if raw is None:
            out.append(None)
            continue
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return [], f"Task {i} 'files' must be an array of 'path' or 'path:start-end' strings."
        specs = []
        for item in raw:
            spec, err = parse_file_spec(item, i)
            if err:
                return [], err
            specs.append(spec)
        out.append(specs or None)
    return out, None


def _fence(body: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", body)), default=0)
    return "`" * max(3, longest + 1)


def _refusal(spec: FileSpec, base: Path) -> tuple[Optional[Path], Optional[str]]:
    """The resolved file, or why it cannot be inlined (outside the workspace, missing, denied, binary)."""
    from agent.file_safety import get_read_block_error
    from tools.binary_extensions import has_binary_extension

    resolved = (base / Path(spec.path).expanduser()).resolve()
    if not resolved.is_relative_to(base):
        return None, f"'files' entry {spec.path!r} is outside the child's workspace {base}."
    if not resolved.is_file():
        return None, f"'files' entry {spec.path!r} is not a file under {base}."
    blocked = get_read_block_error(str(resolved))
    if blocked:
        return None, blocked
    with open(resolved, "rb") as stream:
        head = stream.read(_SNIFF_BYTES)
    if has_binary_extension(str(resolved)) or b"\0" in head:
        return None, f"'files' entry {spec.path!r} is a binary file; describe it in 'context' instead."
    return resolved, None


def _render_one(spec: FileSpec, resolved: Path, budget: int) -> tuple[str, InlinedFile]:
    from agent.redact import _is_secret_file_arg, redact_sensitive_text
    from tools.file_tools_read_tracking import _file_metadata

    version = _file_metadata(str(resolved))
    lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    start = spec.start or 1
    end = min(spec.end or len(lines), len(lines))
    wanted = lines[start - 1:end] if start <= len(lines) else []
    shown = []
    used = 0
    for n, line in enumerate(wanted[:MAX_FILE_LINES], start):
        row = f"{n}|{line}\n"
        if used + len(row.encode("utf-8")) > budget:
            break
        shown.append(row)
        used += len(row.encode("utf-8"))
    raw = "".join(shown)
    body = redact_sensitive_text(raw, file_read=True, secret_file=_is_secret_file_arg(str(resolved)))
    missing = len(wanted) - len(shown)
    if missing:
        body += f"[truncated: {missing} more lines, read_file to see the rest]\n"
    last = start + len(shown) - 1
    label = spec.path if spec.start is None else f"{spec.path}:{spec.start}-{spec.end}"
    fence = _fence(body)
    span = InlinedFile(str(resolved), start, max(last, start), spec.start is None, not missing and body == raw, version)
    return f"{label}\n{fence}\n{body}{fence}\n", span


def resolve_task_files(
    task_specs: list[Optional[list[FileSpec]]], workspace: Optional[str],
) -> tuple[list[Optional[TaskFiles]], Optional[str]]:
    """Every task's ``FILES`` block; one refusal fails the whole call before any child spawns."""
    out: list[Optional[TaskFiles]] = []
    for i, specs in enumerate(task_specs):
        files, err = read_task_files(specs, workspace) if specs else (None, None)
        if err:
            return [], f"Task {i} {err}"
        out.append(files)
    return out, None


def read_task_files(specs: list[FileSpec], workspace: Optional[str]) -> tuple[Optional[TaskFiles], Optional[str]]:
    """Read *specs* against *workspace* into one ``FILES`` block, or ``(None, error)`` for the first refusal."""
    if not workspace:
        return None, "'files' needs a local workspace directory for the child; put the paths in 'context' instead."
    base = Path(workspace).resolve()
    parts, spans = [], []
    budget = MAX_TOTAL_BYTES
    for spec in specs:
        resolved, err = _refusal(spec, base)
        if resolved is None:
            return None, err
        text, span = _render_one(spec, resolved, budget)
        budget -= len(text.encode("utf-8"))
        parts.append(text)
        spans.append(span)
    return TaskFiles(f"{FILES_HEADING}\n" + "\n".join(parts), tuple(spans)), None


def with_files(context: Optional[str], files: Optional[TaskFiles]) -> Optional[str]:
    """The child's context with the ``FILES`` block after it. It rides inside the context string, so the rebuild
    spec (action='continue', auto-resume) and the brief's cache-registered prefix carry it unchanged."""
    if files is None:
        return context
    return f"{context.rstrip()}\n\n{files.block}" if context and context.strip() else files.block


def mark_files_read(task_id: str, files: Optional[TaskFiles]) -> None:
    """Record each fully inlined, unchanged span in the child's read tracker, so a read_file of the same range gets
    the dedup notice instead of the bytes again. Truncated or redacted spans stay unmarked: the child must be able
    to read the real rest. A whole file is keyed the way read_file keys a default read (offset 1, default limit)."""
    if files is None:
        return
    from tools import file_state
    from tools.file_operations_common import DEFAULT_READ_LIMIT, normalize_read_pagination
    from tools.file_tools_read_tracking import (
        _cap_read_tracker_data, _file_metadata, _mark_full_write_baseline, _read_tracker_lock, _task_data,
    )

    for span in files.spans:
        if not span.complete or span.version is None or _file_metadata(span.resolved) != span.version:
            continue
        key = (span.resolved, *normalize_read_pagination(
            span.start, DEFAULT_READ_LIMIT if span.whole_file else span.end - span.start + 1))
        with _read_tracker_lock:
            task_data = _task_data(task_id)
            task_data["dedup"][key] = span.version
            task_data["dedup_generation_reads"].add(key)
            _cap_read_tracker_data(task_data)
        file_state.record_read(task_id, span.resolved, partial=not span.whole_file)
        if span.whole_file:
            _mark_full_write_baseline(span.resolved, task_id)
