"""delegate_task ``files`` inlining: the parent names ``path`` or ``path:start-end`` ranges and the child's first turn
carries their current bytes, so the child starts with the code it needs instead of re-reading it.

The ranges are read once at dispatch against the workspace the child is told about, capped per file and in total,
and marked as already read in the child's read_file dedup state (``mark_files_read``).
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

MAX_FILE_LINES = 400
MAX_TOTAL_BYTES = 60_000
FILES_HEADING = "FILES (already read for you, current as of dispatch):"
_RANGE_RE = re.compile(r"^(?P<path>.+?):(?P<start>\d+)-(?P<end>\d+)$")
_SNIFF_BYTES = 8192
_MIN_ENTRY_BYTES = 512  # below this an entry would be all label and marker; also the skipped-list marker's room
_MAX_SKIPPED_NAMES = 8
_CHUNK = 64 * 1024


@dataclass(frozen=True)
class FileSpec:
    """One requested entry; ``start``/``end`` are 1-based inclusive, ``None`` for the whole file."""
    path: str
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass(frozen=True)
class InlinedFile:
    """What the child was shown: the resolved path, the line span, whether every requested line fit unredacted,
    and the exact snapshot it came from (``_file_version`` shape: metadata + sha256; None when the file moved
    during the read), which the read tracker is tied to."""
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


def _bounded_lines(stream: Any, digest: Any, keep: Any):
    """Yield ``(n, head, size)`` per ``\\n``-terminated line (and a non-empty unterminated tail), reading
    ``_CHUNK``-byte blocks so a newline-free line is never held whole: *head* is at most ``keep(n)`` bytes of the
    line and *size* its full length. Every byte goes through *digest*."""
    n, held, size, limit = 1, bytearray(), 0, keep(1)
    while chunk := stream.read(_CHUNK):
        digest.update(chunk)
        pos = 0
        while (nl := chunk.find(b"\n", pos)) != -1:
            held += chunk[pos:min(nl + 1, pos + max(0, limit - len(held)))]
            size += nl + 1 - pos
            yield n, bytes(held), size
            n, held, size, pos = n + 1, bytearray(), 0, nl + 1
            limit = keep(n)
        held += chunk[pos:pos + max(0, limit - len(held))]
        size += len(chunk) - pos
    if size:
        yield n, bytes(held), size


def _read_lines(resolved: Path, start: int, end: Optional[int], budget: int) -> tuple[list[str], int, bytes]:
    """Lines ``start..end`` split on ``\\n`` only (read_file's rule), streamed in fixed-size chunks so neither a
    huge file nor one huge line is ever loaded; at most ``MAX_FILE_LINES`` rows within ``budget`` bytes. Returns
    ``(rows, wanted, sha256)``: *wanted* counts the requested lines that exist and *sha256* covers the whole file
    (the snapshot the child saw)."""
    digest = hashlib.sha256()
    rows: list[str] = []
    wanted = used = 0

    def keep(n: int) -> int:
        if n < start or (end is not None and n > end) or len(rows) >= MAX_FILE_LINES or used >= budget:
            return 0
        # Slack covers what rendering strips (BOM, \r, \n): a line longer than this can never fit the budget.
        return budget - used + 8

    with open(resolved, "rb") as stream:
        for n, head, size in _bounded_lines(stream, digest, keep):
            if n < start or (end is not None and n > end):
                continue
            wanted += 1
            if len(rows) >= MAX_FILE_LINES or used >= budget:
                continue
            if len(head) < size:
                used = budget
                continue
            text = head.decode("utf-8", errors="replace").rstrip("\n").removesuffix("\r")
            row = f"{n}|{text.removeprefix(chr(0xFEFF)) if n == 1 else text}\n"
            if used + len(row.encode("utf-8")) > budget:
                used = budget
                continue
            rows.append(row)
            used += len(row.encode("utf-8"))
    return rows, wanted, digest.digest()


def _render_one(spec: FileSpec, resolved: Path, budget: int) -> tuple[str, InlinedFile]:
    body_budget = budget - len(spec.path) - 128
    text, span, used = _render_within(spec, resolved, body_budget)
    # A long backtick run in the body widens both fences past the reserve. Budgeting below the rows actually shown
    # drops at least one row per pass, even when the next row was already too long to fit.
    while (over := len(text.encode("utf-8")) - budget) > 0 and body_budget > 0:
        body_budget = min(body_budget, used) - over
        text, span, used = _render_within(spec, resolved, body_budget)
    return text, span


def _render_within(spec: FileSpec, resolved: Path, body_budget: int) -> tuple[str, InlinedFile, int]:
    from agent.redact import _is_secret_file_arg, redact_sensitive_text
    from tools.file_tools_read_tracking import _file_metadata

    label = spec.path if spec.start is None else f"{spec.path}:{spec.start}-{spec.end}"
    start = spec.start or 1
    before = _file_metadata(str(resolved))
    rows, wanted, digest = _read_lines(resolved, start, spec.end, body_budget - (len(label) - len(spec.path)))
    if spec.start is not None and wanted == 0:
        raise ValueError(f"'files' entry {label!r} starts beyond the end of the file.")
    stable = before is not None and _file_metadata(str(resolved)) == before
    raw = "".join(rows)
    body = redact_sensitive_text(raw, file_read=True, secret_file=_is_secret_file_arg(str(resolved)))
    missing = wanted - len(rows)
    if missing:
        body += f"[truncated: {missing} more lines, read_file to see the rest]\n"
    fence = _fence(body)
    version = (*before, digest) if stable else None
    span = InlinedFile(str(resolved), start, max(start + len(rows) - 1, start), spec.start is None,
                       not missing and body == raw, version)
    return f"{label}\n{fence}\n{body}{fence}\n", span, len(raw.encode("utf-8"))


def _files_are_local() -> bool:
    """The block is read from this host, so it is the child's view only when the child's file tools run here too.
    ``_get_env_config`` bridges a config-only backend (``hermes serve``, ACP) that no terminal call has bridged yet."""
    from tools.environments.sandbox_scope import hosted_scope
    from tools.terminal_tool import _get_env_config
    return _get_env_config()["env_type"] == "local" and hosted_scope() is None


def resolve_task_files(
    task_list: list[dict[str, Any]], files: Any, parent_agent: Any,
) -> tuple[list[Optional[TaskFiles]], Optional[str]]:
    """Every task's ``FILES`` block, read against the workspace the child is told about; one refusal fails the
    whole call before any child spawns. Refused under ``delegation.worktree_isolation``: the child then edits a
    fresh worktree off HEAD, so the parent checkout's bytes (and its paths) are not the files it works on."""
    task_specs, err = coerce_task_files(task_list, files)
    if err or not any(task_specs):
        return [None] * len(task_list) if not err else [], err
    from tools.delegate_tool_config import _get_worktree_isolation
    from tools.delegate_tool_progress import _resolve_workspace_hint
    if _get_worktree_isolation():
        return [], ("'files' is not supported with delegation.worktree_isolation (the child works in a fresh "
                    "worktree, not the files read here); put the paths in 'context' instead.")
    if not _files_are_local():
        return [], ("'files' needs the local terminal backend (on ssh/docker/modal the child's tools see the backend's "
                    "files, not the ones read here); put the paths in 'context' instead.")
    workspace = _resolve_workspace_hint(parent_agent)
    out: list[Optional[TaskFiles]] = []
    for i, specs in enumerate(task_specs):
        files, err = read_task_files(specs, workspace) if specs else (None, None)
        if err:
            return [], f"Task {i} {err}"
        out.append(files)
    return out, None


def read_task_files(specs: list[FileSpec], workspace: Optional[str]) -> tuple[Optional[TaskFiles], Optional[str]]:
    """Read *specs* against *workspace* into one ``FILES`` block, or ``(None, error)`` for the first refusal. Every
    rendered byte (labels, fences, markers) counts against ``MAX_TOTAL_BYTES``; entries past it are listed by name."""
    if not workspace:
        return None, "'files' needs a local workspace directory for the child; put the paths in 'context' instead."
    base = Path(workspace).resolve()
    resolved_specs = []
    for spec in specs:
        resolved, err = _refusal(spec, base)
        if resolved is None:
            return None, err
        resolved_specs.append((spec, resolved))
    parts, spans, skipped = [], [], []
    # Room for the skipped-names marker is held back whenever a later entry could be skipped.
    budget = MAX_TOTAL_BYTES - len(FILES_HEADING) - 1 - (_MIN_ENTRY_BYTES if len(resolved_specs) > 1 else 0)
    for spec, resolved in resolved_specs:
        label = spec.path if spec.start is None else f"{spec.path}:{spec.start}-{spec.end}"
        if budget < _MIN_ENTRY_BYTES:
            skipped.append(label)
            continue
        try:
            text, span = _render_one(spec, resolved, budget)
        except ValueError as exc:
            return None, str(exc)
        # A label (with its fences and marker) longer than what is left cannot be cut to fit: list it instead.
        if len(text.encode("utf-8")) > budget:
            skipped.append(label)
            continue
        budget -= len(text.encode("utf-8")) + 1
        parts.append(text)
        spans.append(span)
    if skipped:
        names = ", ".join(skipped[:_MAX_SKIPPED_NAMES]) + (f" +{len(skipped) - _MAX_SKIPPED_NAMES} more"
                                                          if len(skipped) > _MAX_SKIPPED_NAMES else "")
        shown = names.encode()[:_MIN_ENTRY_BYTES - 96].decode(errors="ignore")
        parts.append(f"[not inlined, over the {MAX_TOTAL_BYTES}-byte cap: {shown}; "
                     "read_file to see them]\n")
    return TaskFiles(f"{FILES_HEADING}\n" + "\n".join(parts), tuple(spans)), None


def with_files(context: Optional[str], files: Optional[TaskFiles]) -> Optional[str]:
    """The child's context with the ``FILES`` block after it. It rides inside the context string, so the rebuild
    spec (action='continue', auto-resume) and the brief's cache-registered prefix carry it unchanged."""
    if files is None:
        return context
    return f"{context.rstrip()}\n\n{files.block}" if context and context.strip() else files.block


def mark_files_read(task_id: str, files: Optional[TaskFiles]) -> None:
    """Record each fully inlined, unredacted span in the child's read tracker, tied to the snapshot the child was
    shown: a read_file of the same range on those bytes gets the dedup notice, and a whole file becomes the write
    baseline only while the disk still holds exactly those bytes. A span whose file changed since is left unmarked,
    so the child's next read returns the real content and a stale whole-file write is still refused."""
    if files is None:
        return
    from tools import file_state
    from tools.file_operations_common import DEFAULT_READ_LIMIT, normalize_read_pagination
    from tools.file_tools_read_tracking import (
        _cap_read_tracker_data, _file_version, _read_tracker_lock, _task_data,
    )

    for span in files.spans:
        if not span.complete or span.version is None or _file_version(span.resolved) != span.version:
            continue
        st = os.stat(span.resolved)
        if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) != span.version[:-1]:
            continue
        key = (span.resolved, *normalize_read_pagination(
            span.start, DEFAULT_READ_LIMIT if span.whole_file else span.end - span.start + 1))
        with _read_tracker_lock:
            task_data = _task_data(task_id)
            task_data["dedup"][key] = span.version[:-1]
            task_data["dedup_generation_reads"].add(key)
            if span.whole_file:
                task_data["full_write_baselines"][span.resolved] = span.version
            _cap_read_tracker_data(task_data)
        # The stamp is the shown snapshot's mtime (the stat that matched it), so a later change reads as stale.
        file_state.get_registry().record_read(task_id, span.resolved, partial=not span.whole_file, mtime=st.st_mtime)
