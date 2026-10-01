"""Per-task read/search bookkeeping for the file tools.

Process-lifetime state behind read_file/search_files/write_file/patch.
Per task_id ``_read_tracker``
stores: ``last_key``/``consecutive`` (loop detection; reset by any OTHER tool
call), ``read_history`` (diagnostics), ``dedup`` (key -> file metadata; survives context
compression), ``dedup_generation_reads`` (keys whose full content was served since
the last compaction boundary; cleared on compression so one recovery read returns
full content), ``dedup_hits`` (stub-loop breaker), ``read_timestamps``
(staleness warnings), ``read_coverage`` (per resolved path: the line ranges the
task has paged through at one file version — contiguous pages that reach the last line
count as a whole-file read), ``full_write_baselines`` (resolved paths whose
whole-file content this task saw via unredacted read_file page(s) or wrote via
write_file; required before write_file may overwrite an existing file — patch
never qualifies) and ``not_found`` (short-TTL negative cache). Every
container is hard-capped (``_cap_read_tracker_data``) so long sessions stay small.
"""

import hashlib
import logging
import os
import stat
import threading
import time

from tools.file_state import _evict_oldest
from tools.file_tools_paths import _authoritative_workspace_root, _resolve_path_for_task

logger = logging.getLogger("tools.file_tools")

_read_tracker_lock = threading.Lock()
_read_tracker: dict = {}

# Consecutive patch failures per (task_id, resolved_path); escalates the hint
# when the model keeps failing the same file. Reset on a successful patch.
_patch_failure_lock = threading.Lock()
_patch_failure_tracker: dict = {}  # {task_id: {resolved_path: count}}
_PATCH_FAILURE_PATHS_CAP = 64

# Only the most recent reads matter for dedup, loop detection and external-edit
# warnings; caps bound accretion regardless of session length.
_READ_HISTORY_CAP = 500
_DEDUP_CAP = 1000
_READ_TIMESTAMPS_CAP = 1000
_FULL_WRITE_BASELINES_CAP = 1000
_NOT_FOUND_CAP = 500
_NOT_FOUND_TTL_SECONDS = 60.0  # a path that didn't exist may be created soon


def _task_data(task_id: str) -> dict:
    """Get-or-create the tracker entry for *task_id*, back-filling missing containers
    (search_tool / tests create partial entries). Lock must be held."""
    task_data = _read_tracker.setdefault(task_id, {
        "last_key": None, "consecutive": 0, "read_history": set()})
    for key in ("dedup", "dedup_hits", "read_timestamps", "read_coverage", "full_write_baselines"):
        task_data.setdefault(key, {})
    task_data.setdefault("dedup_generation_reads", set())
    return task_data


def _record_patch_failure(task_id: str, resolved_path: str) -> int:
    """Increment and return the consecutive-failure count for this path."""
    with _patch_failure_lock:
        task_failures = _patch_failure_tracker.setdefault(task_id, {})
        # Evict the oldest entry once a task has failed on many distinct files.
        if resolved_path not in task_failures:
            _evict_oldest(task_failures, _PATCH_FAILURE_PATHS_CAP - 1)
        task_failures[resolved_path] = task_failures.get(resolved_path, 0) + 1
        return task_failures[resolved_path]


def _reset_patch_failures(task_id: str, resolved_paths: list) -> None:
    """Clear consecutive-failure counts for the given paths."""
    if not resolved_paths:
        return
    with _patch_failure_lock:
        task_failures = _patch_failure_tracker.get(task_id)
        for rp in resolved_paths if task_failures else ():
            task_failures.pop(rp, None)


def _cap_read_tracker_data(task_data: dict) -> None:
    """Enforce size caps on the per-task sub-containers. Call with ``_read_tracker_lock`` held."""
    # Caps are read at call time so tests can monkeypatch the module constants.
    for key, cap in (
        ("read_history", _READ_HISTORY_CAP),
        ("dedup", _DEDUP_CAP),
        ("dedup_hits", _DEDUP_CAP),
        ("dedup_generation_reads", _DEDUP_CAP),
        ("read_timestamps", _READ_TIMESTAMPS_CAP),
        ("read_coverage", _READ_TIMESTAMPS_CAP),
        ("full_write_baselines", _FULL_WRITE_BASELINES_CAP),
        ("not_found", _NOT_FOUND_CAP)):
        container = task_data.get(key)
        if container is not None and len(container) > cap:
            _evict_oldest(container, cap)


def _resolved_or_none(filepath: str, task_id: str) -> str | None:
    try:
        return str(_resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        return None


def _pop_not_found(op: str, resolved_str: str, task_id: str) -> None:
    """Drop the negative-cache entry for *(op, resolved_str)*. Lock must be held."""
    task_data = _read_tracker.get(task_id)
    nf = task_data.get("not_found") if task_data else None
    if nf:
        nf.pop((op, resolved_str), None)


def _check_not_found_cache(op: str, resolved_str: str, task_id: str) -> str | None:
    """Return cached not-found JSON for *(op, resolved_str)* if still fresh.

    *op* is "read" or "search" (different error JSON shapes). Evicted by TTL,
    by write_file/patch on the path, or by any other tool call.
    """
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        entry = (task_data.get("not_found") or {}).get((op, resolved_str)) if task_data else None
        if entry is None:
            return None
        ts, cached_json = entry
        if time.monotonic() - ts > _NOT_FOUND_TTL_SECONDS:
            _pop_not_found(op, resolved_str, task_id)
            return None
    # "check → create → read" is common, so never serve a stale miss for a path
    # that now exists. The stat runs OUTSIDE the tracker lock: a hung stat on a
    # dead network mount must not stall every task.
    if os.path.exists(resolved_str):
        with _read_tracker_lock:
            _pop_not_found(op, resolved_str, task_id)
        return None
    return cached_json


def _record_not_found(op: str, resolved_str: str, task_id: str, error_json: str) -> None:
    """Cache a not-found error so the next *op* call for *resolved_str* skips I/O."""
    with _read_tracker_lock:
        task_data = _task_data(task_id)
        task_data.setdefault("not_found", {})[(op, resolved_str)] = (time.monotonic(), error_json)
        _cap_read_tracker_data(task_data)


def _bump_consecutive(task_data: dict, key: tuple) -> int:
    """Update last_key/consecutive for *key* and return the new count. Lock must be held."""
    if task_data["last_key"] == key:
        task_data["consecutive"] += 1
    else:
        task_data["last_key"] = key
        task_data["consecutive"] = 1
    return task_data["consecutive"]


def reset_file_dedup(task_id: str = None):
    """Advance the read-dedup generation after context compression (one task, or all
    when ``task_id`` is None). The per-key ``dedup`` metadata map is PRESERVED so unchanged
    files keep returning stubs instead of re-bloating the reclaimed context; the
    generation-read set is cleared so the FIRST unchanged read of each key after
    compaction returns full content the summary may have dropped. Stub-hit counters
    are cleared so the hard block restarts fresh. write_file baselines survive
    exactly like the dedup map does — while the file metadata still matches the
    stamp this task recorded; byte identity is checked before overwriting. A baseline
    whose file changed underneath is dropped
    (the stat runs outside the lock so a hung mount cannot stall other tasks)."""
    with _read_tracker_lock:
        if task_id:
            targets = [_read_tracker[task_id]] if _read_tracker.get(task_id) else []
        else:
            targets = list(_read_tracker.values())
        for task_data in targets:
            if "dedup_hits" in task_data:
                task_data["dedup_hits"].clear()
            task_data.setdefault("dedup_generation_reads", set()).clear()
        candidates = [(task_data, dict(task_data.get("full_write_baselines", {})))
                      for task_data in targets]
    for task_data, baselines in candidates:
        changed = {p for p, version in baselines.items() if _file_metadata(p) != version[:-1]}
        if changed:
            with _read_tracker_lock:
                for p in changed:
                    if task_data["full_write_baselines"].get(p) == baselines[p]:
                        task_data["full_write_baselines"].pop(p, None)


def notify_other_tool_call(task_id: str = "default"):
    """Reset the consecutive read/search counter for a task.

    Called by the dispatcher for every tool OTHER than read_file/search_files.
    Also clears stub-hit counters and the not-found cache: any other tool may
    have created a previously-missing path (or flipped its permissions).
    """
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data:
            task_data["last_key"] = None
            task_data["consecutive"] = 0
            for key in ("dedup_hits", "not_found"):
                if task_data.get(key):
                    task_data[key].clear()


def _invalidate_dedup_for_path(filepath: str, task_id: str) -> None:
    """Evict every dedup entry (all offset/limit ranges) and not-found entry for *filepath*
    after a write, so the next read returns fresh content. Acquires the lock itself."""
    resolved = _resolved_or_none(filepath, task_id)
    if resolved is None:
        return
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data is None:
            return
        dedup = task_data.get("dedup")
        if dedup:
            for k in [k for k in dedup if k[0] == resolved]:
                del dedup[k]
        _pop_not_found("read", resolved, task_id)
        _pop_not_found("search", resolved, task_id)


def _update_read_timestamp(filepath: str, task_id: str) -> None:
    """After a successful write: invalidate dedup and refresh the stored mtime so
    consecutive edits by the same task don't trigger false staleness warnings.

    Also invalidates the dedup cache for the written path so that subsequent reads return fresh content
    (fixes #13144).
    """
    _invalidate_dedup_for_path(filepath, task_id)
    resolved = _resolved_or_none(filepath, task_id)
    if resolved is None:
        return
    try:
        current_mtime = os.path.getmtime(resolved)
    except OSError:
        return
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data is not None:
            task_data.setdefault("read_timestamps", {})[resolved] = current_mtime
            _cap_read_tracker_data(task_data)


def _file_metadata(resolved: str) -> tuple | None:
    try:
        st = os.stat(resolved)
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
    except OSError:
        return None


def _file_version(resolved: str) -> tuple | None:
    """A byte snapshot, not just mtime (editors/copy tools can preserve that)."""
    try:
        if not stat.S_ISREG(os.stat(resolved).st_mode):
            return None
        fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                return None
            digest = hashlib.file_digest(stream, "sha256").digest()
            after = os.stat(resolved)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        version = tuple(getattr(before, name) for name in fields)
        if version == tuple(getattr(after, name) for name in fields):
            return (*version, digest)
        return None
    except OSError:
        return None


def _mark_full_write_baseline(resolved: str, task_id: str, expected_sha256: str | None = None) -> None:
    """Record that *task_id* saw the whole current content of *resolved* (full
    unredacted read_file, or its own successful write_file), so a later
    write_file may replace the file. Acquires the lock itself."""
    version = _file_version(resolved)
    if version is None or (expected_sha256 is not None and version[-1].hex() != expected_sha256):
        return
    with _read_tracker_lock:
        task_data = _task_data(task_id)
        task_data["full_write_baselines"][str(resolved)] = version
        _cap_read_tracker_data(task_data)


def _has_full_write_baseline(resolved: str, task_id: str) -> bool:
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id) or {}
        baseline = task_data.get("full_write_baselines", {}).get(str(resolved))
    return baseline is not None and _file_version(resolved) == baseline


_READ_COVERAGE_RANGES_CAP = 256


def _note_read_coverage(task_data: dict, resolved: str, version: tuple, start: int, end: int,
                        total_lines, redacted: bool) -> tuple[bool, bool]:
    """Merge the page ``start..end`` into this task's coverage of *resolved* and return
    ``(complete, redacted_any)``: whether pages taken at this same *version* now reach from
    line 1 to *total_lines*, and whether any of them came back redacted. A file too large
    for one read_file page (>2000 lines / the char budget) can only ever be seen this
    way, so paging through it must count as a whole-file read. A new version restarts the
    coverage (the earlier pages describe a file that no longer exists). Lock must be held."""
    coverage = task_data.setdefault("read_coverage", {})
    entry = coverage.get(resolved)
    if entry is None or entry["version"] != version or len(entry["ranges"]) > _READ_COVERAGE_RANGES_CAP:
        entry = coverage[resolved] = {"version": version, "ranges": [], "redacted": False}
    entry["redacted"] = entry["redacted"] or redacted
    merged: list[tuple[int, int]] = []
    for s, e in sorted(entry["ranges"] + [(start, end)]):
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    entry["ranges"] = merged
    complete = (isinstance(total_lines, int) and total_lines > 0
                and merged[0][0] <= 1 and merged[0][1] >= total_lines)
    return complete, entry["redacted"]


def _read_mtime_drifted(filepath: str, task_id: str) -> bool:
    """True when the file's mtime changed since this task last read it. False when
    never read, fresh, or unstattable (a deleted file is the write's problem)."""
    resolved = _resolved_or_none(filepath, task_id)
    if resolved is None:
        return False
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        read_mtime = task_data.get("read_timestamps", {}).get(resolved) if task_data else None
    if read_mtime is None:
        return False
    try:
        return os.path.getmtime(resolved) != read_mtime
    except OSError:
        return False


def _check_file_staleness(filepath: str, task_id: str) -> str | None:
    """Warn (don't block) when the file's mtime changed since this task last read it."""
    if _read_mtime_drifted(filepath, task_id):
        return (
            f"Warning: {filepath} was modified since you last read it "
            "(external edit or concurrent agent). The content you read may be "
            "stale. Consider re-reading the file to verify before writing.")
    return None


def _mark_verification_stale(task_id: str, resolved_paths: list[str],
                             session_id: str | None = None) -> None:
    """Best-effort note that successful edits made prior verification stale. cwd: the
    first edited path's recognised project root, else the workspace root, else the first parent."""
    from pathlib import Path

    paths = [p for p in resolved_paths if p]
    if not paths:
        return
    try:
        from agent.coding_context import project_facts_for
        from agent.verification_evidence import mark_workspace_edited

        parents = [str(Path(p).parent) for p in paths]
        cwd = (next((c for c in parents if project_facts_for(c)), None)
               or _authoritative_workspace_root(task_id) or parents[0])
        mark_workspace_edited(session_id=session_id or task_id, cwd=cwd, paths=paths)
    except Exception:
        logger.debug("verification stale marker failed", exc_info=True)
