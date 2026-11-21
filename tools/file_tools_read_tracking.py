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
write_file, or patched from a baseline it already held; required before write_file
may overwrite an existing file), ``blind_patches`` (resolved path -> sha256 a patch
of this task wrote without such a baseline, for the refusal wording) and
``not_found`` (short-TTL negative cache). Every
container is hard-capped (``_cap_read_tracker_data``) so long sessions stay small.
"""

import contextlib
import contextvars
import hashlib
import logging
import os
import stat
import threading
import time

from tools.file_state import _evict_oldest
from tools.file_tools_paths import _authoritative_workspace_root, _resolve_path_for_task

logger = logging.getLogger("tools.file_tools")


class _PerAgentTracker(dict):
    """``{task_id: ...}`` whose keys carry the bound hosted agent's sandbox key, so two hosted
    agents that reuse a task id never share loop counters, read history or patch failures. With
    no hosted scope bound a key is the task id itself, as before."""

    @staticmethod
    def _key(task_id):
        from tools.environments.sandbox_scope import current_sandbox_key
        sandbox = current_sandbox_key()
        return f"{sandbox}:{task_id}" if sandbox else task_id

    def __getitem__(self, task_id):
        return super().__getitem__(self._key(task_id))

    def __setitem__(self, task_id, value):
        super().__setitem__(self._key(task_id), value)

    def __contains__(self, task_id):
        return super().__contains__(self._key(task_id))

    def get(self, task_id, default=None):
        return super().get(self._key(task_id), default)

    def setdefault(self, task_id, default=None):
        return super().setdefault(self._key(task_id), default)

    def pop(self, task_id, *default):
        return super().pop(self._key(task_id), *default)


_read_tracker_lock = threading.Lock()
_read_tracker: dict = _PerAgentTracker()

# Consecutive patch failures per (task_id, resolved_path); escalates the hint
# when the model keeps failing the same file. Reset on a successful patch.
_patch_failure_lock = threading.Lock()
_patch_failure_tracker: dict = _PerAgentTracker()  # {task_id: {resolved_path: count}}
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
    for key in ("dedup", "dedup_hits", "read_timestamps", "read_coverage", "full_write_baselines",
                "blind_patches", "remote_write_baselines"):
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
        ("blind_patches", _FULL_WRITE_BASELINES_CAP),
        ("remote_write_baselines", _FULL_WRITE_BASELINES_CAP),
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
    # dead network mount must not stall every task. A sandbox path cannot be
    # checked here, so its miss is never served from the cache.
    if not _on_host(resolved_str) or os.path.exists(resolved_str):
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


# Set while an ``execute_code`` cell dispatches a tool. The cell consumes the result in code, so
# the "you already have this" stub only breaks it (``result["content"]`` raises), and the read
# shows the model nothing, so it must not count as content the conversation has seen.
_programmatic_read: contextvars.ContextVar[bool] = contextvars.ContextVar("programmatic_file_read", default=False)


@contextlib.contextmanager
def programmatic_file_read():
    token = _programmatic_read.set(True)
    try:
        yield
    finally:
        _programmatic_read.reset(token)


def _dedup_lookup(task_id: str, dedup_key: tuple) -> tuple[dict, object, bool]:
    """``(task_data, cached_version, stub_eligible)`` for a read. Eligible means the model was
    already served this region since the last compaction boundary (the summary may have dropped
    exact bytes, so the first read after one serves content); never for a programmatic read."""
    with _read_tracker_lock:
        task_data = _task_data(task_id)
        return (task_data, task_data["dedup"].get(dedup_key),
                dedup_key in task_data["dedup_generation_reads"] and not _programmatic_read.get())


def _note_read_served(task_data: dict, dedup_key: tuple, path: str, offset: int, limit: int) -> tuple[int, dict]:
    """Record that a read's content was served; returns ``(consecutive count, dedup map)``. Lock
    must be held. A programmatic read leaves the model-facing state alone: count 1 and a
    throwaway map, so a later read by the model is never told it already has these bytes."""
    if _programmatic_read.get():
        return 1, {}  # the map is discarded: nothing is remembered as shown to the model
    task_data["dedup_hits"].pop(dedup_key, None)
    task_data["dedup_generation_reads"].add(dedup_key)
    return _bump_consecutive(task_data, ("read", path, offset, limit)), task_data["dedup"]


def _bump_consecutive(task_data: dict, key: tuple) -> int:
    """Update last_key/consecutive for *key* and return the new count. Lock must be held.
    A cell repeating a read or search in a loop is code, not a model stuck re-asking: count 1."""
    if _programmatic_read.get():
        return 1
    if task_data["last_key"] == key:
        task_data["consecutive"] += 1
    else:
        task_data["last_key"] = key
        task_data["consecutive"] = 1
    return task_data["consecutive"]


def reset_file_dedup(task_id: str | None = None):
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
    if resolved is None or not _on_host(resolved):
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


def _on_host(resolved: str) -> bool:
    """Whether *resolved* names a file on this host. A hosted agent's paths name its sandbox: the
    bookkeeping below (dedup, staleness, baselines) never stats or opens them here."""
    from tools.environments.sandbox_scope import hosted_scope
    return hosted_scope() is None


def _file_metadata(resolved: str) -> tuple | None:
    if not _on_host(resolved):
        return None
    try:
        st = os.stat(resolved)
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
    except OSError:
        return None


def _file_version(resolved: str) -> tuple | None:
    """A byte snapshot, not just mtime (editors/copy tools can preserve that)."""
    if not _on_host(resolved):
        return None
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
        task_data["blind_patches"].pop(str(resolved), None)
        _cap_read_tracker_data(task_data)


def _has_full_write_baseline(resolved: str, task_id: str) -> bool:
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id) or {}
        baseline = task_data.get("full_write_baselines", {}).get(str(resolved))
    return baseline is not None and _file_version(resolved) == baseline


def _known_full_content_sha256(resolved_paths, task_id: str) -> dict:
    """``{resolved: sha256 hex}`` for each path whose recorded full-write baseline still
    matches the bytes on disk. Call under the per-path locks, before a patch reads."""
    with _read_tracker_lock:
        baselines = dict((_read_tracker.get(task_id) or {}).get("full_write_baselines", {}))
    known = {}
    for resolved in {str(r) for r in resolved_paths if r}:
        baseline = baselines.get(resolved)
        if baseline is not None and _file_version(resolved) == baseline:
            known[resolved] = baseline[-1].hex()
    return known


def _carry_full_write_baselines(task_id: str, known: dict, writes, path_to_resolved: dict) -> None:
    """After this task's own successful patch: keep the whole-file baseline of each written
    path when the task knew the exact bytes the patch read (``known``, from
    ``_known_full_content_sha256``) or the patch created the file (read sha ``""``). Then
    the task knows every byte it wrote. ``writes`` is ``PatchResult._writes`` in order, so
    a second edit of the same path chains from the first. The mark re-hashes the disk and
    requires the bytes the patch wrote, so a writer landing after the patch leaves no
    baseline. A write the task could not vouch for is noted in ``blind_patches``."""
    lookup = {raw: r for raw, r in path_to_resolved.items() if r}
    lookup.update({r: r for r in lookup.values()})
    known = dict(known)
    written: dict = {}
    for path, read_sha, written_sha in writes:
        resolved = lookup.get(path)
        if resolved is None:
            continue  # a backend-side path (sandbox, translated mount): no host baseline
        vouched = read_sha == "" or (read_sha is not None and read_sha == known.get(resolved))
        known[resolved] = written_sha if vouched else None
        written[resolved] = written_sha
    for resolved, written_sha in written.items():
        if written_sha is None:
            continue
        if known[resolved] is not None:
            _mark_full_write_baseline(resolved, task_id, written_sha)
            continue
        with _read_tracker_lock:
            task_data = _task_data(task_id)
            task_data["blind_patches"][resolved] = written_sha
            _cap_read_tracker_data(task_data)


def _is_own_blind_patch(resolved: str, task_id: str) -> bool:
    """True when the bytes on disk are exactly what this task's own patch wrote without a
    full-content baseline (a blind patch, or one after a partial view)."""
    with _read_tracker_lock:
        written_sha = ((_read_tracker.get(task_id) or {}).get("blind_patches") or {}).get(str(resolved))
    if written_sha is None:
        return False
    version = _file_version(resolved)
    return version is not None and version[-1].hex() == written_sha


_READ_COVERAGE_RANGES_CAP = 256


def _note_read_coverage(task_data: dict, resolved: str | tuple[str, str], version: tuple | str, start: int, end: int,
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


def _record_remote_read(task_id: str, key: tuple[str, str], digest: str, *, start: int,
                        end: int, total_lines: int | None, redacted: bool) -> None:
    """Record a backend file version only after the same version bracketed the read."""
    with _read_tracker_lock:
        data = _task_data(task_id)
        baselines = data["remote_write_baselines"]
        if baselines.get(key) == digest:
            return
        complete, redacted_any = _note_read_coverage(
            data, key, digest, start, end, total_lines, redacted)
        if total_lines == 0 and start == 1:
            complete = True
        if complete and not redacted_any:
            baselines[key] = digest
        else:
            baselines.pop(key, None)
        _cap_read_tracker_data(data)


def _remote_write_baseline(task_id: str, key: tuple[str, str]) -> str | None:
    with _read_tracker_lock:
        data = _read_tracker.get(task_id) or {}
        return data.get("remote_write_baselines", {}).get(key)


def _mark_remote_write_baseline(task_id: str, key: tuple[str, str], digest: str | None) -> None:
    with _read_tracker_lock:
        baselines = _task_data(task_id)["remote_write_baselines"]
        if digest is None:
            baselines.pop(key, None)
        else:
            baselines[key] = digest
        _cap_read_tracker_data(_task_data(task_id))


def _read_mtime_drifted(filepath: str, task_id: str) -> bool:
    """True when the file's mtime changed since this task last read it. False when
    never read, fresh, or unstattable (a deleted file is the write's problem)."""
    resolved = _resolved_or_none(filepath, task_id)
    if resolved is None or not _on_host(resolved):
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
    if not paths or not _on_host(paths[0]):  # project detection reads the workspace's manifests
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
