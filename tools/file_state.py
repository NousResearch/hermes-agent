"""Cross-agent file state coordination.

Prevents mangled edits when concurrent subagents (same process, same
filesystem) touch the same file. Complements the single-agent path-overlap
check in ``run_agent._should_parallelize_tool_batch`` — this module catches
the case where subagent B writes a file that subagent A already read, so
A's next write would overwrite B's changes with stale content.

Design
------
A process-wide singleton ``FileStateRegistry`` tracks, per resolved path:

  * per-agent read stamps: {task_id: {path: (mtime, read_ts, partial)}}
  * last writer globally: {path: (task_id, write_ts)}
  * per-path ``threading.Lock`` for read→modify→write critical sections

Three public hooks are used by the file tools:

  * ``record_read(task_id, path, *, partial)`` — called by read_file
  * ``note_write(task_id, path)`` — called after write_file / patch
  * ``check_stale(task_id, path)`` — called BEFORE write_file / patch

Plus ``lock_path(path)`` — a context-manager returning a per-path lock to
wrap the whole read→modify→write block. And ``writes_since(task_id,
since_ts, paths)`` for the subagent-completion reminder in delegate_tool.

All methods are no-ops when ``HERMES_DISABLE_FILE_STATE_GUARD=1`` is set.

This module is intentionally separate from ``_read_tracker`` in
``file_tools.py`` — that tracker is per-task and handles consecutive-read
loop detection, which is a different concern.
"""
from __future__ import annotations

import itertools
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ── Public stamp type ────────────────────────────────────────────────
# (mtime, read_ts, partial).  partial=True when read_file returned a
# windowed view (offset > 1 or limit < total_lines) — writes that happen
# after a partial read should still warn so the model re-reads in full.
ReadStamp = Tuple[float, float, bool]

# Number of resolved-path entries retained per agent.  Bounded to keep
# long sessions from accumulating unbounded state.  On overflow we drop
# the oldest entries by insertion order.
_MAX_PATHS_PER_AGENT = 4096

# Global last-writer map cap.  Same policy.
_MAX_GLOBAL_WRITERS = 4096

# Top-level task entries retained in the read-stamp registry.  This is a
# defense-in-depth cap for long-lived gateway/API processes; normal task
# completion removes entries explicitly via ``cleanup_task``.
_MAX_TASKS = 4096


@dataclass
class _TrackedLock:
    lock: threading.Lock
    users: int = 0


class FileStateRegistry:
    """Process-wide coordinator for cross-agent file edits."""

    def __init__(self) -> None:
        self._reads: Dict[str, Dict[str, ReadStamp]] = defaultdict(dict)
        self._last_writer: Dict[str, Tuple[str, float]] = {}
        self._path_locks: Dict[str, _TrackedLock] = {}
        self._meta_lock = threading.Lock()  # guards _path_locks
        self._state_lock = threading.Lock()  # guards _reads + _last_writer

    # ── Path lock management ────────────────────────────────────────
    def _retain_lock(self, resolved: str) -> _TrackedLock:
        with self._meta_lock:
            tracked = self._path_locks.get(resolved)
            if tracked is None:
                tracked = _TrackedLock(threading.Lock())
                self._path_locks[resolved] = tracked
            tracked.users += 1
            return tracked

    def _release_lock(self, resolved: str, tracked: _TrackedLock) -> None:
        with self._meta_lock:
            tracked.users -= 1
            if tracked.users <= 0 and self._path_locks.get(resolved) is tracked:
                self._path_locks.pop(resolved, None)

    @contextmanager
    def lock_path(self, resolved: str):
        """Acquire the per-path lock for a read→modify→write section.

        Same process, same filesystem — threads on the same path serialize.
        Different paths proceed in parallel.  Locks are retained only while
        active/waiting so unique-path writes cannot grow process memory
        indefinitely.
        """
        if _disabled():
            yield
            return

        tracked = self._retain_lock(resolved)
        tracked.lock.acquire()
        try:
            yield
        finally:
            tracked.lock.release()
            self._release_lock(resolved, tracked)

    # ── Read/write accounting ───────────────────────────────────────
    def record_read(
        self,
        task_id: str,
        resolved: str,
        *,
        partial: bool = False,
        mtime: Optional[float] = None,
    ) -> None:
        if _disabled():
            return
        if mtime is None:
            try:
                mtime = os.path.getmtime(resolved)
            except OSError:
                return
        now = time.time()
        with self._state_lock:
            # Move task_id to newest position (LRU) so active tasks are not
            # evicted before idle ones when _reads hits _MAX_TASKS.
            agent_reads = self._reads.pop(task_id, None)
            if agent_reads is None:
                agent_reads = {}
            self._reads[task_id] = agent_reads
            agent_reads[resolved] = (float(mtime), now, bool(partial))
            _cap_dict(agent_reads, _MAX_PATHS_PER_AGENT)
            _cap_dict(self._reads, _MAX_TASKS)

    def note_write(
        self,
        task_id: str,
        resolved: str,
        *,
        mtime: Optional[float] = None,
    ) -> None:
        """Record a successful write.

        Updates the global last-writer map AND this agent's own read stamp
        (a write is an implicit read — the agent now knows the current
        content).
        """
        if _disabled():
            return
        if mtime is None:
            try:
                mtime = os.path.getmtime(resolved)
            except OSError:
                return
        now = time.time()
        with self._state_lock:
            self._last_writer[resolved] = (task_id, now)
            _cap_dict(self._last_writer, _MAX_GLOBAL_WRITERS)
            # Writer's own view is now up-to-date.
            # Move task_id to newest position (LRU) so active tasks are not
            # evicted before idle ones when _reads hits _MAX_TASKS.
            agent_reads = self._reads.pop(task_id, None)
            if agent_reads is None:
                agent_reads = {}
            self._reads[task_id] = agent_reads
            agent_reads[resolved] = (float(mtime), now, False)
            _cap_dict(agent_reads, _MAX_PATHS_PER_AGENT)
            _cap_dict(self._reads, _MAX_TASKS)

    def check_stale(self, task_id: str, resolved: str) -> Optional[str]:
        """Return a model-facing warning if this write would be stale.

        Three staleness classes, in order of severity:

          1. Sibling subagent wrote this file after this agent's last read.
          2. External/unknown change (mtime differs from our last read).
          3. Agent never read the file (write-without-read).

        Returns ``None`` when the write is safe.  Does not raise — callers
        decide whether to block or warn.
        """
        if _disabled():
            return None
        with self._state_lock:
            stamp = self._reads.get(task_id, {}).get(resolved)
            last_writer = self._last_writer.get(resolved)

        # Case 3: never read AND we have no write record — net-new file or
        # first touch by this agent.  Let existing _check_sensitive_path
        # and file-exists logic handle it; nothing to warn about here.
        if stamp is None and last_writer is None:
            return None

        try:
            current_mtime = os.path.getmtime(resolved)
        except OSError:
            # File doesn't exist — write will create it; not stale.
            return None

        # Case 1: sibling subagent modified after our last read.
        if last_writer is not None:
            writer_tid, writer_ts = last_writer
            if writer_tid != task_id:
                if stamp is None:
                    return (
                        f"{resolved} was modified by sibling subagent "
                        f"{writer_tid!r} but this agent never read it. "
                        "Read the file before writing to avoid overwriting "
                        "the sibling's changes."
                    )
                read_ts = stamp[1]
                if writer_ts > read_ts:
                    return (
                        f"{resolved} was modified by sibling subagent "
                        f"{writer_tid!r} at {_fmt_ts(writer_ts)} — after "
                        f"this agent's last read at {_fmt_ts(read_ts)}. "
                        "Re-read the file before writing."
                    )

        # Case 2: external / unknown modification (mtime drifted).
        if stamp is not None:
            read_mtime, _read_ts, partial = stamp
            if current_mtime != read_mtime:
                return (
                    f"{resolved} was modified since you last read it "
                    "on disk (external edit or unrecorded writer). "
                    "Re-read the file before writing."
                )
            if partial:
                return (
                    f"{resolved} was last read with offset/limit pagination "
                    "(partial view). Re-read the whole file before "
                    "overwriting it."
                )

        # Case 3b: agent truly never read the file.
        if stamp is None:
            return (
                f"{resolved} was not read by this agent. "
                "Read the file first so you can write an informed edit."
            )

        return None

    # ── Reminder helper for delegate_tool ───────────────────────────
    def writes_since(
        self,
        exclude_task_id: str,
        since_ts: float,
        paths: Iterable[str],
    ) -> Dict[str, List[str]]:
        """Return ``{writer_task_id: [paths]}`` for writes done after
        ``since_ts`` by agents OTHER than ``exclude_task_id``.

        Used by delegate_task to append a "subagent modified files the
        parent previously read" reminder to the delegation result.
        """
        if _disabled():
            return {}
        paths_set = set(paths)
        out: Dict[str, List[str]] = defaultdict(list)
        with self._state_lock:
            for p, (writer_tid, ts) in self._last_writer.items():
                if writer_tid == exclude_task_id:
                    continue
                if ts < since_ts:
                    continue
                if p in paths_set:
                    out[writer_tid].append(p)
        return dict(out)

    def known_reads(self, task_id: str) -> List[str]:
        """Return the list of resolved paths this agent has read."""
        if _disabled():
            return []
        with self._state_lock:
            return list(self._reads.get(task_id, {}).keys())

    def cleanup_task(self, task_id: str) -> None:
        """Forget read stamps owned by a completed task.

        Runs unconditionally — even when the kill-switch is active — so
        tasks that recorded stamps before the guard was disabled still
        release their ``_reads`` entries and cannot cause unbounded growth.
        """
        with self._state_lock:
            self._reads.pop(task_id, None)

    # ── Testing hooks ───────────────────────────────────────────────
    def clear(self) -> None:
        """Reset all state.  Intended for tests only."""
        with self._state_lock:
            self._reads.clear()
            self._last_writer.clear()
        with self._meta_lock:
            self._path_locks.clear()


# ── Module-level singleton + helpers ─────────────────────────────────
_registry = FileStateRegistry()


def get_registry() -> FileStateRegistry:
    return _registry


def _disabled() -> bool:
    # Re-read each call so tests can toggle via monkeypatch.setenv.
    return os.environ.get("HERMES_DISABLE_FILE_STATE_GUARD", "").strip() == "1"


def _fmt_ts(ts: float) -> str:
    # Short relative wall-clock for error messages; avoids pulling in
    # datetime formatting overhead on the hot path.
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _cap_dict(d: dict, limit: int) -> None:
    """Trim a dict to ``limit`` entries by dropping insertion-order oldest."""
    over = len(d) - limit
    if over <= 0:
        return
    # Snapshot only the first `over` keys rather than the full keyset so
    # trimming costs O(over) instead of O(len(d)) on each call.
    for key in list(itertools.islice(iter(d), over)):
        d.pop(key, None)


# ── Convenience wrappers (short names used at call sites) ────────────
def record_read(task_id: str, resolved_or_path: str | Path, *, partial: bool = False) -> None:
    _registry.record_read(task_id, str(resolved_or_path), partial=partial)


def note_write(task_id: str, resolved_or_path: str | Path) -> None:
    _registry.note_write(task_id, str(resolved_or_path))


def check_stale(task_id: str, resolved_or_path: str | Path) -> Optional[str]:
    return _registry.check_stale(task_id, str(resolved_or_path))


def lock_path(resolved_or_path: str | Path):
    return _registry.lock_path(str(resolved_or_path))


def writes_since(
    exclude_task_id: str,
    since_ts: float,
    paths: Iterable[str | Path],
) -> Dict[str, List[str]]:
    return _registry.writes_since(exclude_task_id, since_ts, [str(p) for p in paths])


def known_reads(task_id: str) -> List[str]:
    return _registry.known_reads(task_id)


def cleanup_task(task_id: str) -> None:
    _registry.cleanup_task(task_id)


__all__ = [
    "FileStateRegistry",
    "get_registry",
    "record_read",
    "note_write",
    "check_stale",
    "lock_path",
    "writes_since",
    "known_reads",
    "cleanup_task",
]
