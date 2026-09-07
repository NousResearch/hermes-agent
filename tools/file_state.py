"""Cross-agent file state coordination.

Prevents mangled edits when concurrent subagents (same process, same
filesystem) touch the same file: B writes a file A already read, so A's next
write would clobber B's changes. Complements the single-agent path-overlap
check in ``agent.tool_dispatch_helpers._should_parallelize_tool_batch``. A process-wide
``FileStateRegistry`` tracks per-agent read stamps, the global last writer and
a per-path lock; every method is a no-op under ``HERMES_DISABLE_FILE_STATE_GUARD=1``.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# (mtime, read_ts, partial). partial=True when read_file returned a windowed
# view (offset > 1 or limit < total_lines) — a later write should still warn
# so the model re-reads in full.
ReadStamp = Tuple[float, float, bool]

# Bounded so long sessions don't accumulate unbounded state.
_MAX_PATHS_PER_AGENT = 4096
_MAX_GLOBAL_WRITERS = 4096


def _disabled() -> bool:
    # Re-read each call so tests can toggle via monkeypatch.setenv.
    return os.environ.get("HERMES_DISABLE_FILE_STATE_GUARD", "").strip() == "1"


def _mtime_or_none(resolved: str) -> Optional[float]:
    try:
        return os.path.getmtime(resolved)
    except OSError:
        return None


def _fmt_ts(ts: float) -> str:
    # Short wall-clock for warnings; avoids datetime formatting on the hot path.
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _evict_oldest(container, cap: int) -> None:
    """Pop entries until *container* is within *cap* (sets: arbitrary; dicts: oldest
    by insertion order). An eviction only costs one redundant re-send or staleness check."""
    for _ in range(len(container) - cap):
        try:
            if isinstance(container, set):
                container.pop()
            else:
                container.pop(next(iter(container)))
        except (StopIteration, KeyError):
            break


class FileStateRegistry:
    """Process-wide coordinator for cross-agent file edits."""

    def __init__(self) -> None:
        self._reads: Dict[str, Dict[str, ReadStamp]] = defaultdict(dict)
        self._last_writer: Dict[str, Tuple[str, float]] = {}
        self._path_locks: Dict[str, threading.Lock] = {}
        self._path_lock_users: Dict[str, int] = {}
        self._meta_lock = threading.Lock()  # guards _path_locks
        self._state_lock = threading.Lock()  # guards _reads + _last_writer

    @contextmanager
    def lock_path(self, resolved: str):
        """Per-path lock: threads on the same path serialize, different paths proceed.
        The lock entry is dropped once the last holder/waiter exits."""
        with self._meta_lock:
            lock = self._path_locks.setdefault(resolved, threading.Lock())
            self._path_lock_users[resolved] = self._path_lock_users.get(resolved, 0) + 1
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            with self._meta_lock:
                users = self._path_lock_users[resolved] - 1
                if users:
                    self._path_lock_users[resolved] = users
                else:
                    self._path_lock_users.pop(resolved, None)
                    self._path_locks.pop(resolved, None)

    def _stamp(self, task_id: str, resolved: str, mtime: float, now: float, partial: bool) -> None:
        """Caller holds ``_state_lock``."""
        agent_reads = self._reads[task_id]
        agent_reads[resolved] = (float(mtime), now, bool(partial))
        _evict_oldest(agent_reads, _MAX_PATHS_PER_AGENT)

    def record_read(self, task_id: str, resolved: str, *, partial: bool = False,
                    mtime: Optional[float] = None) -> None:
        if _disabled():
            return
        mtime = _mtime_or_none(resolved) if mtime is None else mtime
        if mtime is None:
            return
        with self._state_lock:
            self._stamp(task_id, resolved, mtime, time.time(), partial)

    def note_write(self, task_id: str, resolved: str, *, mtime: Optional[float] = None) -> None:
        """Record a successful write: global last-writer AND this agent's own
        read stamp (a write is an implicit read of the current content)."""
        if _disabled():
            return
        mtime = _mtime_or_none(resolved) if mtime is None else mtime
        if mtime is None:
            return
        now = time.time()
        with self._state_lock:
            self._last_writer[resolved] = (task_id, now)
            _evict_oldest(self._last_writer, _MAX_GLOBAL_WRITERS)
            self._stamp(task_id, resolved, mtime, now, False)

    def check_stale(self, task_id: str, resolved: str) -> Optional[str]:
        """Model-facing warning if this write would be stale, else ``None``. Severity
        order: sibling wrote after our read > mtime drift / partial read > never read."""
        if _disabled():
            return None
        with self._state_lock:
            stamp = self._reads.get(task_id, {}).get(resolved)
            last_writer = self._last_writer.get(resolved)

        if stamp is None and last_writer is None:  # net-new file / first touch
            return None
        current_mtime = _mtime_or_none(resolved)
        if current_mtime is None:
            return None  # file doesn't exist — write creates it; not stale

        if last_writer is not None:
            writer_tid, writer_ts = last_writer
            if writer_tid != task_id:
                if stamp is None:
                    return (
                        f"{resolved} was modified by sibling subagent "
                        f"{writer_tid!r} but this agent never read it. "
                        "Read the file before writing to avoid overwriting "
                        "the sibling's changes.")
                read_ts = stamp[1]
                if writer_ts > read_ts:
                    return (
                        f"{resolved} was modified by sibling subagent "
                        f"{writer_tid!r} at {_fmt_ts(writer_ts)} — after "
                        f"this agent's last read at {_fmt_ts(read_ts)}. "
                        "Re-read the file before writing.")

        if stamp is not None:
            read_mtime, _read_ts, partial = stamp
            if current_mtime != read_mtime:
                return (
                    f"{resolved} was modified since you last read it "
                    "on disk (external edit or unrecorded writer). "
                    "Re-read the file before writing.")
            if partial:
                return (
                    f"{resolved} was last read with offset/limit pagination "
                    "(partial view). Re-read the whole file before "
                    "overwriting it.")
            return None

        return (
            f"{resolved} was not read by this agent. "
            "Read the file first so you can write an informed edit.")

    def writes_since(self, exclude_task_id: str, since_ts: float,
                     paths: Iterable[str]) -> Dict[str, List[str]]:
        """``{writer_task_id: [paths]}`` for writes after ``since_ts`` by agents
        other than ``exclude_task_id`` (delegate_task's "subagent modified files
        you previously read" reminder)."""
        if _disabled():
            return {}
        paths_set = set(paths)
        out: Dict[str, List[str]] = defaultdict(list)
        with self._state_lock:
            for p, (writer_tid, ts) in self._last_writer.items():
                if writer_tid != exclude_task_id and ts >= since_ts and p in paths_set:
                    out[writer_tid].append(p)
        return dict(out)

    def known_reads(self, task_id: str) -> List[str]:
        """Resolved paths this agent has read."""
        if _disabled():
            return []
        with self._state_lock:
            return list(self._reads.get(task_id, {}).keys())

    def forget_task(self, task_id: str) -> None:
        """Release read stamps owned by a task after its lifecycle ends."""
        with self._state_lock:
            self._reads.pop(task_id, None)

    def clear(self) -> None:
        """Reset all state. Intended for tests only."""
        with self._state_lock:
            self._reads.clear()
            self._last_writer.clear()
        with self._meta_lock:
            self._path_locks.clear()
            self._path_lock_users.clear()


_registry = FileStateRegistry()


def get_registry() -> FileStateRegistry:
    return _registry


# Convenience wrappers (short names used at call sites).
def record_read(task_id: str, resolved_or_path: str | Path, *, partial: bool = False) -> None:
    _registry.record_read(task_id, str(resolved_or_path), partial=partial)


def note_write(task_id: str, resolved_or_path: str | Path) -> None:
    _registry.note_write(task_id, str(resolved_or_path))


def check_stale(task_id: str, resolved_or_path: str | Path) -> Optional[str]:
    return _registry.check_stale(task_id, str(resolved_or_path))


def lock_path(resolved_or_path: str | Path):
    return _registry.lock_path(str(resolved_or_path))


def writes_since(exclude_task_id: str, since_ts: float, paths: Iterable[str | Path]) -> Dict[str, List[str]]:
    return _registry.writes_since(exclude_task_id, since_ts, [str(p) for p in paths])


def known_reads(task_id: str) -> List[str]:
    return _registry.known_reads(task_id)


__all__ = [
    "FileStateRegistry",
    "get_registry",
    "record_read",
    "note_write",
    "check_stale",
    "lock_path",
    "writes_since",
    "known_reads"]
