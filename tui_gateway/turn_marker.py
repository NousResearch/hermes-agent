"""Durable interrupted-turn markers for the desktop/TUI auto-continue path.

A running turn's progress lives only in process memory (the agent flushes to
SQLite at turn end, not mid-turn), so an app/backend/machine death mid-turn
leaves no durable trace of the interrupted prompt. This sidecar is that
trace: a marker is written when a turn starts running and cleared when the
turn concludes — success, handled error, or interrupt all clear it, so only
a process death leaves one behind. ``session.resume`` reads the marker to
decide whether to auto-continue the interrupted turn (see
``_maybe_schedule_auto_continue`` in ``tui_gateway/server.py``).

Markers are stored per ``HERMES_HOME`` (callers pass the session's home so
profile sessions keep their state in their own profile directory) and the
file is bounded: writes prune entries older than ``_MAX_AGE_SECS`` and cap
the total count, so an unlucky streak of crashes can't grow it unboundedly.

Every function is best-effort by design — marker bookkeeping must never
break a turn — so I/O errors degrade to "no marker" instead of raising.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_MARKER_DIR = "desktop"
_MARKER_FILE = "interrupted_turns.json"
_MAX_AGE_SECS = 24 * 3600
_MAX_ENTRIES = 32
# Enough to re-submit any realistic prompt; guards the sidecar against a
# pathological multi-megabyte paste being journaled on every turn.
_MAX_PROMPT_CHARS = 64_000
# A small backwards clock adjustment is harmless.  A marker hours or days in
# the future is not evidence of a current interrupted turn, though, and must
# not crowd real entries out of the bounded journal or trigger replay.
_MAX_FUTURE_SKEW_SECS = 5 * 60

_lock = threading.Lock()
_lock_warning_paths: set[str] = set()


def _marker_path(home: Path | str) -> Path:
    return Path(home) / _MARKER_DIR / _MARKER_FILE


def _marker_lock_path(home: Path | str) -> Path:
    return Path(home) / _MARKER_DIR / ".interrupted_turns.lock"


@contextmanager
def _process_lock(home: Path | str) -> Iterator[None]:
    """Serialize marker read-modify-write transactions across backends.

    A dashboard restart can briefly overlap the old backend while it drains.
    The process-local lock alone lets those processes both read the same file
    and then atomically replace it with different snapshots, losing one turn.
    Keep the advisory lock on a stable adjacent inode (never the replaced JSON
    inode) so it remains authoritative throughout the transaction.
    """
    path = _marker_lock_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    locked = False
    try:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                # Windows byte-range locks require the byte to exist.
                if not handle.read(1):
                    handle.seek(0)
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
        except (ImportError, OSError):
            # Keep the historical best-effort marker behavior on exotic
            # filesystems/platforms that do not offer advisory locks.
            path_key = str(path)
            if path_key not in _lock_warning_paths:
                _lock_warning_paths.add(path_key)
                logger.warning(
                    "turn-marker cross-process lock unavailable at %s; "
                    "using process-local serialization",
                    path,
                    exc_info=True,
                )
        yield
    finally:
        try:
            if locked and os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif locked:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _started_at(entry: dict) -> float | None:
    try:
        value = float(entry.get("started_at") or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _load(path: Path) -> dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        logger.debug("unreadable turn-marker file %s; starting fresh", path, exc_info=True)
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _prune(entries: dict[str, dict], now: float) -> dict[str, dict]:
    fresh: dict[str, dict] = {}
    for key, entry in entries.items():
        started_at = _started_at(entry)
        if started_at is None:
            continue
        age = now - started_at
        if -_MAX_FUTURE_SKEW_SECS <= age <= _MAX_AGE_SECS:
            fresh[key] = entry
    if len(fresh) <= _MAX_ENTRIES:
        return fresh
    newest = sorted(
        fresh.items(),
        key=lambda item: _started_at(item[1]) or 0,
        reverse=True,
    )[:_MAX_ENTRIES]
    return dict(newest)


def _store(path: Path, entries: dict[str, dict]) -> None:
    if not entries:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".turn-marker-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # The file fsync above makes the payload durable; persisting the
        # containing directory makes the replacement name durable too.  On a
        # sudden machine reset, omitting this step can resurrect the old name
        # or lose the marker entirely even though the temp file was synced.
        if os.name != "nt":
            try:
                dir_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                # Some network filesystems reject directory fsync.  The
                # atomic replacement already succeeded; preserve that useful
                # state instead of treating bookkeeping as a failed write.
                logger.debug("directory fsync unavailable for %s", path.parent)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def record_turn_start(
    home: Path | str, session_key: str, prompt: str, *, attempts: int = 0
) -> None:
    """Persist the marker for a turn that is about to run.

    ``attempts`` counts how many auto-continues led to this run: 0 for a
    user-initiated turn, N for the Nth automatic re-run — the crash-loop
    breaker reads it back on the next resume.
    """
    if not session_key or not prompt:
        return
    now = time.time()
    entry = {
        "attempts": max(0, int(attempts)),
        "prompt": prompt[:_MAX_PROMPT_CHARS],
        "started_at": now,
    }
    try:
        with _lock:
            with _process_lock(home):
                path = _marker_path(home)
                entries = _prune(_load(path), now)
                entries[session_key] = entry
                # Apply the bound after insertion too.  Pruning only the old
                # snapshot allowed every successful write to hold 33 entries.
                _store(path, _prune(entries, now))
    except Exception:
        logger.debug("failed to record turn marker for %s", session_key, exc_info=True)


def clear_turn_marker(home: Path | str, session_key: str) -> None:
    """Remove the marker once its turn concluded (any outcome the client saw)."""
    if not session_key:
        return
    try:
        with _lock:
            with _process_lock(home):
                path = _marker_path(home)
                entries = _load(path)
                if session_key not in entries:
                    return
                del entries[session_key]
                _store(path, entries)
    except Exception:
        logger.debug("failed to clear turn marker for %s", session_key, exc_info=True)


def read_turn_marker(home: Path | str, session_key: str) -> dict[str, Any] | None:
    """The marker left by a turn that never concluded, or None."""
    if not session_key:
        return None
    try:
        with _lock:
            entry = _load(_marker_path(home)).get(session_key)
    except Exception:
        return None
    if not isinstance(entry, dict):
        return None
    prompt = str(entry.get("prompt") or "")
    if not prompt.strip():
        return None
    try:
        started_at = _started_at(entry)
        if started_at is None:
            return None
        attempts = max(0, int(entry.get("attempts") or 0))
    except (TypeError, ValueError, OverflowError):
        return None
    return {"attempts": attempts, "prompt": prompt, "started_at": started_at}
