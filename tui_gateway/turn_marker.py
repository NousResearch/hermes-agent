"""Durable interrupted-turn markers for the desktop/TUI auto-continue path. A running turn's progress
lives only in process memory (the agent flushes to SQLite at turn end), so a marker is written at turn
start and cleared on any conclusion — only a process death leaves one behind, and ``session.resume``
reads it (``_maybe_schedule_auto_continue``). Stored per ``HERMES_HOME`` (profile-aware); writes prune
entries older than ``_MAX_AGE_SECS`` and cap the count so a crash streak can't grow the file. Every
function is best-effort — marker bookkeeping must never break a turn — so I/O errors degrade to "no
marker" instead of raising."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_MAX_AGE_SECS = 24 * 3600
_MAX_ENTRIES = 32
# Enough to re-submit any realistic prompt; guards against a multi-megabyte paste being journaled.
_MAX_PROMPT_CHARS = 64_000

_lock = threading.Lock()


def _marker_path(home: Path | str) -> Path:
    return Path(home) / "desktop" / "interrupted_turns.json"


def _started_at(entry: dict) -> float:
    return float(entry.get("started_at") or 0)


def _load(path: Path) -> dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        logger.debug("unreadable turn-marker file %s; starting fresh", path, exc_info=True)
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def _prune(entries: dict[str, dict], now: float) -> dict[str, dict]:
    fresh = {k: e for k, e in entries.items() if now - _started_at(e) <= _MAX_AGE_SECS}
    if len(fresh) <= _MAX_ENTRIES:
        return fresh
    return dict(sorted(fresh.items(), key=lambda item: _started_at(item[1]), reverse=True)[:_MAX_ENTRIES])


def _store(path: Path, entries: dict[str, dict]) -> None:
    if not entries:
        path.unlink(missing_ok=True)
        return
    atomic_json_write(path, entries, indent=None, mode=0o600)


def _update(home: Path | str, session_key: str, mutate, what: str) -> None:
    """Load → ``mutate(entries)`` → store under the lock; ``mutate`` returns None to skip the write."""
    try:
        with _lock:
            path = _marker_path(home)
            entries = mutate(_load(path))
            if entries is not None:
                _store(path, entries)
    except Exception:
        logger.debug("failed to %s turn marker for %s", what, session_key, exc_info=True)


def record_turn_start(
    home: Path | str, session_key: str, prompt: str, *, attempts: int = 0,
    auto_continue: bool = True, writer: dict[str, Any] | None = None,
) -> None:
    """Persist the marker for a turn that is about to run. ``attempts`` = how many auto-continues led to
    this run (0 for a user-initiated turn); the crash-loop breaker reads it back on the next resume.

    ``writer`` (preemptible leases) stamps the recording holder's identity (lease_id +
    epoch): clears become conditional compare-and-deletes so a DYING holder can never
    retire the NEW owner's marker, and a steal can detect + force-clear a foreign one.
    Absent ``writer`` = pre-change/foreign recorder; identity-less markers stay clearable
    by anyone (they cannot belong to a live identity-checked writer)."""
    if not session_key or not prompt:
        return
    now = time.time()
    entry: dict[str, Any] = {"attempts": max(0, int(attempts)), "prompt": prompt[:_MAX_PROMPT_CHARS], "started_at": now,
                             "auto_continue": bool(auto_continue)}
    if isinstance(writer, dict) and writer.get("lease_id"):
        entry["lease_id"] = str(writer.get("lease_id"))
        entry["epoch"] = int(writer.get("epoch") or 1)
    _update(home, session_key, lambda entries: {**_prune(entries, now), session_key: entry}, "record")


def _marker_writer(entry: Any) -> dict[str, Any] | None:
    """The stored entry's writer identity, or None for an identity-less (legacy) marker."""
    if not isinstance(entry, dict) or not entry.get("lease_id"):
        return None
    return {"lease_id": str(entry.get("lease_id")), "epoch": int(entry.get("epoch") or 1)}


def clear_turn_marker(
    home: Path | str, session_key: str, *, writer: dict[str, Any] | None = None, force: bool = False
) -> None:
    """Remove the marker once its turn concluded (any outcome the client saw).

    Conditional compare-and-delete: a ``writer``-identified clear removes the stored
    marker ONLY when it still matches that identity (or is identity-less legacy) — the
    dying holder of a stolen session must not retire the new owner's marker. ``force``
    (and plain writer-less calls, for legacy call sites) clears unconditionally: the
    steal path's deliberate foreign-clear, and stale-marker cleanup at resume."""
    if not session_key:
        return
    if force or writer is None:
        _update(home, session_key, lambda e: {k: v for k, v in e.items() if k != session_key} if session_key in e else None, "clear")
        return

    def _conditional(entries: dict[str, dict]) -> dict[str, dict] | None:
        stored = entries.get(session_key)
        if stored is None:
            return None
        stored_writer = _marker_writer(stored)
        if stored_writer is not None and stored_writer != {"lease_id": str(writer.get("lease_id")), "epoch": int(writer.get("epoch") or 1)}:
            return None  # not our marker anymore: the new owner's marker stays
        return {k: v for k, v in entries.items() if k != session_key}

    _update(home, session_key, _conditional, "clear")


def read_turn_marker(home: Path | str, session_key: str) -> dict[str, Any] | None:
    """The marker left by a turn that never concluded, or None. Carries the writer
    identity (when recorded) so the auto-continue kickoff can bail if the lease epoch
    moved between schedule and dispatch."""
    if not session_key:
        return None
    try:
        with _lock:
            entry = _load(_marker_path(home)).get(session_key)
        prompt = str(entry.get("prompt") or "") if isinstance(entry, dict) else ""
        if not prompt.strip():
            return None
        marker: dict[str, Any] = {
            "attempts": max(0, int(entry.get("attempts") or 0)), "prompt": prompt,
            "started_at": _started_at(entry),
            "auto_continue": bool(entry.get("auto_continue", True))}
        if (writer := _marker_writer(entry)) is not None:
            marker.update(writer)
        return marker
    except Exception:
        return None
