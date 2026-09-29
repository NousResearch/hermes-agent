"""Durable interrupted-turn markers for the desktop/TUI auto-continue path. A running turn's progress
lives only in process memory (the agent flushes to SQLite at turn end), so a marker is written at turn
start and cleared on any conclusion — only a process death leaves one behind, and ``session.resume``
reads it (``_maybe_schedule_auto_continue``). Stored per ``HERMES_HOME`` (profile-aware); writes prune
entries older than ``_MAX_AGE_SECS`` and cap the count so a crash streak can't grow the file. Every
legacy bookkeeping function is best-effort — marker writes must never break a turn — so the legacy
reader degrades I/O errors to "no marker". A marker carries its writer's pid and optional start time;
its presence alone does not prove the writer died. Exact attachment uses a separate read-only inspector
that reports unknown recovery instead of inferring terminal work from absent or unreadable state."""

from __future__ import annotations

import json
import logging
import math
import os
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


def _writer_identity() -> dict:
    """Best-effort identity of the process writing this marker. ``writer_pid`` alone already answers "who wrote
    this, and are they still alive?" (``marker_writer_state``); ``writer_start_time`` pairs the pid with its create
    time so a recycled pid cannot pass as the original writer. Identity is bookkeeping, never turn-critical, so
    every failure degrades to a bare pid."""
    identity = {"writer_pid": os.getpid()}
    try:
        from hermes_cli.active_sessions import _own_start_time
        start = _own_start_time()
        if start is not None:
            identity["writer_start_time"] = float(start)
    except Exception:
        pass
    return identity


def marker_writer_state(entry: dict) -> str:
    """``"alive"`` / ``"dead"`` / ``"unknown"``: is the process that wrote this marker still running?

    A marker is durable proof a turn started — never proof its writer died. Two backends sharing one HERMES_HOME
    break that assumption: A is mid-turn on session S while B resumes S, and B used to read A's marker as crash
    evidence and start a second turn over it (#94778). Liveness comes from ``active_sessions._pid_liveness``
    (pid + start time, so a reused pid reads dead), and "unknown" is the safe answer: it leaves the marker alone
    without claiming its writer is gone.
    """
    if not isinstance(entry, dict):
        return "unknown"
    pid = entry.get("writer_pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return "unknown"
    try:
        from hermes_cli.active_sessions import _pid_liveness
        live = _pid_liveness(pid, entry.get("writer_start_time"))
    except Exception:
        return "unknown"
    return "unknown" if live is None else ("alive" if live else "dead")


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


def record_turn_start(home: Path | str, session_key: str, prompt: str, *, attempts: int = 0,
                      auto_continue: bool = True, notification_category: str | None = None) -> None:
    """Persist the marker for a turn that is about to run. ``attempts`` = how many auto-continues led to
    this run (0 for a user-initiated turn); the crash-loop breaker reads it back on the next resume."""
    if not session_key or not prompt:
        return
    now = time.time()
    entry = {"attempts": max(0, int(attempts)), "prompt": prompt[:_MAX_PROMPT_CHARS], "started_at": now,
             "auto_continue": bool(auto_continue), **_writer_identity()}
    if notification_category == "diagnostic":
        entry["notification_category"] = notification_category
    # Identity only — never the prompt: this log is read on crash triage and must not carry turn content.
    logger.debug("turn marker recorded for session %s by writer pid %s", session_key, entry["writer_pid"])
    _update(home, session_key, lambda entries: {**_prune(entries, now), session_key: entry}, "record")


def clear_turn_marker(home: Path | str, session_key: str) -> None:
    """Remove the marker once its turn concluded (any outcome the client saw)."""
    if session_key:
        _update(home, session_key, lambda e: {k: v for k, v in e.items() if k != session_key} if session_key in e else None, "clear")


def read_turn_marker(home: Path | str, session_key: str) -> dict[str, Any] | None:
    """The marker left by a turn that never concluded, or None."""
    if not session_key:
        return None
    try:
        with _lock:
            entry = _load(_marker_path(home)).get(session_key)
        prompt = str(entry.get("prompt") or "") if isinstance(entry, dict) else ""
        if not prompt.strip():
            return None
        return {"attempts": max(0, int(entry.get("attempts") or 0)), "prompt": prompt, "started_at": _started_at(entry),
                "auto_continue": bool(entry.get("auto_continue", True)),
                # Writer identity when present: extra keys only, so a marker written by an older build still reads.
                **{k: entry[k] for k in ("writer_pid", "writer_start_time") if entry.get(k) is not None},
                **({"notification_category": "diagnostic"}
                   if entry.get("notification_category") == "diagnostic" else {})}
    except Exception:
        return None


def inspect_turn_recovery(home: Path | str, session_key: str) -> tuple[str, str]:
    """Read-only, fail-closed counterpart to best-effort ``read_turn_marker``.

    Absence is UNKNOWN, not a terminal receipt. Never return the old prompt,
    clear a marker, prune stale work, or infer permission from freshness/policy.
    """
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate recovery identity")
            obj[key] = value
        return obj

    try:
        with _lock, open(_marker_path(home), encoding="utf-8-sig") as stream:
            raw = stream.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            return "unknown", "invalid_recovery_state"
        entries = json.loads(raw, object_pairs_hook=unique_object)
        # The writer prunes to _MAX_ENTRIES before adding the new turn, so a
        # freshly written file can legitimately contain one extra entry.
        if not isinstance(entries, dict) or len(entries) > _MAX_ENTRIES + 1:
            return "unknown", "invalid_recovery_state"
        for key, entry in entries.items():
            if not isinstance(key, str) or not key or not isinstance(entry, dict):
                return "unknown", "invalid_recovery_state"
            if set(entry) - {"prompt", "started_at", "attempts", "auto_continue", "notification_category",
                              "writer_pid", "writer_start_time"}:
                return "unknown", "invalid_recovery_state"
            if "notification_category" in entry and entry["notification_category"] != "diagnostic":
                return "unknown", "invalid_recovery_state"
            started, attempts, prompt = entry.get("started_at"), entry.get("attempts"), entry.get("prompt")
            if (type(started) not in (int, float) or not math.isfinite(started) or started <= 0
                    or type(attempts) is not int or attempts < 0
                    or not isinstance(prompt, str) or not prompt.strip() or len(prompt) > _MAX_PROMPT_CHARS
                    or type(entry.get("auto_continue", True)) is not bool):
                return "unknown", "invalid_recovery_state"
            if "writer_pid" in entry and (
                    type(entry["writer_pid"]) is not int or not 0 < entry["writer_pid"] <= 0xFFFFFFFF):
                return "unknown", "invalid_recovery_state"
            if "writer_start_time" in entry and (
                    "writer_pid" not in entry or type(entry["writer_start_time"]) not in (int, float)
                    or not math.isfinite(entry["writer_start_time"]) or entry["writer_start_time"] <= 0):
                return "unknown", "invalid_recovery_state"
        marker = entries.get(session_key)
        if marker is None:
            return "unknown", "no_terminal_receipt"
        if "writer_pid" not in marker:
            return "unknown", "writer_liveness_unknown"
        writer = marker_writer_state(marker)
        if writer == "dead":
            return "interrupted", "interrupted_turn"
        if writer == "alive":
            return "unknown", "marker_writer_alive"
        return "unknown", "writer_liveness_unknown"
    except FileNotFoundError:
        return "unknown", "no_terminal_receipt"
    except (OSError, ValueError, TypeError, OverflowError):
        return "unknown", "invalid_recovery_state"
