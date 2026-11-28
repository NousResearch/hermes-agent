"""Durable interrupted-turn markers for the desktop/TUI auto-continue path. A running turn's progress
lives only in process memory (the agent flushes to SQLite at turn end), so a marker is written at turn
start and cleared on any conclusion — only a process death leaves one behind, and ``session.resume``
reads it (``_maybe_schedule_auto_continue``). Stored per ``HERMES_HOME`` (profile-aware); writes prune
entries older than ``_MAX_AGE_SECS`` and cap the count so a crash streak can't grow the file. Every
function is best-effort — marker bookkeeping must never break a turn — so I/O errors degrade to "no
marker" instead of raising. A marker also carries its writer's pid + start time (``marker_writer_state``): "a
marker exists" only implies the writer died when no live sibling backend wrote it (#94778).

A marker additionally records the immutable identity of the turn that created it — a per-turn
``turn_token`` plus the writer's pid + start time. ``clear_turn_marker`` / ``touch_turn_marker`` are
scoped to that identity, so a SUPERSEDED runtime's turn end, Stop, or heartbeat can never delete,
refresh or revive a NEWER owner's marker for the same stored session key (they share one HERMES_HOME
and one session key across processes); only the turn that wrote a marker may mutate it.

Two DIFFERENT questions are asked of this file, so there are two readers:

* ``read_turn_marker`` — the crash auto-continue contract. A marker with no prompt TEXT is nothing to
  re-submit, so it reads as None and a blank/attachment-only turn is never auto-continued.
* ``read_turn_marker_entry`` / ``turn_in_flight`` — the cross-process "is a turn running for this
  chat?" contract used by the Desktop lease takeover (``session_lifecycle._holder_turn_in_flight``).
  Every turn that can run records a marker (blank prompt included), so this reader has NO prompt
  requirement and reports "unreadable marker file" separately from "no marker"
  (``MarkerUnreadable``), letting a fail-closed caller tell the two apart."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_MAX_AGE_SECS = 24 * 3600
_MAX_ENTRIES = 32
# Enough to re-submit any realistic prompt; guards against a multi-megabyte paste being journaled.
_MAX_PROMPT_CHARS = 64_000
# Minimum gap between progress touches: a busy turn emits many events and every touch is a locked
# read-modify-write of a file sibling backends share. 15s is far below the staleness threshold and
# far above the event rate.
_PROGRESS_MIN_INTERVAL_SECS = 15.0

_lock = threading.Lock()
_last_touch_lock = threading.Lock()
_last_touch: dict[tuple[str, str], float] = {}
_file_lock_holder = threading.local()


class MarkerUnreadable(Exception):
    """The durable turn-marker FILE exists but cannot be read or parsed.

    Raised by ``read_turn_marker_entry`` so a caller that must fail closed can tell "the marker file
    is unreadable" (it may name a live turn) from "there is no marker" (nothing was recorded).
    ``read_turn_marker`` keeps its best-effort contract and never raises.
    """


def _marker_path(home: Path | str) -> Path:
    return Path(home) / "desktop" / "interrupted_turns.json"


def _cross_process_lock(path: Path):
    """Sibling backends on one HERMES_HOME (#94778) don't share ``_lock``; without this flock one's stale
    load → store resurrects a marker the other just cleared (a finished turn auto-continues again)."""
    from hermes_cli.auth import _file_lock
    return _file_lock(path.with_name(f"{path.name}.lock"), _file_lock_holder, 5.0,
                      f"Timed out waiting for the turn-marker lock ({path})")


def _started_at(entry: dict) -> float:
    return float(entry.get("started_at") or 0)


def _progress_at(entry: dict) -> float:
    """Last sign of life for a recorded turn: the throttle-stamped progress, else when it started.

    An entry written before the heartbeat existed carries no ``progress_at``; falling back to
    ``started_at`` keeps an old build honest (a long turn then reads as no progress) rather than
    inventing liveness.
    """
    return float(entry.get("progress_at") or entry.get("started_at") or 0)


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


# The immutable identity a marker records about the turn that created it. ``turn_token`` is the
# precise anchor (unique per turn, minted at creation); the writer pid + create time is the
# process-level fallback, and the only identity an older build's marker can carry.
_IDENTITY_FIELDS = ("turn_token", "writer_pid", "writer_start_time")


def marker_identity(entry: dict[str, Any] | None) -> dict[str, Any]:
    """The immutable writer/turn identity ``entry`` carries, or ``{}`` when it records none.

    Feed it back to ``clear_turn_marker`` / ``touch_turn_marker`` as ``identity`` (to mutate exactly
    the entry a reader just saw), or build the same shape from the ``turn_token`` that
    ``record_turn_start`` returned (to mutate the caller's own turn).
    """
    if not isinstance(entry, dict):
        return {}
    return {field: entry[field] for field in _IDENTITY_FIELDS if entry.get(field) is not None}


def _entry_matches_identity(entry: dict[str, Any] | None, identity: dict[str, Any] | None) -> bool:
    """True only when ``entry`` is provably the turn ``identity`` names.

    Every field the caller pins must match. An absent or differently-valued field on the stored entry
    is a MISMATCH, never a wildcard: a marker written by a build that recorded no writer/turn identity
    (a legacy marker) must not be claimed by a caller that has one, or "no identity" would read as
    proof of ownership and a superseded runtime could clear a newer owner's marker. An empty caller
    identity claims nothing, and so does a missing marker.
    """
    if not isinstance(entry, dict) or not identity:
        return False
    pinned = [(field, identity.get(field)) for field in _IDENTITY_FIELDS if identity.get(field) is not None]
    if not pinned:
        return False
    return all(entry.get(field) == value for field, value in pinned)


def _load_strict(path: Path) -> dict[str, dict] | None:
    """Markers keyed by session id; ``None`` when the FILE could not be read or parsed as an object.

    A missing file is ``{}`` (no marker was ever written) — only an existing-but-unreadable file is
    the fail-closed signal, which is exactly the distinction ``MarkerUnreadable`` carries.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        logger.debug("unreadable turn-marker file %s", path, exc_info=True)
        return None
    if not isinstance(data, dict):
        return None
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _load(path: Path) -> dict[str, dict]:
    """``_load_strict`` degraded to "no marker": the WRITE path repairs a broken file with a fresh one."""
    return _load_strict(path) or {}


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
        path = _marker_path(home)
        with _lock, _cross_process_lock(path):
            entries = mutate(_load(path))
            if entries is not None:
                _store(path, entries)
    except Exception:
        logger.debug("failed to %s turn marker for %s", what, session_key, exc_info=True)


def record_turn_start(home: Path | str, session_key: str, prompt: str, *, attempts: int = 0,
                      auto_continue: bool = True, notification_category: str | None = None,
                      turn_token: str | None = None) -> str:
    """Persist the marker for a turn that is about to run. ``attempts`` = how many auto-continues led to
    this run (0 for a user-initiated turn); the crash-loop breaker reads it back on the next resume.

    Returns the turn's immutable identity token, recorded as ``turn_token``. It is minted here (or
    adopted from ``turn_token``) and every later mutation of this marker — clear, retirement, heartbeat
    — is scoped to it, so a superseded writer's turn end cannot touch a newer owner's marker for the
    same session key.

    The prompt may be BLANK — an attachment-only or programmatic turn still runs, and it must still
    leave cross-process proof that "a turn is in flight" for this chat (see ``turn_in_flight``). A
    non-string prompt (a client that sent structured content) is recorded as blank text rather than
    refused. ``read_turn_marker`` keeps its non-empty-prompt contract, so a blank marker is never
    auto-continued.
    """
    if not session_key:
        return ""
    text = prompt if isinstance(prompt, str) else ""
    token = str(turn_token) if turn_token else uuid.uuid4().hex
    now = time.time()
    entry = {"attempts": max(0, int(attempts)), "prompt": text[:_MAX_PROMPT_CHARS], "started_at": now,
             "progress_at": now, "turn_token": token,
             "auto_continue": bool(auto_continue), **_writer_identity()}
    if notification_category == "diagnostic":
        entry["notification_category"] = notification_category
    # Identity only — never the prompt: this log is read on crash triage and must not carry turn content.
    logger.debug("turn marker recorded for session %s by writer pid %s", session_key, entry["writer_pid"])
    _update(home, session_key, lambda entries: {**_prune(entries, now), session_key: entry}, "record")
    return token


def clear_turn_marker(home: Path | str, session_key: str, *,
                      identity: dict[str, Any] | None = None) -> None:
    """Remove the marker once its turn concluded (any outcome the client saw).

    Scoped to ``identity`` — the immutable writer/turn identity captured when the marker was created
    (``marker_identity``, or ``{"turn_token": token}`` from ``record_turn_start``). A superseded
    runtime's turn end / Stop therefore cannot delete a NEWER owner's marker for the same stored
    session key. ``identity`` defaults to THIS process's own writer identity, so a caller that has no
    turn identity can still retire a marker it wrote itself, but never a peer's.
    """
    if not session_key:
        return
    ident = _writer_identity() if identity is None else identity

    def mutate(entries: dict[str, dict]) -> dict[str, dict] | None:
        if not _entry_matches_identity(entries.get(session_key), ident):
            return None
        return {k: v for k, v in entries.items() if k != session_key}

    _update(home, session_key, mutate, "clear")


def touch_turn_marker(home: Path | str, session_key: str, *, now: float | None = None,
                      min_interval: float = _PROGRESS_MIN_INTERVAL_SECS,
                      identity: dict[str, Any] | None = None) -> bool:
    """Record that the turn this chat already has in flight is still making progress.

    Called from the turn's own progress events (tool steps, streamed deltas), never from a timer:
    a timer keeps touching a turn that is wedged inside one call, which is exactly the case this
    signal has to detect.

    Throttled in process BEFORE any file access: a chatty turn emits dozens of events per second and
    every real touch needs the cross-process file lock, so only a genuinely due touch pays for it.
    Only an EXISTING marker OWNED BY ``identity`` is refreshed — a touch never creates or revives one,
    and a foreign/stale heartbeat cannot refresh a NEWER owner's marker for the same session key.
    ``identity`` defaults to this process's own writer identity. Best effort: every failure answers
    False.
    """
    if not session_key:
        return False
    stamp = time.time() if now is None else float(now)
    ident = _writer_identity() if identity is None else identity
    slot = (str(home), session_key)
    with _last_touch_lock:
        if stamp - _last_touch.get(slot, 0.0) < min_interval:
            return False
    wrote = False

    def mutate(entries: dict[str, dict]) -> dict[str, dict] | None:
        nonlocal wrote
        entry = entries.get(session_key)
        if not isinstance(entry, dict) or not _entry_matches_identity(entry, ident):
            return None
        if stamp - _progress_at(entry) < min_interval:
            return None
        entry["progress_at"] = stamp
        wrote = True
        return entries

    _update(home, session_key, mutate, "touch")
    if wrote:
        with _last_touch_lock:
            _last_touch[slot] = stamp
    return wrote

def turn_marker_stale(entry: dict[str, Any] | None, *, stale_after: float,
                      now: float | None = None) -> bool:
    """True when a recorded turn has shown NO progress for ``stale_after`` seconds.

    This is the only way to tell "long turn" from "hung turn": ``started_at`` alone cannot, and
    nothing else about the holder is observable from outside its process. Absent/!dict entries are
    not stale (there is no turn to judge), and the caller decides what an unreadable file means.
    """
    if not isinstance(entry, dict):
        return False
    stamp = time.time() if now is None else float(now)
    return stamp - _progress_at(entry) > float(stale_after)


def read_turn_marker_entry(home: Path | str, session_key: str) -> dict[str, Any] | None:
    """The RAW durable marker recorded for ``session_key``, prompt included even when blank, or None
    when no marker exists. Raise :class:`MarkerUnreadable` when the marker file exists but cannot be
    read or parsed, so a caller that must fail closed can tell that from "no marker at all"
    (``read_turn_marker`` deliberately cannot: it never raises)."""
    if not session_key:
        return None
    path = _marker_path(home)
    with _lock:
        entries = _load_strict(path)
    if entries is None:
        raise MarkerUnreadable(f"unreadable turn-marker file {path}")
    entry = entries.get(session_key)
    return entry if isinstance(entry, dict) else None


def _entry_writer_pid(entry: dict[str, Any]) -> int:
    try:
        return int(entry.get("writer_pid") or 0)
    except (TypeError, ValueError):
        return 0


def turn_in_flight(home: Path | str, session_key: str, *, writer_pid: int | None = None) -> bool:
    """Is a turn recorded in flight for ``session_key``? No prompt requirement — every turn that can
    run records a marker, blank prompt included (see ``record_turn_start``).

    ``writer_pid``, when given, is the process whose turn we are asking about (the lease holder):
    a marker naming it answers "alive writer -> in flight, provably dead writer -> not". A marker
    naming somebody ELSE is NOT this writer's turn and reports False here — the caller decides what
    an unattributable marker means (``session_lifecycle._holder_turn_in_flight`` refuses when the
    holder's profile can run isolated turns, whose compute-host CHILD writes the marker under its
    own pid). Raises :class:`MarkerUnreadable` for an unreadable marker file.
    """
    entry = read_turn_marker_entry(home, session_key)
    if not isinstance(entry, dict):
        return False
    if writer_pid is not None and _entry_writer_pid(entry) != writer_pid:
        return False
    return marker_writer_state(entry) != "dead"


def read_turn_marker(home: Path | str, session_key: str) -> dict[str, Any] | None:
    """The marker left by a turn that never concluded, or None.

    This is the CRASH AUTO-CONTINUE contract, not the "is a turn in flight" one: a marker with no
    prompt TEXT (an attachment-only turn, or a marker from a build that recorded no prompt) is
    nothing to re-submit, so it reads as None and is never auto-continued. Use
    ``read_turn_marker_entry``/``turn_in_flight`` to ask the liveness question. Best-effort: an
    unreadable file answers None, exactly like "no marker".
    """
    if not session_key:
        return None
    try:
        entry = read_turn_marker_entry(home, session_key)
        if not isinstance(entry, dict):
            return None
        prompt = str(entry.get("prompt") or "")
        if not prompt.strip():
            return None
        return {"attempts": max(0, int(entry.get("attempts") or 0)), "prompt": prompt, "started_at": _started_at(entry),
                "auto_continue": bool(entry.get("auto_continue", True)),
                # Writer/turn identity when present: extra keys only, so a marker written by an older
                # build still reads. The turn token lets a crash-recovery reader retire exactly the
                # marker it judged (and a newer owner's marker is left alone).
                **{k: entry[k] for k in ("turn_token", "writer_pid", "writer_start_time") if entry.get(k) is not None},
                **({"notification_category": "diagnostic"}
                   if entry.get("notification_category") == "diagnostic" else {})}
    except Exception:
        return None
