"""Read-only, source-side observation plane for Hermes sessions.

This module deliberately does not import the controller dispatcher.  It is a
small second sink for gateway events: projection happens before an event is
stored or delivered, and observer failures are isolated from the controller
transport.
"""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

SCHEMA_SESSIONS = "iris.observe.sessions.v1"
SCHEMA_SNAPSHOT = "iris.observe.snapshot.v1"
SCHEMA_EVENT = "iris.observe.event.v1"
MAX_EVENTS = 10_000
MAX_AGE_SECONDS = 15 * 60
MAX_TEXT = 4_000
MAX_JOURNALS = 128
MAX_SUBSCRIBERS_PER_SESSION = 8

_ALLOWED_KINDS = frozenset({
    "transcript.row.upsert", "message.start", "message.delta", "message.complete",
    "tool.phase", "status.current", "waiting_for_primary_client", "session.metadata",
    "session.state", "session.ended", "lineage.changed", "gap", "cursor.advance",
    "error.safe", "unknown_activity",
})
_HIDDEN_KINDS = frozenset({
    "approval.request", "clarify.request", "sudo.request", "secret.request",
    "terminal.read.request", "preview.read.request", "window.read.request",
    "mcp.setup.request", "tool.output", "reasoning.available", "moa.reference",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any) -> str:
    text = str(value or "")
    return text[:MAX_TEXT]


def logical_session_id(internal_id: str) -> str:
    """Return a stable opaque ID without exposing the durable/runtime ID."""
    digest = hashlib.sha256(("hermes-observer-v1:" + str(internal_id)).encode()).hexdigest()
    return "ls_" + digest[:32]


def _safe_payload(kind: str, payload: Any) -> dict:
    """Allowlist fields instead of recursively copying an untrusted envelope."""
    if not isinstance(payload, dict):
        return {}
    if kind in {"message.start", "message.delta", "message.complete"}:
        allowed = {"text", "status", "transcript_item_id", "correlation_id"}
    elif kind == "transcript.row.upsert":
        allowed = {"durable_item_id", "role", "text", "created_at", "display_kind", "correlation_id"}
    elif kind == "tool.phase":
        allowed = {"activity_id", "name", "phase", "duration_ms"}
    elif kind == "status.current":
        allowed = {"text", "kind"}
    elif kind in {"session.metadata", "session.state"}:
        allowed = {"title", "profile", "model", "state", "confirmed_at"}
    elif kind == "session.ended":
        allowed = {"reason", "ended_at"}
    elif kind == "gap":
        allowed = {"reason", "requested_stream_id", "requested_after_seq", "retention_floor_seq", "rehydrate_required"}
    elif kind == "lineage.changed":
        allowed = {"lineage_generation", "reason"}
    elif kind == "error.safe":
        allowed = {"message", "code"}
    else:
        allowed = set()
    result: dict[str, Any] = {}
    for key in allowed:
        if key not in payload:
            continue
        value = payload[key]
        if key in {"text", "name", "message", "reason", "title", "profile", "model", "state", "kind", "code", "phase", "role", "display_kind", "correlation_id", "activity_id", "durable_item_id", "transcript_item_id", "requested_stream_id", "ended_at", "created_at", "confirmed_at"}:
            result[key] = _safe_text(value)
        elif key in {"duration_ms", "lineage_generation", "requested_after_seq", "retention_floor_seq"}:
            try:
                result[key] = int(value)
            except (TypeError, ValueError):
                continue
        elif key == "rehydrate_required":
            result[key] = bool(value)
    return result


@dataclass(frozen=True)
class ObserverRecord:
    logical_session_id: str
    stream_id: str
    seq: int
    event_id: str
    captured_at: str
    event_class: str
    kind: str
    visibility: str
    payload: dict

    def as_dict(self) -> dict:
        return {"schema": SCHEMA_EVENT, "logical_session_id": self.logical_session_id,
                "stream_id": self.stream_id, "seq": self.seq, "event_id": self.event_id,
                "captured_at": self.captured_at, "class": self.event_class, "kind": self.kind,
                "visibility": self.visibility, "payload": self.payload}


class ObserverJournal:
    """Bounded per-session journal. Records are projected before insertion."""

    def __init__(self, *, max_events: int = MAX_EVENTS, max_age_seconds: float = MAX_AGE_SECONDS) -> None:
        self.stream_id = "st_" + secrets.token_urlsafe(18)
        self.started_at = _now()
        self._max_events = max(1, int(max_events))
        self._max_age = max(1.0, float(max_age_seconds))
        self._records: deque[tuple[float, ObserverRecord]] = deque()
        self._seq = 0
        self._lock = threading.RLock()

    @property
    def head(self) -> int:
        with self._lock:
            return self._seq

    @property
    def floor(self) -> int:
        with self._lock:
            self._prune_locked()
            return self._records[0][1].seq if self._records else self._seq + 1

    def _prune_locked(self) -> None:
        cutoff = time.monotonic() - self._max_age
        while self._records and self._records[0][0] < cutoff:
            self._records.popleft()

    def append(self, record_factory: Callable[[str, int], ObserverRecord]) -> ObserverRecord:
        with self._lock:
            self._seq += 1
            record = record_factory(self.stream_id, self._seq)
            now = time.monotonic()
            self._records.append((now, record))
            self._prune_locked()
            while len(self._records) > self._max_events:
                self._records.popleft()
            return record

    def replay(self, after_seq: int) -> tuple[list[ObserverRecord], str | None]:
        with self._lock:
            self._prune_locked()
            if after_seq < self.floor - 1:
                return [], "retention_exceeded"
            return [r for _, r in self._records if r.seq > after_seq], None


class ObserverPlane:
    """Fan-out capture, safe snapshots, replay, and observer-only state."""

    def __init__(self, session_reader: Callable[[], Iterable[dict]] | None = None,
                 transcript_reader: Callable[[str], Iterable[dict]] | None = None) -> None:
        self._session_reader = session_reader or (lambda: ())
        self._transcript_reader = transcript_reader or (lambda _sid: ())
        self._journals: dict[str, ObserverJournal] = {}
        self._observers: dict[str, list[queue.Queue]] = {}
        self._lock = threading.RLock()

    def _journal(self, logical_id: str) -> ObserverJournal:
        with self._lock:
            if logical_id not in self._journals and len(self._journals) >= MAX_JOURNALS:
                self._journals.pop(next(iter(self._journals)))
            return self._journals.setdefault(logical_id, ObserverJournal())

    def rotate_stream(self, logical_id: str) -> ObserverJournal:
        """Start a new epoch after restart, destructive rewind, or lineage change."""
        with self._lock:
            old_observers = self._observers.pop(logical_id, [])
            journal = ObserverJournal()
            self._journals[logical_id] = journal
            gap = ObserverRecord(logical_id, journal.stream_id, 1, f"{journal.stream_id}:1", _now(), "control", "gap", "display", {"reason": "stream_restarted", "rehydrate_required": True})
            for target in old_observers:
                try:
                    target.put_nowait(gap)
                except Exception:
                    pass
            return journal

    def list_sessions(self, limit: int = 20) -> dict:
        rows = []
        for source in list(self._session_reader())[:max(0, min(int(limit), 20))]:
            if not isinstance(source, dict):
                continue
            internal = str(source.get("id") or source.get("session_id") or "")
            if not internal:
                continue
            logical = logical_session_id(internal)
            row = {"logical_session_id": logical, "title": _safe_text(source.get("title")),
                   "preview": _safe_text(source.get("preview")),
                   "profile": _safe_text(source.get("profile") or "default"),
                   "model": _safe_text(source.get("model")),
                   "last_activity": _safe_text(source.get("last_activity") or source.get("last_active") or _now()),
                   "state": (source.get("state") if source.get("state") in {"starting", "working", "waiting", "idle", "ended", "not_live", "unknown"} else ("working" if source.get("running") else "not_live")),
                   "history_state": "available"}
            if source.get("state_confirmed_at"):
                row["state_confirmed_at"] = source["state_confirmed_at"]
            rows.append(row)
        return {"schema": SCHEMA_SESSIONS, "snapshot_id": "sp_" + secrets.token_urlsafe(12),
                "next_cursor": None, "sessions": rows}

    def snapshot(self, logical_id: str) -> dict | None:
        source_id = None
        source = None
        for candidate in self._session_reader():
            if not isinstance(candidate, dict):
                continue
            internal = str(candidate.get("id") or candidate.get("session_id") or "")
            if logical_session_id(internal) == logical_id:
                source_id, source = internal, candidate
                break
        if source_id is None:
            return None
        assert isinstance(source, dict)
        journal = self._journal(logical_id)
        # Capturing head before reading durable rows is the snapshot barrier.
        through = journal.head
        rows = []
        for raw in self._transcript_reader(source_id):
            if not isinstance(raw, dict):
                continue
            projected = _safe_payload("transcript.row.upsert", raw)
            if projected.get("role") in {"user", "assistant"} and projected.get("text") is not None:
                rows.append(projected)
        try:
            generation = max(1, int(source.get("lineage_generation") or 1))
        except (TypeError, ValueError):
            generation = 1
        return {"schema": SCHEMA_SNAPSHOT, "logical_session_id": logical_id,
                "lineage_generation": generation,
                "session": _safe_payload("session.metadata", source),
                "transcript": rows, "stream": {"stream_id": journal.stream_id,
                "through_seq": through, "retention_floor_seq": journal.floor,
                "journal_started_at": journal.started_at},
                "completeness": {"durable_transcript": "complete",
                "transient_events": "complete_from_cursor", "reason": None}}

    def capture(self, event: str, internal_session_id: str, payload: dict | None = None) -> ObserverRecord | None:
        if not internal_session_id:
            return None
        logical = logical_session_id(internal_session_id)
        hidden = event in _HIDDEN_KINDS
        known = event in _ALLOWED_KINDS
        kind = event if known else ("cursor.advance" if hidden else "unknown_activity")
        projected = _safe_payload(kind, payload)
        visibility = "display" if known and not hidden else "hidden"
        event_class = "control" if kind in {"cursor.advance", "unknown_activity", "gap", "lineage.changed"} else ("activity" if kind in {"tool.phase", "status.current"} else "transcript")
        # Serialize append, observer registration, and epoch rotation. This
        # makes journal-before-fanout and replay-to-live a single boundary.
        with self._lock:
            if event == "lineage.changed" and logical in self._journals:
                self.rotate_stream(logical)
            journal = self._journal(logical)
            with journal._lock:
                record = journal.append(lambda stream, seq: ObserverRecord(logical, stream, seq, f"{stream}:{seq}", _now(), event_class, kind, visibility, projected))
                targets = list(self._observers.get(logical, ()))
                for target in targets:
                    try:
                        target.put_nowait(record)
                    except Exception:
                        # Slow observers are isolated and may miss replay; controller is unaffected.
                        pass
        return record

    def subscribe(self, logical_id: str, after_seq: int = 0, max_queue: int = 256, stream_id: str | None = None) -> tuple[queue.Queue, list[ObserverRecord], str | None]:
        q: queue.Queue = queue.Queue(maxsize=max(1, int(max_queue)))
        # Hold the journal lock through replay registration.  Capture appends
        # under the same lock, so an event cannot land between the replay head
        # and observer registration (the hydrate/subscribe race).
        with self._lock:
            journal = self._journals.get(logical_id)
            if journal is None:
                known = any(logical_session_id(str(row.get("id") or row.get("session_id") or "")) == logical_id
                            for row in self._session_reader() if isinstance(row, dict))
                if not known:
                    return q, [], "source_unavailable"
                journal = self._journal(logical_id)
            with journal._lock:
                replay, gap = journal.replay(int(after_seq))
                if stream_id and stream_id != journal.stream_id:
                    gap = "stream_restarted"
                    replay = []
                if gap is None:
                    if len(self._observers.get(logical_id, ())) >= MAX_SUBSCRIBERS_PER_SESSION:
                        return q, [], "source_unavailable"
                    self._observers.setdefault(logical_id, []).append(q)
        return q, replay, gap

    def unsubscribe(self, logical_id: str, q: queue.Queue) -> None:
        with self._lock:
            observers = self._observers.get(logical_id, [])
            if q in observers:
                observers.remove(q)
            if not observers:
                self._observers.pop(logical_id, None)


# Shared observer state is defined in this controller-independent module so
# importing the observer router never initializes the controller dispatcher.
observer_plane = ObserverPlane()


__all__ = ["ObserverPlane", "ObserverJournal", "ObserverRecord", "logical_session_id", "SCHEMA_SESSIONS", "SCHEMA_SNAPSHOT", "SCHEMA_EVENT", "observer_plane"]
