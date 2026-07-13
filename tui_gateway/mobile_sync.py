"""Revisioned conversation snapshots and bounded session-event replay.

The synchronization boundary is one lock per live session stream.  Callers
mutate snapshot-visible state, allocate the matching event sequence, and queue
that event for transport while this lock is held.  Snapshot capture takes the
conversation history lock first and this stream lock second.  A snapshot's
``watermark`` therefore covers every state transition published before it;
clients install the snapshot at that watermark and apply only events with a
larger sequence.

Replay is intentionally process-local and bounded.  A process restart changes
``server_instance_id`` and rebuilding a live session changes ``stream_id``;
either condition produces an explicit reset instead of pretending replay is
complete.
"""

from __future__ import annotations

import copy
import json
import threading
import uuid
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

SYNC_SCHEMA_MAJOR = 1
EVENT_SCHEMA_MAJOR = 1
DEFAULT_REPLAY_MAX_EVENTS = 512
DEFAULT_REPLAY_MAX_BYTES = 1024 * 1024


class SessionEventStream:
    """Thread-safe synchronization authority for one live Hermes session."""

    def __init__(
        self,
        server_instance_id: str,
        *,
        max_events: int = DEFAULT_REPLAY_MAX_EVENTS,
        max_bytes: int = DEFAULT_REPLAY_MAX_BYTES,
    ) -> None:
        self.server_instance_id = str(server_instance_id)
        self.stream_id = str(uuid.uuid4())
        self.max_events = max(1, int(max_events))
        self.max_bytes = max(1, int(max_bytes))
        self._lock = threading.RLock()
        self._sequence = 0
        self._revision = 1
        self._discarded_through = 0
        self._replay_bytes = 0
        self._replay: deque[tuple[int, int, dict[str, Any]]] = deque()
        self._active_tools: dict[str, dict[str, Any]] = {}
        self._pending_interactions: dict[str, dict[str, Any]] = {}

    def track_tool(self, descriptor: dict[str, Any]) -> None:
        tool_id = str(descriptor.get("tool_id") or "")
        if tool_id:
            self._active_tools[tool_id] = copy.deepcopy(descriptor)

    def finish_tool(self, tool_id: str) -> None:
        self._active_tools.pop(str(tool_id), None)

    def track_pending_interaction(self, descriptor: dict[str, Any]) -> None:
        interaction_id = str(
            descriptor.get("request_id") or descriptor.get("approval_id") or ""
        )
        if interaction_id:
            self._pending_interactions[interaction_id] = copy.deepcopy(descriptor)

    def finish_pending_interaction(self, request_id: str) -> None:
        """Remove state that has no legacy completion event.

        The revision changes so the next authoritative snapshot cannot compare
        equal to one that still contained the interaction.  The future
        addressable-approval slice can replace this with an explicit lifecycle
        event without changing the synchronization envelope.
        """
        with self._lock:
            if self._pending_interactions.pop(str(request_id), None) is not None:
                self._revision += 1

    def mutate(self, update: Callable[[SessionEventStream], None]) -> None:
        """Apply snapshot-only state when the legacy protocol emits no event."""
        with self._lock:
            update(self)
            self._revision += 1

    @contextmanager
    def transition(self, update: Callable[[SessionEventStream], None]):
        """Keep a state update and the caller's event publication atomic."""
        with self._lock:
            update(self)
            yield

    def publish(
        self,
        event: str,
        session_id: str,
        payload: dict[str, Any] | None,
        deliver: Callable[[dict[str, Any]], Any],
        *,
        update: Callable[[SessionEventStream], None] | None = None,
    ) -> dict[str, Any]:
        """Allocate, retain, and deliver one event in monotonic wire order."""
        with self._lock:
            retained_payload = copy.deepcopy(payload) if payload is not None else None
            if update is not None:
                update(self)
            self._sequence += 1
            self._revision += 1
            params: dict[str, Any] = {
                "type": event,
                "session_id": session_id,
                "schema_major": EVENT_SCHEMA_MAJOR,
                "stream_id": self.stream_id,
                "sequence": self._sequence,
            }
            if retained_payload is not None:
                params["payload"] = retained_payload
            frame = {"jsonrpc": "2.0", "method": "event", "params": params}
            retained = copy.deepcopy(frame)
            try:
                size = len(
                    json.dumps(
                        retained,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            except (TypeError, ValueError):
                # A malformed event cannot be replayed. Preserve the explicit
                # gap invariant even if its producer catches serialization.
                self._discard_all_through(self._sequence)
                raise
            self._retain(self._sequence, size, retained)
            # Keep delivery under the stream lock so concurrent publishers
            # cannot put sequence N+1 on the transport before sequence N.
            deliver(frame)
            return copy.deepcopy(frame)

    def synchronization(
        self,
        snapshot_factory: Callable[[dict[str, Any]], dict[str, Any]],
        cursor: object = None,
    ) -> dict[str, Any]:
        """Capture one authoritative snapshot and classify cursor recovery."""
        with self._lock:
            watermark = self._sequence
            snapshot = snapshot_factory({
                "active_tools": copy.deepcopy(list(self._active_tools.values())),
                "pending_interactions": copy.deepcopy(
                    list(self._pending_interactions.values())
                ),
            })
            snapshot.update({
                "schema_major": SYNC_SCHEMA_MAJOR,
                "server_instance_id": self.server_instance_id,
                "stream_id": self.stream_id,
                "revision": self._revision,
                "watermark": watermark,
            })
            recovery = self._recovery(cursor, watermark)
            return {
                "schema_major": SYNC_SCHEMA_MAJOR,
                "snapshot": snapshot,
                "recovery": recovery,
            }

    def cursor(self) -> dict[str, Any]:
        with self._lock:
            return self._cursor(self._sequence)

    def _retain(self, sequence: int, size: int, frame: dict[str, Any]) -> None:
        if size > self.max_bytes:
            self._discard_all_through(sequence)
            return
        self._replay.append((sequence, size, frame))
        self._replay_bytes += size
        while (
            len(self._replay) > self.max_events or self._replay_bytes > self.max_bytes
        ):
            evicted_sequence, evicted_size, _ = self._replay.popleft()
            self._replay_bytes -= evicted_size
            self._discarded_through = max(
                self._discarded_through,
                evicted_sequence,
            )

    def _discard_all_through(self, sequence: int) -> None:
        self._replay.clear()
        self._replay_bytes = 0
        self._discarded_through = max(self._discarded_through, sequence)

    def _cursor(self, sequence: int) -> dict[str, Any]:
        return {
            "server_instance_id": self.server_instance_id,
            "stream_id": self.stream_id,
            "sequence": sequence,
        }

    def _recovery(self, cursor: object, watermark: int) -> dict[str, Any]:
        current = self._cursor(watermark)
        base: dict[str, Any] = {
            "cursor": current,
            "events": [],
            "snapshot_required": True,
        }
        if not isinstance(cursor, dict):
            return {**base, "outcome": "reset", "reason": "cursor_missing"}
        if cursor.get("server_instance_id") != self.server_instance_id:
            return {
                **base,
                "outcome": "reset",
                "reason": "server_instance_changed",
            }
        if cursor.get("stream_id") != self.stream_id:
            return {**base, "outcome": "reset", "reason": "stream_changed"}
        raw_sequence = cursor.get("sequence")
        if isinstance(raw_sequence, bool) or not isinstance(raw_sequence, int):
            return {**base, "outcome": "reset", "reason": "cursor_invalid"}
        sequence = raw_sequence
        if sequence < 0 or sequence > watermark:
            return {**base, "outcome": "reset", "reason": "cursor_invalid"}
        if sequence < self._discarded_through:
            return {
                **base,
                "outcome": "gap",
                "reason": "replay_evicted",
                "available_after": self._discarded_through,
            }
        events = [
            copy.deepcopy(frame)
            for event_sequence, _size, frame in self._replay
            if sequence < event_sequence <= watermark
        ]
        return {
            **base,
            "outcome": "complete",
            "events": events,
            "snapshot_required": False,
        }
