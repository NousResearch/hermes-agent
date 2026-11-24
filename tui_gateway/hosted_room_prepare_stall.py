"""Bound the room driver's retries of a room preparation that keeps failing, and say so in the room.

A Send returns the stored message even when preparing the room afterwards fails (#143239), and the
driver retries the preparation on its next passes. A failure that never clears (a member whose
profile was deleted, a full room log) would otherwise retry forever with the error only in the
gateway log: the user sees "sent" and no member ever answers. After ``STALL_ATTEMPTS`` consecutive
failures the room gets ONE ``member.unavailable`` event (Desktop and the Webapp render it), the
room's status offers Retry, and the driver stops preparing that room until the next Send or Retry.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms

logger = logging.getLogger(__name__)

STALL_ATTEMPTS = 3
RETRY_TASK_PREFIX = "prepare:"
_EVENT_PREFIX = "prepare-stalled:"
_MAX_ERROR_CHARS = 300
# Contention or an ownership move, not a broken room: never counted toward the stall.
_TRANSIENT = (
    sqlite3.OperationalError, hosted_rooms.AuthorityConflictError, hosted_rooms.EventCursorConflictError,
    hosted_rooms.RoomNotFoundError, driver.LeaseHeldError, driver.StaleLeaseError, driver.RoomUnavailableError)


@dataclass
class _Streak:
    failures: int = 0
    error: str | None = None
    stalled: bool = False


def _describe(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text if len(text) <= _MAX_ERROR_CHARS else text[:_MAX_ERROR_CHARS - 1] + "…"


class PrepareStallTracker:
    """Per-room consecutive preparation failures; thread-safe (driver room threads and RPC threads)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rooms: dict[str, _Streak] = {}

    def stalled(self, room_id: str) -> bool:
        with self._lock:
            streak = self._rooms.get(room_id)
            return bool(streak and streak.stalled)

    def last_error(self, room_id: str) -> str | None:
        with self._lock:
            streak = self._rooms.get(room_id)
            return streak.error if streak else None

    def clear(self, room_id: str) -> None:
        with self._lock:
            self._rooms.pop(room_id, None)

    def failed(self, room_id: str, exc: BaseException) -> bool:
        """Record one failure; True exactly once, when this failure stalls the room."""
        with self._lock:
            streak = self._rooms.setdefault(room_id, _Streak())
            streak.error = _describe(exc)
            if isinstance(exc, _TRANSIENT) or streak.stalled:
                return False
            streak.failures += 1
            streak.stalled = streak.failures >= STALL_ATTEMPTS
            return streak.stalled

    def stall(self, room_id: str, exc: BaseException) -> None:
        """A failed explicit Retry: stalled again at once, without a second room event."""
        with self._lock:
            self._rooms[room_id] = _Streak(failures=STALL_ATTEMPTS, error=_describe(exc), stalled=True)



def retry_action(room_id: str, room: Mapping[str, Any]) -> dict[str, Any]:
    """A stalled room's Retry row in ``groups.state``; ``member_id`` names who was not woken."""
    names = [str(m.get("handle") or m.get("member_id") or "") for m in room.get("members") or ()]
    return {"kind": "retry", "member_id": ", ".join(n for n in names if n) or room_id,
            "task_id": f"{RETRY_TASK_PREFIX}{room_id}", "execution_generation": 1}


def report_stall(db_path: Any, room_id: str, exc: BaseException) -> None:
    """Append the room's one stall notice; a stall already reported last in the log is not repeated."""
    try:
        room = hosted_rooms.room_state(db_path, room_id=room_id)
        latest_seq = int(room["latest_seq"])
        tail = hosted_rooms.read_events(
            db_path, room_id=room_id, since_seq=max(0, latest_seq - 1), limit=1)["events"]
        if tail and str(tail[-1]["event_id"]).startswith(_EVENT_PREFIX):
            return
        gateway_id, epoch = str(room["authority_gateway_id"]), int(room["authority_epoch"])
        hosted_rooms.append_event(
            db_path, room_id=room_id, event_id=f"{_EVENT_PREFIX}{latest_seq}", kind="member.unavailable",
            actor={"kind": "gateway", "id": gateway_id},
            payload={"error": f"Members could not be woken: {_describe(exc)}", "reason_code": "prepare_failed"},
            authority_gateway_id=gateway_id, authority_epoch=epoch, control_reserve=True)
    except Exception:
        # The Retry row in the room's status still shows the stall.
        logger.warning("hosted room %s: recording the stalled preparation failed", room_id, exc_info=True)
