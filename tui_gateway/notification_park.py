"""Bounded holding area for addressed TUI notifications whose owner session is not live.

Every TUI/Desktop/dashboard session's poller drains ONE process-wide completion queue. A session can end
(explicit close, a reap, a backend-side teardown) while a ``notify_on_complete`` process or an async delegation
it started is still running, and ``resume`` brings the very same ``session_key`` back. Whichever poller drained
such an event first used to drop it (#118653, #129071). It is parked here instead, for a bounded window, and the
owner's own poller claims it on its first pass after it is live again, through the same ownership gate delivery
uses (the caller's ``owns`` predicate), so no foreign session ever adopts it.

Parking keeps the event OUT of ``completion_queue``: a foreign event left in the queue keeps every live poller's
``take_ready`` returning at once and spins those loops. The caps are total AND per owner, so a session that is gone
for good cannot spend the whole budget.

Progress notices (``watch_*``, ``heartbeat``) yield to results: a process completion or async delegation that finds
its owner's slots (or the total) full displaces the oldest such notice, same owner first, so a chatty
``watch_patterns`` process cannot crowd out the completion the user is waiting for.

A startup replay (``restored``) delegation is not parked: its row's owner process is dead, so the orphan sweep
re-offers it until its owner is back. A live delegation's parked copy is its only in-memory one, so the park holds
it in the ledger's held-copy registry (``async_delegation.mark_copy_held``) for as long as it owns it: the orphan
sweep, which cannot see this list, then neither warns that the copy was lost nor queues a duplicate. The copy is
marked in ``park`` (inside the routing step, before the poller's ``release`` of its dequeue hold) and unmarked
exactly once when it leaves: after the claiming poller has routed it (``unhold``), or when the park lets go of it
(TTL, ``clear``). A delegation the park lets go of is a lost result, logged at WARNING as an unparked drop always
was, with its offer cleared (``return_completion_offer``) so the sweep re-offers the still-pending row.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

RETENTION_SECONDS = 600.0
MAX_PARKED = 16
MAX_PARKED_PER_KEY = 4
_parked: list[tuple[float, dict]] = []
_lock = threading.Lock()


def _parkable(evt: dict) -> bool:
    return not (evt.get("type") == "async_delegation" and evt.get("restored"))


def _expire_locked(now: float) -> list:
    """Drop expired entries (caller holds ``_lock``); returns them for ``_release`` outside the lock."""
    expired = [evt for deadline, evt in _parked if deadline <= now]
    if expired:
        _parked[:] = [(deadline, evt) for deadline, evt in _parked if deadline > now]
    return expired


def _is_progress(evt: dict) -> bool:
    evt_type = str(evt.get("type") or "")
    return evt_type == "heartbeat" or evt_type.startswith("watch_")


def _release(events) -> None:
    """Events the park let go of. A delegation is a lost result (WARNING, as an unparked drop logs it) whose offer
    is cleared and whose hold ends, so the orphan sweep re-offers the still-pending row; see the module docstring."""
    if not events:
        return
    from tools.async_delegation import return_completion_offer
    for evt in events:
        is_delegation = evt.get("type") == "async_delegation"
        if is_delegation:
            return_completion_offer(evt)
        else:
            _settle_dropped(evt)
        unhold([evt])
        (logger.warning if is_delegation else logger.debug)(
            "Dropping parked %s notification past its retention window (key=%r)",
            evt.get("type", "completion"), str(evt.get("session_key") or ""))


def _settle_dropped(evt: dict) -> None:
    """A process notice the park lets go of is a deliberate drop: settle it so its process stops reading as
    awaiting a notification (``ProcessRegistry.settle_notification`` ignores every other event type)."""
    from tools.process_registry import process_registry
    process_registry.settle_notification(evt)


def _key(evt: dict) -> str:
    return str(evt.get("session_key") or "")


def _admit_locked(evt: dict, key: str) -> tuple[bool, list]:
    """Room for ``evt`` (caller holds ``_lock``): a result displaces the oldest progress notice when full, same
    owner first (only a same-owner victim frees a per-owner slot). Returns (admitted, displaced)."""
    key_full = bool(key) and sum(1 for _d, held in _parked if _key(held) == key) >= MAX_PARKED_PER_KEY
    if not key_full and len(_parked) < MAX_PARKED:
        return True, []
    if _is_progress(evt):
        return False, []
    progress = [i for i, (_d, held) in enumerate(_parked) if _is_progress(held)]
    victims = [i for i in progress if _key(_parked[i][1]) == key] or ([] if key_full else progress)
    if not victims:
        return False, []
    return True, [_parked.pop(victims[0])[1]]


def park(evt: dict) -> bool:
    """Retain an addressed event whose owner session is not live. False when the park cannot take it; the caller
    then disposes of the event exactly as it did before the park existed."""
    if not _parkable(evt):
        return False
    from tools.async_delegation import completion_copy_id, mark_copy_held
    now = time.monotonic()
    with _lock:
        expired = _expire_locked(now)
        admitted, displaced = _admit_locked(evt, _key(evt))
        if admitted:
            mark_copy_held(completion_copy_id(evt))
            _parked.append((now + RETENTION_SECONDS, evt))
    _release(expired)
    for held in displaced:
        logger.debug("Dropping parked %s notification to make room for a %s (key=%r)",
                     held.get("type"), evt.get("type", "completion"), _key(held))
        _settle_dropped(held)
    return admitted


def claim(owns: Callable[[dict], bool]) -> list:
    """Parked events ``owns`` accepts, in arrival order. ``owns`` may read the lineage store, so it runs on a
    snapshot outside the lock; an event is handed out only if it is still parked, so exactly one poller takes it.
    Each claimed delegation's held-copy mark passes to the caller, who MUST pass the result to ``unhold`` once
    routing returns (a re-park in between takes its own mark)."""
    if not _parked:
        return []
    with _lock:
        expired = _expire_locked(time.monotonic())
        snapshot = [evt for _deadline, evt in _parked]
    _release(expired)
    mine = {id(evt) for evt in snapshot if owns(evt)}
    if not mine:
        return []
    with _lock:
        claimed = [evt for _deadline, evt in _parked if id(evt) in mine]
        _parked[:] = [(deadline, evt) for deadline, evt in _parked if id(evt) not in mine]
    return claimed


def unhold(events) -> None:
    """End the held-copy mark ``park`` took for each of ``events`` (a no-op for anything but a result copy)."""
    if events:
        from tools.async_delegation import completion_copy_id, unmark_copy_held
        for evt in events:
            unmark_copy_held(completion_copy_id(evt))


def parked() -> list:
    with _lock:
        return [evt for _deadline, evt in _parked]


def clear() -> None:
    with _lock:
        dropped = [evt for _deadline, evt in _parked]
        _parked.clear()
    unhold(dropped)
