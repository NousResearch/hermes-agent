"""Gateway busy-turn choice registry.

Discord and other interactive gateways use this tiny process-local registry to
turn a button click into one of the existing busy-turn actions: queue, steer,
interrupt, or cancel.  It deliberately stores only pending choices for the
current gateway process; restart expiry is fine because the current running turn
is also process-local.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import itertools
import time
from typing import Any, Awaitable, Callable, Dict, Optional

Handler = Callable[[str, Any], Awaitable[str] | str]


@dataclass
class BusyTurnChoiceEntry:
    session_key: str
    event: Any
    handler: Handler
    created_at: float
    resolved: bool = False


_entries: Dict[str, BusyTurnChoiceEntry] = {}
_counter = itertools.count(1)

_VALID_CHOICES = {"queue", "steer", "interrupt", "cancel"}
_ENTRY_TTL_SECS = 5 * 60


def _prune(now: Optional[float] = None) -> None:
    """Drop expired entries so timed-out Discord prompts don't leak events."""
    current = time.time() if now is None else now
    expired = [
        choice_id
        for choice_id, entry in _entries.items()
        if entry.resolved or current - entry.created_at > _ENTRY_TTL_SECS
    ]
    for choice_id in expired:
        _entries.pop(choice_id, None)


def register(session_key: str, event: Any, handler: Handler) -> str:
    """Register a pending busy-turn prompt and return a choice id."""
    now = time.time()
    _prune(now)
    choice_id = f"busy-{int(now * 1000)}-{next(_counter)}"
    _entries[choice_id] = BusyTurnChoiceEntry(
        session_key=session_key,
        event=event,
        handler=handler,
        created_at=now,
    )
    return choice_id

async def resolve(choice_id: str, choice: str) -> Optional[str]:
    """Resolve *choice_id* with one of queue/steer/interrupt/cancel."""
    normalized = str(choice or "").strip().lower()
    if normalized not in _VALID_CHOICES:
        raise ValueError(f"Unknown busy-turn choice: {choice!r}")
    _prune()
    entry = _entries.pop(choice_id, None)
    if entry is None or entry.resolved:
        return None
    entry.resolved = True
    result = entry.handler(normalized, entry.event)
    if inspect.isawaitable(result):
        result = await result
    return str(result or "")


def discard(choice_id: str) -> None:
    """Discard a pending choice without applying an action."""
    _entries.pop(choice_id, None)


def pending_count() -> int:
    _prune()
    return len(_entries)
