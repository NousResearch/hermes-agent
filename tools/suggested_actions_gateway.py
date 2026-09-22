"""Gateway-side suggested-actions primitive (non-blocking action registry).

The ``suggest_actions`` tool attaches tappable actions to a message and returns
immediately — the agent does not block. When the user taps an action later, the
adapter needs to recover the *payload* for that action and inject it as a fresh
user turn.

Telegram caps ``callback_data`` at 64 bytes, so we cannot round-trip an
arbitrary payload inside the button. Instead each action set is registered here
under a short ``action_set_id``; buttons carry ``sa:<set_id>:<index>`` and the
tap handler looks the payload back up.

State is module-level (same shape as ``tools.clarify_gateway`` and
``tools.approval``) so platform adapters can resolve a tapped action without a
back-reference to the ``GatewayRunner``.

Unlike clarify there is **no blocking Event** and **no timeout thread**: the
agent already finished its turn. Entries live until (a) they are resolved by a
tap, (b) their session is cleared, or (c) the per-session cap evicts the oldest
set. This bounds memory without needing a reaper.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Keep at most this many action sets per session; oldest evicted first. A user
# rarely taps actions from an ancient message, and unbounded growth would leak.
_MAX_SETS_PER_SESSION = 20


@dataclass
class _ActionSet:
    """One registered set of suggested actions for a message."""
    set_id: str
    session_key: str
    message: str
    actions: List[dict]  # [{"label": str, "payload": str}, ...]
    created_at: float = field(default_factory=time.monotonic)

    def payload_for(self, index: int) -> Optional[str]:
        if 0 <= index < len(self.actions):
            return self.actions[index].get("payload")
        return None

    def label_for(self, index: int) -> Optional[str]:
        if 0 <= index < len(self.actions):
            return self.actions[index].get("label")
        return None


_lock = threading.RLock()
# set_id → _ActionSet  (primary lookup for button callbacks)
_sets: Dict[str, _ActionSet] = {}
# session_key → list[set_id]  (FIFO; for eviction + session cleanup)
_session_index: Dict[str, List[str]] = {}


def register(
    set_id: str,
    session_key: str,
    message: str,
    actions: List[dict],
) -> _ActionSet:
    """Register an action set so a later tap can recover its payload.

    Enforces the per-session cap by evicting the oldest set when the session
    exceeds :data:`_MAX_SETS_PER_SESSION`.
    """
    entry = _ActionSet(
        set_id=set_id,
        session_key=session_key,
        message=message,
        actions=list(actions or []),
    )
    with _lock:
        _sets[set_id] = entry
        ids = _session_index.setdefault(session_key, [])
        ids.append(set_id)
        # Evict oldest sets beyond the cap.
        while len(ids) > _MAX_SETS_PER_SESSION:
            oldest = ids.pop(0)
            _sets.pop(oldest, None)
    return entry


def resolve(set_id: str, index: int) -> Optional[str]:
    """Return the payload for a tapped action, or ``None`` if unknown.

    The set is intentionally NOT removed on resolve: a message may offer
    several actions and the user could tap more than one. Sets are reclaimed by
    session cleanup or the per-session eviction cap.
    """
    with _lock:
        entry = _sets.get(set_id)
    if entry is None:
        return None
    return entry.payload_for(index)


def get_set(set_id: str) -> Optional[_ActionSet]:
    with _lock:
        return _sets.get(set_id)


def clear_session(session_key: str) -> int:
    """Drop every registered action set for a session. Returns count cleared.

    Called on session-boundary cleanup (``/new``, gateway shutdown, cached
    agent eviction) so stale sets don't accumulate.
    """
    with _lock:
        ids = list(_session_index.pop(session_key, []) or [])
        for sid in ids:
            _sets.pop(sid, None)
    return len(ids)


def has_pending(session_key: str) -> bool:
    with _lock:
        ids = _session_index.get(session_key) or []
        return any(_sets.get(sid) is not None for sid in ids)
