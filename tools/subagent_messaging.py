"""In-process message bus for live subagents.

Lets one live child send a text message to another live child (or to its
parent) without going through the completion message channel. Designed for
short, opaque payloads — control signals, follow-up tasks, partial results
to merge.

Bus characteristics:

* **In-process.** No IPC. Designed for a single Python runtime holding
  multiple live subagents. Remote/backend boundaries are out of scope.
* **Per-recipient queue.** Each registered subagent gets an unbounded
  FIFO. ``peek_inbox`` is non-destructive, ``drain_inbox`` pops all.
* **Optional TTL + cap.** The bus prunes expired messages and enforces a
  per-recipient cap so a runaway producer cannot OOM the runtime.
* **Ownership-aware.** ``send_message`` and ``peek_inbox`` require the
  sender to have registered the bus endpoint first. This matches the
  delegate_tool control plane: only descendants of the same parent can
  talk to each other, never foreign spawn trees.
* **Defensive.** Every public function never raises into the agent loop;
  failures return ``(False, reason)`` so callers can log and continue.

This module does **not** replace the completion message. Subagents still
deliver a final result. The bus is for live cross-talk while they run.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default budgets. Override per-bus-instance via the constructor.
_DEFAULT_MAX_MESSAGES = 256
_DEFAULT_TTL_SECONDS = 3600  # 1 hour

# Module-level default registry. Tests instantiate their own.
_global_bus: Optional["SubagentMessageBus"] = None
_global_lock = threading.Lock()


def get_bus() -> "SubagentMessageBus":
    """Return the process-wide bus, creating it on first call."""
    global _global_bus
    if _global_bus is None:
        with _global_lock:
            if _global_bus is None:
                _global_bus = SubagentMessageBus()
    return _global_bus


def reset_bus() -> None:
    """Drop the process-wide bus. Tests only; never call from production."""
    global _global_bus
    _global_bus = None


class SubagentMessageBus:
    """Thread-safe FIFO message bus keyed by subagent_id.

    Each registered subagent gets its own deque. ``send_message`` writes
    to the recipient's queue; ``peek_inbox`` and ``drain_inbox`` read from
    the recipient's queue. Queues respect ``max_messages`` (drop oldest
    on overflow) and ``ttl_seconds`` (drop on read if expired).

    The bus is intentionally simple — a dict of deques under a lock. No
    persistence, no cross-process sync, no backpressure beyond the cap.
    """

    def __init__(
        self,
        *,
        max_messages: int = _DEFAULT_MAX_MESSAGES,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        if max_messages < 1:
            raise ValueError(f"max_messages must be >= 1, got {max_messages}")
        if ttl_seconds < 1:
            raise ValueError(f"ttl_seconds must be >= 1, got {ttl_seconds}")
        self._queues: Dict[str, Deque[Dict[str, Any]]] = {}
        self._max_messages = int(max_messages)
        self._ttl_seconds = int(ttl_seconds)
        self._lock = threading.Lock()
        self._sent_count = 0
        self._received_count = 0
        self._dropped_count = 0

    def register(self, subagent_id: str) -> bool:
        """Idempotently register a recipient. Returns True on success."""
        sid = (subagent_id or "").strip()
        if not sid:
            return False
        with self._lock:
            self._queues.setdefault(sid, deque())
        return True

    def unregister(self, subagent_id: str) -> bool:
        """Drop a recipient's queue. Returns True if it existed."""
        sid = (subagent_id or "").strip()
        if not sid:
            return False
        with self._lock:
            return self._queues.pop(sid, None) is not None

    def is_registered(self, subagent_id: str) -> bool:
        sid = (subagent_id or "").strip()
        if not sid:
            return False
        with self._lock:
            return sid in self._queues

    def send_message(
        self,
        from_sid: str,
        to_sid: str,
        message: str,
        *,
        kind: str = "note",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Enqueue a message to a registered recipient.

        Returns ``(True, "")`` on success, ``(False, reason)`` otherwise.
        Never raises. The sender must be registered too (so a typo from
        an unregistered caller is caught before queuing).
        """
        from_id = (from_sid or "").strip()
        to_id = (to_sid or "").strip()
        body = message if isinstance(message, str) else str(message or "")
        if not from_id:
            return False, "from_sid is empty"
        if not to_id:
            return False, "to_sid is empty"
        if not body:
            return False, "message is empty"
        if len(body) > 65536:
            return False, f"message too long ({len(body)} > 65536 chars)"
        with self._lock:
            if from_id not in self._queues:
                return False, f"sender '{from_id}' is not registered"
            if to_id not in self._queues:
                return False, f"recipient '{to_id}' is not registered"
            envelope = {
                "from": from_id,
                "to": to_id,
                "kind": str(kind or "note"),
                "message": body,
                "metadata": dict(metadata) if isinstance(metadata, dict) else {},
                "sent_at": time.time(),
                "expires_at": time.time() + self._ttl_seconds,
            }
            q = self._queues[to_id]
            q.append(envelope)
            self._sent_count += 1
            # Enforce cap: drop oldest first.
            while len(q) > self._max_messages:
                q.popleft()
                self._dropped_count += 1
        return True, ""

    def peek_inbox(
        self,
        subagent_id: str,
        *,
        limit: int = 50,
        drop_expired: bool = True,
    ) -> List[Dict[str, Any]]:
        """Non-destructive view of the inbox. Expired messages may be pruned.

        Always returns a list (possibly empty). Never raises.
        """
        sid = (subagent_id or "").strip()
        if not sid:
            return []
        cap = max(1, min(int(limit), 1000))
        with self._lock:
            q = self._queues.get(sid)
            if q is None:
                return []
            now = time.time()
            if drop_expired:
                self._prune_locked(q, now)
            snapshot = list(q)[-cap:]
            self._received_count += len(snapshot)
        return snapshot

    def drain_inbox(
        self,
        subagent_id: str,
        *,
        drop_expired: bool = True,
    ) -> List[Dict[str, Any]]:
        """Pop all messages and return them. Empty list if none."""
        sid = (subagent_id or "").strip()
        if not sid:
            return []
        with self._lock:
            q = self._queues.get(sid)
            if q is None:
                return []
            now = time.time()
            if drop_expired:
                self._prune_locked(q, now)
            out = list(q)
            q.clear()
            self._received_count += len(out)
        return out

    def stats(self) -> Dict[str, int]:
        """Return counters (no PII; safe to log)."""
        with self._lock:
            return {
                "recipients": len(self._queues),
                "sent": self._sent_count,
                "received": self._received_count,
                "dropped": self._dropped_count,
                "max_messages_per_recipient": self._max_messages,
                "ttl_seconds": self._ttl_seconds,
            }

    def _prune_locked(self, q: Deque[Dict[str, Any]], now: float) -> None:
        while q and q[0].get("expires_at", 0) <= now:
            q.popleft()
            self._dropped_count += 1


# --- Module-level convenience wrappers ------------------------------------

def send_message(
    from_sid: str,
    to_sid: str,
    message: str,
    *,
    kind: str = "note",
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Send via the global bus. See :class:`SubagentMessageBus.send_message`."""
    return get_bus().send_message(from_sid, to_sid, message, kind=kind, metadata=metadata)


def peek_inbox(subagent_id: str, *, limit: int = 50) -> List[Dict[str, Any]]:
    """Peek the global bus. See :class:`SubagentMessageBus.peek_inbox`."""
    return get_bus().peek_inbox(subagent_id, limit=limit)


def drain_inbox(subagent_id: str) -> List[Dict[str, Any]]:
    """Drain the global bus. See :class:`SubagentMessageBus.drain_inbox`."""
    return get_bus().drain_inbox(subagent_id)


def register(subagent_id: str) -> bool:
    """Register a recipient on the global bus."""
    return get_bus().register(subagent_id)


def unregister(subagent_id: str) -> bool:
    """Drop a recipient from the global bus."""
    return get_bus().unregister(subagent_id)


def stats() -> Dict[str, int]:
    """Return global bus stats."""
    return get_bus().stats()
