"""Prompt-cache warm-up gate for batch subagents (``delegation.prefix_stagger_seconds``).

Siblings in one batch send the same tools[] and system prompt; only their first user turn (context, then
goal) differs. Launched together,
every child's first request misses the provider cache and pays the full prefix. A provider cache entry is
readable once the writing response starts streaming, so the gate lets the batch's first child (the leader)
run at once and holds the siblings that share its route until the leader's first response has started, it
has finished, or the cap elapses, whichever comes first. The wait is gone the moment the cache can be read,
so the cost of opting in is one time-to-first-token, not a fixed per-child sleep.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

# Set by the streaming paths on the first chunk of a response (chat completions and Responses).
_FIRST_CHUNK_ACTIVITY = "receiving stream response"
_POLL_S = 0.05


def _route_key(child: Any) -> tuple:
    return tuple(getattr(child, attr, None) for attr in ("provider", "base_url", "model", "api_mode"))


def _caches_prompts(child: Any) -> bool:
    """Only routes Hermes marks for prompt caching benefit; for the rest the wait would be pure latency."""
    return getattr(child, "_use_prompt_caching", False) is True


def _first_response_started(child: Any) -> bool:
    return (getattr(child, "session_api_calls", 0) or 0) >= 1 or (
        getattr(child, "_last_activity_desc", None) == _FIRST_CHUNK_ACTIVITY)


def _stop_requested(child: Any) -> bool:
    return getattr(child, "_interrupt_requested", False) is True


class PrefixCacheGate:
    """Holds a batch's cache-sharing siblings until the leader's first response has started."""

    def __init__(self, leader: Any, max_wait_s: float) -> None:
        self.leader, self.max_wait_s = leader, max_wait_s
        self._leader_route = _route_key(leader)
        self._leader_done = threading.Event()

    @classmethod
    def for_children(cls, children: list, max_wait_s: float) -> Optional[PrefixCacheGate]:
        """A gate for ``children`` (``(index, task, child)`` tuples), or None when there is nothing to share."""
        # The leader is always task 0 (it is submitted first, so it always gets a worker): an uncached task 0
        # leaves the batch ungated even when later tasks share a cached route with each other.
        if max_wait_s <= 0 or len(children) < 2 or not _caches_prompts(children[0][2]):
            return None
        return cls(children[0][2], max_wait_s)

    def follows(self, child: Any) -> bool:
        return child is not self.leader and _caches_prompts(child) and _route_key(child) == self._leader_route

    def run(self, child: Any, run: Callable[[], Any], *, parent: Any = None) -> Optional[Any]:
        """Run ``run`` for ``child`` after the gate opens. Returns None, without running, when the child or
        ``parent`` was stopped during the wait (the caller reports it interrupted)."""
        if child is self.leader:
            try:
                return run()
            finally:
                self._leader_done.set()
        if self.follows(child) and not self._wait(child, parent):
            return None
        return run()

    def _wait(self, child: Any, parent: Any) -> bool:
        deadline = time.monotonic() + self.max_wait_s
        while not (self._leader_done.is_set() or _first_response_started(self.leader)):
            if _stop_requested(child) or (parent is not None and _stop_requested(parent)):
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._leader_done.wait(min(_POLL_S, remaining))
        return True
