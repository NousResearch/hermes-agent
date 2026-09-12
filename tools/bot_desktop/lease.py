"""Who may drive a profile's Bot Desktop screen: the agent (default) or exactly one human viewer.

The lease is the single truth shared by the RFB bridge (drops human input from non-holders), the
``computer_use`` tool (refuses to act while a human holds control — the person may be typing a
credential, so even screenshots are refused; fail closed rather than trusting the agent to pause
itself) and the Desktop UI (Watch / Take over / Hand back).

Per-process, keyed by ``hermes_home_key()`` so multiplexed profiles never share a lease. Handoff
requests raised by the agent (``request_handoff``) are how it asks for hands and later learns the
human is done: ``wait_for_release`` blocks the tool call until the lease returns to the agent.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from hermes_constants import hermes_home_key

AGENT = "agent"
HUMAN = "human"


class HumanHasControl(RuntimeError):
    """Raised by screen-driving tools while a human holds the lease."""


@dataclass
class Lease:
    holder: str = AGENT
    viewer_id: Optional[str] = None
    since: float = field(default_factory=time.time)
    reason: str = ""
    pending_handoff: Optional[str] = None  # agent's reason for asking, until the human takes over

    def as_dict(self) -> Dict[str, object]:
        return {"holder": self.holder, "viewer_id": self.viewer_id, "since": self.since,
                "reason": self.reason, "pending_handoff": self.pending_handoff}


_lock = threading.Condition()
_leases: Dict[str, Lease] = {}
_listeners: List[Callable[[str, Lease], None]] = []


def _key(profile_key: Optional[str]) -> str:
    return profile_key or hermes_home_key()


def get(profile_key: Optional[str] = None) -> Lease:
    with _lock:
        return _leases.setdefault(_key(profile_key), Lease())


def on_change(listener: Callable[[str, Lease], None]) -> Callable[[], None]:
    """Subscribe to lease transitions (gateway broadcasts them to Desktop clients)."""
    with _lock:
        _listeners.append(listener)

    def _off() -> None:
        with _lock:
            if listener in _listeners:
                _listeners.remove(listener)
    return _off


def _notify(key: str, lease: Lease) -> None:
    for cb in list(_listeners):
        try:
            cb(key, lease)
        except Exception:  # a broken subscriber must not wedge the handoff
            pass


def acquire(viewer_id: str, *, profile_key: Optional[str] = None, reason: str = "") -> Lease:
    """Human ``viewer_id`` takes control. Last writer wins: a second viewer evicts the first, and the
    RFB bridge closes the evicted socket so its UI drops to view-only."""
    key = _key(profile_key)
    with _lock:
        lease = _leases.setdefault(key, Lease())
        lease.holder, lease.viewer_id, lease.since, lease.reason = HUMAN, viewer_id, time.time(), reason
        lease.pending_handoff = None
        _lock.notify_all()
    _notify(key, lease)
    return lease


def release(viewer_id: Optional[str] = None, *, profile_key: Optional[str] = None) -> Lease:
    """Return control to the agent. With ``viewer_id`` only that holder may release (a stale viewer
    closing its window must not yank control from the one who took over after it)."""
    key = _key(profile_key)
    with _lock:
        lease = _leases.setdefault(key, Lease())
        if viewer_id is not None and lease.holder == HUMAN and lease.viewer_id != viewer_id:
            return lease
        lease.holder, lease.viewer_id, lease.since, lease.reason = AGENT, None, time.time(), ""
        lease.pending_handoff = None  # "hand back" answers an open request even if nobody formally took over
        _lock.notify_all()
    _notify(key, lease)
    return lease


def request_handoff(reason: str, *, profile_key: Optional[str] = None) -> Lease:
    """Agent asks a human to take over (login, 2FA, CAPTCHA, payment). Recorded so the UI can show
    why and the bridge can page the user; control itself still flips only on ``acquire``."""
    key = _key(profile_key)
    with _lock:
        lease = _leases.setdefault(key, Lease())
        lease.pending_handoff = reason
        _lock.notify_all()
    _notify(key, lease)
    return lease


def wait_for_release(*, timeout: float, profile_key: Optional[str] = None) -> bool:
    """Block until the agent holds the lease (and no handoff is pending) or ``timeout`` elapses.
    True when control is back with the agent."""
    key = _key(profile_key)
    deadline = time.monotonic() + timeout
    with _lock:
        while True:
            lease = _leases.setdefault(key, Lease())
            if lease.holder == AGENT and lease.pending_handoff is None:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _lock.wait(remaining)


def human_holds(profile_key: Optional[str] = None) -> bool:
    return get(profile_key).holder == HUMAN


def viewer_may_send_input(viewer_id: str, *, profile_key: Optional[str] = None) -> bool:
    lease = get(profile_key)
    return lease.holder == HUMAN and lease.viewer_id == viewer_id


def assert_agent_may_act(profile_key: Optional[str] = None) -> None:
    lease = get(profile_key)
    if lease.holder == HUMAN:
        raise HumanHasControl(
            "A human has taken over this desktop (they may be entering a credential). Screen actions and "
            "captures are refused until they hand control back; call computer_use action='wait_for_human' "
            "to block until then.")


def _reset_for_tests() -> None:
    with _lock:
        _leases.clear()
        _listeners.clear()
