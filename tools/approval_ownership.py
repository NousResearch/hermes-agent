"""Execution-scoped lifetimes for messaging-gateway approval delivery.

A detached worker retains its own owner before submission; copying the parent's
ContextVar alone would retain a route that closes when the parent returns. Legacy
TUI/API registrations still use session-wide unregister, not detached delivery.
Queue admission and owner closure share approval._lock; transport calls never do.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable


@dataclass(eq=False)
class GatewayApprovalOwner:
    session_key: str
    notify_cb: Callable | None
    cancelled: str | None = None

    def __call__(self, data: dict) -> None:
        from tools import approval

        with approval._lock:
            notify = None if self.cancelled else self.notify_cb
        if notify is not None:
            notify(data)

    def close(self, cause: str = "the execution ended before the prompt was answered") -> None:
        from tools import approval

        with approval._lock:
            self._close_locked(approval, cause)

    def _close_locked(self, approval, cause: str) -> None:
        if self.cancelled:
            return
        self.cancelled = cause
        self.notify_cb = None
        _owners.discard(self)
        # A previous turn must not unregister a newer turn's delivery route.
        if approval._gateway_notify_cbs.get(self.session_key) is self:
            approval._gateway_notify_cbs.pop(self.session_key, None)
        queue = approval._gateway_queues.get(self.session_key, [])
        for entry in list(queue):
            if entry.owner is self:
                queue.remove(entry)
                entry.cancelled = cause
                entry.event.set()
        if not queue:
            approval._gateway_queues.pop(self.session_key, None)


gateway_approval_owner: ContextVar[GatewayApprovalOwner | None] = ContextVar("gateway_approval_owner", default=None)
_owners: set[GatewayApprovalOwner] = set()  # approval._lock protects owner state


def current_gateway_approval_owner(session_key: str) -> GatewayApprovalOwner | None:
    owner = gateway_approval_owner.get()
    return owner if owner is not None and owner.session_key == session_key else None


def register_gateway_approval_owner(session_key: str, notify_cb: Callable) -> GatewayApprovalOwner:
    from tools import approval

    owner = GatewayApprovalOwner(session_key, notify_cb)
    with approval._lock:
        _owners.add(owner)
        approval._gateway_notify_cbs[session_key] = owner
    return owner


def retain_gateway_approval_owner(session_key: str) -> GatewayApprovalOwner | None:
    """Fork the bound execution's route, never a newer turn's session registration.

    A closed context produces a closed owner rather than falling back to another
    turn's callback. Acquire on the submitting thread, before it can finish.
    """
    from tools import approval

    parent = current_gateway_approval_owner(session_key)
    if parent is None:
        return None
    with approval._lock:
        owner = GatewayApprovalOwner(session_key, parent.notify_cb, parent.cancelled)
        if not owner.cancelled:
            _owners.add(owner)
        return owner


def close_gateway_approval_owners_locked(session_key: str, cause: str) -> None:
    """Full session teardown, including detached owners. Caller holds approval._lock."""
    from tools import approval

    for owner in list(_owners):
        if owner.session_key == session_key:
            owner._close_locked(approval, cause)
