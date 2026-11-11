"""Dispatch-owned approval transport; never extends a completed turn's stream lifetime.

A background child takes a lease at dispatch (``tools.async_delegation``). While a turn of its session
has a notifier registered, the child's prompt goes through that notifier, exactly as before. Once none
is (the turn that dispatched it ended), the lease's own transport delivers it: the ``background_notify``
the gateway turn runner published (``gateway/run_turn_runner_approval_transport.py``). A surface that
publishes none leaves the lease without a transport, and the request fails closed instead of parking an
approval nobody can see (#133514).
"""
from contextvars import ContextVar
from weakref import WeakSet, ref

_current = ContextVar("approval_notify_lease", default=None)
_leases = WeakSet()  # Guarded by approval._lock; does not keep completed workers alive.


class NotifyLease:
    def __init__(self, session_key, callback, origin=None):
        self.session_key = session_key
        self.callback = callback
        # Identity only: a strong reference would keep the dispatching turn (and its history) alive.
        self._origin = ref(origin) if origin is not None else None
        self.active = True

    def _target_locked(self):
        """The dispatching turn's own notifier while it runs (its typing, stream boundary and
        canonical projection apply), else the lease's transport; never a later turn's generation."""
        from tools import approval
        if not self.active:
            return None
        registered = approval._gateway_notify_cbs.get(self.session_key)
        origin = self._origin() if self._origin is not None else None
        if registered is not None and (registered is origin or self.callback is None):
            return registered
        return self.callback

    def deliverable(self):
        """False only while the lease is live but nothing could show its prompt."""
        from tools import approval
        with approval._lock:
            return not self.active or self._target_locked() is not None

    def notify(self, data):
        from tools import approval
        with approval._lock:
            target = self._target_locked()
        if target is None:
            raise RuntimeError("Approval notification unavailable: background task ended or transport missing")
        target(data)

    def release(self):
        from tools import approval
        with approval._lock:
            self.active = False
            self.callback = self._origin = None
            _leases.discard(self)
            queue = approval._gateway_queues.get(self.session_key, [])
            for entry in list(queue):
                if entry.owner is self:
                    queue.remove(entry)
                    entry.cancelled = "background task ended"
                    entry.event.set()
            if not queue:
                approval._gateway_queues.pop(self.session_key, None)


def acquire(session_key):
    from tools import approval
    with approval._lock:
        # Only a transport built to outlive its turn is kept; a turn-scoped callback is not.
        origin = approval._gateway_notify_cbs.get(session_key)
        lease = NotifyLease(session_key, getattr(origin, "background_notify", None), origin)
        _leases.add(lease)
        return lease


def revoke_session_locked(session_key):
    """Called by clear_session under approval._lock before waking pending waits."""
    for lease in list(_leases):
        if lease.session_key == session_key:
            lease.active = False
            lease.callback = lease._origin = None
            _leases.discard(lease)


def stranded_denial(session_key, is_gateway, noun, pattern_key, description):
    """The fail-closed result for a background child's request no chat can show (its dispatching
    turn ended and its surface left it no transport), else None. ``pending_approval`` there would be
    a prompt nobody can see or ``/approve``."""
    lease = current(session_key)
    if not is_gateway or lease is None or lease.deliverable():
        return None
    from tools.approval import _denied
    return _denied(
        "BLOCKED: no approval channel is available to ask the user — the gateway turn "
        f"that delivers approval prompts has ended, so this {noun} did not run "
        "and no prompt can be answered from this context. Do NOT retry or rephrase; "
        "run it from an interactive turn instead.",
        pattern_key=pattern_key, description=description, outcome="notify_failed", noun=noun)


def current(session_key):
    lease = _current.get()
    return lease if lease is not None and lease.session_key == session_key else None


def run(lease, fn):
    token = _current.set(lease)
    try:
        return fn()
    finally:
        _current.reset(token)
        lease.release()
