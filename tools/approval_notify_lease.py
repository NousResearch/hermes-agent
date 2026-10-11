"""Dispatch-owned approval transport; never extends a completed turn's stream lifetime."""
from contextvars import ContextVar
from weakref import WeakSet

_current = ContextVar("approval_notify_lease", default=None)
_leases = WeakSet()  # Guarded by approval._lock; does not keep completed workers alive.


class NotifyLease:
    def __init__(self, session_key, callback):
        self.session_key = session_key
        self.callback = callback
        self.active = True

    def notify(self, data):
        from tools import approval
        with approval._lock:
            active, callback = self.active, self.callback
        if not active or callback is None:
            raise RuntimeError("Approval notification unavailable: background task ended or transport missing")
        callback(data)

    def release(self):
        from tools import approval
        with approval._lock:
            self.active = False
            self.callback = None
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
        callback = approval._gateway_notify_cbs.get(session_key)
        callback = getattr(callback, "background_notify", callback)
        lease = NotifyLease(session_key, callback)
        _leases.add(lease)
        return lease


def revoke_session_locked(session_key):
    """Called by clear_session under approval._lock before waking pending waits."""
    for lease in list(_leases):
        if lease.session_key == session_key:
            lease.active = False
            lease.callback = None
            _leases.discard(lease)


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
