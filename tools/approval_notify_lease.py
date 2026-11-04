"""Dispatch-owned approval transport; never extends a completed turn's stream lifetime."""
from contextvars import ContextVar
from weakref import WeakSet

_current = ContextVar("approval_notify_lease", default=None)
_leases = WeakSet()  # Guarded by approval._lock; does not keep completed workers alive.


class NotifyLease:
    def __init__(self, session_key, callback, cancelled=None):
        self.session_key = session_key
        self.callback = callback
        self._active = True
        self.cancelled = cancelled
        self.canonical_owner = None

    @property
    def active(self):
        return self._active and not (self.cancelled is not None and self.cancelled.is_set())

    @active.setter
    def active(self, value):
        self._active = value

    def permits(self, request_id):
        return self.active and (self.canonical_owner is None or self.canonical_owner.permits(request_id))

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


def acquire(session_key, dispatch_id=None, *, cancelled=None):
    from tools import approval
    with approval._lock:
        callback = approval._gateway_notify_cbs.get(session_key)
        factory = getattr(callback, "acquire_background", None)
        callback = getattr(callback, "background_notify", callback)
        lease = NotifyLease(session_key, callback, cancelled)
        _leases.add(lease)
    # Minting takes the stream lock: never call it under the tools lock.
    if factory is not None:
        try:
            callback = factory(lease, dispatch_id)
            with approval._lock:
                if lease.active:
                    lease.callback = callback
        except Exception:
            lease.release()
            raise
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
