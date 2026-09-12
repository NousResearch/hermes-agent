"""Process-local approval delivery leases for detached delegate_task children.

A copied ContextVar is not a notifier lifetime. Acquire before dispatch returns,
revoke on child completion/cancel, and bind only inside that child's worker.
Queue decisions remain tools.approval's responsibility. Nothing here approves a
tool, persists a rule, replays an operation, or grants broker consent.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable

from hermes_constants import get_hermes_home
from tools.approval_context import get_current_session_key, _get_session_platform


@dataclass(frozen=True)
class ApprovalOwner:
    profile: str
    session: str
    platform: str
    actor: str
    chat: str
    thread: str


def owner_for_source(source, session_key):
    """Bind an already-authenticated SessionSource, not arbitrary truthy values.

    SessionSource uses Platform and string IDs. SDK numeric IDs are converted by
    adapters before this boundary (e.g. Telegram._build_message_event). Do not
    coerce raw integers, bools, containers or value-wrapper objects into authority.
    Only an absent/empty optional thread is represented by the empty string.
    """
    from gateway.config import Platform
    from gateway.session import SessionSource
    if not isinstance(source, SessionSource):
        return None
    if not isinstance(source.platform, Platform) or source.is_bot is not False:
        return None
    platform = source.platform.value
    required = (session_key, platform, source.user_id, source.chat_id)
    if any(type(value) is not str or not value or value != value.strip()
           or not value.isprintable() for value in required):
        return None
    thread = source.thread_id
    if thread is None:
        thread = ''
    elif type(thread) is not str or thread != thread.strip() or (thread and not thread.isprintable()):
        return None
    return ApprovalOwner(str(get_hermes_home().resolve()), session_key, platform,
                         source.user_id, source.chat_id, thread)


@dataclass(eq=False)
class _Route:
    owner: ApprovalOwner | None
    notify: Callable | None
    active: bool = True


@dataclass(eq=False)
class ChildApprovalLease(_Route):
    child_id: str = ''


_origin = ContextVar('background_approval_origin', default=None)
_child = ContextVar('background_approval_child', default=None)
_leases: set[ChildApprovalLease] = set()  # protected by approval._lock


@contextmanager
def gateway_approval_origin(owner, notify):
    route = _Route(owner, notify)
    token = _origin.set(route)
    try:
        yield
    finally:
        from tools import approval
        with approval._lock:
            route.active = False
        _origin.reset(token)


def _matches(route, session_key):
    owner = route.owner
    return bool(route.active and owner and route.notify and owner.session == session_key
                and owner.profile == str(get_hermes_home().resolve())
                and owner.platform == _get_session_platform())


def acquire_child_approval(child_id):
    from tools import approval
    with approval._lock:
        # Nested children may derive a new lease from their still-live owner,
        # not from a stale parent-turn registration in a copied context.
        route = _child.get() or _origin.get()
        valid = bool(child_id and route and _matches(route, get_current_session_key()))
        lease = ChildApprovalLease(route.owner if valid else None,
                                   route.notify if valid else None, valid, str(child_id or ''))
        if valid:
            _leases.add(lease)
        return lease


def current_child_approval():
    return _child.get()


def child_notify(session_key):
    from tools import approval
    with approval._lock:
        lease = _child.get()
        return lease.notify if lease and _matches(lease, session_key) else None


@contextmanager
def child_approval_scope(lease):
    token = _child.set(lease)
    try:
        yield
    finally:
        release_child_approval(lease)
        _child.reset(token)


def _release_locked(lease, approval):
    lease.active = False
    _leases.discard(lease)
    for key, queue in list(approval._gateway_queues.items()):
        for entry in list(queue):
            if entry.lease is lease:
                queue.remove(entry)
                entry.result = 'deny'
                entry.event.set()
        if not queue:
            approval._gateway_queues.pop(key, None)


def release_child_approval(lease):
    if lease is None:
        return
    from tools import approval
    with approval._lock:
        _release_locked(lease, approval)


def clear_child_approvals(session_key):
    from tools import approval
    profile = str(get_hermes_home().resolve())
    with approval._lock:
        for lease in list(_leases):
            if lease.owner.profile == profile and lease.owner.session == session_key:
                _release_locked(lease, approval)


def lease_matches(lease, session_key):
    """Caller holds approval._lock when admitting/resolving a queue entry."""
    return _matches(lease, session_key)


def response_matches(entry, owner, request_id, choice, resolve_all):
    lease = entry.lease
    return bool(lease and lease.active and owner is not None and lease.owner == owner
                and request_id == entry.data['request_id'] and not resolve_all
                and choice in {'once', 'deny'})
