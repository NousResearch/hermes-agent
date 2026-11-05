"""Session store handles and their storage interface handles. Bodies are rebound onto server.py's globals at install
time (method_ctx.bind_module), so they reference server.py globals bare.

Every store this gateway opens is itself a storage interface handle. Where a caller holds a store and wants its handle,
the launch store is its own handle and a dedicated store gets a reference of its own on the same profile, released with
:func:`_release_persistence`. A store that is not a handle has none (None).
"""

from __future__ import annotations

import contextlib

from .method_ctx import bind_module


def _is_handle(db) -> bool:
    from persistence_assembly import Persistence
    return isinstance(db, Persistence)


def _profile_session_db(profile_home):
    """``(db, owns)``: a DEDICATED handle on ``profile_home``'s state.db, else the shared launch db."""
    if profile_home:
        from persistence_assembly import open_persistence, profile_scope
        return open_persistence(profile_scope(home=Path(profile_home)), eager=True), True
    return _get_db(), False


def _release_db(db) -> None:
    with contextlib.suppress(Exception):
        db.close()


def _launch_persistence():
    """The launch store ``_get_db()`` holds, its own handle; None while no launch store is held or while it is not a
    handle."""
    return _db if _is_handle(_db) else None


def _persistence_beside(db):
    """The writer handle for ``db``: the launch store for the launch store, else a reference of its own on db's
    profile, released with :func:`_release_persistence`; None without a store or for one that is not a handle."""
    if db is None:
        return None
    if db is _db:
        return _launch_persistence()
    if _is_handle(db):
        from persistence_assembly import open_persistence
        return open_persistence(db.scope)
    return None


def _release_persistence(handle) -> None:
    """Release a handle from :func:`_persistence_beside`; the launch store stays open."""
    if handle is not None and handle is not _db:
        with contextlib.suppress(Exception):
            handle.close()


@contextlib.contextmanager
def _persistence_for(db):
    """``db`` for one call when it is a handle (a store is its own, in its own access mode); None otherwise."""
    yield db if _is_handle(db) else None


def register(server) -> None:
    """Publish this module's helpers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
