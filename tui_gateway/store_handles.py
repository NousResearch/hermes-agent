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


def _bootstrap_served_profile_store(profile_home: Path) -> None:
    """A served profile whose first RPC is a read (Bot Chat registry lookup on a fresh bot) has no
    store yet. Browsing never creates a store, but THIS process is the profile's session
    owner (``_open_profile_session_db`` acquires the writer for every agent build), so it seeds the
    store it is about to own instead of answering ``storage_unavailable`` until a turn happens."""
    from persistence_assembly import open_persistence, profile_scope, store_exists, store_location
    scope = profile_scope(home=Path(profile_home))
    location = store_location(scope)
    try:
        if store_exists(scope) if location is None else location.stat().st_size > 0:
            return
    except FileNotFoundError:
        pass
    open_persistence(scope, eager=True).close()


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
