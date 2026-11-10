"""Session store handles and the storage interface handles beside them. Bodies are rebound onto server.py's globals at
install time (method_ctx.bind_module), so they reference server.py globals bare.

Every store this gateway opens is itself a storage interface handle. Where a caller holds a store and wants the handle
beside it, the launch store is its own handle and a dedicated store gets a reference of its own on the same profile,
released with :func:`_release_persistence`. A store that is not a handle (a legacy store object a caller passes in)
gets a handle opened beside it on the same file and in the same access mode; a handle that cannot open is None and the
store beside it is unaffected.
"""

from __future__ import annotations

import contextlib

from .method_ctx import bind_module

# The launch store ``_get_db()`` holds and the handle beside it.
_launch_beside = (None, None)


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
    """The handle beside the launch store, opened on first use and kept as long as that store; None while no launch
    store is held or while the handle cannot open."""
    global _launch_beside
    from persistence_beside import close_beside, open_beside
    store, handle = _launch_beside
    if _db is None or _is_handle(_db):
        return _db
    if store is not _db or handle is None:
        close_beside(handle)
        handle = open_beside(_launch_home())
        _launch_beside = (_db, handle)
    return handle


def _persistence_beside(db):
    """The writer handle beside ``db``: the launch handle beside the launch store, else a handle of its own on db's
    file, released with :func:`_release_persistence`; None without a store."""
    from persistence_beside import open_beside
    if db is None:
        return None
    if db is _db:
        return _launch_persistence()
    if _is_handle(db):
        from persistence_assembly import open_persistence
        return open_persistence(db.scope)
    path = getattr(db, "db_path", None)
    return open_beside(Path(path).parent) if isinstance(path, (str, Path)) else None


def _release_persistence(handle) -> None:
    """Release a handle from :func:`_persistence_beside`; the launch handle stays open."""
    from persistence_beside import close_beside
    if handle is not _launch_beside[1] and handle is not _db:
        close_beside(handle)


@contextlib.contextmanager
def _persistence_for(db):
    """The handle beside ``db`` for one call, in db's access mode (a reader beside a read-only store), released on
    exit; the launch handle stays open. A store that is a handle is its own."""
    from persistence_beside import close_beside, open_reader_beside
    if _is_handle(db):
        yield db
        return
    handles = contextlib.ExitStack()
    path = getattr(db, "db_path", None)
    if getattr(db, "read_only", False) is True and isinstance(path, (str, Path)):
        handle = open_reader_beside(Path(path), handles)
    else:
        handle = _persistence_beside(db)
        handles.callback(_release_persistence, handle)
    try:
        yield handle
    finally:
        close_beside(handles)


def register(server) -> None:
    """Publish this module's helpers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
