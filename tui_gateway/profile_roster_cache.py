"""Memos for ``profiles.list``'s per-profile fields, each keyed on the file(s) it derives from.

The Bots roster polls ``profiles.list`` every 5s PER CONNECTION, and two groups of fields on every
row are recomputed from disk each time even though nothing has changed:

- the **session fields** (``last_session`` / ``worker_session`` / ``canonical_session``) — a
  read-only ``state.db`` open plus a listing query plus the canonical-title lookup, per profile;
- the **ui_meta fields** (``ui_meta`` / ``ui_meta_revisions``) — a second parse of that profile's
  ``profile.yaml``, which the listing body already parsed once for description/display_name.

Both are pure functions of the files they read, so while those files have not moved there is
nothing to recompute: an idle fleet re-derived identical rows every five seconds for every bot.

Signatures carry each file's ``(mtime_ns, size)`` so a write landing inside one mtime tick still
invalidates; ``profile.yaml`` also carries its inode, because the atomic writers rename a temp file
into place. A profile missing the file a memo keys on is never cached — reading it costs nothing,
and one written later must be picked up.

Only DERIVED values live here. The raw readers keep their uncached contract: ``_read_profile_yaml``
is also used by the ui_meta CAS writer, which reads that document, mutates it and writes it back, and
a stale read there would overwrite a newer file.

This is a module of its own because ``methods_profiles``'s bodies are rebound onto ``server.py``'s
globals (``method_ctx.bind_module``), which COPIES module-level dicts rather than sharing them — a
cache declared there would be written to one copy and read from another.
"""
from __future__ import annotations

import copy
import sqlite3
from pathlib import Path
from typing import Any, Callable, Optional

from utils import mtime_is_racy

# Each keyed by resolved profile path. One entry per profile the roster has painted; a removed
# profile leaves one stale entry, so a memo is dropped whole once it grows past any plausible fleet.
_SESSION_CACHE: dict[str, tuple[tuple, dict]] = {}
_UI_META_CACHE: dict[str, tuple[tuple, dict]] = {}
# The canonical gateway's rows also depend on who reads them (its owner filter): keyed by
# (profile path, reader).
_READER_SESSION_CACHE: dict[tuple, tuple[tuple, dict]] = {}
# ``live_message_count`` per (state.db path, session id) -> (display-set fingerprint, count).
_LIVE_COUNT_CACHE: dict[tuple[str, str], tuple[tuple, int]] = {}
_MAX_ENTRIES = 512

_STORE_FILES = ("state.db", "state.db-wal")


def _file_parts(path: Path, *, with_inode: bool = False) -> Optional[tuple]:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (path.name, stat.st_mtime_ns, stat.st_size) + ((stat.st_ino,) if with_inode else ())


def _cached(cache: dict, key: str, signature: Optional[tuple], compute: Callable[[], dict],
            copier: Callable[[dict], dict]) -> dict[str, Any]:
    """``compute()``'s fields, reused while *signature* holds. No signature means no cache."""
    if signature is None:
        return compute()
    hit = cache.get(key)
    if hit is not None and hit[0] == signature:
        return copier(hit[1])
    fields = compute()
    if len(cache) >= _MAX_ENTRIES:
        cache.clear()
    cache[key] = (signature, copier(fields))
    return fields


def store_signature(profile_path: str | Path) -> Optional[tuple]:
    """``(name, mtime_ns, size)`` per session-store file, or None when the profile has no store.

    The pair the change watcher already trusts for ``sessions.changed``. None too while a file's
    mtime is fresh enough that a same-tick, same-size WAL rewrite could still hide behind it.
    """
    base = Path(profile_path)
    parts = [p for p in (_file_parts(base / name) for name in _STORE_FILES) if p is not None]
    # Our own read-only open creates an EMPTY -wal sidecar on a store that never had one; it
    # carries no frames, so it must not read as "the store moved" on the very next poll.
    parts = [p for p in parts if not (p[0].endswith("-wal") and p[2] == 0)]
    if any(mtime_is_racy(p[1]) for p in parts):
        return None
    return tuple(parts) or None


def profile_yaml_signature(profile_dir: str | Path) -> Optional[tuple]:
    """``(name, mtime_ns, size, inode)`` of the profile's ``profile.yaml``, or None when absent."""
    return _file_parts(Path(profile_dir) / "profile.yaml", with_inode=True)


def cached_session_fields(profile_path: str | Path, compute: Callable[[], dict]) -> dict[str, Any]:
    """``compute()``'s fields, reused while the profile's session store has not moved."""
    return _cached(_SESSION_CACHE, str(profile_path), store_signature(profile_path), compute, dict)


def cached_reader_session_fields(profile_path: str | Path, reader: tuple,
                                 compute: Callable[[], dict]) -> dict[str, Any]:
    """``cached_session_fields`` for fields that also depend on *reader*, a hashable principal key."""
    return _cached(_READER_SESSION_CACHE, (str(profile_path), reader), store_signature(profile_path),
                   compute, dict)


def cached_ui_meta_fields(profile_dir: str | Path, compute: Callable[[], dict]) -> dict[str, Any]:
    """``compute()``'s fields, reused while the profile's ``profile.yaml`` has not changed.

    Copied deeply: ``ui_meta`` is a nested mapping the caller hands to a client.
    """
    return _cached(_UI_META_CACHE, str(profile_dir), profile_yaml_signature(profile_dir),
                   compute, copy.deepcopy)


def live_count_signature(db, session_id: str) -> tuple:
    """Cheap fingerprint of what ``db.display_message_count(session_id)`` counts.

    The count reads every display row of the session, table rows included (its visibility filter
    parses ``display_metadata``, its tool-copy filter probes sibling rows), which on a long chat in a
    large store costs from tens of milliseconds warm to seconds on cold pages, once per chat per
    ``profiles.list``. This reads only the partial index ``idx_messages_display_page`` (the display
    set, ``active = 1 OR compacted = 1``) plus the session's own row: the set's size, how much of it is
    still ``active`` (compaction archives a row without changing the size), its highest id (an
    append), its ``display_order`` groups (the count is one per group, and a backfill or re-fold
    moves rows between groups without touching any revision), and the write-side
    ``transcript_revision`` (the triggers bump it on a content, role or display-metadata rewrite).
    Raises ``sqlite3.Error`` on a store without these columns.
    """
    from hermes_state_messages import _DISPLAY_ACTIVE_CLAUSE

    row = db._read_one(
        "SELECT COUNT(*), COALESCE(SUM(active), 0), COALESCE(MAX(id), 0),"
        " COUNT(DISTINCT display_order), COUNT(display_order),"
        " (SELECT COALESCE(transcript_revision, 0) FROM sessions WHERE id = ?)"
        f" FROM messages WHERE session_id = ?{_DISPLAY_ACTIVE_CLAUSE}", (session_id, session_id))
    return tuple(int(v or 0) for v in row)


def cached_live_count(db, session_id: str) -> int:
    """``db.display_message_count(session_id)``, reused while the session's display set has not moved.

    ``profiles.list`` asks for the canonical chat and the newest chat of every profile on every
    poll, and on a store that background workers write all the time the store-wide memo above
    never holds. The fingerprint is read BEFORE the count, so a write landing between the two is
    cached under the older fingerprint and recounted on the next call, never served under the
    newer one. No fingerprint (an older schema, an in-memory store) means no memo: it just counts.
    """
    session_id = str(session_id)
    path = getattr(db, "db_path", None)
    if not path or str(path) == ":memory:":
        return db.display_message_count(session_id)
    try:
        signature = live_count_signature(db, session_id)
    except sqlite3.Error:
        return db.display_message_count(session_id)
    key = (str(path), session_id)
    hit = _LIVE_COUNT_CACHE.get(key)
    if hit is not None and hit[0] == signature:
        return hit[1]
    count = db.display_message_count(session_id)
    if len(_LIVE_COUNT_CACHE) >= _MAX_ENTRIES:
        _LIVE_COUNT_CACHE.clear()
    _LIVE_COUNT_CACHE[key] = (signature, count)
    return count


def invalidate(profile_path: str | Path | None = None) -> None:
    """Drop one profile's memos, or all of them. For tests and for a caller that knows better."""
    for cache in (_SESSION_CACHE, _UI_META_CACHE):
        if profile_path is None:
            cache.clear()
        else:
            cache.pop(str(profile_path), None)
    store = None if profile_path is None else str(Path(profile_path) / "state.db")
    for key in [key for key in _LIVE_COUNT_CACHE if store is None or key[0] == store]:
        _LIVE_COUNT_CACHE.pop(key, None)
    for key in [key for key in _READER_SESSION_CACHE if profile_path is None or key[0] == str(profile_path)]:
        _READER_SESSION_CACHE.pop(key, None)
