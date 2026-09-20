"""Memo for ``profiles.list``'s per-profile session fields, keyed on the session store's signature.

The Bots roster polls ``profiles.list`` every 5s PER CONNECTION, and its session fields
(``last_session`` / ``worker_session`` / ``canonical_session``) are read from that profile's
``state.db`` — an open plus a listing query plus the canonical-title lookup, per profile, per poll.
They are a pure function of that store, so while the store has not moved there is nothing to
recompute: an idle fleet re-derived the same rows every five seconds for every bot it has.

The signature is the one the change watcher already trusts for ``sessions.changed`` — the newest
mtime across ``state.db`` and its ``-wal`` sidecar — plus each file's size, so a write that lands
inside one mtime tick still invalidates. A profile with no store is not cached: it has nothing to
read, so the call is already cheap and a future store must be picked up.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

# Keyed by resolved profile path. One entry per profile the roster has painted; a removed profile
# leaves one stale entry, so the whole memo is dropped once it grows past any plausible fleet.
_CACHE: dict[str, tuple[tuple, dict]] = {}
_MAX_ENTRIES = 512

_STORE_FILES = ("state.db", "state.db-wal")


def store_signature(profile_path: "str | Path") -> Optional[tuple]:
    """``(name, mtime_ns, size)`` per session-store file, or None when the profile has no store."""
    base = Path(profile_path)
    parts: list[tuple[str, int, int]] = []
    for name in _STORE_FILES:
        try:
            stat = (base / name).stat()
        except OSError:
            continue
        parts.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(parts) or None


def cached_session_fields(profile_path: "str | Path", compute: Callable[[], dict]) -> dict[str, Any]:
    """``compute()``'s fields, reused while the profile's session store has not moved."""
    signature = store_signature(profile_path)
    if signature is None:
        return compute()

    key = str(profile_path)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == signature:
        return dict(hit[1])

    fields = compute()
    if len(_CACHE) >= _MAX_ENTRIES:
        _CACHE.clear()
    _CACHE[key] = (signature, dict(fields))
    return dict(fields)


def invalidate(profile_path: "str | Path | None" = None) -> None:
    """Drop one profile's memo, or all of them. For tests and for a caller that knows better."""
    if profile_path is None:
        _CACHE.clear()
    else:
        _CACHE.pop(str(profile_path), None)
