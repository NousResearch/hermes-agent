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
from pathlib import Path
from typing import Any, Callable, Optional

# Each keyed by resolved profile path. One entry per profile the roster has painted; a removed
# profile leaves one stale entry, so a memo is dropped whole once it grows past any plausible fleet.
_SESSION_CACHE: dict[str, tuple[tuple, dict]] = {}
_UI_META_CACHE: dict[str, tuple[tuple, dict]] = {}
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


def store_signature(profile_path: "str | Path") -> Optional[tuple]:
    """``(name, mtime_ns, size)`` per session-store file, or None when the profile has no store.

    The pair the change watcher already trusts for ``sessions.changed``.
    """
    base = Path(profile_path)
    parts = [p for p in (_file_parts(base / name) for name in _STORE_FILES) if p is not None]
    return tuple(parts) or None


def profile_yaml_signature(profile_dir: "str | Path") -> Optional[tuple]:
    """``(name, mtime_ns, size, inode)`` of the profile's ``profile.yaml``, or None when absent."""
    return _file_parts(Path(profile_dir) / "profile.yaml", with_inode=True)


def cached_session_fields(profile_path: "str | Path", compute: Callable[[], dict]) -> dict[str, Any]:
    """``compute()``'s fields, reused while the profile's session store has not moved."""
    return _cached(_SESSION_CACHE, str(profile_path), store_signature(profile_path), compute, dict)


def cached_ui_meta_fields(profile_dir: "str | Path", compute: Callable[[], dict]) -> dict[str, Any]:
    """``compute()``'s fields, reused while the profile's ``profile.yaml`` has not changed.

    Copied deeply: ``ui_meta`` is a nested mapping the caller hands to a client.
    """
    return _cached(_UI_META_CACHE, str(profile_dir), profile_yaml_signature(profile_dir),
                   compute, copy.deepcopy)


def invalidate(profile_path: "str | Path | None" = None) -> None:
    """Drop one profile's memos, or all of them. For tests and for a caller that knows better."""
    for cache in (_SESSION_CACHE, _UI_META_CACHE):
        if profile_path is None:
            cache.clear()
        else:
            cache.pop(str(profile_path), None)
