"""Memo for ``profiles.list``'s ui_meta fields, keyed on that profile's ``profile.yaml``.

The Bots roster polls ``profiles.list`` every 5s per connection, and each row's
``ui_meta`` / ``ui_meta_revisions`` come from parsing the profile's ``profile.yaml`` — a second
parse of the same file in the same request, since ``hermes_cli.profiles.read_profile_meta`` has
already read it for the row's description and display name. They are a pure function of that file,
so while it has not moved there is nothing to recompute.

Only the YAML-derived fields are cached. ``has_avatar`` stays live (three ``is_file()`` stats,
cheaper than keying a cache on them), and the raw reader keeps its uncached contract: the ui_meta
CAS writer does a read-modify-write of that same document, and a stale read there would overwrite a
newer file.

Lives in its own module because ``methods_profiles``'s bodies are rebound onto ``server.py``'s
globals (``method_ctx.bind_module``), which copies module-level dicts rather than sharing them — a
cache declared there would be written to one copy and read from another.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Optional

# Keyed by resolved profile dir. One entry per profile the roster paints; a removed profile leaves
# one stale entry, so the whole memo is dropped once it grows past any plausible fleet.
_CACHE: dict[str, tuple[tuple, dict]] = {}
_MAX_ENTRIES = 512


def profile_yaml_signature(profile_dir: "str | Path") -> Optional[tuple]:
    """``(mtime_ns, size, inode)`` of the profile's ``profile.yaml``, or None when it is absent.
    The atomic writers rename a temp file into place, so a rewrite always lands a new inode even
    inside one mtime tick."""
    try:
        stat = (Path(profile_dir) / "profile.yaml").stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


def cached_ui_meta_fields(profile_dir: "str | Path", compute: Callable[[], dict]) -> dict[str, Any]:
    """``compute()``'s fields, reused while the profile's ``profile.yaml`` has not changed.

    A profile without one is never cached: there is nothing to parse, and a file written later must
    be picked up. The copy is deep — ``ui_meta`` is a nested mapping the caller hands to a client.
    """
    signature = profile_yaml_signature(profile_dir)
    if signature is None:
        return compute()

    key = str(profile_dir)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return copy.deepcopy(cached[1])

    fields = compute()
    if len(_CACHE) >= _MAX_ENTRIES:
        _CACHE.clear()
    _CACHE[key] = (signature, copy.deepcopy(fields))
    return fields


def invalidate(profile_dir: "str | Path | None" = None) -> None:
    """Drop one profile's memo, or all of them. For tests and for a caller that knows better."""
    if profile_dir is None:
        _CACHE.clear()
    else:
        _CACHE.pop(str(profile_dir), None)
