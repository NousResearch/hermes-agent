"""Profile-scoped filesystem allowlist (hard rule, enforced in tool code).

A client-facing or otherwise untrusted profile should not be able to read the
operator's private files (finances, credentials, other clients' projects) even
if the model is prompted or tricked into trying. Prompt-level guardrails are
model-obeyed and therefore soft; this module enforces the boundary inside the
file-tool chokepoints *before* any file I/O, so it cannot be talked around.

Mechanism:
  * A restricted profile is one that appears as a key under the config block
    ``profile_fs_allowlist`` in the root ``config.yaml``. Its value is the list
    of directory roots that profile is allowed to read/write/search.
  * A path is permitted only if its fully-resolved realpath (symlinks
    followed) equals, or is nested under, one of those roots. Resolving the
    realpath first defeats symlink escapes out of an allowed root into a
    denied directory, and the nesting check uses a trailing separator so an
    allowed root ``/data/foo`` does not also match a sibling ``/data/foobar``.
  * Profiles NOT listed in ``profile_fs_allowlist`` are unrestricted — the
    guard is a pure pass-through, so installs that never set the key (and the
    default profile) see no behavior change.

The allowlist is read from the root (default ``HERMES_HOME``) ``config.yaml``
rather than the active profile's, so it is stable across per-turn profile
switches under a multiplexed gateway. In a single-profile install the root
config *is* the active config, so the same code path applies unchanged.

Fail-closed: for a restricted profile, if a path check errors it is denied. A
profile with no allowlist entry is never affected.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

# Cache: (allowlist_map). Populated once per process from config; profile
# routing can switch the *active* profile per turn, but the allowlist map
# itself is static config, so caching it is safe.
_lock = threading.Lock()
_allowlist_cache: dict | None = None


def _load_allowlist() -> dict:
    """Return ``{profile_name: [resolved_root, ...]}`` from config.

    Roots are tilde-expanded and realpath-resolved once so per-call checks are
    cheap string comparisons.
    """
    global _allowlist_cache
    with _lock:
        if _allowlist_cache is not None:
            return _allowlist_cache
        mapping: dict[str, list[str]] = {}
        try:
            # Read the ROOT (default HERMES_HOME) config, NOT the active
            # profile's. Under a multiplexed gateway the active profile
            # switches per turn; a per-profile config read would see only
            # whichever profile happened to be active on first call and cache
            # a partial map. The root config.yaml is shared by all profiles,
            # so the allowlist there is global and stable.
            import yaml
            from hermes_cli.profiles import _get_default_hermes_home

            root_cfg_path = _get_default_hermes_home() / "config.yaml"
            with open(root_cfg_path, "r", encoding="utf-8") as fh:
                root_cfg = yaml.safe_load(fh) or {}
            raw = root_cfg.get("profile_fs_allowlist", {}) or {}
            for profile, roots in raw.items():
                resolved_roots: list[str] = []
                for root in roots or []:
                    try:
                        expanded = os.path.expanduser(str(root))
                        resolved_roots.append(os.path.realpath(expanded))
                    except Exception:
                        # A malformed root entry must not silently widen access;
                        # skip it (fail-closed for that entry).
                        continue
                # Normalize the profile key the same way Hermes does on disk.
                mapping[str(profile).strip().lower()] = resolved_roots
        except Exception:
            mapping = {}
        _allowlist_cache = mapping
        return mapping


def _active_profile() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return (get_active_profile_name() or "default").strip().lower()
    except Exception:
        return "default"


def _is_under(child_real: str, root_real: str) -> bool:
    """True if *child_real* is *root_real* or nested beneath it."""
    if child_real == root_real:
        return True
    # Ensure a trailing separator so /a/note is not matched by root /a/no.
    root_with_sep = root_real.rstrip(os.sep) + os.sep
    return child_real.startswith(root_with_sep)


def check_path_allowed(path: str, *, base_dir: str | Path | None = None) -> str | None:
    """Return an error string if the active profile may not touch *path*.

    ``None`` means allowed (either the profile is unrestricted, or the path is
    inside an allowlisted root). *path* may be relative; if so it is anchored
    to *base_dir* (the task cwd) before resolution.
    """
    profile = _active_profile()
    allowlist = _load_allowlist()
    roots = allowlist.get(profile)
    if roots is None:
        # Profile is not restricted — no enforcement.
        return None

    try:
        expanded = os.path.expanduser(str(path))
        if not os.path.isabs(expanded) and base_dir is not None:
            expanded = os.path.join(os.fspath(base_dir), expanded)
        # realpath follows symlinks on every component, so a symlink planted
        # inside an allowed root that points at a denied dir resolves to the
        # denied realpath and is rejected.
        real = os.path.realpath(expanded)
    except Exception:
        return (
            f"Access denied: '{path}' could not be resolved for the "
            f"'{profile}' profile's restricted filesystem policy."
        )

    for root in roots:
        if _is_under(real, root):
            return None

    allowed = ", ".join(roots) if roots else "(none)"
    return (
        f"Access denied: the '{profile}' profile may only access: {allowed}. "
        f"The path '{path}' is outside that boundary and cannot be read, "
        f"searched, or written."
    )


def reset_cache() -> None:
    """Test/hook helper — drop the cached allowlist so config is re-read."""
    global _allowlist_cache
    with _lock:
        _allowlist_cache = None
