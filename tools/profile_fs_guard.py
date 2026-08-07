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
  * A path is permitted only if its fully-resolved path equals, or is nested
    under, one of those roots. The nesting check uses a trailing separator so
    an allowed root ``/data/foo`` does not also match a sibling
    ``/data/foobar``.
  * Path resolution is *backend-aware*. On a local backend the path is
    realpath-resolved, so a symlink planted inside an allowed root that points
    at a denied directory resolves to the denied target and is refused. Under
    a container backend (docker/singularity/modal/daytona) the tool paths are
    guest paths and host symlink dereference would rewrite them incorrectly —
    ``/workspace`` is commonly a host symlink — so the path is normalized
    lexically instead, matching ``file_tools._normalize_without_host_deref``.
    Lexical normalization still collapses ``..`` before the boundary test, so
    traversal out of an allowed root is refused on both backends.
  * Profiles NOT listed in ``profile_fs_allowlist`` are unrestricted — the
    guard is a pure pass-through, so installs that never set the key (and the
    default profile) see no behavior change.

The allowlist is read from the root (default ``HERMES_HOME``) ``config.yaml``
rather than the active profile's, so it is stable across per-turn profile
switches under a multiplexed gateway. In a single-profile install the root
config *is* the active config, so the same code path applies unchanged.

Fail-closed, in both directions that matter:
  * If the policy file exists but cannot be read or parsed, the loader raises
    rather than degrading to ``{}`` — a corrupt policy must not silently
    unrestrict every profile it was meant to confine. Callers surface the
    error as a denial.
  * For a restricted profile, a path that cannot be resolved is denied.
A genuinely absent policy (no ``config.yaml``, or no ``profile_fs_allowlist``
key) is the documented no-op default and yields an empty map, so installs that
never opt in are unaffected.
"""

from __future__ import annotations

import os
import posixpath
import threading
from pathlib import Path, PurePosixPath
from typing import NamedTuple

# Cache: (allowlist_map). Populated once per process from config; profile
# routing can switch the *active* profile per turn, but the allowlist map
# itself is static config, so caching it is safe.
_lock = threading.Lock()
_allowlist_cache: dict | None = None


class _Root(NamedTuple):
    """An allowlisted root, pre-resolved in both comparison forms.

    ``real`` is the host realpath (symlinks followed), used on local
    backends. ``lexical`` is the syntactically-normalized path, used under
    container backends where the path is a guest path the host must not
    dereference. See ``check_path_allowed``.
    """

    real: str
    lexical: str


class PolicyError(RuntimeError):
    """The allowlist policy exists but could not be read or parsed.

    Raised instead of degrading to an empty (== unrestricted) map, so a
    corrupt or unreadable policy can never silently drop the boundary it was
    written to enforce. Callers convert this into a denial.
    """


def _resolve_root(root) -> _Root:
    """Pre-resolve one allowlist root into both comparison forms."""
    expanded = os.path.expanduser(str(root))
    return _Root(
        real=os.path.realpath(expanded),
        lexical=str(PurePosixPath(posixpath.normpath(expanded))),
    )


def _load_allowlist() -> dict:
    """Return ``{profile_name: [resolved_root, ...]}`` from config.

    Roots are tilde-expanded and realpath-resolved once so per-call checks are
    cheap string comparisons.

    Raises ``PolicyError`` if the root ``config.yaml`` exists but cannot be
    read or parsed, or if ``profile_fs_allowlist`` is present but malformed.
    A missing config file, or a config with no ``profile_fs_allowlist`` key,
    is the documented opt-out and returns an empty map.
    """
    global _allowlist_cache
    with _lock:
        if _allowlist_cache is not None:
            return _allowlist_cache

        # Read the ROOT (default HERMES_HOME) config, NOT the active
        # profile's. Under a multiplexed gateway the active profile
        # switches per turn; a per-profile config read would see only
        # whichever profile happened to be active on first call and cache
        # a partial map. The root config.yaml is shared by all profiles,
        # so the allowlist there is global and stable.
        import yaml
        from hermes_cli.profiles import _get_default_hermes_home

        try:
            root_cfg_path = _get_default_hermes_home() / "config.yaml"
        except Exception as exc:
            raise PolicyError(f"could not locate the root config: {exc}") from exc

        try:
            with open(root_cfg_path, "r", encoding="utf-8") as fh:
                root_cfg = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            # No root config at all -> no policy declared -> no restrictions.
            # This is the ordinary state of an install that never opted in.
            _allowlist_cache = {}
            return _allowlist_cache
        except Exception as exc:
            # Unreadable/corrupt policy: refuse to guess. Denying is
            # recoverable (fix the YAML); silently unrestricting is not.
            raise PolicyError(f"could not read {root_cfg_path}: {exc}") from exc

        if not isinstance(root_cfg, dict):
            raise PolicyError(f"{root_cfg_path} did not parse to a mapping")

        raw = root_cfg.get("profile_fs_allowlist") or {}
        if not raw:
            _allowlist_cache = {}
            return _allowlist_cache
        if not isinstance(raw, dict):
            raise PolicyError(
                "profile_fs_allowlist must be a mapping of "
                f"profile -> [roots], got {type(raw).__name__}"
            )

        mapping: dict[str, list[str]] = {}
        for profile, roots in raw.items():
            if roots is None:
                roots = []
            if isinstance(roots, (str, bytes)) or not isinstance(roots, (list, tuple)):
                # A bare string here is almost certainly a typo for a
                # one-element list. Guessing could widen or narrow the
                # boundary silently, so make the operator say what they mean.
                raise PolicyError(
                    f"profile_fs_allowlist['{profile}'] must be a list of "
                    f"directory roots, got {type(roots).__name__}"
                )
            resolved_roots: list[_Root] = []
            for root in roots:
                try:
                    # Keep BOTH forms. A root must be compared against a path
                    # resolved the same way: realpath vs realpath on a local
                    # backend, lexical vs lexical under a container backend.
                    # Realpath'ing a root while normalizing the path lexically
                    # (or vice versa) makes an allowed path look denied when
                    # the root itself sits behind a host symlink.
                    resolved_roots.append(_resolve_root(root))
                except Exception as exc:
                    raise PolicyError(
                        f"profile_fs_allowlist['{profile}'] contains an "
                        f"unusable root {root!r}: {exc}"
                    ) from exc
            # Normalize the profile key the same way Hermes does on disk.
            mapping[str(profile).strip().lower()] = resolved_roots

        _allowlist_cache = mapping
        return mapping


def _active_profile() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return (get_active_profile_name() or "default").strip().lower()
    except Exception:
        return "default"


def _is_under(child_real: str, root_real: str, *, sep: str = os.sep) -> bool:
    """True if *child_real* is *root_real* or nested beneath it."""
    if child_real == root_real:
        return True
    # Ensure a trailing separator so /a/note is not matched by root /a/no.
    root_with_sep = root_real.rstrip(sep) + sep
    return child_real.startswith(root_with_sep)


def _uses_container_paths(task_id: str) -> bool:
    """Whether *task_id* runs on a backend whose paths are guest paths.

    Delegates to ``file_tools`` so the guard and the tools it protects agree
    on exactly one definition of "container backend". Import is local and
    defensive to avoid a circular import at module load (file_tools imports
    this module).
    """
    try:
        from tools.file_tools import _uses_container_paths as _impl

        return bool(_impl(task_id))
    except Exception:
        # Unknown backend -> treat as local. Local resolution is the stricter
        # of the two (it dereferences symlinks), so this cannot widen access.
        return False


def check_path_allowed(
    path: str,
    *,
    base_dir: str | Path | None = None,
    task_id: str = "default",
) -> str | None:
    """Return an error string if the active profile may not touch *path*.

    ``None`` means allowed (either the profile is unrestricted, or the path is
    inside an allowlisted root). *path* may be relative; if so it is anchored
    to *base_dir* (the task cwd) before resolution. *task_id* selects the
    execution backend, which decides whether host symlinks are dereferenced
    (see the module docstring).
    """
    profile = _active_profile()
    try:
        allowlist = _load_allowlist()
    except PolicyError as exc:
        # The policy is declared but broken. Deny rather than fall open: the
        # whole point of this module is that the boundary holds even when
        # something else has gone wrong.
        return (
            f"Access denied: the filesystem allowlist policy could not be "
            f"loaded ({exc}). Refusing file access for the '{profile}' "
            f"profile until the policy is valid."
        )
    roots = allowlist.get(profile)
    if roots is None:
        # Profile is not restricted — no enforcement.
        return None

    containerized = _uses_container_paths(task_id)
    try:
        expanded = os.path.expanduser(str(path))
        if containerized:
            # Guest path: normalize lexically, never touch the host FS. This
            # mirrors file_tools._normalize_without_host_deref so the guard
            # tests the same string the tool will actually dispatch. ``..`` is
            # still collapsed, so escaping an allowed root remains impossible;
            # what we give up is symlink resolution, which is meaningless for
            # a path the host cannot interpret anyway.
            if not posixpath.isabs(expanded) and base_dir is not None:
                expanded = posixpath.join(str(base_dir), expanded)
            real = str(PurePosixPath(posixpath.normpath(expanded)))
            sep = "/"
        else:
            if not os.path.isabs(expanded) and base_dir is not None:
                expanded = os.path.join(os.fspath(base_dir), expanded)
            # realpath follows symlinks on every component, so a symlink
            # planted inside an allowed root that points at a denied dir
            # resolves to the denied realpath and is rejected.
            real = os.path.realpath(expanded)
            sep = os.sep
    except Exception:
        return (
            f"Access denied: '{path}' could not be resolved for the "
            f"'{profile}' profile's restricted filesystem policy."
        )

    for root in roots:
        # Compare like with like: the lexical root under a container backend,
        # the realpath root locally.
        root_str = root.lexical if containerized else root.real
        if _is_under(real, root_str, sep=sep):
            return None

    allowed = (
        ", ".join(r.lexical if containerized else r.real for r in roots)
        if roots
        else "(none)"
    )
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
