"""Shared path validation helpers for tool implementations.

Extracts the ``resolve() + relative_to()`` and ``..`` traversal check
patterns previously duplicated across skill_manager_tool, skills_tool,
skills_hub, cronjob_tools, and credential_files.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _is_path_redirect(path: Path) -> bool:
    """True when *path* is a symlink or (on Windows) a directory junction.

    Either form lets an attacker who can write into a directory tree
    redirect subsequent filesystem operations to content outside it.
    ``is_junction`` only exists on Python 3.12+ Windows; gate with
    ``hasattr``.
    """
    return path.is_symlink() or (
        hasattr(path, "is_junction") and path.is_junction()  # type: ignore[union-attr]
    )


def _has_symlink_component(path: Path, root: Path) -> Optional[str]:
    """Walk *path* component-by-component below *root* and return an error
    message if any intermediate component is a symlink or junction.

    This catches symlink redirects *before* ``Path.resolve()`` follows
    them, so that a path like ``skills/symlink_dir/file`` where
    ``symlink_dir -> /tmp`` is rejected even though ``Path.resolve()``
    would reveal the escape.

    Walks the *lexical* components of *path* relative to *root* — not the
    resolved components — so that a redirect at e.g. ``root/link`` is seen
    as ``link``, not as ``real``, and is caught before ``resolve()`` hides it.
    Dangling symlinks (target does not exist) are rejected too: ``is_symlink()``
    returns True for them and existence is not required to detect the redirect.
    """
    # Build the lexical relative path without resolving root itself;
    # anchoring on root.resolve() only to strip the root prefix safely.
    try:
        root_resolved = root.resolve()
    except OSError as exc:
        return f"Cannot resolve root path: {exc}"

    # Compute lexical relative parts: make path absolute relative to root,
    # then strip the resolved root prefix from the resolved path — but walk
    # the *original* (unresolved) parts so symlinks are visible.
    try:
        # Normalise to absolute without following symlinks.
        abs_path = (root / path).resolve() if not path.is_absolute() else path
        lexical_rel = abs_path.relative_to(root_resolved)
    except (ValueError, OSError):
        # Path is already outside root — the caller's relative_to check
        # will handle this; no need to walk components.
        return None

    # Walk lexical components from root using the *unresolved* root as base
    # so intermediate symlinks are visible to is_symlink().
    target = root
    for part in lexical_rel.parts:
        target = target / part
        # Check redirect without requiring target to exist: is_symlink() is
        # True for dangling symlinks, so skip the exists() guard entirely.
        if _is_path_redirect(target):
            return (
                f"Path component '{part}' is a symlink or junction, "
                f"which could redirect operations outside the allowed directory"
            )
    return None


def validate_within_dir(path: Path, root: Path, *,
                        check_symlink_components: bool = False) -> Optional[str]:
    """Ensure *path* resolves to a location within *root*.

    Returns an error message string if validation fails, or ``None`` if the
    path is safe.  Uses ``Path.resolve()`` to follow symlinks and normalize
    ``..`` components.

    When *check_symlink_components* is True, walks the path
    component-by-component and refuses any intermediate symlink/junction
    before the final resolution.  This catches symlink redirects that
    ``Path.resolve()`` would silently follow.

    Usage::

        error = validate_within_dir(user_path, allowed_root)
        if error:
            return json.dumps({"error": error})
    """
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        resolved.relative_to(root_resolved)
    except (ValueError, OSError) as exc:
        return f"Path escapes allowed directory: {exc}"

    if check_symlink_components:
        symlink_error = _has_symlink_component(path, root)
        if symlink_error:
            return symlink_error

    return None


def has_traversal_component(path_str: str) -> bool:
    """Return True if *path_str* contains ``..`` traversal components.

    Quick check for obvious traversal attempts before doing full resolution.
    """
    parts = Path(path_str).parts
    return ".." in parts
