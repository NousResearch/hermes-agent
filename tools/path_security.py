"""Shared path validation helpers for tool implementations.

Extracts the ``resolve() + relative_to()`` and ``..`` traversal check
patterns previously duplicated across skill_manager_tool, skills_tool,
skills_hub, cronjob_tools, and credential_files.
"""

import logging
import urllib.parse
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def validate_within_dir(path: Path, root: Path) -> Optional[str]:
    """Ensure *path* resolves to a location within *root*.

    Returns an error message string if validation fails, or ``None`` if the
    path is safe.  Uses ``Path.resolve()`` to follow symlinks and normalize
    ``..`` components.

    Also rejects:
    - Null bytes in the path string, which truncate paths in C-based OS APIs
      and can lead to unintended file access.
    - URL-encoded traversal sequences (``%2e%2e``, ``%2f``, etc.) that survive
      a plain ``..`` check but resolve to an escape after decoding.

    Usage::

        error = validate_within_dir(user_path, allowed_root)
        if error:
            return json.dumps({"error": error})
    """
    path_str = str(path)

    # Reject null bytes — they truncate paths in C-based OS APIs.
    if "\x00" in path_str or "%00" in path_str.lower():
        return f"Path contains null byte (possible injection attempt): {path}"

    # Reject URL-encoded traversal. pathlib does not decode percent-encoding,
    # so %2e%2e would pass the resolve() check as a literal directory name
    # but could be decoded upstream to ".." by a web layer.
    # Also catches double-encoded sequences (%252e%252e → %2e%2e → ..)
    # mirroring the double-decode logic already in has_traversal_component().
    decoded = urllib.parse.unquote(path_str)
    decoded_twice = urllib.parse.unquote(decoded)
    for _decoded, _label in ((decoded, "URL-decoded"), (decoded_twice, "Double-URL-decoded")):
        if _decoded != path_str:
            if ".." in Path(_decoded).parts:
                return (
                    f"{_label} path contains traversal component: {_decoded!r}. "
                    "Pass a plain (non-percent-encoded) path."
                )

    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        resolved.relative_to(root_resolved)
    except (ValueError, OSError) as exc:
        return f"Path escapes allowed directory: {exc}"
    return None


def has_traversal_component(path_str: str) -> bool:
    """Return True if *path_str* contains ``..`` traversal components.

    Quick check for obvious traversal attempts before doing full resolution.
    Also catches URL-encoded forms (``%2e%2e``, ``%252e%252e``).
    """
    parts = Path(path_str).parts
    if ".." in parts:
        return True

    # URL-decoded checks — catches %2e%2e and double-encoded %252e%252e
    try:
        decoded_once = urllib.parse.unquote(path_str)
        decoded_twice = urllib.parse.unquote(decoded_once)
    except Exception:
        return False

    for decoded in (decoded_once, decoded_twice):
        if decoded != path_str:
            if ".." in Path(decoded).parts:
                return True

    return False
