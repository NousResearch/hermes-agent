"""Shared path validation helpers for tool implementations (skills, cron, credential files)."""

import re
from pathlib import Path
from typing import Optional


def validate_within_dir(path: Path, root: Path) -> Optional[str]:
    """Error message if *path* does not resolve inside *root* (symlinks and ``..`` followed)."""
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError) as exc:
        return f"Path escapes allowed directory: {exc}"
    return None


def has_traversal_component(path_str: str) -> bool:
    """Cheap pre-check for a literal ``..`` component before full resolution."""
    return ".." in Path(path_str).parts


# Control chars + Unicode line separators (NEL, LS, PS): newline-bearing paths are legal POSIX
# names, but they corrupt line-delimited protocols (MEDIA: tags) and forge log lines. Same
# class as _LOG_UNSAFE_CHARS in gateway.platforms.base.
_UNSAFE_PATH_CHARS = re.compile(r"[\x00-\x1f\x7f\x85\u2028\u2029]")


def has_unsafe_path_chars(path_str: str) -> bool:
    """True when *path_str* contains control characters or line separators."""
    return bool(_UNSAFE_PATH_CHARS.search(path_str))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import logging  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'logger': ('tools.approval', 'logger'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
