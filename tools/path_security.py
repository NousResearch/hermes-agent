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
