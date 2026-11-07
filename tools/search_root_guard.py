"""Refuse search_files roots whose recursive walk would read a whole disk.

A root is refused when it IS, or CONTAINS, a broad root: the filesystem root,
the user's home, or an operator-listed root (``HERMES_SEARCH_BROAD_ROOTS``,
``os.pathsep``-separated, plus ``file_search.broad_roots`` in config.yaml).
Pure path arithmetic: the check itself never touches the disk.
"""

import os
from typing import Optional

ENV_VAR = "HERMES_SEARCH_BROAD_ROOTS"


def _config_roots() -> list[str]:
    try:
        from hermes_cli.config import load_config_readonly
        roots = (load_config_readonly().get("file_search") or {}).get("broad_roots") or []
    except Exception:
        return []
    return [str(r) for r in roots] if isinstance(roots, list) else []


def _broad_roots() -> list[str]:
    listed = [r for r in os.environ.get(ENV_VAR, "").split(os.pathsep) if r.strip()]
    roots = ["/", os.path.expanduser("~"), *listed, *_config_roots()]
    return [os.path.normpath(os.path.expanduser(r.strip())) for r in roots]


def broad_root_error(path: str) -> Optional[str]:
    """Refusal message when ``path`` (absolute) is or contains a broad root."""
    if not path or not os.path.isabs(path):
        return None
    target = os.path.normpath(path)
    for root in _broad_roots():
        if os.path.commonpath([target, root]) == target:
            what = "it is a broad root" if target == root else f"it contains the broad root {root}"
            return (f"Refusing to search {target}: {what}, and a recursive walk there reads "
                    "GiBs and starves every other process on the disk. Narrow `path` to the "
                    "task or project directory.")
    return None
