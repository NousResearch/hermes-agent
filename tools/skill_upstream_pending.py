"""Stage a refused autonomous write to a bundled/hub skill as an upstream patch.

The background-review fork improves the skills it learns from, but a bundled or hub-installed
skill's local copy IS the upstream copy: ``tools/skills_sync._dir_hash`` matches that copy by an
exact hash, so a local write detaches the skill from upstream (``user_modified``) and no later
update can reconcile it. Refusing the write would throw the fork's lesson away, so the change is
diverted here instead — a human-readable unified diff under
``<HERMES_HOME>/pending/skill-upstream/<skill>/`` that reaches every install once it lands
upstream.
"""

from __future__ import annotations

import difflib
import time
from pathlib import Path
from typing import Any, Optional

_PENDING_PARTS = ("pending", "skill-upstream")


def _pending_dir(name: str) -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home().joinpath(*_PENDING_PARTS, name)


def _read_text(path: Path) -> str:
    """Current file text, ``""`` for a file that is not there (a new supporting file)."""
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""


def _unified_diff(relative_path: str, old: str, new: str) -> str:
    """Unified diff with ``a/``/``b/`` prefixes, so ``git apply -p1`` accepts it as-is."""
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{relative_path}", tofile=f"b/{relative_path}"))


def intended_change(action: str, skill_dir: Path, *, content: Optional[str] = None,
                    file_path: Optional[str] = None, file_content: Optional[str] = None,
                    old_string: Optional[str] = None, new_string: Optional[str] = None,
                    replace_all: bool = False):
    """``(relative_path, old_text, new_text)`` for the write that was refused, or ``None`` when the
    intended content cannot be reconstructed — an unmatchable patch has no content to stage."""
    relative = (file_path or "SKILL.md") if action in {"edit", "patch"} else file_path
    if not relative:
        return None
    old = _read_text(skill_dir / relative)
    if action == "edit":
        return None if content is None else (relative, old, content)
    if action == "write_file":
        return None if file_content is None else (relative, old, file_content)
    if action == "remove_file":
        return relative, old, ""
    if action == "patch":
        if not old_string or new_string is None:
            return None
        from tools.fuzzy_match import fuzzy_find_and_replace
        new, _count, _strategy, match_error = fuzzy_find_and_replace(
            old, old_string, new_string, replace_all)
        return None if match_error else (relative, old, new)
    return None


def _header(name: str, label: str, action: str, relative_path: str, filename: str) -> str:
    return (
        f"# Upstream patch staged by Hermes' background review ({action} on skill '{name}').\n"
        f"#\n"
        f"# '{name}' is {label}, so the local copy is upstream's, matched by an exact hash —\n"
        f"# writing it here would strand the skill as `user_modified`. The review fork's lesson\n"
        f"# is kept in this patch instead of being dropped.\n"
        f"#\n"
        f"# Apply from the skill's directory in the upstream checkout:\n"
        f"#   git apply -p1 {filename}    ({relative_path})\n"
        f"#\n")


def stage_write_as_upstream_patch(name: str, action: str, skill_dir: Path, *, label: str = "bundled",
                                  **write_args: Any) -> Optional[Path]:
    """Write the refused write to ``<HERMES_HOME>/pending/skill-upstream/<name>/<stamp>.patch``.

    Returns the patch path, or ``None`` when there is nothing to stage (the intended content could
    not be reconstructed, or the write would have been a no-op).
    """
    change = intended_change(action, skill_dir, **write_args)
    if change is None:
        return None
    relative_path, old, new = change
    diff = _unified_diff(relative_path, old, new)
    if not diff:
        return None
    from hermes_constants import mkdir_under_hermes_home
    from utils import atomic_write_bytes
    directory = mkdir_under_hermes_home(_pending_dir(name))
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = directory / f"{stamp}.patch"
    counter = 1
    while path.exists():  # two refusals in the same second must not overwrite each other
        counter += 1
        path = directory / f"{stamp}-{counter}.patch"
    # Bytes, not text: a CRLF patch file is what `git apply` complains about, and the EOL of a
    # diff is part of its format.
    atomic_write_bytes(path, (_header(name, label, action, relative_path, path.name) + diff).encode("utf-8"))
    return path
