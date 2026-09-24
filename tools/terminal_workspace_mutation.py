"""Detect host Git-workspace mutations made by terminal commands.

The terminal accepts arbitrary shells and scripts, so command parsing cannot say
whether a command writes. Instead, compare cheap Git snapshots around a local
foreground command. This catches scripts and Git commands alike without making
claims about container or remote filesystems Hermes cannot inspect directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hermes_cli._subprocess_compat import bounded_git_probe

_GIT_TIMEOUT = 2.5


@dataclass(frozen=True)
class GitWorkspaceSnapshot:
    root: Path
    head: str
    signatures: dict[str, tuple[int, int]]


def _git(cwd: str | Path, *args: str) -> str:
    return bounded_git_probe(["git", "-C", str(cwd), *args], timeout=_GIT_TIMEOUT)


def _status_paths(root: Path) -> list[str]:
    # Porcelain v2 entries start with a non-whitespace record type. That matters
    # because bounded_git_probe strips outer whitespace from stdout; porcelain
    # v1 starts an unstaged entry with a space and would lose its first byte.
    raw = _git(root, "status", "--porcelain=v2", "-z", "--untracked-files=all")
    if not raw:
        return []
    fields = raw.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if not entry:
            continue
        record_type = entry[0]
        if record_type == "1":
            parts = entry.split(" ", 8)
            if len(parts) == 9:
                paths.append(parts[8])
        elif record_type == "2":
            parts = entry.split(" ", 9)
            if len(parts) == 10:
                paths.append(parts[9])
            if index < len(fields):
                paths.append(fields[index])
                index += 1
        elif record_type == "u":
            parts = entry.split(" ", 10)
            if len(parts) == 11:
                paths.append(parts[10])
        elif record_type == "?":
            paths.append(entry[2:])
    return paths


def _signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (-1, -1)
    return (stat.st_mtime_ns, stat.st_size)


def _head_paths(root: Path, before: str, after: str) -> list[str]:
    if not before or not after or before == after:
        return []
    return [
        path
        for path in _git(root, "diff", "--name-only", "--no-ext-diff", "--no-textconv", before, after).splitlines()
        if path
    ]


def capture_git_workspace(cwd: str | Path | None) -> GitWorkspaceSnapshot | None:
    """Capture a bounded snapshot for the Git workspace containing *cwd*."""
    if not cwd:
        return None
    root_text = _git(cwd, "rev-parse", "--show-toplevel")
    if not root_text:
        return None
    root = Path(root_text).resolve()
    signatures = {
        path: _signature(root / path)
        for path in _status_paths(root)
        if path
    }
    return GitWorkspaceSnapshot(root=root, head=_git(root, "rev-parse", "HEAD"), signatures=signatures)


def detect_git_workspace_mutation(
    before: GitWorkspaceSnapshot | None, cwd: str | Path | None
) -> dict[str, object] | None:
    """Return standard mutation metadata when the workspace changed."""
    if before is None:
        return None
    after = capture_git_workspace(cwd)
    if after is None or after.root != before.root:
        return None

    paths = set(_head_paths(before.root, before.head, after.head))
    all_dirty = set(before.signatures) | set(after.signatures)
    paths.update(
        path for path in all_dirty
        if before.signatures.get(path) != after.signatures.get(path)
    )
    if not paths and before.head == after.head:
        return None
    return {
        "workspace": str(before.root),
        "operation": "unknown",
        "paths": [str((before.root / path).resolve()) for path in sorted(paths)],
    }
