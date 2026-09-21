"""Near-duplicate hint for write_file: a NEW file whose lines mostly match an existing sibling.

A model that cannot anchor a patch, or lost track of a file it already wrote, tends to create
a parallel copy next to the original (``utils2.py``, ``handler_new.ts``, ``config.v2.yaml``).
The write succeeds, nothing is red, and the repo now has two drifting copies. Inspired by
Muse Code's Write tool, which flags when a new file nearly duplicates an existing one.

Advisory only: the write always proceeds; the hint rides on the success result. Runs only for
the local environment (the host filesystem is the write target there) and only for files that
did not exist before the write. Same-directory, same-extension siblings within a size band are
compared as line multisets (linear), never with a sequence diff (quadratic on repeated lines).
"""

from __future__ import annotations

import os
from collections import Counter

_MIN_CHARS = 200
_MAX_CHARS = 200_000
_MIN_LINES = 5
# A sibling more than this far from the new file's size cannot reach the similarity floor.
_SIZE_TOLERANCE = 0.35
# Closest-size siblings actually read; bounds the I/O on a directory of hundreds of files.
_MAX_CANDIDATES = 24
_MIN_SIMILARITY = 0.85


def _is_local_env(file_ops) -> bool:
    from tools.environments.local import LocalEnvironment
    return isinstance(getattr(file_ops, "env", None), LocalEnvironment)


def find_near_duplicate(resolved: str, new_content: str) -> tuple[str, float, int] | None:
    """``(sibling_name, similarity, shared_lines)`` for the most similar same-extension sibling of a
    NOT-yet-existing ``resolved`` path when it clears the floor; else None. Pure host-filesystem logic."""
    if not resolved or os.path.lexists(resolved) or not (_MIN_CHARS <= len(new_content) <= _MAX_CHARS):
        return None
    new_lines = new_content.splitlines()
    if len(new_lines) < _MIN_LINES:
        return None
    directory, name = os.path.split(resolved)
    ext = os.path.splitext(name)[1]
    if not ext:  # extensionless siblings (Makefile, LICENSE, Dockerfile) rarely mean a copy
        return None
    approx_size = len(new_content.encode("utf-8", errors="replace"))
    low, high = approx_size * (1 - _SIZE_TOLERANCE), approx_size * (1 + _SIZE_TOLERANCE)
    candidates: list[tuple[int, str]] = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name == name or entry.name.startswith(".") or not entry.name.endswith(ext):
                    continue
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    size = entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
                if low <= size <= high:
                    candidates.append((abs(size - approx_size), entry.path))
    except OSError:
        return None
    candidates.sort()
    new_counter, n_new = Counter(new_lines), len(new_lines)
    best: tuple[str, float, int] | None = None
    for _, sibling in candidates[:_MAX_CANDIDATES]:
        try:
            with open(sibling, "rb") as fh:
                raw = fh.read(_MAX_CHARS * 4)
        except OSError:
            continue
        if b"\x00" in raw:
            continue
        old_lines = raw.decode("utf-8", errors="replace").splitlines()
        if not old_lines:
            continue
        shared = sum((Counter(old_lines) & new_counter).values())
        ratio = shared / max(len(old_lines), n_new)
        if ratio >= _MIN_SIMILARITY and (best is None or ratio > best[1]):
            best = (os.path.basename(sibling), ratio, shared)
    return best


def near_duplicate_hint(file_ops, resolved: str | None, new_content: str) -> str | None:
    """Hint text for write_file's result, or None. Call BEFORE the write (existence is the gate)."""
    if not resolved or not _is_local_env(file_ops):
        return None
    try:
        match = find_near_duplicate(resolved, new_content)
    except Exception:
        return None
    if not match:
        return None
    sibling, ratio, shared = match
    return (
        f"New file {os.path.basename(resolved)!r} is {ratio:.0%} line-identical to existing sibling {sibling!r} "
        f"({shared:,} shared lines). If this was meant to replace or extend that file, edit it with patch instead "
        "of leaving a parallel copy; if both files are intended, ignore this."
    )
