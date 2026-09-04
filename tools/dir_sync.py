"""In-place directory synchronization.

Refreshes the contents of *dst* from *src* without ever removing or
recreating *dst* itself.  A directory inode must stay stable when it is
bind-mounted into a long-lived sandbox (e.g. ``skills/`` in a persistent
Docker container): deleting and recreating the source directory empties
the mount and the sandbox sees a dead directory (hermes-agent#53630,
#73842).
"""

from __future__ import annotations

import logging
import os
import shutil
import stat as stat_module
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def sync_dir_in_place(
    src: Path,
    dst: Path,
    skip_names: set[str] | frozenset[str] | None = None,
) -> None:
    """Mirror *src* into *dst*, preserving the *dst* directory inode.

    Guarantees:

    - ``dst`` itself is never removed, recreated, or replaced — only its
      contents change, so existing bind mounts keep working.  (The inode
      guarantee is for the top-level *dst* directory; sub-directories may
      legitimately be removed or recreated when they go stale or change
      type between syncs.)
    - Symlinks found in *dst* are removed (a mirror must never contain
      symlinks, otherwise a later copy could write through one).
    - Symlinks in *src*: file symlinks are dereferenced (their content is
      copied as a regular file, matching the historical ``copytree``
      behavior); symlinks to directories are skipped (the historical
      ``copytree`` failed the whole install on those).
    - File <-> directory type transitions are handled by removing the
      stale entry before writing the new one.
    - Files are written atomically (same-directory temp + rename), so a
      concurrent reader never observes a partially written file.
    - Unchanged files (same mtime and size) are left untouched.
    - Empty directories left behind after deletions are pruned.

    *skip_names* (optional) excludes top-level entries of *src* by name,
    mirroring the caller's ownership rules.
    """
    skip = set(skip_names) if skip_names else set()
    if src.resolve() == dst.resolve():
        raise ValueError(f"sync_dir_in_place: src and dst resolve to the same path ({src})")
    dst.mkdir(parents=True, exist_ok=True)

    # Phase 1: expected state (rel path -> "file" | "dir") from src.
    expected: dict[str, str] = {}
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        if rel.parts and rel.parts[0] in skip:
            continue
        if item.is_symlink():
            try:
                st = item.stat()  # follows the link
            except OSError:
                continue  # dangling link — drop it
            if stat_module.S_ISDIR(st.st_mode):
                continue  # directory symlinks unsupported — drop it
            expected[rel.as_posix()] = "file"
            continue
        expected[rel.as_posix()] = "dir" if item.is_dir() else "file"

    # Phase 2: remove stale, type-conflicting, or symlinked entries from dst.
    existing = sorted(dst.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    for entry in existing:
        rel = entry.relative_to(dst)
        if rel.parts and rel.parts[0] in skip:
            # Skipped top-level entry (user-owned): never touched, so its
            # contents survive the sync.
            continue
        if entry.is_symlink():
            # Never allow symlinks to survive in the mirror — a later copy
            # would write through them to their external targets.
            entry.unlink(missing_ok=True)
            continue
        expected_kind = expected.get(rel.as_posix())
        if expected_kind is None:
            # Stale entry — not present in src anymore.
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
        elif expected_kind == "file" and entry.is_dir():
            # dir -> file transition.
            shutil.rmtree(entry, ignore_errors=True)
        elif expected_kind == "dir" and not entry.is_dir():
            # file -> dir transition.
            entry.unlink(missing_ok=True)

    # Phase 3: copy current src entries into dst.
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        if rel.parts and rel.parts[0] in skip:
            continue
        target = dst / rel
        if item.is_symlink():
            try:
                st = item.stat()
            except OSError:
                continue  # dangling link — nothing to copy
            if stat_module.S_ISDIR(st.st_mode):
                continue  # directory symlinks unsupported — skip
            # File symlink: fall through — copy2 dereferences it into a
            # regular file, matching historical copytree behavior.
        elif item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        elif not item.is_file():
            continue
        # Skip files whose mtime + size are unchanged (avoids redundant I/O
        # on large trees such as skills/).
        try:
            if target.is_file():
                src_stat = item.stat()
                dst_stat = target.stat()
                if src_stat.st_mtime == dst_stat.st_mtime and src_stat.st_size == dst_stat.st_size:
                    continue
        except OSError:
            pass
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: same-directory temp file + rename, so concurrent
        # readers (running sandboxes, agent subprocesses) never observe a
        # partially written file.
        fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=".hermes-sync-", suffix=".tmp")
        os.close(fd)
        try:
            shutil.copy2(item, tmp_path)
            os.replace(tmp_path, target)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # Phase 4: prune empty directories that have no counterpart in src.
    for dirpath in sorted(
        (p for p in dst.rglob("*") if p.is_dir() and not p.is_symlink()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        rel = dirpath.relative_to(dst)
        if rel.parts and rel.parts[0] in skip:
            continue
        if expected.get(rel.as_posix()) == "dir":
            continue
        try:
            dirpath.rmdir()
        except OSError:
            pass
