"""Safe link-mode primitives for bundled-skill synchronization.

This module owns source qualification, manifest link metadata, and link-to-copy
transitions.  Callers must pass the active skills root and copy function so the
core sync module remains the authority for profile-scoped paths and cache
filtering.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Callable, Optional, Tuple

SYMLINK_SENTINEL = "@symlink"


def is_link_manifest_entry(value: str) -> bool:
    """Return whether a decoded manifest value represents link ownership."""
    return value == SYMLINK_SENTINEL or value.startswith(f"{SYMLINK_SENTINEL}:")


def link_manifest_value(target: Optional[Path] = None) -> str:
    """Encode a link entry, optionally retaining its normalized target."""
    if target is None:
        return SYMLINK_SENTINEL
    return f"{SYMLINK_SENTINEL}:{target.resolve(strict=False)}"


def link_manifest_target(value: str) -> Optional[Path]:
    """Decode the recorded target from a link entry, if this manifest has one."""
    if not is_link_manifest_entry(value) or value == SYMLINK_SENTINEL:
        return None
    return Path(value.removeprefix(f"{SYMLINK_SENTINEL}:"))


def parse_manifest_value(mode: str, payload: str) -> str:
    """Decode one v3 mode/payload pair into the core manifest representation."""
    mode, payload = mode.strip(), payload.strip()
    if mode != "symlink":
        return payload
    return link_manifest_value(Path(payload)) if payload else SYMLINK_SENTINEL


def format_manifest_value(value: str) -> Tuple[str, str]:
    """Encode the core representation as one v3 mode/payload pair."""
    if is_link_manifest_entry(value):
        target = link_manifest_target(value)
        return "symlink", str(target) if target is not None else ""
    return "copy", value


def _link_target(dest: Path) -> Path:
    """Return a symlink's normalized absolute target without requiring it to exist."""
    raw = Path(os.readlink(dest))
    return (raw if raw.is_absolute() else dest.parent / raw).resolve(strict=False)


def owns_symlink(
    dest: Path, manifest_value: str, expected_source: Optional[Path] = None
) -> bool:
    """Prove that ``dest`` is the link recorded by the manifest.

    Target-bearing v3 entries remain provable after the target becomes dangling.
    Legacy target-less v3 entries are accepted only while they still point at the
    currently expected bundled source.  A user-retargeted link is never owned.
    """
    if not dest.is_symlink() or not is_link_manifest_entry(manifest_value):
        return False
    try:
        actual = _link_target(dest)
        recorded = link_manifest_target(manifest_value)
        expected = (
            expected_source.resolve(strict=False)
            if expected_source is not None
            else None
        )
    except (OSError, RuntimeError):
        return False
    return actual == (
        recorded.resolve(strict=False) if recorded is not None else expected
    )


def _git_toplevel(path: Path) -> Optional[Path]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).resolve(strict=False)


def can_link_from_source(source_dir: Path, skills_dir: Path) -> bool:
    """Allow links only from a persistent Git tree disjoint from the active skills tree."""
    try:
        source = source_dir.resolve(strict=True)
        destination = skills_dir.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    if (
        not source.is_dir()
        or source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        return False
    root = _git_toplevel(source)
    return root is not None and (source == root or source.is_relative_to(root))


def git_is_dirty(path: Path) -> bool:
    """Conservatively report dirty when Git cannot prove the source tree clean."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return result.returncode != 0 or bool(result.stdout.strip())


def create_symlink(source: Path, dest: Path) -> str:
    """Create a directory link and return its target-bearing manifest value."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    target = source.resolve(strict=True)
    os.symlink(target, dest, target_is_directory=True)
    return link_manifest_value(target)


def retarget_owned_symlink(dest: Path, target: Path, manifest_value: str) -> str:
    """Retarget a proven-owned link, restoring its old target if creation fails."""
    if not owns_symlink(dest, manifest_value):
        raise ValueError(f"refusing to retarget unowned symlink at {dest}")
    old_target = os.readlink(dest)
    new_target = target.resolve(strict=True)
    dest.unlink()
    try:
        os.symlink(new_target, dest, target_is_directory=True)
    except OSError:
        with suppress(OSError):
            os.symlink(old_target, dest, target_is_directory=True)
        raise
    return link_manifest_value(new_target)


def replace_owned_symlink_with_copy(
    source: Path,
    dest: Path,
    manifest_value: str,
    copy_dir: Callable[[Path, Path], None],
) -> None:
    """Stage a copy, then atomically replace a proven-owned link.

    The link is checked both before and after staging.  If the final rename
    fails, its exact prior target is restored for a later retry.
    """
    if not owns_symlink(dest, manifest_value):
        raise ValueError(f"refusing to replace unowned symlink at {dest}")
    stage_root = Path(
        tempfile.mkdtemp(prefix=f".{dest.name}-link-copy-", dir=dest.parent)
    )
    staged = stage_root / dest.name
    old_target = os.readlink(dest)
    try:
        copy_dir(source, staged)
        if not owns_symlink(dest, manifest_value):
            raise ValueError(f"symlink changed while staging copy at {dest}")
        dest.unlink()
        try:
            staged.rename(dest)
        except OSError:
            with suppress(OSError):
                os.symlink(old_target, dest, target_is_directory=True)
            raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
