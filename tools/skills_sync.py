#!/usr/bin/env python3
"""
Skills Sync -- Manifest-based seeding and updating of bundled skills.

Copies bundled skills from the repo's skills/ directory into ~/.hermes/skills/
and uses a manifest to track which skills have been offered.

Behavior:
  - NEW skills (not in manifest): copied to user dir, added to manifest.
  - EXISTING skills (in manifest, present in user dir): UPDATED from bundled.
  - DELETED by user (in manifest, absent from user dir): respected -- not re-added.
  - REMOVED from bundled (in manifest, gone from repo): cleaned from manifest.

The manifest lives at ~/.hermes/skills/.bundled_manifest and is a simple
newline-delimited list of skill names that have been offered to the user.
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
SKILLS_DIR = HERMES_HOME / "skills"
MANIFEST_FILE = SKILLS_DIR / ".bundled_manifest"


def _get_bundled_dir() -> Path:
    """Locate the bundled skills/ directory in the repo."""
    return Path(__file__).parent.parent / "skills"


def _read_manifest() -> set:
    """Read the set of skill names already offered to the user."""
    if not MANIFEST_FILE.exists():
        return set()
    try:
        return set(
            line.strip()
            for line in MANIFEST_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, IOError):
        return set()


def _write_manifest(names: set):
    """Write the manifest file."""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(
        "\n".join(sorted(names)) + "\n",
        encoding="utf-8",
    )


def _discover_bundled_skills(bundled_dir: Path) -> List[Tuple[str, Path]]:
    """
    Find all SKILL.md files in the bundled directory.
    Returns list of (skill_name, skill_directory_path) tuples.
    """
    skills = []
    if not bundled_dir.exists():
        return skills

    for skill_md in bundled_dir.rglob("SKILL.md"):
        path_str = str(skill_md)
        if "/.git/" in path_str or "/.github/" in path_str or "/.hub/" in path_str:
            continue
        skill_dir = skill_md.parent
        skill_name = skill_dir.name
        skills.append((skill_name, skill_dir))

    return skills


def _compute_relative_dest(skill_dir: Path, bundled_dir: Path) -> Path:
    """
    Compute the destination path in SKILLS_DIR preserving the category structure.
    e.g., bundled/skills/mlops/axolotl -> ~/.hermes/skills/mlops/axolotl
    """
    rel = skill_dir.relative_to(bundled_dir)
    return SKILLS_DIR / rel


def _dir_hash(directory: Path) -> str:
    """Compute a hash of all file contents in a directory for change detection."""
    hasher = hashlib.md5()
    try:
        for fpath in sorted(directory.rglob("*")):
            if fpath.is_file():
                rel = fpath.relative_to(directory)
                hasher.update(str(rel).encode("utf-8"))
                hasher.update(fpath.read_bytes())
    except (OSError, IOError):
        pass
    return hasher.hexdigest()


def sync_skills(quiet: bool = False) -> dict:
    """
    Sync bundled skills into ~/.hermes/skills/ using the manifest.

    - NEW skills (not in manifest): copied to user dir, added to manifest.
    - EXISTING skills (in manifest, present in user dir): updated from bundled.
    - DELETED by user (in manifest, absent from user dir): respected, not re-added.
    - REMOVED from bundled (in manifest, gone from repo): cleaned from manifest.

    Returns:
        dict with keys: copied (list), updated (list), skipped (int),
                        cleaned (list), total_bundled (int)
    """
    bundled_dir = _get_bundled_dir()
    if not bundled_dir.exists():
        return {"copied": [], "updated": [], "skipped": 0, "cleaned": [], "total_bundled": 0}

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest()
    bundled_skills = _discover_bundled_skills(bundled_dir)
    bundled_names = {name for name, _ in bundled_skills}

    copied = []
    updated = []
    skipped = 0

    for skill_name, skill_src in bundled_skills:
        dest = _compute_relative_dest(skill_src, bundled_dir)

        if skill_name not in manifest:
            # New skill -- never offered before
            try:
                if dest.exists():
                    # User already has a skill with the same name (unlikely but possible)
                    skipped += 1
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(skill_src, dest)
                    copied.append(skill_name)
                    if not quiet:
                        print(f"  + {skill_name}")
            except (OSError, IOError) as e:
                if not quiet:
                    print(f"  ! Failed to copy {skill_name}: {e}")
            manifest.add(skill_name)

        elif dest.exists():
            # Existing skill in manifest AND on disk -- check for updates
            src_hash = _dir_hash(skill_src)
            dst_hash = _dir_hash(dest)
            if src_hash != dst_hash:
                try:
                    shutil.rmtree(dest)
                    shutil.copytree(skill_src, dest)
                    updated.append(skill_name)
                    if not quiet:
                        print(f"  ↑ {skill_name} (updated)")
                except (OSError, IOError) as e:
                    if not quiet:
                        print(f"  ! Failed to update {skill_name}: {e}")
            else:
                skipped += 1

        else:
            # In manifest but not on disk -- user deleted it, respect that
            skipped += 1

    # Clean stale manifest entries (skills removed from bundled dir)
    cleaned = sorted(manifest - bundled_names)
    manifest -= set(cleaned)

    # Also copy DESCRIPTION.md files for categories (if not already present)
    for desc_md in bundled_dir.rglob("DESCRIPTION.md"):
        rel = desc_md.relative_to(bundled_dir)
        dest_desc = SKILLS_DIR / rel
        if not dest_desc.exists():
            try:
                dest_desc.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(desc_md, dest_desc)
            except (OSError, IOError) as e:
                logger.debug("Could not copy %s: %s", desc_md, e)

    _write_manifest(manifest)

    return {
        "copied": copied,
        "updated": updated,
        "skipped": skipped,
        "cleaned": cleaned,
        "total_bundled": len(bundled_skills),
    }


if __name__ == "__main__":
    print("Syncing bundled skills into ~/.hermes/skills/ ...")
    result = sync_skills(quiet=False)
    print(f"\nDone: {len(result['copied'])} new, {len(result['updated'])} updated, "
          f"{result['skipped']} unchanged, {len(result['cleaned'])} cleaned from manifest, "
          f"{result['total_bundled']} total bundled.")
