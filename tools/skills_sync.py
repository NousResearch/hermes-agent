#!/usr/bin/env python3
"""
Skills Sync -- Manifest-based seeding of bundled skills into ~/.hermes/skills/.

On fresh install: copies all bundled skills from the repo's skills/ directory
into ~/.hermes/skills/ and records every skill name in a manifest file.

On update: copies only NEW bundled skills (names not in the manifest) so that
user deletions are permanent and user modifications are never overwritten.

The manifest lives at ~/.hermes/skills/.bundled_manifest and is a simple
newline-delimited list of skill names that have been offered to the user.
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple


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


def sync_skills(quiet: bool = False) -> dict:
    """
    Sync bundled skills into ~/.hermes/skills/ using the manifest.

    - Skills whose names are already in the manifest are skipped (even if deleted by user).
    - New skills (not in manifest) are copied to SKILLS_DIR and added to the manifest.

    Returns:
        dict with keys: copied (list of names), skipped (int), total_bundled (int)
    """
    bundled_dir = _get_bundled_dir()
    if not bundled_dir.exists():
        return {"copied": [], "skipped": 0, "total_bundled": 0}

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest()
    bundled_skills = _discover_bundled_skills(bundled_dir)
    copied = []
    skipped = 0

    for skill_name, skill_src in bundled_skills:
        if skill_name in manifest:
            skipped += 1
            continue

        dest = _compute_relative_dest(skill_src, bundled_dir)
        try:
            if dest.exists():
                # Skill dir exists (maybe user created one with same name) -- don't overwrite
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

    # Also copy DESCRIPTION.md files for categories (if not already present)
    for desc_md in bundled_dir.rglob("DESCRIPTION.md"):
        rel = desc_md.relative_to(bundled_dir)
        dest_desc = SKILLS_DIR / rel
        if not dest_desc.exists():
            try:
                dest_desc.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(desc_md, dest_desc)
            except (OSError, IOError):
                pass

    _write_manifest(manifest)

    return {
        "copied": copied,
        "skipped": skipped,
        "total_bundled": len(bundled_skills),
    }


if __name__ == "__main__":
    print("Syncing bundled skills into ~/.hermes/skills/ ...")
    result = sync_skills(quiet=False)
    print(f"\nDone: {len(result['copied'])} new, {result['skipped']} skipped, "
          f"{result['total_bundled']} total bundled.")
