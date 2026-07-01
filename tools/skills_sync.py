#!/usr/bin/env python3
"""
Skills Sync -- Manifest-based seeding and updating of bundled skills.

Copies bundled skills from the repo's skills/ directory into ~/.hermes/skills/
and uses a manifest to track which skills have been synced and their origin hash.

Manifest format (v2): each line is "skill_name:origin_hash" where origin_hash
is the MD5 of the bundled skill at the time it was last synced to the user dir.
Old v1 manifests (plain names without hashes) are auto-migrated.

Update logic:
  - NEW skills (not in manifest): copied to user dir, origin hash recorded.
  - EXISTING skills (in manifest, present in user dir):
      * If user copy matches origin hash: user hasn't modified it → safe to
        update from bundled if bundled changed. New origin hash recorded.
      * If user copy differs from origin hash: user customized it → SKIP.
  - DELETED by user (in manifest, absent from user dir): respected, not re-added.
  - REMOVED from bundled (in manifest, gone from repo): cleaned from manifest.

The manifest lives at ~/.hermes/skills/.bundled_manifest.
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from hermes_constants import get_bundled_skills_dir, get_hermes_home
from agent.skill_utils import is_excluded_skill_path
from typing import Dict, List, Tuple
from utils import atomic_replace

logger = logging.getLogger(__name__)


HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
MANIFEST_FILE = SKILLS_DIR / ".bundled_manifest"


def _get_bundled_dir() -> Path:
    """Locate the bundled skills/ directory.

    Checks HERMES_BUNDLED_SKILLS env var first (set by Nix wrapper),
    then a wheel-installed data dir, then falls back to the relative
    path from this source file.
    """
    return get_bundled_skills_dir(Path(__file__).parent.parent / "skills")


def _read_manifest() -> Dict[str, str]:
    """
    Read the manifest as a dict of {skill_name: origin_hash}.

    Handles both v1 (plain names) and v2 (name:hash) formats.
    v1 entries get an empty hash string which triggers migration on next sync.
    """
    if not MANIFEST_FILE.exists():
        return {}
    try:
        result = {}
        for line in MANIFEST_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                # v2 format: name:hash
                name, _, hash_val = line.partition(":")
                result[name.strip()] = hash_val.strip()
            else:
                # v1 format: plain name — empty hash triggers migration
                result[line] = ""
        return result
    except (OSError, IOError):
        return {}


def _write_manifest(entries: Dict[str, str]):
    """Write the manifest file atomically in v2 format (name:hash).

    Uses a temp file + os.replace() to avoid corruption if the process
    crashes or is interrupted mid-write.
    """
    import tempfile

    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = "\n".join(f"{name}:{hash_val}" for name, hash_val in sorted(entries.items())) + "\n"

    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(MANIFEST_FILE.parent),
            prefix=".bundled_manifest_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            atomic_replace(tmp_path, MANIFEST_FILE)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("Failed to write skills manifest %s: %s", MANIFEST_FILE, e, exc_info=True)


def _read_skill_name(skill_md: Path, fallback: str) -> str:
    """Read the name field from SKILL.md YAML frontmatter, falling back to *fallback*."""
    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return fallback
    in_frontmatter = False
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped == "---":
            if in_frontmatter:
                break
            in_frontmatter = True
            continue
        if in_frontmatter and stripped.startswith("name:"):
            value = stripped.split(":", 1)[1].strip().strip("\"'")
            if value:
                return value
    return fallback


def _discover_bundled_skills(bundled_dir: Path) -> List[Tuple[str, Path]]:
    """
    Find all SKILL.md files in the bundled directory.
    Returns list of (skill_name, skill_directory_path) tuples.
    """
    skills = []
    if not bundled_dir.exists():
        return skills

    for skill_md in bundled_dir.rglob("SKILL.md"):
        if is_excluded_skill_path(skill_md):
            continue
        skill_dir = skill_md.parent
        skill_name = _read_skill_name(skill_md, skill_dir.name)
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

    Returns:
        dict with keys: copied (list), updated (list), skipped (int),
                        user_modified (list), cleaned (list), total_bundled (int)
    """
    bundled_dir = _get_bundled_dir()
    if not bundled_dir.exists():
        return {
            "copied": [], "updated": [], "skipped": 0,
            "user_modified": [], "cleaned": [], "total_bundled": 0,
        }

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest()
    bundled_skills = _discover_bundled_skills(bundled_dir)
    bundled_names = {name for name, _ in bundled_skills}

    copied = []
    updated = []
    user_modified = []
    skipped = 0

    for skill_name, skill_src in bundled_skills:
        dest = _compute_relative_dest(skill_src, bundled_dir)
        bundled_hash = _dir_hash(skill_src)

        if skill_name not in manifest:
            # ── New skill — never offered before ──
            try:
                if dest.exists():
                    # User already has a skill with the same name — don't overwrite.
                    # Only baseline in the manifest when the on-disk copy is
                    # byte-identical to bundled (e.g. a reset that re-syncs, or
                    # a coincidentally identical install); that case is harmless
                    # to track. If the copy differs (custom skill, hub-installed,
                    # or user-edited) skip the manifest write: recording
                    # bundled_hash there would poison update detection by making
                    # user_hash != origin_hash read as "user-modified" on every
                    # subsequent sync, permanently blocking bundled updates.
                    skipped += 1
                    if _dir_hash(dest) == bundled_hash:
                        manifest[skill_name] = bundled_hash
                    elif not quiet:
                        print(
                            f"  ⚠ {skill_name}: bundled version shipped but you "
                            f"already have a local skill by this name — yours "
                            f"was kept. Run `hermes skills reset {skill_name}` "
                            f"to replace it with the bundled version."
                        )
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(skill_src, dest)
                    copied.append(skill_name)
                    manifest[skill_name] = bundled_hash
                    if not quiet:
                        print(f"  + {skill_name}")
            except (OSError, IOError) as e:
                if not quiet:
                    print(f"  ! Failed to copy {skill_name}: {e}")
                # Do NOT add to manifest — next sync should retry

        elif dest.exists():
            # ── Existing skill — in manifest AND on disk ──
            origin_hash = manifest.get(skill_name, "")
            user_hash = _dir_hash(dest)

            if not origin_hash:
                # v1 migration: no origin hash recorded. Set baseline from
                # user's current copy so future syncs can detect modifications.
                manifest[skill_name] = user_hash
                if user_hash == bundled_hash:
                    skipped += 1  # already in sync
                else:
                    # Can't tell if user modified or bundled changed — be safe
                    skipped += 1
                continue

            if user_hash != origin_hash:
                # User modified this skill — don't overwrite their changes
                user_modified.append(skill_name)
                if not quiet:
                    print(f"  ~ {skill_name} (user-modified, skipping)")
                continue

            # User copy matches origin — check if bundled has a newer version
            if bundled_hash != origin_hash:
                try:
                    # Move old copy to a backup so we can restore on failure
                    backup = dest.with_suffix(".bak")
                    shutil.move(str(dest), str(backup))
                    try:
                        shutil.copytree(skill_src, dest)
                        manifest[skill_name] = bundled_hash
                        updated.append(skill_name)
                        if not quiet:
                            print(f"  ↑ {skill_name} (updated)")
                        # Remove backup after successful copy
                        shutil.rmtree(backup, ignore_errors=True)
                    except (OSError, IOError):
                        # Restore from backup
                        if backup.exists() and not dest.exists():
                            shutil.move(str(backup), str(dest))
                        raise
                except (OSError, IOError) as e:
                    if not quiet:
                        print(f"  ! Failed to update {skill_name}: {e}")
            else:
                skipped += 1  # bundled unchanged, user unchanged

        else:
            # ── In manifest but not on disk — user deleted it ──
            skipped += 1

    # Clean stale manifest entries (skills removed from bundled dir)
    cleaned = sorted(set(manifest.keys()) - bundled_names)
    for name in cleaned:
        del manifest[name]

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

    # Deploy scripts from skill scripts/ subdirectories to ~/.hermes/scripts/
    _scripts_home = HERMES_HOME / "scripts"
    try:
        deploy_skill_scripts(SKILLS_DIR, _scripts_home)
    except Exception:
        pass  # non-fatal

    return {
        "copied": copied,
        "updated": updated,
        "skipped": skipped,
        "user_modified": user_modified,
        "cleaned": cleaned,
        "total_bundled": len(bundled_skills),
    }


def deploy_skill_scripts(skills_home: Path, scripts_dir: Path) -> int:
    """Deploy or update scripts from skill scripts/ subdirectories to ~/.hermes/scripts/.

    Scans all installed skills for a scripts/ subdirectory and copies any .py
    files found into *scripts_dir*.  Files are only written when the content
    has changed, making repeated calls fast.

    Args:
        skills_home: Path to ~/.hermes/skills (or equivalent).
        scripts_dir: Path to ~/.hermes/scripts (or equivalent).

    Returns:
        Number of scripts deployed or updated this call.
    """
    import hashlib

    def _file_md5(path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

    scripts_dir.mkdir(parents=True, exist_ok=True)
    deployed = 0

    # Scan both flat (skills_home/<skill>/scripts/) and
    # category-nested (skills_home/<cat>/<skill>/scripts/) layouts.
    for scripts_subdir in sorted(skills_home.rglob("scripts")):
        if not scripts_subdir.is_dir():
            continue
        for src in sorted(scripts_subdir.glob("*.py")):
            dst = scripts_dir / src.name
            if dst.exists() and _file_md5(src) == _file_md5(dst):
                continue  # unchanged — skip
            try:
                import shutil
                shutil.copy2(src, dst)
                dst.chmod(0o755)
                deployed += 1
            except OSError:
                pass  # non-fatal — log if logger is available

    return deployed


def reset_bundled_skill(name: str, restore: bool = False) -> dict:
    """
    Reset a bundled skill's manifest tracking so future syncs work normally.

    When a user edits a bundled skill, subsequent syncs mark it as
    ``user_modified`` and skip it forever — even if the user later copies
    the bundled version back into place, because the manifest still holds
    the *old* origin hash. This function breaks that loop.

    Args:
        name: The skill name (matches the manifest key / skill frontmatter name).
        restore: If True, also delete the user's copy in SKILLS_DIR and let
                 the next sync re-copy the current bundled version. If False
                 (default), only clear the manifest entry — the user's
                 current copy is preserved but future updates work again.

    Returns:
        dict with keys:
          - ok: bool, whether the reset succeeded
          - action: one of "manifest_cleared", "restored", "not_in_manifest",
                    "bundled_missing"
          - message: human-readable description
          - synced: dict from sync_skills() if a sync was triggered, else None
    """
    manifest = _read_manifest()
    bundled_dir = _get_bundled_dir()
    bundled_skills = _discover_bundled_skills(bundled_dir)
    bundled_by_name = dict(bundled_skills)

    in_manifest = name in manifest
    is_bundled = name in bundled_by_name

    if not in_manifest and not is_bundled:
        return {
            "ok": False,
            "action": "not_in_manifest",
            "message": (
                f"'{name}' is not a tracked bundled skill. Nothing to reset. "
                f"(Hub-installed skills use `hermes skills uninstall`.)"
            ),
            "synced": None,
        }

    # Step 1: drop the manifest entry so next sync treats it as new
    if in_manifest:
        del manifest[name]
        _write_manifest(manifest)

    # Step 2 (optional): delete the user's copy so next sync re-copies bundled
    deleted_user_copy = False
    if restore:
        if not is_bundled:
            return {
                "ok": False,
                "action": "bundled_missing",
                "message": (
                    f"'{name}' has no bundled source — manifest entry cleared "
                    f"but cannot restore from bundled (skill was removed upstream)."
                ),
                "synced": None,
            }
        # The destination mirrors the bundled path relative to bundled_dir.
        dest = _compute_relative_dest(bundled_by_name[name], bundled_dir)
        if dest.exists():
            try:
                shutil.rmtree(dest)
                deleted_user_copy = True
            except (OSError, IOError) as e:
                return {
                    "ok": False,
                    "action": "manifest_cleared",
                    "message": (
                        f"Cleared manifest entry for '{name}' but could not "
                        f"delete user copy at {dest}: {e}"
                    ),
                    "synced": None,
                }

    # Step 3: run sync to re-baseline (or re-copy if we deleted)
    synced = sync_skills(quiet=True)

    if restore and deleted_user_copy:
        action = "restored"
        message = f"Restored '{name}' from bundled source."
    elif restore:
        # Nothing on disk to delete, but we re-synced — acts like a fresh install
        action = "restored"
        message = f"Restored '{name}' (no prior user copy, re-copied from bundled)."
    else:
        action = "manifest_cleared"
        message = (
            f"Cleared manifest entry for '{name}'. Future `hermes update` runs "
            f"will re-baseline against your current copy and accept upstream changes."
        )

    return {"ok": True, "action": action, "message": message, "synced": synced}


if __name__ == "__main__":
    print("Syncing bundled skills into ~/.hermes/skills/ ...")
    result = sync_skills(quiet=False)
    parts = [
        f"{len(result['copied'])} new",
        f"{len(result['updated'])} updated",
        f"{result['skipped']} unchanged",
    ]
    if result["user_modified"]:
        names = result["user_modified"]
        MAX_SHOW = 5
        shown = ", ".join(names[:MAX_SHOW])
        if len(names) > MAX_SHOW:
            shown += f", +{len(names) - MAX_SHOW} more"
        parts.append(f"{len(names)} user-modified (kept): {shown}")
    if result["cleaned"]:
        parts.append(f"{len(result['cleaned'])} cleaned from manifest")
    print(f"\nDone: {', '.join(parts)}. {result['total_bundled']} total bundled.")
