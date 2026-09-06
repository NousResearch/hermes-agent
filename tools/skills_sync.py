#!/usr/bin/env python3
"""Skills Sync -- manifest-based seeding and updating of bundled skills. Copies repo skills/ into
~/.hermes/skills/, tracking each synced skill's origin hash in .bundled_manifest (v2 "name:hash"
lines; v1 plain names auto-migrate). NEW skills are copied and recorded; EXISTING skills update
only when bundled changed AND the user copy still matches the origin hash (else user-customized
-> SKIP); user-DELETED skills are not re-added; upstream-REMOVED ones leave the manifest."""

import hashlib
import logging
import os
import shutil
import stat
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

# Force UTF-8 stdout/stderr: GBK-style Windows locales can't encode the glyphs
# printed here (✓ ↑ →), and install.ps1 parses this script's stdout as UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        with suppress(ValueError, TypeError):
            _stream.reconfigure(encoding="utf-8", errors="replace")
from hermes_constants import get_bundled_skills_dir, get_hermes_home, get_optional_skills_dir
from agent.skill_utils import ESSENTIAL_SKILLS, is_excluded_skill_path
from tools.skill_usage import _read_skill_name, read_suppressed_names
from tools.skills_sync_bundled_ops import _is_tracked_user_modification
from tools.skills_sync_optional import _backfill_optional_provenance, _read_hub_install_paths
from utils import atomic_write_text

logger = logging.getLogger(__name__)

HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
MANIFEST_FILE = SKILLS_DIR / ".bundled_manifest"

# Import-time snapshots backing the call-time accessors: long-lived multi-profile runtimes
# retarget HERMES_HOME after import, and frozen constants would resolve (and for
# reset_bundled_skill() DELETE) against the wrong profile. Accessors honor an explicitly
# patched module global and otherwise re-resolve on every call.
# Same bug class and same fix as skills_tool (f8723c478) and skill_manager_tool (c6a3d412d): long-lived
# multi-profile runtimes (Dashboard console, TUI/Desktop backend, cron, kanban workers) import this module
# once under the launch HERMES_HOME and later scope requests to a different profile via
# set_hermes_home_override(). See #65828.
_HERMES_HOME_AT_IMPORT = HERMES_HOME
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR
_MANIFEST_FILE_AT_IMPORT = MANIFEST_FILE


def _live(configured, at_import: Path, fallback) -> Path:
    """The patched module global if it changed since import, else the live value."""
    return Path(configured) if Path(configured) != at_import else fallback()


def _hermes_home() -> Path:
    return _live(HERMES_HOME, _HERMES_HOME_AT_IMPORT, get_hermes_home)


def _skills_dir() -> Path:
    return _live(SKILLS_DIR, _SKILLS_DIR_AT_IMPORT, lambda: _hermes_home() / "skills")


def _manifest_file() -> Path:
    return _live(MANIFEST_FILE, _MANIFEST_FILE_AT_IMPORT, lambda: _skills_dir() / ".bundled_manifest")


# Written by `hermes profile create --no-skills` / installer `--no-skills`: sync seeds only
# essential skills. Mirrors hermes_cli.profiles.NO_BUNDLED_SKILLS_MARKER (no CLI import here).
NO_BUNDLED_SKILLS_MARKER = ".no-bundled-skills"


def _get_bundled_dir() -> Path:  # HERMES_BUNDLED_SKILLS env first, then repo-relative
    return get_bundled_skills_dir(Path(__file__).parent.parent / "skills")


def _get_optional_dir() -> Path:
    return get_optional_skills_dir(Path(__file__).parent.parent / "optional-skills")


def _rel_skills_posix(path: Path) -> str:
    return path.relative_to(_skills_dir()).as_posix()


def _iter_skill_mds(root: Path, sort: bool = False) -> Iterator[Path]:
    """Yield every non-excluded SKILL.md under ``root`` (nothing when it does not exist)."""
    found = root.rglob("SKILL.md") if root.exists() else iter(())
    for skill_md in sorted(found) if sort else found:
        if not is_excluded_skill_path(skill_md):
            yield skill_md


def _iter_active_skill_mds(sort: bool = False) -> Iterator[Path]:
    """Yield every non-excluded SKILL.md in the user's skills tree."""
    return _iter_skill_mds(_skills_dir(), sort)


def _build_external_skill_index() -> Set[str]:
    """Names (directory and frontmatter) of every skill provided by external_dirs,
    so sync_skills never shadows an externally-delegated skill."""
    from agent.skill_utils import get_external_skills_dirs, _external_dirs_cache_clear
    _external_dirs_cache_clear()  # so a config edit (or a test patch) is seen
    external_names: Set[str] = set()
    for ext_dir in get_external_skills_dirs():
        for skill_md in _iter_skill_mds(ext_dir):
            external_names.update({skill_md.parent.name, _read_skill_name(skill_md, "")})
    external_names.discard("")
    return external_names


def _read_manifest() -> Dict[str, str]:
    """``{skill_name: origin_hash}``; v1 plain-name lines get an empty hash (migrates next sync)."""
    try:
        lines = _manifest_file().read_text(encoding="utf-8").splitlines() if _manifest_file().exists() else []
    except OSError:
        return {}
    pairs = (line.partition(":") for line in map(str.strip, lines) if line)
    return {name.strip(): hash_val.strip() for name, _, hash_val in pairs}


def _read_suppressed_names() -> set:
    """Built-in skills the curator pruned — must NOT be re-seeded (tests patch this name)."""
    return read_suppressed_names()


def _write_manifest(entries: Dict[str, str]):
    """Atomic v2 write, preserving an existing file's mode/owner (not mkstemp's 0600)."""
    _manifest_file().parent.mkdir(parents=True, exist_ok=True)
    try:
        data = "".join(f"{n}:{h}\n" for n, h in sorted(entries.items()))
        atomic_write_text(_manifest_file(), data, tmp_prefix=".bundled_manifest_", preserve_mode=True)
    except Exception as e:
        logger.debug("Failed to write skills manifest %s: %s", _manifest_file(), e, exc_info=True)


def _discover_bundled_skills(bundled_dir: Path) -> List[Tuple[str, Path]]:
    """``(skill_name, skill_dir)`` per SKILL.md under the bundled dir. Exclusions are evaluated
    relative to the bundled tree: the install prefix itself may contain ``venv``/``site-packages``
    (which once made wheel installs discover zero skills)."""
    if not bundled_dir.exists():
        return []
    return [
        (_read_skill_name(md, md.parent.name), md.parent)
        for md in bundled_dir.rglob("SKILL.md")
        if not is_excluded_skill_path(md.relative_to(bundled_dir), root=bundled_dir)]


def _compute_relative_dest(skill_dir: Path, bundled_dir: Path) -> Path:
    """Destination preserving category structure (bundled/mlops/axolotl -> skills/mlops/axolotl)."""
    return _skills_dir() / skill_dir.relative_to(bundled_dir)


def _dir_hash(directory: Path) -> str:
    """MD5 over relative paths + contents of every file in a directory."""
    hasher = hashlib.md5()
    with suppress(OSError):
        for fpath in sorted(directory.rglob("*")):
            if fpath.is_file():
                hasher.update(str(fpath.relative_to(directory)).encode("utf-8"))
                hasher.update(fpath.read_bytes())
    return hasher.hexdigest()


def _move_dir(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))


def _copy_dir(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)


def _recover_renamed_skill(st: "_SyncState", skill_name: str, dest: Path) -> Optional[str]:
    """Move a bundled skill's stale copy to its new canonical path after an upstream RENAME /
    RECATEGORIZATION (else it is misread as user-deleted and stranded forever). Only a copy
    byte-identical to the origin hash — proof *we* placed it — moves. Returns rel source path."""
    origin_hash = st.manifest.get(skill_name, "")
    if not origin_hash:
        return None
    if st.active_index is None:  # by frontmatter name
        st.active_index = {}
        for md in _iter_active_skill_mds():
            st.active_index.setdefault(_read_skill_name(md, md.parent.name), []).append(md.parent)
        st.hub_paths = _read_hub_install_paths()
    for candidate in st.active_index.get(skill_name, []):
        if candidate == dest or not candidate.is_dir():
            continue
        try:
            rel = _rel_skills_posix(candidate)
        except ValueError:
            continue
        if rel in st.hub_paths:  # the hub owns its install paths
            continue
        if _dir_hash(candidate) != origin_hash:  # moving a customized copy would edit user work
            st.say(
                f"  ⚠ {skill_name}: upstream moved this skill to {_rel_skills_posix(dest)}, but your "
                f"modified copy at {rel} was kept — it will not receive updates. "
                f"Run `hermes skills reset {skill_name} --restore` to move to the new location.")
            continue
        try:
            _move_dir(candidate, dest)
        except OSError:
            logger.warning("Could not relocate renamed skill %s -> %s", candidate, dest, exc_info=True)
            return None
        logger.info("Relocated renamed bundled skill: %s -> %s", candidate, dest)
        st.say(f"  → {skill_name} (moved {rel} → {_rel_skills_posix(dest)})")
        return rel
    return None


@dataclass
class _SyncState:
    """Mutable accumulator threaded through one sync_skills() run."""
    manifest: Dict[str, str]
    quiet: bool
    skipped: int = 0
    copied: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)
    user_modified: List[str] = field(default_factory=list)
    suppressed: List[str] = field(default_factory=list)
    relocated: List[str] = field(default_factory=list)
    shadowed_by_external: List[str] = field(default_factory=list)
    active_index: Optional[Dict[str, List[Path]]] = None  # rename-recovery indexes are expensive on
    hub_paths: Set[str] = field(default_factory=set)  # bind mounts: built lazily, only when needed

    def say(self, msg: str) -> None:
        if not self.quiet:
            print(msg)


def _recover_orphan_backup(dest: Path) -> None:
    """If an interrupted update left the user's only copy in ``dest.bak`` with dest gone, move it
    back so the skill isn't misread as user-deleted."""
    orphan = dest.with_suffix(".bak")
    if orphan.exists() and not dest.exists():
        try:
            _move_dir(orphan, dest)
            logger.info("Recovered orphaned skill backup: %s", orphan)
        except OSError:
            logger.warning("Could not recover orphaned skill backup %s", orphan, exc_info=True)


def _defer_to_external(st: _SyncState, skill_name: str, dest: Path, bundled_hash: str) -> None:
    """An external_dirs source provides this skill; a local copy would be a name collision the
    loader refuses. Defer for ALL manifest states; remove a stale local shadow from an earlier
    sync only when byte-identical (a user's own skill differs)."""
    st.shadowed_by_external.append(skill_name)
    st.skipped += 1
    st.say(f"  ⇢ {skill_name} (deferred to external_dirs, not written to local tree)")
    if dest.exists() and _dir_hash(dest) == bundled_hash:
        _rmtree_writable(dest)
        st.say(f"  ✓ removed stale shadow of {skill_name}")
        st.manifest.pop(skill_name, None)


def _install_new_skill(st: _SyncState, skill_name: str, skill_src: Path, dest: Path, bundled_hash: str) -> None:
    """Handle a skill never offered before (not in manifest)."""
    try:
        if dest.exists():
            # Never overwrite a same-named user skill. Baseline the manifest only when
            # byte-identical: a differing copy's bundled_hash reads as "user-modified" forever.
            st.skipped += 1
            if _dir_hash(dest) == bundled_hash:
                st.manifest[skill_name] = bundled_hash
            else:
                st.say(
                    f"  ⚠ {skill_name}: bundled version shipped but you already have a local skill "
                    f"by this name — yours was kept. Run `hermes skills reset {skill_name}` to "
                    f"replace it with the bundled version.")
        else:
            _copy_dir(skill_src, dest)
            st.copied.append(skill_name)
            st.manifest[skill_name] = bundled_hash
            st.say(f"  + {skill_name}")
    except OSError as e:
        st.say(f"  ! Failed to copy {skill_name}: {e}")  # not in manifest — next sync retries


def _replace_skill_dir(skill_src: Path, dest: Path) -> None:
    """Replace ``dest`` with a fresh copy of ``skill_src`` via a .bak sibling; restore on failure."""
    backup = dest.with_suffix(".bak")
    if backup.exists():  # a stale .bak would make shutil.move() nest dest INSIDE it
        _rmtree_writable(backup)
    shutil.move(str(dest), str(backup))
    try:
        shutil.copytree(skill_src, dest)
    except OSError:
        if backup.exists():  # clear a partially-written dest so it can't shadow/block the restore
            if dest.exists():
                try:
                    _rmtree_writable(dest)
                except OSError:
                    logger.warning("Could not clear partial copy %s during restore", dest,
                                   exc_info=True)
            if not dest.exists():
                shutil.move(str(backup), str(dest))
        raise
    try:
        _rmtree_writable(backup)
    except OSError:
        logger.debug("Could not remove backup %s", backup, exc_info=True)


def _update_existing_skill(st: _SyncState, skill_name: str, skill_src: Path, dest: Path, bundled_hash: str) -> None:
    """Handle a skill that is in the manifest AND on disk."""
    origin_hash = st.manifest.get(skill_name, "")
    if origin_hash and bundled_hash == origin_hash:  # bundled unchanged: skip without hashing the user copy
        st.skipped += 1
        return
    user_hash = _dir_hash(dest)
    if not origin_hash:  # v1 migration: baseline from user's copy (can't tell edit from upstream)
        st.manifest[skill_name] = user_hash
        st.skipped += 1
        return
    if _is_tracked_user_modification(origin_hash, user_hash):
        st.user_modified.append(skill_name)
        st.say(f"  ~ {skill_name} (user-modified, skipping)")
        return
    # bundled changed and the user copy is pristine -> update
    try:
        _replace_skill_dir(skill_src, dest)
    except OSError as e:
        st.say(f"  ! Failed to update {skill_name}: {e}")
        return
    st.manifest[skill_name] = bundled_hash
    st.updated.append(skill_name)
    st.say(f"  ↑ {skill_name} (updated)")


def _seed_category_descriptions(bundled_dir: Path, only_dirs: Optional[Set[Path]]) -> None:
    """Copy category DESCRIPTION.md files not already present; ``only_dirs`` restricts
    seeding to the essential skills' categories on opted-out profiles."""
    for desc_md in bundled_dir.rglob("DESCRIPTION.md"):
        dest_desc = _skills_dir() / desc_md.relative_to(bundled_dir)
        if (only_dirs is not None and dest_desc.parent not in only_dirs) or dest_desc.exists():
            continue
        try:
            dest_desc.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(desc_md, dest_desc)
        except OSError as e:
            logger.debug("Could not copy %s: %s", desc_md, e)


def sync_skills(quiet: bool = False) -> dict:
    """Sync bundled skills into ~/.hermes/skills/ using the manifest; returns the per-category
    result dict. Opted-out profiles seed ONLY ESSENTIAL_SKILLS (the system prompt always
    points at ``hermes-agent``)."""
    essential_only = (_hermes_home() / NO_BUNDLED_SKILLS_MARKER).exists()
    if essential_only and not quiet:
        print("  (profile opted out of bundled skills via .no-bundled-skills — seeding essential skills only)")
    bundled_dir = _get_bundled_dir()
    if not bundled_dir.exists():
        return {"copied": [], "updated": [], "skipped": 0, "user_modified": [], "cleaned": [],
                "suppressed": [], "total_bundled": 0, "optional_provenance_backfilled": []}
    _skills_dir().mkdir(parents=True, exist_ok=True)
    bundled_skills = _discover_bundled_skills(bundled_dir)
    if essential_only:
        bundled_skills = [(name, src) for name, src in bundled_skills if name in ESSENTIAL_SKILLS]
    suppressed = _read_suppressed_names()
    external_index = _build_external_skill_index()
    st = _SyncState(manifest=_read_manifest(), quiet=quiet)

    for skill_name, skill_src in bundled_skills:
        # Curator-pruned built-ins must not resurrect on every update; essentials are exempt.
        if skill_name in suppressed and skill_name not in ESSENTIAL_SKILLS:
            st.suppressed.append(skill_name)
            continue
        dest = _compute_relative_dest(skill_src, bundled_dir)
        bundled_hash = _dir_hash(skill_src)
        # Recoveries run BEFORE classification so a missing dest isn't misread as user-deleted.
        _recover_orphan_backup(dest)
        if not dest.exists() and skill_name in st.manifest and _recover_renamed_skill(st, skill_name, dest):
            st.relocated.append(skill_name)
        if skill_name in external_index:
            _defer_to_external(st, skill_name, dest, bundled_hash)
        elif skill_name not in st.manifest:
            _install_new_skill(st, skill_name, skill_src, dest, bundled_hash)
        elif dest.exists():
            _update_existing_skill(st, skill_name, skill_src, dest, bundled_hash)
        else:
            st.skipped += 1  # in manifest but not on disk — user deleted it
    # Clean manifest entries for skills removed upstream. Skipped when opted out: bundled_skills
    # is only the essential set there, so cleaning would drop tracking for everything else.
    cleaned = [] if essential_only else sorted(set(st.manifest) - {name for name, _ in bundled_skills})
    for name in cleaned:
        del st.manifest[name]
    _seed_category_descriptions(
        bundled_dir,
        {_compute_relative_dest(src, bundled_dir).parent for _, src in bundled_skills} if essential_only else None)
    _write_manifest(st.manifest)
    return {
        "copied": st.copied, "updated": st.updated, "skipped": st.skipped, "user_modified": st.user_modified,
        "cleaned": cleaned, "suppressed": st.suppressed, "relocated": st.relocated,
        "total_bundled": len(bundled_skills),
        "optional_provenance_backfilled": _backfill_optional_provenance(quiet=quiet),
        "shadowed_by_external": st.shadowed_by_external,
        "skipped_opt_out": essential_only}  # lets callers report "opted out", not a normal sync


def _rmtree_writable(path: Path) -> None:
    """rmtree that first makes read-only entries writable (Nix/deb/rpm keep r-x dirs; unlinking
    a child needs a writable parent, so chmod both). Scope guard: refuses anything not a STRICT
    child of the active skills root (bad join / missing HERMES_HOME / malicious manifest entry).

    Handles immutable package sources (Nix store, deb/rpm installs) that preserve read-only permissions on
    copied files *and* directories (``r-xr-xr-x``). Removing a child requires write permission on its parent
    directory, so the retry handler makes the failing path **and its parent** writable before re-attempting.
    See #34860, #34972.
    """
    target = Path(path).resolve()
    skills_root = _skills_dir().resolve()
    if skills_root not in target.parents:
        raise ValueError(f"refusing to rmtree {target!r}: not strictly under {skills_root!r} (scope guard — see #48200)")

    def _on_error(func, fpath, exc_info):
        for p in (os.path.dirname(fpath), fpath):
            with suppress(OSError):
                os.chmod(p, stat.S_IRWXU)
        func(fpath)
    shutil.rmtree(path, onerror=_on_error)


if __name__ == "__main__":
    print("Syncing bundled skills into ~/.hermes/skills/ ...")
    result = sync_skills(quiet=False)
    parts = [f"{len(result['copied'])} new", f"{len(result['updated'])} updated", f"{result['skipped']} unchanged"]
    if names := result["user_modified"]:
        shown = ", ".join(names[:5]) + (f", +{len(names) - 5} more" if len(names) > 5 else "")
        parts.append(f"{len(names)} user-modified (kept): {shown}")
    if result["cleaned"]:
        parts.append(f"{len(result['cleaned'])} cleaned from manifest")
    if backfilled := result.get("optional_provenance_backfilled"):
        parts.append(f"{len(backfilled)} official optional backfilled")
    print(f"\nDone: {', '.join(parts)}. {result['total_bundled']} total bundled.")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from pathlib import PurePosixPath  # noqa: F401,E402
from datetime import datetime  # noqa: F401,E402
import json  # noqa: F401,E402
from datetime import timezone  # noqa: F401,E402

def is_bundled_skills_opt_out() -> bool:
    """Return True if the active profile carries the opt-out marker."""
    return (_hermes_home() / NO_BUNDLED_SKILLS_MARKER).exists()


_PLUGIN_COMPAT_LAZY = {
    'atomic_replace': ('utils', 'atomic_replace'),
    'diff_bundled_skill': ('tools.skills_sync_bundled_ops', 'diff_bundled_skill'),
    'list_user_modified_bundled_skills': ('tools.skills_sync_bundled_ops', 'list_user_modified_bundled_skills'),
    'remove_pristine_bundled_skills': ('tools.skills_sync_bundled_ops', 'remove_pristine_bundled_skills'),
    'reset_bundled_skill': ('tools.skills_sync_bundled_ops', 'reset_bundled_skill'),
    'restore_official_optional_skill': ('tools.skills_sync_optional', 'restore_official_optional_skill'),
    'set_bundled_skills_opt_out': ('tools.skills_sync_bundled_ops', 'set_bundled_skills_opt_out'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
