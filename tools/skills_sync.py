#!/usr/bin/env python3
"""Skills Sync -- manifest-based seeding and updating of bundled skills. Copies repo skills/ into
~/.hermes/skills/, tracking each synced skill's origin hash in .bundled_manifest (v2 "name:hash"
lines; v1 plain names auto-migrate). NEW skills are copied and recorded; EXISTING skills update
only when bundled changed AND the user copy still matches the origin hash (else user-customized
-> SKIP); user-DELETED skills are not re-added; upstream-REMOVED ones leave the manifest once no active or
archived copy remains (the entry is that copy's only built-in provenance record)."""

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
from agent.skill_utils import (
    ESSENTIAL_SKILLS, EXCLUDED_SKILL_DIRS, SKILL_SUPPORT_DIRS, is_excluded_skill_path,
    parse_frontmatter,
)
from tools.skill_usage import read_skill_name
from tools.skills_sync_optional import (
    _backfill_optional_provenance, _ignore_runtime_cache, _is_runtime_cache, _read_hub_install_paths,
)
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
    from hermes_cli._startup_fast import project_root_str
    return get_bundled_skills_dir(Path(project_root_str()) / "skills")


def _get_optional_dir() -> Path:
    from hermes_cli._startup_fast import project_root_str
    return get_optional_skills_dir(Path(project_root_str()) / "optional-skills")


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


def _iter_skill_mds_following_links() -> Iterator[Path]:
    """Cycle-safe SKILL.md walk that, unlike ``_iter_active_skill_mds``, also follows symlinked
    category dirs — the loader's view. Prunes EXCLUDED_SKILL_DIRS and skill-root support dirs
    with the exact loader pattern (``agent.skill_utils.iter_skill_index_files`` /
    ``agent.prompt_builder._build_skills_manifest``). A directory already visited (by real
    inode) is never descended into twice, so symlink cycles terminate and an aliased category
    yields no duplicates. Read-only; feeds diagnostics only, never a move/delete decision
    (#143470)."""
    visited: set[tuple[int, int]] = set()
    for cur, dirs, files in os.walk(_skills_dir(), followlinks=True):
        try:
            info = os.stat(cur)
        except OSError:
            dirs[:] = []
            continue
        if (info.st_dev, info.st_ino) in visited:
            dirs[:] = []
            continue
        visited.add((info.st_dev, info.st_ino))
        has_skill_md = "SKILL.md" in files
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDED_SKILL_DIRS and not (has_skill_md and d in SKILL_SUPPORT_DIRS)
        ]
        if has_skill_md:
            yield Path(cur) / "SKILL.md"


def _build_external_skill_index() -> set[str]:
    """Names (directory and frontmatter) of every skill provided by external_dirs,
    so sync_skills never shadows an externally-delegated skill."""
    from agent.skill_utils import get_external_skills_dirs, _external_dirs_cache_clear
    _external_dirs_cache_clear()  # so a config edit (or a test patch) is seen
    external_names: set[str] = set()
    for ext_dir in get_external_skills_dirs():
        for skill_md in _iter_skill_mds(ext_dir):
            external_names.update({skill_md.parent.name, read_skill_name(skill_md, "")})
    external_names.discard("")
    return external_names


def _read_manifest() -> dict[str, str]:
    """``{skill_name: origin_hash}``; v1 plain-name lines get an empty hash (migrates next sync)."""
    try:
        result = {}
        for line in _manifest_file().read_text(encoding="utf-8-sig").splitlines():
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
    except OSError:
        return {}


def _read_suppressed_names() -> set:
    """Built-in skills the curator pruned — must NOT be re-seeded on sync.

    Delegates to ``tools.skill_usage`` (single source of truth) and falls back
    to reading ``~/.hermes/skills/.curator_suppressed`` directly if that import
    is unavailable in a packaged/update context.
    """
    try:
        from tools.skill_usage import read_suppressed_names

        return read_suppressed_names()
    except Exception:
        path = _skills_dir() / ".curator_suppressed"
        if not path.exists():
            return set()
        names = set()
        try:
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    names.add(line)
        except OSError:
            pass
        return names


def _write_manifest(entries: dict[str, str]):
    """Atomic v2 write, preserving an existing file's mode/owner (not mkstemp's 0600)."""
    from hermes_constants import mkdir_under_hermes_home
    mkdir_under_hermes_home(_manifest_file().parent)
    try:
        data = "".join(f"{n}:{h}\n" for n, h in sorted(entries.items()))
        atomic_write_text(_manifest_file(), data, tmp_prefix=".bundled_manifest_", preserve_mode=True)
    except Exception as e:
        logger.debug("Failed to write skills manifest %s: %s", _manifest_file(), e, exc_info=True)


def _discover_bundled_skills(bundled_dir: Path) -> list[tuple[str, Path]]:
    """``(skill_name, skill_dir)`` per SKILL.md under the bundled dir. Exclusions are evaluated
    relative to the bundled tree: the install prefix itself may contain ``venv``/``site-packages``
    (which once made wheel installs discover zero skills)."""
    if not bundled_dir.exists():
        return []
    return [
        (read_skill_name(md, md.parent.name), md.parent)
        for md in bundled_dir.rglob("SKILL.md")
        if not is_excluded_skill_path(md.relative_to(bundled_dir), root=bundled_dir)]


def _compute_relative_dest(skill_dir: Path, bundled_dir: Path) -> Path:
    """Destination preserving category structure (bundled/mlops/axolotl -> skills/mlops/axolotl)."""
    return _skills_dir() / skill_dir.relative_to(bundled_dir)


def dir_hash(directory: Path, *, include_runtime_cache: bool = False) -> str:
    """MD5 of package paths/content, excluding generated runtime state.

    The legacy option is only for proving an exact pre-filter origin match.
    Keep the original path encoding so clean existing manifests remain valid.
    """
    hasher = hashlib.md5()
    with suppress(OSError):
        for fpath in sorted(directory.rglob("*")):
            if (include_runtime_cache or not _is_runtime_cache(fpath, directory)) and fpath.is_file():
                hasher.update(str(fpath.relative_to(directory)).encode("utf-8"))
                hasher.update(fpath.read_bytes())
    return hasher.hexdigest()


def _matches_origin_hash(directory: Path, origin_hash: str, user_hash: Optional[str] = None) -> bool:
    """Prove unchanged package ownership against a clean OR exact legacy hash.

    Never re-baseline a differing package merely because it contains a cache:
    if legacy cached bytes changed/disappeared, the old origin cannot be proven.
    A genuinely edited package must remain protected in that case.
    """
    if not origin_hash:
        return False
    current = dir_hash(directory) if user_hash is None else user_hash
    return current == origin_hash or dir_hash(directory, include_runtime_cache=True) == origin_hash


def _move_dir(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))


def _loader_visible_name(skill_md: Path) -> Optional[str]:
    """The loader's YAML frontmatter ``name`` for a SKILL.md — the identity the agent actually
    invokes. Differs from ``read_skill_name``'s line-based read on commented (``name: demo # x``),
    escape-encoded and block-scalar names (#143470): those load as ``demo`` but read as garbage
    line-wise. Parses the complete file like the loader — no truncation, or a frontmatter fence
    past the cut would again hide a name the loader recognizes. Diagnostic/recovery lookup only;
    discovery/manifest identities keep ``read_skill_name``."""
    try:
        frontmatter, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8-sig", errors="replace"))
    except OSError:
        return None
    name = frontmatter.get("name")
    return name if isinstance(name, str) and name else None


def _reached_through_category_symlink(directory: Path) -> bool:
    """True when the path from the skills root down to ``directory`` traverses at least one
    symlinked directory — the #143470 scenario the warning describes. A plain real directory
    is visible to every consumer and must never be labeled symlink-only-reachable."""
    root = _skills_dir()
    current = directory
    while current != root and current != current.parent:
        if current.is_symlink():
            return True
        current = current.parent
    return False


def _warn_invisible_renamed_copy(st: "_SyncState", skill_name: str, dest: Path, *, v1_entry: bool = False) -> None:
    """#143470: the loader follows symlinked category dirs, so a skill whose only copy sits under
    one is ACTIVE for the agent yet invisible to the rglob-based active index — an upstream
    rename then reads as a user deletion. READ-ONLY fallback diagnostic: first match only (one
    warning per run), cycle-safe, and sync never moves, deletes, or resets content reached
    through a symlink. No ``reset`` advice either: reset's own scan cannot see the copy, and
    moving it is not sync's call — the safe remedy (replace the category symlink with a real
    directory containing the copy) is stated instead. The wording is deliberately generic about
    the copy's contents: deciding "pristine vs modified" would hash external content reached
    through the symlink, a cost the diagnostic does not need — user edits are untouched either
    way. A candidate matches on the manifest's line-based name read (which itself falls back
    to the directory basename when the frontmatter has no name) or on the loader's YAML name —
    the diagnostic answers "can the agent still address this copy?", so it must not miss a
    copy the loader invokes under a name the line reader garbles, and must not claim a
    directory that only shares a basename.

    The warning is APPENDED to ``st.relocation_warnings`` (rendered by quiet production
    callers from the sync result) and only additionally printed via ``st.say`` when not
    quiet."""
    for md in _iter_skill_mds_following_links():
        if md.parent == dest or not md.parent.is_dir():
            continue
        # Identity = how the agent can address this copy: the manifest's line-based name read
        # (which itself falls back to the directory basename when the frontmatter has no name)
        # or the loader's YAML name. A directory whose basename merely equals the tracked
        # skill name is NOT that skill by loader identity and is deliberately not matched.
        if (
            read_skill_name(md, md.parent.name) != skill_name
            and _loader_visible_name(md) != skill_name
        ):
            continue
        try:
            rel = _rel_skills_posix(md.parent)
        except ValueError:
            continue
        if _hub_owned(st, md.parent, rel):
            # The hub owns this copy (install path or inode alias) but not every copy of the
            # identity — skip it and keep searching: a later non-hub copy still deserves its
            # warning, order-independently.
            continue
        if not _reached_through_category_symlink(md.parent):
            # A plain visible directory is not the stranding scenario — the label
            # "only reachable through a category symlink" must never be applied to one.
            continue
        if v1_entry:
            remedy = (
                f"Move the copy to {_rel_skills_posix(dest)} to resume updates: a v1 manifest "
                f"entry records no origin hash, so sync cannot authorize an automatic "
                f"relocation and baselines from your copy once it is at the new location.")
        else:
            remedy = (
                "Replace the category symlink with a real directory containing the copy to "
                "make it visible again: a pristine copy then relocates on the next sync, and "
                "a modified one is kept at the old path with your edits preserved. Removing "
                "the symlink alone would strand it — the retained manifest entry makes sync "
                "read the missing destination as a user deletion.")
        message = (
            f"{skill_name}: upstream moved this skill to {_rel_skills_posix(dest)}, but the copy "
            f"at {rel} is only reachable through a category symlink, so it was not automatically "
            f"relocated. It is left untouched and will not receive updates. {remedy}")
        st.relocation_warnings.append(message)
        st.say(f"  ⚠ {message}")
        return


def _copy_dir(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, ignore=_ignore_runtime_cache)


def _ensure_recovery_index(st: "_SyncState") -> dict[str, list[Path]]:
    """Lazily build the shared recovery lookup: visible copies by every name form the loader
    or the manifest can address them by, plus hub ownership as install-path strings AND as
    real-directory inodes — an alias of a hub-installed dir is hub-managed whichever symlink
    reached it first, so ownership must not depend on traversal order."""
    if st.active_index is None:
        st.active_index = {}
        for md in _iter_active_skill_mds():
            line_name = read_skill_name(md, md.parent.name)
            st.active_index.setdefault(line_name, []).append(md.parent)
            loader_name = _loader_visible_name(md)  # loader parity, see _warn_invisible_renamed_copy
            if loader_name and loader_name != line_name:
                st.active_index.setdefault(loader_name, []).append(md.parent)
        st.hub_paths = _read_hub_install_paths()
        for rel in st.hub_paths:
            try:
                info = (_skills_dir() / rel).stat()
            except OSError:
                continue  # dangling lock path: no directory, nothing to alias
            st.hub_inodes.add((info.st_dev, info.st_ino))
    return st.active_index


def _hub_owned(st: "_SyncState", directory: Path, rel: str) -> bool:
    """True when a copy is hub-managed: by install-path string or by real-directory inode (a
    hub install reached through an alias path is still the hub's). Applied consistently by
    the diagnostic, the relocation loop, and the visible-copy guard."""
    if rel in st.hub_paths:
        return True
    try:
        info = directory.stat()
    except OSError:
        return False
    return (info.st_dev, info.st_ino) in st.hub_inodes


def _visible_non_hub_copy(skill_name: str, dest: Path, active_index: dict[str, list[Path]], st: "_SyncState") -> bool:
    """True when a visible copy exists that recovery could actually act on. A hub-owned
    install path is visible but untouchable — it must not suppress the read-only diagnostic
    for a separate stranded copy behind a category symlink."""
    for candidate in active_index.get(skill_name, []):
        if candidate == dest or not candidate.is_dir():
            continue
        try:
            rel = _rel_skills_posix(candidate)
        except ValueError:
            continue
        if not _hub_owned(st, candidate, rel):
            return True
    return False


def _recover_renamed_skill(st: "_SyncState", skill_name: str, dest: Path) -> Optional[str]:
    """Move a bundled skill's stale copy to its new canonical path after an upstream RENAME /
    RECATEGORIZATION (else it is misread as user-deleted and stranded forever). Only a copy
    byte-identical to the origin hash — proof *we* placed it — moves. Returns rel source path."""
    origin_hash = st.manifest.get(skill_name, "")
    active_index = _ensure_recovery_index(st)
    if not origin_hash:
        # v1 plain-name entry (empty hash): no hash can authorize a move. The read-only
        # #143470 diagnostic applies only when no actionable visible copy exists — exactly
        # like the v2 path below; a visible copy at the old path is the ordinary kept case.
        if not _visible_non_hub_copy(skill_name, dest, active_index, st):
            _warn_invisible_renamed_copy(st, skill_name, dest, v1_entry=True)
        return None
    for candidate in active_index.get(skill_name, []):
        if candidate == dest or not candidate.is_dir():
            continue
        try:
            rel = _rel_skills_posix(candidate)
        except ValueError:
            continue
        if _hub_owned(st, candidate, rel):  # the hub owns its install paths, aliases included
            continue
        if not _matches_origin_hash(candidate, origin_hash):  # moving a customized copy would edit user work
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
    if not _visible_non_hub_copy(skill_name, dest, active_index, st):
        # No actionable visible copy — but the loader sees through symlinked category dirs, so
        # an invisible copy may still exist (a hub-owned install alone does not count).
        # Read-only diagnostic; never migrates (#143470).
        _warn_invisible_renamed_copy(st, skill_name, dest)
    return None


@dataclass
class _SyncState:
    """Mutable accumulator threaded through one sync_skills() run."""
    manifest: dict[str, str]
    quiet: bool
    skipped: int = 0
    copied: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    user_modified: list[str] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)
    relocated: list[str] = field(default_factory=list)
    relocation_warnings: list[str] = field(default_factory=list)  # #143470: deferred rename diagnostics
    shadowed_by_external: list[str] = field(default_factory=list)
    active_index: Optional[dict[str, list[Path]]] = None  # rename-recovery indexes are expensive on
    hub_paths: set[str] = field(default_factory=set)  # bind mounts: built lazily, only when needed
    hub_inodes: set[tuple[int, int]] = field(default_factory=set)  # hub-managed dirs by real inode (#143470)

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
    if dest.exists() and dir_hash(dest) == bundled_hash:
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
            if dir_hash(dest) == bundled_hash:
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
        shutil.copytree(skill_src, dest, ignore=_ignore_runtime_cache)
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
    user_hash = dir_hash(dest)
    if not origin_hash:  # v1 migration: baseline from user's copy (can't tell edit from upstream)
        st.manifest[skill_name] = user_hash
        st.skipped += 1
        return
    if not _matches_origin_hash(dest, origin_hash, user_hash):
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


def _seed_category_descriptions(bundled_dir: Path, only_dirs: Optional[set[Path]]) -> None:
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
                "suppressed": [], "total_bundled": 0, "optional_provenance_backfilled": [],
                "relocation_warnings": []}
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
        bundled_hash = dir_hash(skill_src)
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
    # Clean manifest entries for skills removed upstream once no copy is left. A dropped built-in still
    # on disk (active or archived) keeps its entry: it is the only provenance record, and without it the
    # copy reads as agent-authored ("Learned", editable) (#95415). Skipped when opted out: bundled_skills
    # is only the essential set there, so cleaning would drop tracking for everything else.
    removed = [] if essential_only else sorted(set(st.manifest) - {name for name, _ in bundled_skills})
    present = set()
    if removed:  # curator archive is flat: directory name == skill name
        archive = _skills_dir() / ".archive"
        present = {read_skill_name(md, md.parent.name) for md in _iter_active_skill_mds()} | (
            {p.name for p in archive.iterdir() if p.is_dir()} if archive.is_dir() else set())
    cleaned = [name for name in removed if name not in present]
    for name in cleaned:
        del st.manifest[name]
    _seed_category_descriptions(
        bundled_dir,
        {_compute_relative_dest(src, bundled_dir).parent for _, src in bundled_skills} if essential_only else None)
    _write_manifest(st.manifest)
    return {
        "copied": st.copied, "updated": st.updated, "skipped": st.skipped, "user_modified": st.user_modified,
        "cleaned": cleaned, "suppressed": st.suppressed, "relocated": st.relocated,
        "relocation_warnings": st.relocation_warnings,
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
