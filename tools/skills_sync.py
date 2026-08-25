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
from agent.skill_utils import is_excluded_skill_path
from typing import Dict, List, Optional, Set, Tuple
from utils import atomic_replace, atomic_write_text

logger = logging.getLogger(__name__)

HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
MANIFEST_FILE = SKILLS_DIR / ".bundled_manifest"

# Import-time snapshots backing the call-time accessors below. Same bug class
# and same fix as skills_tool (f8723c478) and skill_manager_tool (c6a3d412d):
# long-lived multi-profile runtimes (Dashboard console, TUI/Desktop backend,
# cron, kanban workers) import this module once under the launch HERMES_HOME
# and later scope requests to a different profile via
# set_hermes_home_override(). Frozen module constants would then resolve —
# and for reset_bundled_skill() DELETE — against the wrong profile's skills
# root (#65828). The accessors honor an explicitly patched module global
# (tests, and web_server's _profile_scope retargeting) and otherwise
# re-resolve from the live profile-scoped HERMES_HOME on every call.
_HERMES_HOME_AT_IMPORT = HERMES_HOME
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR
_MANIFEST_FILE_AT_IMPORT = MANIFEST_FILE


def _hermes_home() -> Path:
    """Return the active profile's HERMES_HOME at call time."""
    configured = Path(HERMES_HOME)
    if configured != _HERMES_HOME_AT_IMPORT:
        return configured
    return get_hermes_home()


def _skills_dir() -> Path:
    """Return the active profile's skills directory at call time."""
    configured = Path(SKILLS_DIR)
    if configured != _SKILLS_DIR_AT_IMPORT:
        return configured
    return _hermes_home() / "skills"


def _manifest_file() -> Path:
    """Return the active profile's bundled-skills manifest at call time."""
    configured = Path(MANIFEST_FILE)
    if configured != _MANIFEST_FILE_AT_IMPORT:
        return configured
    return _skills_dir() / ".bundled_manifest"

# Marker file written by `hermes profile create --no-skills` (named profiles)
# and by the installer's `--no-skills` flag (the default ~/.hermes profile).
# When present in HERMES_HOME, sync_skills() is a no-op so neither the
# installer, `hermes update`, nor a direct sync re-injects bundled skills.
# Delete the file to opt back in. Mirrors
# hermes_cli.profiles.NO_BUNDLED_SKILLS_MARKER (kept as a literal here to
# avoid importing the CLI layer into this low-level sync module).
NO_BUNDLED_SKILLS_MARKER = ".no-bundled-skills"


def _essential_names() -> frozenset:
    """Names of skills that must always exist (see skill_utils.ESSENTIAL_SKILLS)."""
    try:
        from agent.skill_utils import ESSENTIAL_SKILLS
        return ESSENTIAL_SKILLS
    except Exception:
        return frozenset({"hermes-agent"})


def _get_bundled_dir() -> Path:
    """Locate the bundled skills/ directory.

    Checks HERMES_BUNDLED_SKILLS env var first (set by Nix wrapper),
    then falls back to the relative path from this source file.
    """
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
    """
    Read the manifest as a dict of {skill_name: origin_hash}.

    Handles both v1 (plain names) and v2 (name:hash) formats.
    v1 entries get an empty hash string which triggers migration on next sync.
    """
    if not _manifest_file().exists():
        return {}
    try:
        result = {}
        for line in _manifest_file().read_text(encoding="utf-8").splitlines():
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
    pairs = (line.partition(":") for line in map(str.strip, lines) if line)
    return {name.strip(): hash_val.strip() for name, _, hash_val in pairs}


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
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    names.add(line)
        except OSError:
            pass
        return names


def _write_manifest(entries: Dict[str, str]):
    """Write the manifest file atomically in v2 format (name:hash).

    Uses the shared atomic writer so an existing manifest's permission
    bits (and owner, best-effort) survive the replace instead of being
    reset to mkstemp's 0600 — the same mode-preservation contract as the
    skill manager's document writes.
    """
    _manifest_file().parent.mkdir(parents=True, exist_ok=True)
    data = "\n".join(f"{name}:{hash_val}" for name, hash_val in sorted(entries.items())) + "\n"

    try:
        atomic_write_text(
            _manifest_file(),
            data,
            tmp_prefix=".bundled_manifest_",
            preserve_mode=True,
        )
    except Exception as e:
        logger.debug("Failed to write skills manifest %s: %s", _manifest_file(), e, exc_info=True)


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
    """
    Compute the destination path in the skills dir preserving the category structure.
    e.g., bundled/skills/mlops/axolotl -> ~/.hermes/skills/mlops/axolotl
    """
    rel = skill_dir.relative_to(bundled_dir)
    return _skills_dir() / rel


def _dir_hash(directory: Path, *, include_runtime_cache: bool = False) -> str:
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


def _skill_file_list(skill_dir: Path) -> List[str]:
    """List files inside a skill directory in lock-file format."""
    files: List[str] = []
    for fpath in sorted(skill_dir.rglob("*")):
        if fpath.is_file():
            files.append(fpath.relative_to(skill_dir).as_posix())
    return files


def _content_hash(directory: Path) -> str:
    """Return the same hash style the skills hub lock uses, falling back locally."""
    try:
        from tools.skills_guard import content_hash

        return content_hash(directory)
    except Exception:
        # Hashing is provenance metadata only; keep sync resilient if guard
        # dependencies are unavailable in a packaged/update context.
        return _dir_hash(directory)


def _optional_skill_index() -> Dict[str, Tuple[str, str, Path]]:
    """Return official optional skills keyed by folder name and frontmatter name.

    Values are ``(folder_name, install_path, source_dir)``. Multiple keys may
    point to the same skill so callers can accept either the folder slug used
    by the hub lock or the user-facing frontmatter name.
    """
    optional_dir = _get_optional_dir()
    index: Dict[str, Tuple[str, str, Path]] = {}
    if not optional_dir.exists():
        return index
    for skill_md in sorted(optional_dir.rglob("SKILL.md")):
        if is_excluded_skill_path(
            skill_md.relative_to(optional_dir), root=optional_dir
        ):
            continue
        src = skill_md.parent
        try:
            install_path = _safe_rel_install_path(src, optional_dir)
        except ValueError:
            continue
        folder_name = src.name
        frontmatter_name = _read_skill_name(skill_md, folder_name)
        value = (folder_name, install_path, src)
        index[folder_name] = value
        index[frontmatter_name] = value
    return index


def _move_to_restore_backup(path: Path, backup_root: Path) -> str:
    """Move an existing skill directory into a restore backup, preserving rel path."""
    rel = path.relative_to(_skills_dir())
    target = backup_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        suffix = 1
        while target.with_name(f"{target.name}-{suffix}").exists():
            suffix += 1
        target = target.with_name(f"{target.name}-{suffix}")
    shutil.move(str(path), str(target))
    return rel.as_posix()


def restore_official_optional_skill(name: str, *, restore: bool = False) -> dict:
    """Restore one or all official optional skills from repo source.

    ``restore=False`` only performs exact-match provenance backfill. ``restore=True``
    repairs already-mutated/reorganized skills by backing up matching active
    copies and copying the official optional source into its canonical path.
    """
    index = _optional_skill_index()
    if not index:
        return {"ok": False, "message": "No official optional skills directory found.", "restored": [], "backfilled": [], "backed_up": []}

    targets = sorted(set(index.values()), key=lambda item: item[1]) if name in {"all", "*"} else []
    if not targets:
        target = index.get(name)
        if target is None:
            return {"ok": False, "message": f"Official optional skill not found: {name}", "restored": [], "backfilled": [], "backed_up": []}
        targets = [target]

    restored: List[str] = []
    backed_up: List[str] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_root = _skills_dir() / ".restore-backups" / f"official-optional-{timestamp}"

    for folder_name, install_path, src in targets:
        dest = _skills_dir() / Path(*install_path.split("/"))
        src_hash = _dir_hash(src)
        canonical_ok = dest.exists() and _dir_hash(dest) == src_hash

        # Find already-active copies of this official skill by frontmatter name
        # or folder slug, even if curator moved it into another category.
        src_frontmatter = _read_skill_name(src / "SKILL.md", folder_name)
        matches: List[Path] = []
        if _skills_dir().exists():
            for skill_md in sorted(_skills_dir().rglob("SKILL.md")):
                if is_excluded_skill_path(skill_md):
                    continue
                candidate = skill_md.parent
                try:
                    candidate.relative_to(_skills_dir())
                except ValueError:
                    continue
                candidate_name = _read_skill_name(skill_md, candidate.name)
                if candidate == dest:
                    continue
                if candidate.name == folder_name or candidate_name in {folder_name, src_frontmatter}:
                    matches.append(candidate)

        if restore:
            for match in matches:
                if match.exists():
                    backed_up.append(_move_to_restore_backup(match, backup_root))
            if dest.exists() and not canonical_ok:
                backed_up.append(_move_to_restore_backup(dest, backup_root))
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest)
                restored.append(folder_name)
        elif not canonical_ok:
            continue

    backfilled = _backfill_optional_provenance(quiet=True)
    return {
        "ok": True,
        "message": "Official optional skill repair complete.",
        "restored": restored,
        "backfilled": backfilled,
        "backed_up": backed_up,
        "backup_dir": str(backup_root) if backed_up else "",
    }


def _index_installed_skill_dirs_by_name() -> Dict[str, List[Path]]:
    """Index installed skills by directory name with one active-tree scan."""
    index: Dict[str, List[Path]] = {}
    if not _skills_dir().exists():
        return index
    for skill_md in _skills_dir().rglob("SKILL.md"):
        if is_excluded_skill_path(skill_md):
            continue
        candidate = skill_md.parent
        # Never reach outside the skills tree (symlinked/external dirs).
        try:
            candidate.resolve().relative_to(_skills_dir().resolve())
        except (OSError, ValueError):
            continue
        index.setdefault(candidate.name, []).append(candidate)
    return index


def _find_installed_skill_dir_by_name(
    skill_dir_name: str,
    installed_index: Optional[Dict[str, List[Path]]] = None,
) -> Optional[Path]:
    """Locate an installed skill directory by its directory name.

    Used only as a fallback when the repo-derived install path doesn't exist in
    the active tree (upstream recategorized the skill after it was installed).
    Returns None when there is no match, or when the name is AMBIGUOUS — two
    skills sharing a directory name give us no basis to pick one, and guessing
    would write provenance onto the wrong skill. The caller still verifies a
    byte-identical content hash before recording anything.
    """
    if not skill_dir_name or not _skills_dir().exists():
        return None
    if installed_index is None:
        installed_index = _index_installed_skill_dirs_by_name()
    matches = installed_index.get(skill_dir_name, [])
    if len(matches) != 1:
        return None
    return matches[0]


def _backfill_optional_provenance(quiet: bool = False) -> List[str]:
    """Mark already-present official optional skills as hub-installed.

    This covers the migration case where a skill used to be bundled (or was
    manually copied into the active skills tree) and later lives under
    optional-skills/. If the active copy is byte-identical to the official
    optional source, record official hub provenance without copying or
    reinstalling anything. Modified/local skills are left alone.
    """
    optional_dir = _get_optional_dir()
    if not optional_dir.exists():
        return []

    lock_path = _skills_dir() / ".hub" / "lock.json"
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {"version": 1, "installed": {}}
    except (json.JSONDecodeError, OSError):
        data = {"version": 1, "installed": {}}
    installed = data.setdefault("installed", {})
    existing_paths = {
        entry.get("install_path")
        for entry in installed.values()
        if isinstance(entry, dict)
    }

    backfilled: List[str] = []
    changed = False
    installed_dir_index: Optional[Dict[str, List[Path]]] = None
    for skill_md in sorted(optional_dir.rglob("SKILL.md")):
        if is_excluded_skill_path(skill_md):
            continue
        src = skill_md.parent
        try:
            install_path = _safe_rel_install_path(src, optional_dir)
        except ValueError as e:
            logger.debug("Skipping optional skill with unsafe path %s: %s", src, e)
            continue
        lock_name = src.name
        if lock_name in installed or install_path in existing_paths:
            continue
        dest = _skills_dir() / Path(*install_path.split("/"))
        if not dest.exists() or not dest.is_dir():
            # The active tree may hold the same skill under a DIFFERENT
            # category path than the repo uses — categories get reorganized
            # upstream (e.g. mlops/chroma → mlops/vector-databases/chroma)
            # while the already-installed copy keeps its old location. A
            # path-only lookup misses every one of those, so provenance repair
            # silently skips them forever. Fall back to a unique
            # same-directory-name match anywhere in the tree, then still
            # require a byte-identical hash below before claiming provenance.
            if installed_dir_index is None:
                installed_dir_index = _index_installed_skill_dirs_by_name()
            dest = _find_installed_skill_dir_by_name(src.name, installed_dir_index)
            if dest is None:
                continue
            try:
                install_path = _safe_rel_install_path(dest, _skills_dir())
            except ValueError as e:
                logger.debug("Skipping relocated optional skill %s: %s", dest, e)
                continue
        if install_path in existing_paths:
            continue
        if _dir_hash(dest) != _dir_hash(src):
            continue

        timestamp = datetime.now(timezone.utc).isoformat()
        installed[lock_name] = {
            "source": "official",
            "identifier": f"official/{install_path}",
            "trust_level": "builtin",
            "scan_verdict": "backfilled",
            "content_hash": _content_hash(dest),
            "install_path": install_path,
            "files": _skill_file_list(dest),
            "metadata": {"backfilled_from": "optional-skills"},
            "installed_at": timestamp,
            "updated_at": timestamp,
        }
        existing_paths.add(install_path)
        backfilled.append(lock_name)
        changed = True
        if not quiet:
            print(f"  = {lock_name} (official optional provenance backfilled)")

    if changed:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a crash mid-write can't silently wipe all provenance
        # via the JSONDecodeError fallback above (which resets `installed` to
        # an empty dict).
        import tempfile

        payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        fd, tmp_path = tempfile.mkstemp(
            dir=str(lock_path.parent),
            prefix=".lock_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            atomic_replace(tmp_path, lock_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    return backfilled


def _read_hub_install_paths() -> Set[str]:
    """Return install paths recorded in the skills-hub lock, as POSIX strings.

    Hub-installed skills are owned by the hub (``hermes skills uninstall``),
    never by bundled sync. Rename recovery must not move them even when their
    content happens to match a bundled origin hash, or the lock's
    ``install_path`` would point at a directory that no longer exists.
    """
    lock_path = _skills_dir() / ".hub" / "lock.json"
    if not lock_path.exists():
        return set()
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    paths: Set[str] = set()
    for entry in (data.get("installed") or {}).values():
        if isinstance(entry, dict):
            install_path = entry.get("install_path")
            if install_path:
                paths.add(str(install_path).strip("/"))
    return paths


def _index_active_skills() -> Dict[str, List[Path]]:
    """Index every skill in the user's tree by frontmatter name.

    Returns ``{skill_name: [skill_dir, ...]}``. Used by rename recovery to
    locate a bundled skill that upstream moved to a new category/directory.
    """
    index: Dict[str, List[Path]] = {}
    if not _skills_dir().exists():
        return index
    for skill_md in _skills_dir().rglob("SKILL.md"):
        if is_excluded_skill_path(skill_md):
            continue
        skill_dir = skill_md.parent
        name = _read_skill_name(skill_md, skill_dir.name)
        index.setdefault(name, []).append(skill_dir)
    return index


def _recover_renamed_skill(
    skill_name: str,
    origin_hash: str,
    dest: Path,
    active_index: Dict[str, List[Path]],
    hub_paths: Set[str],
    quiet: bool,
) -> Optional[str]:
    """Move a bundled skill's stale copy to its new canonical path.

    When upstream RENAMES or RECATEGORIZES a bundled skill, the manifest key
    (frontmatter name) still matches but ``dest`` is a brand-new path that does
    not exist yet. Without recovery, ``sync_skills()`` falls through to its
    "in manifest but not on disk" branch and misreads the skill as
    *user-deleted*: the old directory is stranded forever and never receives
    another update.

    A stale copy is only moved when it is byte-identical to ``origin_hash`` —
    the hash recorded the last time sync wrote that skill — which proves the
    directory is the copy *we* placed there rather than the user's own work.
    Anything else (user-edited, hub-installed) is left untouched.

    Returns the relative source path when a move happened, else ``None``.
    """
    if not origin_hash:
        return False
    current = _dir_hash(directory) if user_hash is None else user_hash
    return current == origin_hash or _dir_hash(directory, include_runtime_cache=True) == origin_hash


def _move_dir(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))


def _copy_dir(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, ignore=_ignore_runtime_cache)


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
            rel = candidate.relative_to(_skills_dir()).as_posix()
        except ValueError:
            continue
        if rel in st.hub_paths:  # the hub owns its install paths
            continue
        if _dir_hash(candidate) != origin_hash:
            # User customized the copy at the old path. Moving it would edit
            # their work; leaving it avoids a duplicate-name collision. Warn
            # so they can migrate deliberately.
            if not quiet:
                print(
                    f"  ⚠ {skill_name}: upstream moved this skill to "
                    f"{dest.relative_to(_skills_dir()).as_posix()}, but your "
                    f"modified copy at {rel} was kept — it will not receive "
                    f"updates. Run `hermes skills reset {skill_name} --restore` "
                    f"to move to the new location."
                )
            continue
        try:
            _move_dir(candidate, dest)
        except OSError:
            logger.warning("Could not relocate renamed skill %s -> %s", candidate, dest, exc_info=True)
            return None
        logger.info("Relocated renamed bundled skill: %s -> %s", candidate, dest)
        if not quiet:
            print(f"  → {skill_name} (moved {rel} → {dest.relative_to(_skills_dir()).as_posix()})")
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
    user_hash = _dir_hash(dest)
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
    """
    Sync bundled skills into ~/.hermes/skills/ using the manifest.

    Returns:
        dict with keys: copied (list), updated (list), skipped (int),
                        user_modified (list), cleaned (list), total_bundled (int)
    """
    # Opt-out: a profile (named or the default ~/.hermes) that wrote the
    # .no-bundled-skills marker gets zero bundled-skill seeding — EXCEPT the
    # essential skills (agent/skill_utils.ESSENTIAL_SKILLS). The
    # ``hermes-agent`` skill is the agent's own operating manual and the
    # system prompt always points at it, so even a Blank Slate / --no-skills
    # profile keeps that one skill. Returning the empty-result shape with
    # skipped_opt_out lets callers report "opted out" instead of
    # "synced 0 / failed". This is the default-profile counterpart to
    # seed_profile_skills()'s marker check for named profiles.
    essential_only = (_hermes_home() / NO_BUNDLED_SKILLS_MARKER).exists()
    if essential_only and not quiet:
        print(
            "  (profile opted out of bundled skills via .no-bundled-skills — "
            "seeding essential skills only)"
        )

    bundled_dir = _get_bundled_dir()
    if not bundled_dir.exists():
        return {
            "copied": [], "updated": [], "skipped": 0,
            "user_modified": [], "cleaned": [], "suppressed": [], "total_bundled": 0,
            "optional_provenance_backfilled": [],
        }

    _skills_dir().mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest()
    bundled_skills = _discover_bundled_skills(bundled_dir)
    if essential_only:
        # Opted-out profile: only the essential skills are synced.
        bundled_skills = [
            (name, src) for name, src in bundled_skills
            if name in _essential_names()
        ]
    bundled_names = {name for name, _ in bundled_skills}
    suppressed = _read_suppressed_names()
    external_index = _build_external_skill_index()
    st = _SyncState(manifest=_read_manifest(), quiet=quiet)

    for skill_name, skill_src in bundled_skills:
        # Curator-pruned built-ins: do not re-seed. The suppression list
        # (~/.hermes/skills/.curator_suppressed) is written when the curator
        # archives a bundled skill with curator.prune_builtins enabled. Without
        # this skip, every `hermes update` would resurrect a skill the user
        # deliberately pruned. Restoring the skill clears its suppression entry.
        # Essential skills are exempt — they must always come back.
        if skill_name in suppressed and skill_name not in _essential_names():
            suppressed_skipped.append(skill_name)
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
            # ── In manifest but not on disk — user deleted it ──
            skipped += 1

    # Clean stale manifest entries (skills removed from bundled dir).
    # Skip on an opted-out profile: bundled_skills was filtered to the
    # essential set there, and cleaning would drop tracking for every other
    # previously-synced skill still on disk.
    if essential_only:
        cleaned = []
    else:
        cleaned = sorted(set(manifest.keys()) - bundled_names)
        for name in cleaned:
            del manifest[name]

    # Also copy DESCRIPTION.md files for categories (if not already present).
    # On an opted-out profile only the essential skills' own category
    # descriptions are seeded — not the full catalog's.
    _essential_cat_dirs = {
        _compute_relative_dest(src, bundled_dir).parent
        for _, src in bundled_skills
    } if essential_only else None
    for desc_md in bundled_dir.rglob("DESCRIPTION.md"):
        rel = desc_md.relative_to(bundled_dir)
        dest_desc = _skills_dir() / rel
        if _essential_cat_dirs is not None and dest_desc.parent not in _essential_cat_dirs:
            continue
        if not dest_desc.exists():
            try:
                dest_desc.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(desc_md, dest_desc)
            except (OSError, IOError) as e:
                logger.debug("Could not copy %s: %s", desc_md, e)

    _write_manifest(manifest)
    optional_provenance_backfilled = _backfill_optional_provenance(quiet=quiet)

    return {
        "copied": st.copied, "updated": st.updated, "skipped": st.skipped, "user_modified": st.user_modified,
        "cleaned": cleaned, "suppressed": st.suppressed, "relocated": st.relocated,
        "total_bundled": len(bundled_skills),
        "optional_provenance_backfilled": optional_provenance_backfilled,
        "shadowed_by_external": shadowed_by_external,
        # Opted-out profiles still seed essential skills; the flag lets
        # callers report "opted out" rather than a normal full sync.
        "skipped_opt_out": essential_only,
    }


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
    # Every legitimate caller passes a skill directory or its ``.bak``
    # sibling — always a strict child of the skills root. The skills root
    # itself must never be removed: a ``dest`` that collapses to
    # ``SKILLS_DIR`` (e.g. a relative path resolving to ``.``) would wipe
    # every installed skill, and its ``.bak`` sibling lands one level up in
    # ``HERMES_HOME``. Require a strict-child relationship so both escape
    # into the skills root and out of it are refused.
    if skills_root not in target.parents:
        raise ValueError(f"refusing to rmtree {target!r}: not strictly under {skills_root!r} (scope guard — see #48200)")

    def _on_error(func, fpath, exc_info):
        for p in (os.path.dirname(fpath), fpath):
            with suppress(OSError):
                os.chmod(p, stat.S_IRWXU)
        func(fpath)
    shutil.rmtree(path, onerror=_on_error)


def reset_bundled_skill(name: str, restore: bool = False) -> dict:
    """
    Reset a bundled skill's manifest tracking so future syncs work normally.

    When a user edits a bundled skill, subsequent syncs mark it as
    ``user_modified`` and skip it forever — even if the user later copies
    the bundled version back into place, because the manifest still holds
    the *old* origin hash. This function breaks that loop.

    Args:
        name: The skill name (matches the manifest key / skill frontmatter name).
        restore: If True, also delete the user's copy in the skills dir and let
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

    # Step 1 (optional): delete the user's copy so next sync re-copies bundled.
    # Must happen BEFORE manifest deletion so that a failed rmtree does not
    # leave the skill in a manifest-less limbo state (see #34972).
    deleted_user_copy = False
    if restore:
        if not is_bundled:
            return {
                "ok": False,
                "action": "bundled_missing",
                "message": (
                    f"'{name}' has no bundled source — manifest entry preserved "
                    f"but cannot restore from bundled (skill was removed upstream)."
                ),
                "synced": None,
            }
        dest = _compute_relative_dest(bundled_by_name[name], bundled_dir)
        if dest.exists():
            try:
                _rmtree_writable(dest)
                deleted_user_copy = True
            except (OSError, IOError) as e:
                return {
                    "ok": False,
                    "action": "not_reset",
                    "message": (
                        f"Could not delete user copy at {dest}: {e}. "
                        f"Manifest entry preserved — nothing was changed."
                    ),
                    "synced": None,
                }

    # Step 2: drop the manifest entry so next sync treats it as new
    if in_manifest:
        del manifest[name]
        _write_manifest(manifest)

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


def _is_tracked_user_modification(origin_hash: str, user_hash: str) -> bool:
    """Whether an on-disk skill counts as a user modification ``hermes update`` keeps.

    Shared by the sync loop (which decides what to skip) and
    ``list_user_modified_bundled_skills`` (which surfaces the names) so the two
    can never drift. A skill is a tracked modification only when it has a
    recorded origin hash (an un-baselined / v1 entry with an empty hash is not)
    and its current content hash differs from that origin.
    """
    return bool(origin_hash) and user_hash != origin_hash


def list_user_modified_bundled_skills() -> List[dict]:
    """Return the bundled skills that ``hermes update`` keeps because the user
    edited them locally.

    A skill counts as user-modified when its on-disk copy no longer matches the
    origin hash recorded in the manifest the last time it was synced — the exact
    same test the sync loop uses to decide what to skip. This is the discovery
    half of that behavior, so a user can find the names the ``~ N user-modified
    (kept)`` notice only counts.

    Returns a list (sorted by name) of dicts:
        ``{"name": str, "dest": Path, "bundled_src": Path}``
    where ``dest`` is the user's copy and ``bundled_src`` is the current stock
    copy (so callers can diff or restore).
    """
    manifest = _read_manifest()
    if not manifest:
        return []
    bundled_dir = _get_bundled_dir()
    modified: List[dict] = []
    for skill_name, skill_dir in _discover_bundled_skills(bundled_dir):
        origin_hash = manifest.get(skill_name, "")
        # No entry, or a v1 entry not yet baselined (empty hash): not a tracked
        # modification — the next sync handles it.
        if not origin_hash:
            continue
        dest = _compute_relative_dest(skill_dir, bundled_dir)
        if not dest.exists():
            continue
        if _is_tracked_user_modification(origin_hash, _dir_hash(dest)):
            modified.append(
                {"name": skill_name, "dest": dest, "bundled_src": skill_dir}
            )
    modified.sort(key=lambda e: e["name"])
    return modified


def _read_for_diff(path: Path) -> Tuple[Optional[bytes], Optional[str]]:
    """Read a file once for diffing.

    Returns ``(raw_bytes, text)`` where ``text`` is ``None`` if the file is
    binary; ``(None, None)`` if it could not be read. Returning the raw bytes
    lets the caller compare binary files without re-reading them.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None, None
    if b"\x00" in data:
        return data, None
    try:
        return data, data.decode("utf-8")
    except UnicodeDecodeError:
        return data, None


def diff_bundled_skill(name: str) -> dict:
    """Diff a user's copy of a bundled skill against the current stock version.

    Lets a user see exactly what diverged before deciding whether to keep their
    edits or ``hermes skills reset`` back to upstream.

    Returns a dict:
        ``ok`` (bool), ``name`` (str), ``found`` (bool — bundled source exists),
        ``modified`` (bool), ``message`` (str),
        ``diffs``: list of ``{"path": str, "status": str, "diff": str}`` where
        status is one of ``modified`` / ``added`` (only in user copy) /
        ``removed`` (only in bundled) / ``binary``.
    """
    import difflib

    bundled_dir = _get_bundled_dir()
    bundled_by_name = dict(_discover_bundled_skills(bundled_dir))
    bundled_src = bundled_by_name.get(name)
    if bundled_src is None:
        return {
            "ok": False,
            "name": name,
            "found": False,
            "modified": False,
            "diffs": [],
            "message": (
                f"'{name}' is not a tracked bundled skill (no stock version to "
                f"diff against). Hub-installed skills use `hermes skills inspect`."
            ),
        }
    dest = _compute_relative_dest(bundled_src, bundled_dir)
    if not dest.exists():
        return {
            "ok": False,
            "name": name,
            "found": True,
            "modified": False,
            "diffs": [],
            "message": f"No local copy of '{name}' found at {dest}.",
        }

    user_files = set(_skill_file_list(dest))
    stock_files = set(_skill_file_list(bundled_src))

    diffs: List[dict] = []
    for rel in sorted(user_files | stock_files):
        in_user = rel in user_files
        in_stock = rel in stock_files
        user_bytes, user_text = (
            _read_for_diff(dest / rel) if in_user else (None, None)
        )
        stock_bytes, stock_text = (
            _read_for_diff(bundled_src / rel) if in_stock else (None, None)
        )

        if in_user and in_stock:
            if user_text is None or stock_text is None:
                # At least one side is binary — report only if bytes differ
                # (reuse the bytes already read above, no second read).
                if user_bytes != stock_bytes:
                    diffs.append(
                        {"path": rel, "status": "binary", "diff": "<binary file differs>"}
                    )
                continue
            if user_text == stock_text:
                continue
            text = "".join(
                difflib.unified_diff(
                    stock_text.splitlines(keepends=True),
                    user_text.splitlines(keepends=True),
                    fromfile=f"stock/{rel}",
                    tofile=f"yours/{rel}",
                )
            )
            diffs.append({"path": rel, "status": "modified", "diff": text})
        elif in_user:
            diffs.append(
                {"path": rel, "status": "added", "diff": f"+ only in your copy: {rel}"}
            )
        else:
            diffs.append(
                {"path": rel, "status": "removed", "diff": f"- only in stock: {rel}"}
            )

    modified = bool(diffs)
    return {
        "ok": True,
        "name": name,
        "found": True,
        "modified": modified,
        "diffs": diffs,
        "message": (
            f"'{name}' matches the stock version."
            if not modified
            else f"'{name}' differs from the stock version in {len(diffs)} file(s)."
        ),
    }


def set_bundled_skills_opt_out(enabled: bool) -> dict:
    """Toggle the .no-bundled-skills opt-out marker for the active profile.

    When ``enabled`` is True, writes HERMES_HOME/.no-bundled-skills so the
    installer, ``hermes update``, and any direct sync stop seeding bundled
    skills. When False, removes the marker so seeding resumes on the next
    sync. This is the on-disk-state half of ``hermes skills opt-out`` /
    ``opt-in``; removal of already-present skills is a separate, explicit
    step (see ``remove_pristine_bundled_skills``).

    Returns:
        dict with keys: ok (bool), changed (bool), marker (str path),
                        message (str).
    """
    marker = _hermes_home() / NO_BUNDLED_SKILLS_MARKER
    existed = marker.exists()
    try:
        if enabled:
            _hermes_home().mkdir(parents=True, exist_ok=True)
            marker.write_text(
                "This profile opted out of bundled-skill seeding "
                "(`hermes skills opt-out`).\n"
                "Delete this file to re-enable sync on the next `hermes update`.\n",
                encoding="utf-8",
            )
            changed = not existed
            message = (
                "Opted out of bundled skills. Future install / update / sync "
                "runs will not seed bundled skills into this profile."
                if changed
                else "Already opted out — marker was already present."
            )
        else:
            if existed:
                marker.unlink()
            changed = existed
            message = (
                "Opted back in. The next `hermes update` (or `hermes skills "
                "opt-in --sync`) will re-seed bundled skills."
                if changed
                else "Not opted out — no marker to remove."
            )
    except OSError as e:
        return {
            "ok": False, "changed": False, "marker": str(marker),
            "message": f"Could not update opt-out marker at {marker}: {e}",
        }
    return {"ok": True, "changed": changed, "marker": str(marker), "message": message}


def is_bundled_skills_opt_out() -> bool:
    """Return True if the active profile carries the opt-out marker."""
    return (_hermes_home() / NO_BUNDLED_SKILLS_MARKER).exists()


def remove_pristine_bundled_skills(dry_run: bool = False) -> dict:
    """Delete bundled skills that are present, manifest-tracked, AND unmodified.

    Safety is the whole point of this function. A skill on disk is removed
    ONLY when all of these hold:
      - it is recorded in the sync manifest (so it is genuinely a bundled
        skill, not a hub-installed or hand-written one), AND
      - it still exists in the bundled source (so we can hash-compare), AND
      - its on-disk copy is byte-identical to the manifest origin hash
        (so the user has not edited it).

    Anything user-modified, hub-installed, or locally authored is left
    untouched and reported under ``skipped``. The manifest entry for each
    removed skill is dropped so a later opt-in re-seed treats it as new.

    Args:
        dry_run: When True, compute what would be removed without deleting.

    Returns:
        dict with keys: ok (bool), removed (list[str]),
                        skipped (list[dict]) where each dict is
                        {name, reason}, dry_run (bool), message (str).
    """
    manifest = _read_manifest()
    bundled_dir = _get_bundled_dir()
    bundled_by_name = dict(_discover_bundled_skills(bundled_dir))

    removed: List[str] = []
    skipped: List[dict] = []

    for name, origin_hash in sorted(manifest.items()):
        src = bundled_by_name.get(name)
        if src is None:
            # Tracked but no longer bundled upstream — leave it; not ours to judge.
            skipped.append({"name": name, "reason": "no bundled source (removed upstream)"})
            continue
        dest = _compute_relative_dest(src, bundled_dir)
        if not dest.exists():
            # Already gone from disk; just forget the stale manifest entry.
            if not dry_run and name in manifest:
                del manifest[name]
            continue
        on_disk = _dir_hash(dest)
        if on_disk != origin_hash:
            skipped.append({"name": name, "reason": "user-modified (kept)"})
            continue
        # Pristine bundled copy — safe to remove.
        if dry_run:
            removed.append(name)
            continue
        try:
            _rmtree_writable(dest)
        except (OSError, IOError) as e:
            skipped.append({"name": name, "reason": f"delete failed: {e}"})
            continue
        if name in manifest:
            del manifest[name]
        removed.append(name)

    if not dry_run and removed:
        _write_manifest(manifest)

    verb = "Would remove" if dry_run else "Removed"
    message = f"{verb} {len(removed)} pristine bundled skill(s); kept {len(skipped)}."
    return {
        "ok": True, "removed": removed, "skipped": skipped,
        "dry_run": dry_run, "message": message,
    }


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
