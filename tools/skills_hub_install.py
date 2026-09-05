"""Skills Hub install/uninstall/update operations: quarantine staging, install-
target safety (symlink/junction, category-bucket and nested-skill checks),
lock-file-backed uninstall, bundle hashing and upstream update checks.

Split out of ``tools/skills_hub.py``; hub state (paths, ``HubLockFile``, audit log)
is still read from there at call time.
"""

from __future__ import annotations

import logging
import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from agent.skill_utils import is_excluded_skill_path
from tools.skills_guard import ScanResult, content_hash
from tools.skills_hub_github import GitHubAuth
from tools.skills_hub_models import (
    SkillBundle, SkillSource, _normalize_lock_install_path, _validate_bundle_rel_path,
    _validate_install_parent_path, _validate_skill_name,
)

if TYPE_CHECKING:  # origin class; runtime use is via the lazy origin import
    from tools.skills_hub import HubLockFile

# Log-record parity with the origin module.
logger = logging.getLogger("tools.skills_hub")


def _is_path_redirect(path: Path) -> bool:
    """True when ``path`` is a symlink or (on Windows) a directory junction —
    either lets a writer in ``skills/`` redirect a later ``rmtree`` outside it.
    ``is_junction`` only exists on Python 3.12+ Windows; gate with ``hasattr``."""
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _resolve_lock_install_path(install_path: str, skill_name: str) -> Path:
    """Resolve a lock-file install path without allowing escapes from ``SKILLS_DIR``.

    Walks component-by-component refusing symlink/junction redirects (which
    ``Path.resolve`` would silently follow), then rejects both escape-out and
    ``resolved == SKILLS_DIR`` — an empty/``"."`` install_path resolves to the
    skills root and ``rmtree`` there would wipe every installed skill.
    """
    from tools.skills_hub import _skills_dir
    normalized = _normalize_lock_install_path(install_path, skill_name)
    target = skills_dir = _skills_dir()
    skills_root = skills_dir.resolve()
    for part in normalized.split("/"):
        target = target / part
        if _is_path_redirect(target):
            raise ValueError(f"Unsafe install path: {install_path}")
    target = target.resolve()
    if target == skills_root or not target.is_relative_to(skills_root):
        raise ValueError(f"Unsafe install path: {install_path}")
    return target


def quarantine_bundle(bundle: SkillBundle) -> Path:
    """Write a skill bundle to the quarantine directory for scanning."""
    from tools.skills_hub import _quarantine_dir, ensure_hub_dirs
    ensure_hub_dirs()
    skill_name = _validate_skill_name(bundle.name)
    # Validate every path before touching disk so a bad member aborts cleanly.
    validated_files = [(_validate_bundle_rel_path(rel_path), content) for rel_path, content in bundle.files.items()]
    dest = _quarantine_dir() / skill_name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for rel_path, file_content in validated_files:
        file_dest = dest.joinpath(*rel_path.split("/"))
        file_dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(file_content, bytes):
            file_dest.write_bytes(file_content)
        else:
            file_dest.write_text(file_content, encoding="utf-8")
    return dest


def _category_skill_dirs(directory: Path) -> List[str]:
    """Names of non-hidden child dirs holding at least one active SKILL.md
    anywhere below (nested layouts like ``mlops/training/<skill>`` count).

    Vendored/cache/progressive-disclosure paths are pruned via
    :func:`is_excluded_skill_path` so a lone ``node_modules`` or
    ``references/pkg/SKILL.md`` does not make the directory a category.
    Shared with ``hermes_cli.skills_hub._existing_categories``.
    """
    return [
        entry.name for entry in directory.iterdir()
        if entry.is_dir() and not entry.name.startswith(".") and any(
            not is_excluded_skill_path(skill_md.relative_to(directory), root=directory)
            for skill_md in entry.rglob("SKILL.md")
        )
    ]


def _check_install_target(install_dir: Path) -> None:
    """Raise ValueError when installing at ``install_dir`` would destroy other skills.

    - Nesting inside an existing skill dir (``--category <existing-skill>``)
      would make a later update/uninstall of the outer skill rmtree the inner one.
    - A stray regular file at the target: rmtree would raise NotADirectoryError.
    - A category bucket (dir without SKILL.md that holds other skills) must never
      be silently wiped; a dir that directly contains SKILL.md is an existing
      install and stays overwritable (hub installs are lock-guarded in do_install).
    """
    from tools.skills_hub import _skills_dir
    # Refuse to nest a skill inside an existing skill directory. Installing with ``--category
    # <name-of-an-existing-skill>`` would create a hybrid skill-plus-category directory; a later update or
    # uninstall of the outer skill would then rmtree the inner one — the sibling case of the category-bucket
    # wipe reported in issue #75983.
    skills_root = _skills_dir().resolve()
    ancestor = install_dir.parent
    while ancestor != skills_root and ancestor.is_relative_to(skills_root):
        if (ancestor / "SKILL.md").is_file():
            raise ValueError(f"Refusing to install into '{ancestor.name}': it is an "
                             f"existing skill directory, not a category. Choose a different category.")
        ancestor = ancestor.parent
    if not install_dir.exists():
        return
    if not install_dir.is_dir():
        raise ValueError(f"Refusing to install: '{install_dir.name}' already exists "
                         f"and is not a directory. Remove it or choose a different skill name.")
    # Guard against silent data loss when the install target collides with an existing category bucket (a
    # directory that holds other skills). This was reported as GitHub issue #75983: installing a skill with
    # --name matching an existing category directory caused rmtree to wipe all sibling skills. A directory
    # that directly contains SKILL.md is an existing skill installation and stays overwritable
    # (hub-installed skills are additionally guarded by the lock-file check in do_install()). But a
    # directory that contains *other* skill directories is a category bucket and must NOT be silently
    # deleted.
    if not (install_dir / "SKILL.md").exists():
        skill_dirs_in = _category_skill_dirs(install_dir)
        if skill_dirs_in:
            raise ValueError(f"Refusing to overwrite category directory '{install_dir}' "
                             f"which contains {len(skill_dirs_in)} skill(s): {', '.join(sorted(skill_dirs_in))}. "
                             f"Use a different --name or install into a subcategory.")


def install_from_quarantine(
    quarantine_path: Path, skill_name: str, category: str, bundle: SkillBundle, scan_result: ScanResult,
    scan_provenance: Optional[Dict[str, Any]] = None,
) -> Path:
    """Move a scanned skill from quarantine into the skills directory."""
    from tools.skills_hub import HubLockFile, _quarantine_dir, _skills_dir, append_audit_log
    safe_skill_name = _validate_skill_name(skill_name)
    safe_category = _validate_install_parent_path(category) if category else ""
    quarantine_resolved = quarantine_path.resolve()
    if not quarantine_resolved.is_relative_to(_quarantine_dir().resolve()):
        raise ValueError(f"Unsafe quarantine path: {quarantine_path}")

    install_rel_path = f"{safe_category}/{safe_skill_name}" if safe_category else safe_skill_name
    # Same validator the uninstaller uses, so a lock entry can never point at a
    # symlink-redirected target.
    install_dir = _resolve_lock_install_path(install_rel_path, safe_skill_name)
    _check_install_target(install_dir)
    if install_dir.exists():
        shutil.rmtree(install_dir)

    try:
        skill_size = (quarantine_path / "SKILL.md").stat().st_size
    except OSError:
        skill_size = 0
    if skill_size > 100_000:
        logger.warning(
            "Skill '%s' has a large SKILL.md (%s chars). "
            "Large skills consume significant context when loaded. "
            "Consider asking the author to split it into smaller files.",
            safe_skill_name, f"{skill_size:,}",
        )

    # A symlink in the bundle would copy its target into skills/ and leak it
    # to the agent on the next skill_view.
    for entry in quarantine_path.rglob("*"):
        if _is_path_redirect(entry):
            try:
                rel = entry.relative_to(quarantine_resolved)
            except ValueError:
                rel = entry
            raise ValueError(f"Installed skill contains symlinks, which is not allowed: {rel}")

    install_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(quarantine_path), str(install_dir))
    installed_hash = content_hash(install_dir)
    HubLockFile().record_install(
        name=safe_skill_name, source=bundle.source, identifier=bundle.identifier, trust_level=bundle.trust_level,
        scan_verdict=scan_result.verdict, skill_hash=installed_hash,
        install_path=install_dir.resolve().relative_to(_skills_dir().resolve()).as_posix(),
        files=list(bundle.files.keys()), metadata=bundle.metadata,
        scan_provenance=scan_provenance or getattr(scan_result, "scan_provenance", None),
    )
    append_audit_log("INSTALL", safe_skill_name, bundle.source, bundle.trust_level, scan_result.verdict,
                     installed_hash)
    try:
        from tools.skill_usage import record_installed
        record_installed(safe_skill_name)
    except Exception:
        logger.debug("Unable to record skill install lifecycle for %s", safe_skill_name, exc_info=True)
    return install_dir


def uninstall_skill(skill_name: str) -> Tuple[bool, str]:
    """Remove a hub-installed skill. Refuses to remove builtins."""
    from tools.skills_hub import HubLockFile, append_audit_log
    lock = HubLockFile()
    entry = lock.get_installed(skill_name)
    if not entry:
        return False, f"'{skill_name}' is not a hub-installed skill (may be a builtin)"
    # The destructive boundary: whatever reaches rmtree MUST be inside
    # SKILLS_DIR and MUST NOT be SKILLS_DIR itself (see _resolve_lock_install_path).
    try:
        install_path = _resolve_lock_install_path(entry.get("install_path", ""), skill_name)
    except ValueError as exc:
        return False, f"Refusing to uninstall '{skill_name}': {exc}"
    if install_path.exists():
        shutil.rmtree(install_path)
    lock.record_uninstall(skill_name)
    append_audit_log("UNINSTALL", skill_name, entry["source"], entry["trust_level"], "n/a", "user_request")
    return True, f"Uninstalled '{skill_name}' from {entry['install_path']}"


def bundle_content_hash(bundle: SkillBundle) -> str:
    """Deterministic hash of an in-memory bundle.

    MUST stay symmetric with ``tools.skills_guard.content_hash`` (same skill
    from disk), which keys files by POSIX relative path. Windows bundle keys
    carry backslashes, which changed both bytes and sort order and made every
    skill report ``update_available`` forever — normalize before hashing. The
    path is hashed too so swapping contents between two files changes the hash.

    That function keys files by ``relative_to(...).as_posix()`` — forward slashes on every OS. See #62310.
    """
    h = hashlib.sha256()
    normalized = {rel_path.replace("\\", "/"): content for rel_path, content in bundle.files.items()}
    for rel_path in sorted(normalized):
        h.update(rel_path.encode("utf-8"))
        h.update(b"\x00")
        content = normalized[rel_path]
        h.update(content if isinstance(content, bytes) else content.encode("utf-8"))
    return f"sha256:{h.hexdigest()[:16]}"


_SOURCE_ID_ALIASES = {"skills.sh": "skills-sh"}


def _source_matches(source: SkillSource, source_name: str) -> bool:
    return source.source_id() == _SOURCE_ID_ALIASES.get(source_name, source_name)


def check_for_skill_updates(
    name: Optional[str] = None, *, lock: Optional[HubLockFile] = None,
    sources: Optional[List[SkillSource]] = None, auth: Optional[GitHubAuth] = None,
) -> List[dict]:
    """Check installed hub skills for upstream changes.

    Each entry is fetched ONLY from adapters matching its recorded source.
    Falling back to all sources let a same-named skill in a different registry
    satisfy the fetch and silently reassign provenance (names are not
    namespaced across registries), so a missing adapter reports "unavailable".
    """
    from tools.skills_hub import HubLockFile
    from tools.skills_hub_search import create_source_router
    lock = lock or HubLockFile()
    installed = lock.list_installed()
    if name:
        installed = [entry for entry in installed if entry.get("name") == name]
    if sources is None:
        sources = create_source_router(auth=auth)

    results: List[dict] = []
    for entry in installed:
        identifier, source_name = entry.get("identifier", ""), entry.get("source", "")
        row = {"name": entry.get("name", ""), "identifier": identifier, "source": source_name}
        bundle = None
        for src in filter(lambda s: _source_matches(s, source_name), sources):
            try:
                bundle = src.fetch(identifier)
            except Exception:
                bundle = None
            if bundle:
                break
        if not bundle:
            results.append({**row, "status": "unavailable"})
            continue
        current_hash, latest_hash = entry.get("content_hash", ""), bundle_content_hash(bundle)
        results.append({
            **row, "status": "up_to_date" if current_hash == latest_hash else "update_available",
            "current_hash": current_hash, "latest_hash": latest_hash, "bundle": bundle,
        })
    return results
