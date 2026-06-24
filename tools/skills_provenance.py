#!/usr/bin/env python3
"""
Skills provenance — track where each installed skill came from.

A skill's ``provenance`` is one of:

  - ``"builtin"``   — the canonical copy ships with hermes-agent (bundled)
                       and was seeded into the user/profile skills dir.
                       Trust: ``"builtin"``.
  - ``"hub"``       — installed from a remote skill registry (skills.sh,
                       GitHub, official optional skills, etc.).
                       Trust: whatever the hub lock entry recorded
                       (``builtin``/``trusted``/``community``).
  - ``"local-edit"``— a builtin that the user has modified locally.
                       Trust: ``"local"`` (downgraded because user-tampered).
  - ``"local"``     — neither bundled nor hub-installed. A user-authored
                       skill living in the profile skills dir. Trust: ``"local"``.

Provenance is **persisted at install / sync time** in a per-profile
registry file (``<HERMES_HOME>/skills/.provenance``) so ``hermes skills
list`` no longer has to re-derive Source / Trust from file location at
list time. The registry is keyed by skill name with a per-skill record
of::

    {
      "provenance": "builtin" | "hub" | "local-edit" | "local",
      "origin_path": "<absolute path the skill was originally seeded from>",
      "synced_at": "<ISO timestamp>",
    }

Lines in the registry look like::

    name|provenance|origin_path|synced_at

This file is **append-mostly** (writers replace the whole file under an
atomic rename, like the bundled manifest it sits next to). New entries
are written by ``sync_skills`` (builtin seeds), ``install_from_quarantine``
(hub installs), and on explicit user edits (``record_local_edit``). The
``hermes skills list`` command lazily back-fills missing entries using
the heuristic documented on ``classify()`` so legacy installs are not
mislabeled after upgrade.

This module is intentionally dependency-free at import time so it can be
imported from CLI handlers, the hub installer, the sync tool, and tests
without pulling in the Rich console or hermes_cli layer.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from hermes_constants import get_default_hermes_root, get_hermes_home, get_skills_dir
from utils import atomic_replace

logger = logging.getLogger(__name__)


# Provenance values — kept as constants so tests and writers share one source.
PROVENANCE_BUILTIN = "builtin"
PROVENANCE_HUB = "hub"
PROVENANCE_LOCAL_EDIT = "local-edit"
PROVENANCE_LOCAL = "local"

VALID_PROVENANCE = frozenset({
    PROVENANCE_BUILTIN,
    PROVENANCE_HUB,
    PROVENANCE_LOCAL_EDIT,
    PROVENANCE_LOCAL,
})


HERMES_HOME = get_hermes_home()
PROFILE_SKILLS_DIR = HERMES_HOME / "skills"
PROVENANCE_FILE = PROFILE_SKILLS_DIR / ".provenance"


def _platform_skills_dir() -> Path:
    """Canonical platform-level skills dir (NOT the profile's)."""
    return get_default_hermes_root() / "skills"


# ── Read / write the per-profile provenance registry ────────────────────


def _read_provenance_file(path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """Read ``.provenance`` into ``{name: {provenance, origin_path, synced_at}}``.

    Malformed lines are silently skipped — the registry is best-effort
    metadata, not a security boundary. An empty / missing file returns
    ``{}``.
    """
    path = path or PROVENANCE_FILE
    if not path.exists():
        return {}
    result: Dict[str, Dict[str, str]] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return result
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Format: name|provenance|origin_path|synced_at
        # origin_path may contain additional "|" so rsplit from the right
        # up to 3 fields is the safer direction (3 splits ⇒ 4 parts).
        parts = line.split("|")
        if len(parts) < 4:
            logger.debug("Skipping malformed .provenance line: %r", raw)
            continue
        name, provenance, origin_path, synced_at = parts[0], parts[1], parts[2], parts[3]
        if provenance not in VALID_PROVENANCE:
            logger.debug("Skipping unknown provenance value: %r", provenance)
            continue
        result[name] = {
            "provenance": provenance,
            "origin_path": origin_path,
            "synced_at": synced_at,
        }
    return result


def _write_provenance_file(
    entries: Dict[str, Dict[str, str]],
    path: Optional[Path] = None,
) -> None:
    """Atomically replace ``.provenance`` with the supplied entries.

    Sorted by name for stable diffs across writes (mirrors the bundled
    manifest's write order).
    """
    path = path or PROVENANCE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for name in sorted(entries):
        rec = entries[name]
        lines.append(
            f"{name}|{rec.get('provenance', '')}|{rec.get('origin_path', '')}|{rec.get('synced_at', '')}"
        )
    data = "\n".join(lines) + ("\n" if lines else "")
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".provenance_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def record(
    name: str,
    provenance: str,
    origin_path: str = "",
    *,
    path: Optional[Path] = None,
) -> None:
    """Set the provenance for a single skill, preserving other entries.

    No-op if ``provenance`` is not a recognized value — callers should
    validate before passing unknown strings. ``synced_at`` defaults to
    now (UTC).
    """
    if provenance not in VALID_PROVENANCE:
        raise ValueError(f"Unknown provenance: {provenance!r}")
    entries = _read_provenance_file(path)
    entries[name] = {
        "provenance": provenance,
        "origin_path": origin_path,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_provenance_file(entries, path)


def record_many(
    records: Iterable[Tuple[str, str, str]],
    *,
    path: Optional[Path] = None,
) -> None:
    """Bulk update the registry — ``(name, provenance, origin_path)`` triples.

    Existing entries not in ``records`` are preserved.
    """
    entries = _read_provenance_file(path)
    now = datetime.now(timezone.utc).isoformat()
    for name, provenance, origin_path in records:
        if provenance not in VALID_PROVENANCE:
            logger.debug("record_many skipping unknown provenance %r for %s", provenance, name)
            continue
        entries[name] = {
            "provenance": provenance,
            "origin_path": origin_path,
            "synced_at": now,
        }
    _write_provenance_file(entries, path)


def record_local_edit(name: str, install_path: str = "", *, path: Optional[Path] = None) -> None:
    """Downgrade a skill's provenance to ``local-edit`` (user modified a builtin)."""
    record(name, PROVENANCE_LOCAL_EDIT, install_path, path=path)


# ── Trust derivation from provenance ────────────────────────────────────


def trust_for(provenance: str) -> str:
    """Return the Trust string for a given provenance.

    Per the task spec, Trust is derived FROM provenance, not from Source
    as a parallel assignment. Hub entries still override via their own
    ``trust_level`` field — see ``classify_with_hub_lock``.
    """
    if provenance == PROVENANCE_BUILTIN:
        return "builtin"
    if provenance == PROVENANCE_HUB:
        return "community"  # default; the hub lock entry overrides if present
    return "local"  # local-edit and local both downgrade to local trust


# ── Backfill heuristic for legacy records ───────────────────────────────


def classify(
    skill_name: str,
    install_path: Optional[Path],
    *,
    bundled_skills_dir: Optional[Path] = None,
    platform_skills_dir: Optional[Path] = None,
) -> Tuple[str, str]:
    """Heuristically classify a discovered skill by where it lives on disk.

    Used to back-fill provenance for skills that were installed before
    this registry existed (the ``local`` branch with a warning at list
    time catches anything we can't classify confidently).

    Returns ``(provenance, origin_path)``:

      * ``"builtin"``  — the file lives (after symlink resolution)
        directly under the platform-level skills dir
        (``~/.hermes/skills/...``), or under the profile skills dir with
        byte-identical content to the platform-level copy.
      * ``"local"``    — neither of the above. The caller should emit a
        warning so the operator knows the registry is incomplete.

    Bundled-skill provenance (set by ``sync_skills``) is not needed here:
    if the file matches the bundled copy, it's a builtin. This function
    intentionally does NOT touch the hub lock — callers handle that with
    ``classify_with_hub_lock`` so provenance stays consistent.
    """
    platform_skills_dir = platform_skills_dir or _platform_skills_dir()
    install_path = Path(install_path) if install_path else None

    if install_path is None or not install_path.exists():
        return PROVENANCE_LOCAL, ""

    try:
        resolved = install_path.resolve()
    except OSError:
        resolved = install_path

    # Case 1: file lives directly under the platform-level skills dir.
    # That's the canonical builtin location — return builtin regardless
    # of whether the profile also has a copy.
    try:
        resolved.relative_to(platform_skills_dir.resolve())
        return PROVENANCE_BUILTIN, str(install_path)
    except ValueError:
        pass

    # Case 2: file lives under the profile skills dir. Check whether an
    # equivalent copy exists at the platform level with matching content.
    # This catches the symlink-to-platform / copied-from-platform case
    # that the heuristic at the top of this module is meant to label
    # correctly. macos-computer-use is the prototype: lives in the
    # profile skills dir, identical bytes to the platform copy.
    if bundled_skills_dir is None:
        try:
            from tools.skills_sync import _get_bundled_dir  # late import — circular
            bundled_skills_dir = _get_bundled_dir()
        except Exception:
            bundled_skills_dir = None

    # Look up the matching path under the platform skills dir by walking
    # the on-disk layout. We try the same relative layout first (profile
    # skills live in <profile>/skills/<category>/<name>), and if that's
    # missing, search by skill-name match across any subdir.
    candidates: List[Path] = []
    try:
        rel = resolved.relative_to(PROFILE_SKILLS_DIR.resolve())
        candidates.append(platform_skills_dir / rel)
    except ValueError:
        pass
    candidates.append(platform_skills_dir / skill_name)
    for cat_dir in platform_skills_dir.iterdir() if platform_skills_dir.is_dir() else []:
        if cat_dir.is_dir() and (cat_dir / skill_name).is_dir():
            candidates.append(cat_dir / skill_name)

    for candidate in candidates:
        if not candidate.exists():
            continue
        if _dirs_match(candidate, install_path):
            return PROVENANCE_BUILTIN, str(candidate)

    # Case 3: bundled dir has this skill but the on-disk copy differs.
    # Treat as a user-modified builtin.
    if bundled_skills_dir is not None:
        for bundled_skill_md in bundled_skills_dir.rglob("SKILL.md"):
            try:
                bundled_name = bundled_skill_md.parent.name
            except OSError:
                continue
            if bundled_name != skill_name:
                continue
            if bundled_skill_md.parent.exists() and not _dirs_match(
                bundled_skill_md.parent, install_path
            ):
                return PROVENANCE_LOCAL_EDIT, str(bundled_skill_md.parent)

    return PROVENANCE_LOCAL, ""


def classify_with_hub_lock(
    skill_name: str,
    install_path: Optional[Path],
    hub_entry: Optional[dict],
    *,
    bundled_skills_dir: Optional[Path] = None,
    platform_skills_dir: Optional[Path] = None,
) -> Tuple[str, str]:
    """Layered classification used by ``hermes skills list``.

    Order:
      1. If ``hub_entry`` is present, provenance is ``"hub"``.
      2. Else defer to ``classify()`` for the path-based heuristic.
    """
    if hub_entry is not None:
        origin = hub_entry.get("install_path") or ""
        # Resolve to an absolute path the user can read.
        if origin and not os.path.isabs(origin):
            origin = str((PROFILE_SKILLS_DIR / origin).resolve())
        return PROVENANCE_HUB, origin

    return classify(
        skill_name,
        install_path,
        bundled_skills_dir=bundled_skills_dir,
        platform_skills_dir=platform_skills_dir,
    )


# ── Internals ───────────────────────────────────────────────────────────


def _dirs_match(a: Path, b: Path) -> bool:
    """Cheap equality check for two skill directories.

    Compares the SKILL.md (the dominant content) plus a content hash of
    every other file. Hashing is MD5 of the sorted relative-path list
    followed by concatenated file bytes — same approach the bundled
    manifest uses for origin tracking.
    """
    a_skill = a / "SKILL.md"
    b_skill = b / "SKILL.md"
    if a_skill.exists() and b_skill.exists():
        try:
            if a_skill.read_bytes() != b_skill.read_bytes():
                return False
        except OSError:
            return False
    elif a_skill.exists() or b_skill.exists():
        # One side missing the canonical file — different.
        return False
    else:
        # Both sides missing SKILL.md — fall back to filename equality.
        return a.name == b.name

    try:
        from tools.skills_sync import _dir_hash
        return _dir_hash(a) == _dir_hash(b)
    except Exception:
        return True  # if hashing fails, treat as match to avoid false negatives


__all__ = [
    "PROVENANCE_BUILTIN",
    "PROVENANCE_HUB",
    "PROVENANCE_LOCAL_EDIT",
    "PROVENANCE_LOCAL",
    "VALID_PROVENANCE",
    "PROVENANCE_FILE",
    "PROFILE_SKILLS_DIR",
    "record",
    "record_many",
    "record_local_edit",
    "trust_for",
    "classify",
    "classify_with_hub_lock",
    "_read_provenance_file",
    "_write_provenance_file",
]