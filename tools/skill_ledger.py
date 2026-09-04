"""Per-mutation skill audit ledger + single-edit rollback.

Every skill mutation (any actor) appends one JSONL entry to
``~/.hermes/skills/.curator_ledger.jsonl`` with before/after file manifests whose
contents are stored content-addressed (sha256-deduped) under
``~/.hermes/.curator_backups/blobs/``. JSONL, not the state DB: durable, greppable,
survives DB resets. TELEMETRY, NOT A GATE: every public write path swallows and
logs — except ``rollback_entry``, which FAILS CLOSED when its safety capture fails.
"""

from __future__ import annotations

import contextvars
import hashlib
from contextlib import suppress
import json
import logging
import os
import re
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Snapshot-id shape of agent.curator_backup (duplicated to avoid importing the backup stack).
_BACKUP_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z(-\d{2})?$")
# ".archive/<name>-YYYYMMDDHHMMSS" collision suffix added by archive_skill.
_ARCHIVE_TS_SUFFIX_RE = re.compile(r"^(.+)-\d{14}$")
# Rollback of these must restore a COMPLETE package: consolidation may have re-homed
# support files first, so a disk-only capture would restore a hollow skill.
_PACKAGE_RESTORE_ACTIONS = frozenset({"delete", "archive", "purge"})
_VALID_ACTORS = {"curator", "agent", "user"}
_NON_PACKAGE_TOPS = {".curator_backups", ".hub", ".archive"}

# Explicit actor override: the CLI sets "user", the curator walk sets "curator".
_actor_override: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "skill_ledger_actor", default=None)


def set_ledger_actor(actor: Optional[str]) -> contextvars.Token:
    """Bind an explicit actor for this context; reset_ledger_actor(token) in a finally."""
    return _actor_override.set(actor)


def reset_ledger_actor(token: contextvars.Token) -> None:
    _actor_override.reset(token)


def derive_actor() -> str:
    """Explicit override, else background-review provenance -> curator, else agent."""
    override = _actor_override.get()
    if override in _VALID_ACTORS:
        return override
    with suppress(Exception):
        from tools.skill_provenance import is_background_review
        if is_background_review():
            return "curator"
    return "agent"


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def ledger_path() -> Path:
    return _skills_dir() / ".curator_ledger.jsonl"


def blobs_dir() -> Path:
    return get_hermes_home() / ".curator_backups" / "blobs"


def ledger_enabled() -> bool:
    """Config gate ``skills.ledger`` (default True); lazy import keeps this importable without the CLI."""
    try:
        from hermes_cli.config import cfg_get, load_config
        return bool(cfg_get(load_config(), "skills", "ledger", default=True))
    except Exception as e:  # pragma: no cover — best-effort config read
        logger.debug("skill_ledger: config read failed (%s); defaulting on", e)
        return True


def _rel_posix(path: Path | str, root: Path) -> Optional[str]:
    """POSIX path of ``path`` relative to ``root`` (both normalized), or None when outside."""
    try:
        return Path(os.path.normpath(str(path))).relative_to(os.path.normpath(str(root))).as_posix()
    except (ValueError, TypeError):
        return None


def _is_within(root: Path, path: Path) -> bool:
    """True when *path* (normalized, no symlink resolution) sits under *root*."""
    return _rel_posix(path, root) is not None


def _store_blob(data: bytes) -> str:
    """Write *data* keyed by sha256 (existing blob left alone). Returns the hash."""
    digest = hashlib.sha256(data).hexdigest()
    dest = blobs_dir() / digest
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".tmp-{uuid.uuid4().hex[:8]}-{digest}")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    return digest


def read_blob(sha256: str) -> Optional[bytes]:
    """Return blob content or None when missing/invalid."""
    if not sha256 or not all(c in "0123456789abcdef" for c in sha256):
        return None
    with suppress(OSError):
        return (blobs_dir() / sha256).read_bytes() if (blobs_dir() / sha256).exists() else None
    return None


def snapshot_paths(root: Optional[Path], *, complete_package: bool = False) -> List[Dict[str, str]]:
    """{path, sha256} for every file under *root*, each stored as a blob; [] when root is
    None/missing. Raises on I/O failure — callers decide whether that is fatal (rollback safety
    capture) or swallowed (telemetry). ``complete_package`` unions in the newest curator
    tarball's files (disk hashes win)."""
    if root is None:
        return []
    root = Path(root)  # gone from disk -> []; the complete_package fill may still recover it
    files = ([root] if root.is_file()
             else sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [])
    out = [{"path": str(f), "sha256": _store_blob(f.read_bytes())} for f in files]
    return fill_snapshot_from_curator_backup(root, out) if complete_package else out


def _package_rel(root: Path) -> Optional[str]:
    """Relative POSIX path of a skill dir under ``skills/``; None when outside it
    or under backup/hub/archive metadata roots (never a package)."""
    posix = (_rel_posix(root, _skills_dir()) or "").strip("/")
    if not posix or posix.split("/", 1)[0] in _NON_PACKAGE_TOPS:
        return None
    return posix


def _strip_archive_timestamp(name: str) -> str:
    match = _ARCHIVE_TS_SUFFIX_RE.match(name)
    return match.group(1) if match else name


def _skill_md_parents(items: Optional[List[Dict[str, str]]]) -> List[Path]:
    return [p.parent for p in (Path(str(i.get("path", ""))) for i in items or []) if p.name == "SKILL.md"]


def package_prefixes(
    root: Optional[Path] = None, skill: Optional[str] = None,
    before: Optional[List[Dict[str, str]]] = None) -> List[str]:
    """Tar member prefixes of this skill's package: live location under ``skills/``,
    the package parent from the before-state SKILL.md path (rollback fills where
    *root* is gone), the bare skill name, and the name minus an archive suffix."""
    candidates = [_package_rel(Path(root)) if root is not None else None]
    candidates += [_package_rel(p) for p in _skill_md_parents(before)]
    candidates += [skill, _strip_archive_timestamp(skill) if skill else None]
    return list(dict.fromkeys(p for p in ((c or "").strip("/") for c in candidates) if p))


def _read_package_files_from_latest_backup(prefixes: List[str]) -> Dict[str, bytes]:
    """``{posix-relpath: bytes}`` under *prefixes* in the newest ``skills/.curator_backups/*/
    skills.tar.gz``; malicious member names (absolute, ``..`` traversal) are rejected."""
    backups = _skills_dir() / ".curator_backups"
    try:
        children = list(backups.iterdir()) if prefixes and backups.is_dir() else []
    except OSError:
        return {}
    candidates = [
        child / "skills.tar.gz" for child in children
        if child.is_dir() and _BACKUP_ID_RE.match(child.name) and (child / "skills.tar.gz").is_file()]
    if not candidates:
        return {}
    # Parent dirs sort lexicographically == chronologically for the id shape.
    archive = max(candidates, key=lambda p: p.parent.name)
    prefixed = tuple(p if p.endswith("/") else p + "/" for p in prefixes)
    exact = set(prefixes)
    out: Dict[str, bytes] = {}
    try:
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                name = member.name.replace("\\", "/").lstrip("./")
                if (not member.isfile() or not name or name.startswith("/")
                        or ".." in Path(name).parts
                        or (name not in exact and not name.startswith(prefixed))):
                    continue
                extracted = tf.extractfile(member)
                if extracted is not None:
                    out[name] = extracted.read()
    except (OSError, tarfile.TarError) as exc:
        logger.debug("skill_ledger: could not read curator backup package: %s", exc)
        return {}
    return out


def fill_snapshot_from_curator_backup(
    root: Optional[Path], existing: Optional[List[Dict[str, str]]] = None, *,
    skill: Optional[str] = None) -> List[Dict[str, str]]:
    """Union missing skill-package files from the newest curator snapshot. Completeness fill, not
    a gate: failures return *existing* unchanged, and only ABSENT paths are filled. Fill targets go
    where rollback must restore them: under *root* when known (for purge that is
    ``.archive/<name>/``, NOT the live tree), else the live skills dir; the tar's leading
    package-dir segment is stripped when *root* already names the package. Every target must stay
    under ``skills/`` and HERMES_HOME."""
    out = list(existing or [])
    prefixes = package_prefixes(root, skill, out)
    if not prefixes:
        return out
    try:
        extra = _read_package_files_from_latest_backup(prefixes)
    except Exception as exc:
        logger.debug("skill_ledger: backup package fill failed: %s", exc)
        return out
    if not extra:
        return out
    skills = _skills_dir()
    dest_root = Path(root) if root is not None else None
    pkg_names = {dest_root.name, _strip_archive_timestamp(dest_root.name)} if dest_root else set()
    have = {rel for rel in (_rel_posix(str(i.get("path", "")), skills) for i in out) if rel is not None}
    for rel, data in extra.items():
        parts = rel.split("/")
        if dest_root is not None and parts and parts[0] in pkg_names:
            parts = parts[1:]
        if not parts:
            continue
        dest = (dest_root if dest_root is not None else skills).joinpath(*parts)
        if not _is_within(skills, dest) or not _is_within(get_hermes_home(), dest):
            continue
        rel_key = _rel_posix(dest, skills)
        if rel_key is None or rel_key in have:
            continue
        try:
            out.append({"path": str(dest), "sha256": _store_blob(data)})
        except Exception as exc:
            logger.debug("skill_ledger: backup blob store failed for %s: %s", rel, exc)
            continue
        have.add(rel_key)
    return out


def append_entry(
    action: str, skill: str, before: Optional[List[Dict[str, str]]] = None,
    after: Optional[List[Dict[str, str]]] = None, actor: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Append one entry -> id, or None when disabled / write failed (never raises)."""
    if not ledger_enabled():
        return None
    try:
        entry = {
            "id": uuid.uuid4().hex[:12], "ts": datetime.now(timezone.utc).isoformat(),
            "actor": actor if actor in _VALID_ACTORS else derive_actor(),
            "action": action, "skill": skill, "evidence": evidence or {},
            "before": before or [], "after": after or []}
        path = ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry["id"]
    except Exception as e:
        logger.warning("skill_ledger: failed to append entry (%s) — mutation unaffected", e)
        return None


def record_mutation(
    action: str, skill: str, before_root: Optional[Path] = None,
    before: Optional[List[Dict[str, str]]] = None, after_root: Optional[Path] = None,
    actor: Optional[str] = None, evidence: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Mutation hook: after-state from *after_root* (before = pre-captured list or captured from
    *before_root*), then append. NEVER raises. delete/archive/purge capture a COMPLETE package
    (filled from the newest curator backup) so rollback never restores a shell."""
    if not ledger_enabled():
        return None
    try:
        _complete = action in _PACKAGE_RESTORE_ACTIONS
        if before is None:
            before = snapshot_paths(before_root, complete_package=_complete)
        elif _complete:
            before = fill_snapshot_from_curator_backup(before_root, before, skill=skill)
        return append_entry(action, skill, before=before, after=snapshot_paths(after_root),
                            actor=actor, evidence=evidence)
    except Exception as e:
        logger.warning("skill_ledger: record_mutation failed (%s) — mutation unaffected", e)
        return None


def capture_before(
    root: Optional[Path], *, complete_package: bool = False, skill: Optional[str] = None,
) -> Optional[List[Dict[str, str]]]:
    """Best-effort pre-mutation capture; None on failure/disabled (pass straight to
    record_mutation). ``complete_package=True`` for delete/archive/purge."""
    if not ledger_enabled():
        return None
    try:
        captured = snapshot_paths(root)
        return fill_snapshot_from_curator_backup(root, captured, skill=skill) if complete_package else captured
    except Exception as e:
        logger.warning("skill_ledger: before-capture failed (%s) — mutation unaffected", e)
        return None


def list_entries(skill: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read the ledger, newest first. Malformed lines are skipped."""
    try:
        lines = ledger_path().read_text(encoding="utf-8").splitlines()
    except OSError:  # missing or unreadable ledger == empty
        return []
    rows: List[Dict[str, Any]] = []
    for line in lines:
        with suppress(json.JSONDecodeError):
            row = json.loads(line) if line.strip() else None
            if isinstance(row, dict) and (not skill or row.get("skill") == skill):
                rows.append(row)
    rows.reverse()
    return rows[:limit] if limit is not None and limit >= 0 else rows


def get_entry(entry_id: str) -> Optional[Dict[str, Any]]:
    return next((r for r in list_entries() if r.get("id") == entry_id), None) if entry_id else None


def _validate_entry_paths(entry: Dict[str, Any]) -> Optional[str]:
    """Every entry path must be under HERMES_HOME — a hand-edited ledger must not
    become a write-anywhere primitive."""
    home = get_hermes_home()
    for section in ("before", "after"):
        for item in entry.get(section) or []:
            p = Path(str(item.get("path", "")))
            if not _is_within(home, p):
                return f"entry references a path outside {home}: {p}"
    return None


def rollback_entry(entry_id: str) -> Tuple[bool, str]:
    """Restore the before-state of mutation *entry_id*. Fail-closed (mirrors
    agent/curator_backup.rollback): every before-blob must exist BEFORE any change, and a
    pre-rollback safety entry of every touched path's CURRENT state is appended first.

    1. 2. See #63366.
    """
    entry = get_entry(entry_id)
    if entry is None:
        return False, f"no ledger entry with id '{entry_id}'"
    if path_err := _validate_entry_paths(entry):
        return False, f"refusing rollback: {path_err}"
    before = list(entry.get("before") or [])
    after = list(entry.get("after") or [])
    # Historical hollow delete/archive/purge entries (SKILL.md only): fill from the
    # newest curator backup so the complete package is restored. Entry hashes win;
    # only missing paths are added, and the filled set is re-validated.
    if entry.get("action") in _PACKAGE_RESTORE_ACTIONS:
        before = fill_snapshot_from_curator_backup(
            next(iter(_skill_md_parents(before)), None), before,
            skill=str(entry.get("skill") or "") or None)
        if path_err := _validate_entry_paths({**entry, "before": before, "after": after}):
            return False, f"refusing rollback: {path_err}"
    # Pre-check every blob we need so we never fail mid-restore.
    for item in before:
        if read_blob(str(item.get("sha256", ""))) is None:
            return False, (f"missing blob {item.get('sha256')} for {item.get('path')}; "
                           "rollback aborted, nothing was changed")
    # Safety entry: CURRENT state of every touched path, so the rollback itself is undoable.
    touched = {str(i["path"]) for i in before + after if i.get("path")}
    try:
        safety_before = [{"path": p, "sha256": _store_blob(Path(p).read_bytes())}
                         for p in sorted(touched) if Path(p).is_file()]
        safety_id = append_entry(
            "pre-rollback", entry.get("skill", "?"), before=safety_before, after=safety_before,
            evidence={"rollback_target": entry_id})
    except Exception as e:
        return False, (f"pre-rollback safety capture failed ({e}); rollback aborted and "
                       "current skills were not changed")
    if safety_id is None:
        return False, ("pre-rollback safety capture failed (ledger disabled or "
                       "unwritable); rollback aborted and current skills were not changed")
    # Restore: write every before-file, remove files the mutation created.
    before_paths = {str(i["path"]) for i in before}
    for item in before:
        fp = Path(str(item["path"]))
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(read_blob(str(item["sha256"])))  # pre-checked above
    restored, removed = len(before), 0
    for item in after:
        p = str(item.get("path", ""))
        if p and p not in before_paths:
            try:
                if Path(p).is_file():
                    Path(p).unlink()
                    removed += 1
            except OSError as e:
                logger.warning("skill_ledger: could not remove %s during rollback: %s", p, e)
    append_entry(
        "rollback", entry.get("skill", "?"), before=safety_before, after=before,
        evidence={"rollback_target": entry_id, "restored": restored, "removed": removed})
    return True, (
        f"rolled back entry {entry_id} ({entry.get('action')} on "
        f"'{entry.get('skill')}'): {restored} file(s) restored, {removed} removed. "
        f"Safety entry {safety_id} captured the pre-rollback state.")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

ACTOR_AGENT = "agent"

ACTOR_CURATOR = "curator"

ACTOR_USER = "user"
# ---- END PLUGIN-COMPAT ----
