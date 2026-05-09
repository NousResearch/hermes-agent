#!/usr/bin/env python3
"""
Skills Pending Queue — Staging area for background-review skill mutations.

When skills.evolution_mode is "confirm", background-review skill_manage
calls are redirected here instead of writing directly to the live skill
store.  Each pending change is stored as a JSON manifest + skill snapshot
under ~/.hermes/skills/.pending/<id>/.

Human review happens via:
  - `hermes skills review` CLI command
  - `/skills review` slash command
  - Gateway platform notifications (Telegram/Discord/Slack buttons)
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

PENDING_DIR = Path(get_hermes_home()) / "skills" / ".pending"
MAX_PENDING_ENTRIES = 100
PENDING_ID_FORMAT = "%Y%m%d-%H%M%S"


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class PreviousSnapshot:
    """Snapshot of the skill state before the change."""
    exists: bool = False
    skill_dir: str = ""
    file_hashes: dict = field(default_factory=dict)


@dataclass
class SecurityScanResult:
    """Result of the security scan for this pending change."""
    passed: bool = True
    verdict: str = "allow"
    findings: list = field(default_factory=list)


@dataclass
class PendingManifest:
    """Envelope for a pending skill change."""
    id: str
    action: str  # create | edit | patch | delete | write_file | remove_file
    skill_name: str
    skill_category: str = ""
    timestamp: str = ""
    origin: str = "background_review"
    summary: str = ""
    diff: str = ""
    previous_snapshot: PreviousSnapshot = field(default_factory=PreviousSnapshot)
    security_scan: SecurityScanResult = field(default_factory=SecurityScanResult)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> PendingManifest:
        d = dict(d)  # shallow copy
        ps = d.get("previous_snapshot", {})
        d["previous_snapshot"] = PreviousSnapshot(**ps) if isinstance(ps, dict) else PreviousSnapshot()
        sr = d.get("security_scan", {})
        d["security_scan"] = SecurityScanResult(**sr) if isinstance(sr, dict) else SecurityScanResult()
        return cls(**d)

    @classmethod
    def from_file(cls, path: Path) -> PendingManifest:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ── Helpers ───────────────────────────────────────────────────────────

def _generate_pending_id() -> str:
    """Generate a pending change ID: YYYYMMDD-HHMMSS-<4-hex>."""
    ts = datetime.now(timezone.utc).strftime(PENDING_ID_FORMAT)
    rand = hashlib.sha256(
        f"{time.time_ns()}-{id(object())}".encode()
    ).hexdigest()[:4]
    return f"{ts}-{rand}"


def _file_hash(path: Path) -> str:
    """SHA-256 hash of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"sha256:{h.hexdigest()}"


def _snapshot_skill_hashes(skill_dir: Path) -> dict:
    """Map every file in skill_dir to its SHA-256 hash (relative paths as keys)."""
    hashes: dict = {}
    if skill_dir.exists():
        for f in sorted(skill_dir.rglob("*")):
            if f.is_file():
                rel = str(f.relative_to(skill_dir))
                hashes[rel] = _file_hash(f)
    return hashes


def _generate_skill_diff(old_dir: Optional[Path], new_dir: Optional[Path], action: str) -> str:
    """Generate a unified diff between two skill directory snapshots."""
    diffs: list[str] = []

    if action == "delete":
        if old_dir and old_dir.exists():
            for f in sorted(old_dir.rglob("*")):
                if f.is_file():
                    rel = str(f.relative_to(old_dir))
                    try:
                        old_lines = f.read_text(encoding="utf-8").splitlines(keepends=True)
                    except UnicodeDecodeError:
                        continue
                    diff = difflib.unified_diff(
                        old_lines, [], fromfile=f"a/{rel}", tofile="/dev/null"
                    )
                    diff_text = "".join(diff)
                    if diff_text:
                        diffs.append(diff_text)
        return "\n".join(diffs) if diffs else "(empty skill — nothing to diff)"

    if action == "create":
        if new_dir and new_dir.exists():
            for f in sorted(new_dir.rglob("*")):
                if f.is_file():
                    rel = str(f.relative_to(new_dir))
                    try:
                        new_lines = f.read_text(encoding="utf-8").splitlines(keepends=True)
                    except UnicodeDecodeError:
                        continue
                    diff = difflib.unified_diff(
                        [], new_lines, fromfile="/dev/null", tofile=f"b/{rel}"
                    )
                    diff_text = "".join(diff)
                    if diff_text:
                        diffs.append(diff_text)
        return "\n".join(diffs) if diffs else "(empty skill — nothing to diff)"

    # edit / patch / write_file / remove_file — compare old vs new
    all_files: set[str] = set()
    if old_dir and old_dir.exists():
        all_files.update(str(f.relative_to(old_dir)) for f in old_dir.rglob("*") if f.is_file())
    if new_dir and new_dir.exists():
        all_files.update(str(f.relative_to(new_dir)) for f in new_dir.rglob("*") if f.is_file())

    for rel in sorted(all_files):
        old_file = (old_dir / rel) if (old_dir and old_dir.exists()) else None
        new_file = (new_dir / rel) if (new_dir and new_dir.exists()) else None

        try:
            old_lines = old_file.read_text(encoding="utf-8").splitlines(keepends=True) if old_file and old_file.exists() else []
            new_lines = new_file.read_text(encoding="utf-8").splitlines(keepends=True) if new_file and new_file.exists() else []
        except UnicodeDecodeError:
            continue

        diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{rel}", tofile=f"b/{rel}")
        diff_text = "".join(diff)
        if diff_text:
            diffs.append(diff_text)

    return "\n".join(diffs) if diffs else "(no textual changes detected)"


# ── Concurrency ───────────────────────────────────────────────────────

_enqueue_lock = threading.Lock()


# ── Core API ──────────────────────────────────────────────────────────

def enqueue(
    action: str,
    skill_name: str,
    skill_category: str,
    summary: str,
    skill_snapshot_dir: Path,
    diff: str,
    previous_snapshot: PreviousSnapshot,
    security_scan: SecurityScanResult,
    origin: str = "background_review",
) -> dict:
    """Write a pending change to the staging queue.

    Returns ``{"success": True, "pending": True, "pending_id": "..."}``.
    """
    with _enqueue_lock:
        # De-duplication: skip if a pending entry for the same skill+action exists
        entries = list_pending()
        for existing in entries:
            if existing.skill_name == skill_name and existing.action == action:
                return {
                    "success": True,
                    "pending": True,
                    "pending_id": existing.id,
                    "deduplicated": True,
                    "message": f"Pending change for '{skill_name}' ({action}) already exists as {existing.id}",
                }

        # Capacity enforcement — evict oldest entries when over limit
        if len(entries) >= MAX_PENDING_ENTRIES:
            _evict_oldest(len(entries) - MAX_PENDING_ENTRIES + 1)

        pending_id = _generate_pending_id()
        pending_entry_dir = PENDING_DIR / pending_id
        pending_entry_dir.mkdir(parents=True, exist_ok=True)

        # Write skill snapshot
        dest = pending_entry_dir / skill_name
        if skill_snapshot_dir and skill_snapshot_dir.exists():
            shutil.copytree(skill_snapshot_dir, dest, dirs_exist_ok=True)
        else:
            dest.mkdir()

        # Write manifest
        manifest = PendingManifest(
            id=pending_id,
            action=action,
            skill_name=skill_name,
            skill_category=skill_category,
            timestamp=datetime.now(timezone.utc).isoformat(),
            origin=origin,
            summary=summary,
            diff=diff,
            previous_snapshot=previous_snapshot,
            security_scan=security_scan,
        )
        (pending_entry_dir / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")

        logger.info(
            "Skills: pending change enqueued (id=%s, action=%s, skill=%s)",
            pending_id, action, skill_name,
        )

    return {"success": True, "pending": True, "pending_id": pending_id}


def list_pending() -> list[PendingManifest]:
    """List all pending changes, sorted by timestamp (oldest first)."""
    if not PENDING_DIR.exists():
        return []
    results: list[PendingManifest] = []
    for entry in sorted(PENDING_DIR.iterdir()):
        manifest_path = entry / "manifest.json"
        if manifest_path.exists():
            try:
                results.append(PendingManifest.from_file(manifest_path))
            except Exception:
                logger.warning("Could not parse pending manifest: %s", manifest_path)
    return results


def apply_pending(pending_id: str, force: bool = False) -> dict:
    """Apply a pending change: move snapshot to live skill store and refresh cache."""
    entry_dir = PENDING_DIR / pending_id
    if not entry_dir.exists():
        return {"success": False, "error": f"Pending change {pending_id} not found"}

    manifest = PendingManifest.from_file(entry_dir / "manifest.json")
    skills_dir = Path(get_hermes_home()) / "skills"

    # Conflict detection
    if manifest.previous_snapshot.exists and not force:
        conflict = _detect_conflict(manifest)
        if conflict:
            logger.warning(
                "Skills: pending change conflict (id=%s, skill=%s, reason=%s)",
                pending_id, manifest.skill_name, conflict.get("reason", ""),
            )
            return {
                "success": False,
                "conflict": True,
                "conflict_details": conflict,
                "message": "Skill was modified after this pending change was created. Use force=True to override.",
            }

    # Move snapshot to live skill store
    if manifest.skill_category:
        dest = skills_dir / manifest.skill_category / manifest.skill_name
    else:
        dest = skills_dir / manifest.skill_name
    if manifest.action == "delete":
        if dest.exists():
            shutil.rmtree(dest)
        # Clean up empty category directories
        if manifest.skill_category:
            cat_dir = skills_dir / manifest.skill_category
            if cat_dir.exists() and not any(cat_dir.iterdir()):
                cat_dir.rmdir()
    else:
        if dest.exists():
            shutil.rmtree(dest)
        # Look for the snapshot — it may be at entry_dir/skill_name (flat)
        # or entry_dir/category/skill_name (nested, from category-aware _route_to_pending)
        snapshot_src = entry_dir / manifest.skill_name
        if not snapshot_src.exists() and manifest.skill_category:
            snapshot_src = entry_dir / manifest.skill_category / manifest.skill_name
        if snapshot_src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(snapshot_src), str(dest))
        else:
            dest.mkdir(parents=True, exist_ok=True)

    # Clean up pending entry
    shutil.rmtree(entry_dir, ignore_errors=True)

    # Refresh skills prompt cache
    try:
        from agent.prompt_builder import clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=True)
    except Exception:
        pass

    logger.info("Skills: pending change applied (id=%s, action=%s, skill=%s)", pending_id, manifest.action, manifest.skill_name)

    return {"success": True, "applied": pending_id, "action": manifest.action, "skill_name": manifest.skill_name}


def discard_pending(pending_id: str) -> dict:
    """Discard a pending change."""
    entry_dir = PENDING_DIR / pending_id
    if not entry_dir.exists():
        return {"success": False, "error": f"Pending change {pending_id} not found"}
    shutil.rmtree(entry_dir)
    logger.info("Skills: pending change discarded (id=%s)", pending_id)
    return {"success": True, "discarded": pending_id}


def discard_all() -> dict:
    """Discard all pending changes."""
    count = 0
    if PENDING_DIR.exists():
        for entry in list(PENDING_DIR.iterdir()):
            if (entry / "manifest.json").exists():
                shutil.rmtree(entry, ignore_errors=True)
                count += 1
    logger.info("Skills: all pending changes discarded (count=%d)", count)
    return {"success": True, "discarded_count": count}


def apply_all(force: bool = False) -> dict:
    """Apply all pending changes."""
    results: list[dict] = []
    for manifest in list_pending():
        result = apply_pending(manifest.id, force=force)
        results.append(result)
    applied = sum(1 for r in results if r.get("success"))
    conflicts = sum(1 for r in results if r.get("conflict"))
    return {"success": True, "applied": applied, "conflicts": conflicts, "results": results}


def get_diff(pending_id: str) -> dict:
    """Get the diff for a pending change."""
    manifest_path = PENDING_DIR / pending_id / "manifest.json"
    if not manifest_path.exists():
        return {"success": False, "error": f"Pending change {pending_id} not found"}
    manifest = PendingManifest.from_file(manifest_path)
    return {"success": True, "diff": manifest.diff, "manifest": manifest.to_dict()}


def gc_expired(ttl_days: int) -> dict:
    """Garbage-collect pending changes older than *ttl_days*."""
    if not PENDING_DIR.exists() or ttl_days <= 0:
        return {"success": True, "expired_count": 0}

    now = time.time()
    cutoff = now - (ttl_days * 86400)
    count = 0

    for entry in list(PENDING_DIR.iterdir()):
        manifest_path = entry / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = PendingManifest.from_file(manifest_path)
            ts = datetime.fromisoformat(manifest.timestamp).timestamp()
            if ts < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                count += 1
        except (ValueError, OSError):
            pass

    if count:
        logger.info("Skills: expired pending changes cleaned (count=%d)", count)
    return {"success": True, "expired_count": count}


# ── Internal helpers ──────────────────────────────────────────────────

def _detect_conflict(manifest: PendingManifest) -> Optional[dict]:
    """Detect whether the live skill store has diverged from the snapshot."""
    skills_dir = Path(get_hermes_home()) / "skills"
    if manifest.skill_category:
        skill_dir = skills_dir / manifest.skill_category / manifest.skill_name
    else:
        skill_dir = skills_dir / manifest.skill_name

    if not manifest.previous_snapshot.exists:
        # Skill didn't exist before — if it now does, that's a conflict
        if skill_dir.exists():
            return {"reason": "Skill was created by another process after this pending change"}
        return None

    # Skill existed before — if it's now gone, that's a conflict
    if not skill_dir.exists():
        return {"reason": "Skill was deleted by another process after this pending change"}

    # Compare file hashes
    for rel_path, expected_hash in manifest.previous_snapshot.file_hashes.items():
        current_file = skill_dir / rel_path
        if not current_file.exists():
            return {"reason": f"File {rel_path} was deleted", "file": rel_path}
        actual_hash = _file_hash(current_file)
        if actual_hash != expected_hash:
            return {"reason": f"File {rel_path} was modified", "file": rel_path}

    return None


def _evict_oldest(count: int) -> int:
    """Evict the *count* oldest pending entries."""
    entries = list_pending()
    evicted = 0
    for manifest in entries[:count]:
        entry_dir = PENDING_DIR / manifest.id
        if entry_dir.exists():
            shutil.rmtree(entry_dir, ignore_errors=True)
            evicted += 1
    if evicted:
        logger.warning("Skills: pending queue full, evicted %d oldest entries", evicted)
    return evicted
