"""Skill usage telemetry + provenance tracking for the Curator feature.

Tracks per-skill usage metadata in a sidecar JSON file (~/.hermes/skills/.usage.json)
keyed by skill name. Counters are bumped by the existing skill tools (skill_view,
skill_manage); the curator orchestrator reads them to decide lifecycle transitions.

Design notes:
  - Sidecar, not frontmatter. Keeps operational telemetry out of user-authored
    SKILL.md content and avoids conflict pressure for bundled/hub skills.
  - Atomic writes via tempfile + os.replace (same pattern as .bundled_manifest).
  - All counter bumps are best-effort: failures log at DEBUG and return silently.
    A broken sidecar never breaks the underlying tool call.
  - Provenance filter: "agent-created" == not in .bundled_manifest AND not in
    .hub/lock.json. The curator only ever mutates agent-created skills.

Lifecycle states:
    active    -> default
    stale     -> unused > stale_after_days (config)
    archived  -> unused > archive_after_days (config); moved to .archive/
    pinned    -> opt-out from auto transitions (boolean flag, orthogonal to state)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


STATE_ACTIVE = "active"
STATE_STALE = "stale"
STATE_ARCHIVED = "archived"
_VALID_STATES = {STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED}


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def _usage_file() -> Path:
    return _skills_dir() / ".usage.json"


def _archive_dir() -> Path:
    return _skills_dir() / ".archive"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Provenance — which skills are agent-created (and thus eligible for curation)
# ---------------------------------------------------------------------------

def _read_bundled_manifest_names() -> Set[str]:
    """Return the set of skill names that were seeded from the bundled repo.

    Reads ~/.hermes/skills/.bundled_manifest (format: "name:hash" per line).
    Returns empty set if the file is missing or unreadable.
    """
    manifest = _skills_dir() / ".bundled_manifest"
    if not manifest.exists():
        return set()
    names: Set[str] = set()
    try:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            name = line.split(":", 1)[0].strip()
            if name:
                names.add(name)
    except OSError as e:
        logger.debug("Failed to read bundled manifest: %s", e)
    return names


def _read_hub_installed_names() -> Set[str]:
    """Return the set of skill names installed via the Skills Hub.

    Reads ~/.hermes/skills/.hub/lock.json (see tools/skills_hub.py :: HubLockFile).
    """
    lock_path = _skills_dir() / ".hub" / "lock.json"
    if not lock_path.exists():
        return set()
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            installed = data.get("installed") or {}
            if isinstance(installed, dict):
                return {str(k) for k in installed.keys()}
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read hub lock file: %s", e)
    return set()


def list_agent_created_skill_names() -> List[str]:
    """Enumerate skills that were authored by the agent (or user), NOT by a
    bundled or hub-installed source.

    The curator operates exclusively on this set. Bundled / hub skills are
    maintained by their upstream sources and must never be pruned here.
    """
    base = _skills_dir()
    if not base.exists():
        return []
    bundled = _read_bundled_manifest_names()
    hub = _read_hub_installed_names()
    off_limits = bundled | hub

    names: List[str] = []
    # Top-level SKILL.md files (flat layout) AND nested category/skill/SKILL.md
    for skill_md in base.rglob("SKILL.md"):
        # Skip anything under .archive or .hub
        try:
            rel = skill_md.relative_to(base)
        except ValueError:
            continue
        parts = rel.parts
        if parts and (parts[0].startswith(".") or parts[0] == "node_modules"):
            continue
        name = _read_skill_name(skill_md, fallback=skill_md.parent.name)
        if name in off_limits:
            continue
        names.append(name)
    return sorted(set(names))


def _read_skill_name(skill_md: Path, fallback: str) -> str:
    """Parse the `name:` field from a SKILL.md YAML frontmatter."""
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return fallback
    in_frontmatter = False
    for line in text.split("\n"):
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


def is_agent_created(skill_name: str) -> bool:
    """Whether *skill_name* is neither bundled nor hub-installed."""
    off_limits = _read_bundled_manifest_names() | _read_hub_installed_names()
    return skill_name not in off_limits


# ---------------------------------------------------------------------------
# Sidecar I/O
# ---------------------------------------------------------------------------

def _empty_record() -> Dict[str, Any]:
    return {
        "use_count": 0,
        "view_count": 0,
        "last_used_at": None,
        "last_viewed_at": None,
        "patch_count": 0,
        "last_patched_at": None,
        "created_at": _now_iso(),
        "state": STATE_ACTIVE,
        "pinned": False,
        "archived_at": None,
        "negative_claim_confidence": None,
        "negative_claim_ttl_days": None,
        "negative_claim_last_revalidated_at": None,
        "negative_claim_revalidation_due_at": None,
        "negative_claim_summary": None,
        "negative_claim_status": None,
        "negative_claims": [],
    }


def load_usage() -> Dict[str, Dict[str, Any]]:
    """Read the entire .usage.json map. Returns empty dict on missing/corrupt."""
    path = _usage_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    # Defensive: coerce any non-dict values to a fresh empty record
    clean: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            clean[str(k)] = v
    return clean


def save_usage(data: Dict[str, Dict[str, Any]]) -> None:
    """Write the usage map atomically. Best-effort — errors are logged, not raised."""
    path = _usage_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=".usage_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("Failed to write %s: %s", path, e, exc_info=True)


def get_record(skill_name: str) -> Dict[str, Any]:
    """Return the record for *skill_name*, creating a fresh one if missing."""
    data = load_usage()
    rec = data.get(skill_name)
    if not isinstance(rec, dict):
        return _empty_record()
    # Backfill any missing keys so callers don't need to handle old files
    base = _empty_record()
    for k, v in base.items():
        rec.setdefault(k, v)
    return rec


def _mutate(skill_name: str, mutator) -> None:
    """Load, apply *mutator(record)* in place, save. Best-effort.

    Bundled and hub-installed skills are NEVER recorded in the sidecar.
    This keeps .usage.json focused on agent-created skills (the only ones
    the curator considers) and prevents stale counters from hanging around
    for upstream-managed skills.
    """
    if not skill_name:
        return
    try:
        if not is_agent_created(skill_name):
            return
        data = load_usage()
        rec = data.get(skill_name)
        if not isinstance(rec, dict):
            rec = _empty_record()
        mutator(rec)
        data[skill_name] = rec
        save_usage(data)
    except Exception as e:
        logger.debug("skill_usage._mutate(%s) failed: %s", skill_name, e, exc_info=True)


# ---------------------------------------------------------------------------
# Public counter-bump helpers
# ---------------------------------------------------------------------------

def bump_view(skill_name: str) -> None:
    """Bump view_count and last_viewed_at. Called from skill_view()."""
    def _apply(rec: Dict[str, Any]) -> None:
        rec["view_count"] = int(rec.get("view_count") or 0) + 1
        rec["last_viewed_at"] = _now_iso()
    _mutate(skill_name, _apply)


def bump_use(skill_name: str) -> None:
    """Bump use_count and last_used_at. Called when a skill is actively used
    (e.g. loaded into the prompt path or referenced from an assistant turn)."""
    def _apply(rec: Dict[str, Any]) -> None:
        rec["use_count"] = int(rec.get("use_count") or 0) + 1
        rec["last_used_at"] = _now_iso()
    _mutate(skill_name, _apply)


def bump_patch(skill_name: str) -> None:
    """Bump patch_count and last_patched_at. Called from skill_manage (patch/edit)."""
    def _apply(rec: Dict[str, Any]) -> None:
        rec["patch_count"] = int(rec.get("patch_count") or 0) + 1
        rec["last_patched_at"] = _now_iso()
    _mutate(skill_name, _apply)


# ---------------------------------------------------------------------------
# Negative / environment-dependent claim metadata
# ---------------------------------------------------------------------------

def _ensure_aware(dt: Optional[datetime]) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    return _ensure_aware(dt)


_NEGATIVE_COMMAND_PATTERNS = (
    re.compile(r"(?:`([^`]+)`|\b([A-Za-z0-9_.+-]+)\b)\s+command\s+(?:is\s+)?unavailable[^.\n]*(?:[.])?", re.I),
    re.compile(r"command\s+(?:`([^`]+)`|\b([A-Za-z0-9_.+-]+)\b)\s+(?:is\s+)?unavailable[^.\n]*(?:[.])?", re.I),
    re.compile(r"(?:`([^`]+)`|\b([A-Za-z0-9_.+-]+)\b)\s+(?:is\s+)?not\s+(?:installed|available|found)[^.\n]*(?:[.])?", re.I),
)


def _claim_id(kind: str, subject: str, text: str) -> str:
    basis = f"{kind}\0{subject.lower()}\0{text.strip().lower()}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def extract_negative_claims_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract deterministic negative/environment-dependent claims from text.

    This first-pass detector intentionally recognizes only high-confidence
    command-availability claims. Unknown or ambiguous text is left for manual
    review rather than guessed.
    """
    claims: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    source = text or ""
    for pat in _NEGATIVE_COMMAND_PATTERNS:
        for match in pat.finditer(source):
            subject = next((g for g in match.groups() if g), "").strip()
            if not subject or subject.lower() in {"the", "a", "an", "this", "that", "is"}:
                continue
            line_start = source.rfind("\n", 0, match.start()) + 1
            line_end = source.find("\n", match.end())
            if line_end == -1:
                line_end = len(source)
            claim_text = source[line_start:line_end].strip()
            if claim_text.lower().startswith("note:"):
                claim_text = claim_text.split(":", 1)[1].strip()
            claim_text = claim_text.rstrip(".") + "." if claim_text else claim_text
            cid = _claim_id("command_unavailable", subject, claim_text)
            if cid in seen:
                continue
            seen.add(cid)
            claims.append({
                "id": cid,
                "kind": "command_unavailable",
                "subject": subject,
                "text": claim_text,
            })
    return claims


def mark_negative_claim(
    skill_name: str,
    summary: str,
    confidence: Optional[float] = None,
    ttl_days: Optional[int] = None,
    now: Optional[datetime] = None,
) -> None:
    """Record that a skill contains a negative / environment-dependent claim."""
    when = _ensure_aware(now)
    try:
        ttl = int(ttl_days) if ttl_days is not None else None
    except (TypeError, ValueError):
        ttl = None
    due = (when + timedelta(days=ttl)).isoformat() if ttl is not None else None
    clean_summary = str(summary or "").strip() or None
    extracted = extract_negative_claims_from_text(clean_summary or "")
    if not extracted and clean_summary:
        cid = _claim_id("other", "", clean_summary)
        extracted = [{"id": cid, "kind": "other", "subject": "", "text": clean_summary}]

    def _apply(rec: Dict[str, Any]) -> None:
        rec["negative_claim_summary"] = clean_summary
        rec["negative_claim_confidence"] = confidence
        rec["negative_claim_ttl_days"] = ttl
        rec["negative_claim_last_revalidated_at"] = None
        rec["negative_claim_revalidation_due_at"] = due
        rec["negative_claim_status"] = "active"
        existing = rec.get("negative_claims") if isinstance(rec.get("negative_claims"), list) else []
        by_id = {str(c.get("id")): c for c in existing if isinstance(c, dict) and c.get("id")}
        for claim in extracted:
            cid = str(claim.get("id") or "")
            if not cid:
                continue
            by_id[cid] = {
                **by_id.get(cid, {}),
                **claim,
                "confidence": confidence,
                "ttl_days": ttl,
                "status": "active",
                "last_revalidated_at": None,
                "next_revalidate_at": due,
            }
        rec["negative_claims"] = list(by_id.values())

    _mutate(skill_name, _apply)


def update_negative_claim_revalidation(
    skill_name: str,
    status: str,
    confidence: Optional[float] = None,
    summary: Optional[str] = None,
    ttl_days: Optional[int] = None,
    now: Optional[datetime] = None,
) -> None:
    """Update revalidation metadata after a negative claim has been checked."""
    when = _ensure_aware(now)
    valid_statuses = {"active", "refreshed", "disproven", "uncertain"}
    next_status = status if status in valid_statuses else "uncertain"

    def _apply(rec: Dict[str, Any]) -> None:
        rec["negative_claim_status"] = next_status
        rec["negative_claim_last_revalidated_at"] = when.isoformat()
        if confidence is not None:
            rec["negative_claim_confidence"] = confidence
        if summary is not None:
            rec["negative_claim_summary"] = str(summary).strip() or None
        current_ttl = rec.get("negative_claim_ttl_days")
        try:
            ttl = int(ttl_days if ttl_days is not None else current_ttl)
        except (TypeError, ValueError):
            ttl = None
        rec["negative_claim_ttl_days"] = ttl
        rec["negative_claim_revalidation_due_at"] = (
            (when + timedelta(days=ttl)).isoformat() if ttl is not None else None
        )

    _mutate(skill_name, _apply)


def due_negative_claims(
    now: Optional[datetime] = None,
    limit: Optional[int] = None,
    min_confidence: float = 0.0,
) -> List[Dict[str, Any]]:
    """Return negative claims whose TTL has expired and need revalidation."""
    when = _ensure_aware(now)
    rows: List[Dict[str, Any]] = []
    for name, rec in load_usage().items():
        if not isinstance(rec, dict):
            continue
        status = rec.get("negative_claim_status")
        if status in (None, "disproven"):
            continue
        due = _parse_iso(rec.get("negative_claim_revalidation_due_at"))
        if due is None or due > when:
            continue
        try:
            confidence = float(rec.get("negative_claim_confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < min_confidence:
            continue
        base = _empty_record()
        base.update(rec)
        rows.append({"name": str(name), **base})

    rows.sort(key=lambda row: (row.get("negative_claim_revalidation_due_at") or "", row["name"]))
    if limit is not None:
        try:
            n = max(0, int(limit))
            rows = rows[:n]
        except (TypeError, ValueError):
            pass
    return rows


def set_state(skill_name: str, state: str) -> None:
    """Set lifecycle state. No-op if *state* is invalid."""
    if state not in _VALID_STATES:
        logger.debug("set_state: invalid state %r for %s", state, skill_name)
        return
    def _apply(rec: Dict[str, Any]) -> None:
        rec["state"] = state
        if state == STATE_ARCHIVED:
            rec["archived_at"] = _now_iso()
        elif state == STATE_ACTIVE:
            rec["archived_at"] = None
    _mutate(skill_name, _apply)


def set_pinned(skill_name: str, pinned: bool) -> None:
    def _apply(rec: Dict[str, Any]) -> None:
        rec["pinned"] = bool(pinned)
    _mutate(skill_name, _apply)


def should_hide_from_prompt(skill_name: str) -> bool:
    """Return True when *skill_name* should be omitted from prompt-time indexes.

    Stale skills may be hidden from the default skills prompt when the curator
    config enables that policy, but pinned skills remain visible so users can
    protect rare-but-important workflows from recency-based deprioritization.
    This helper is deliberately config-free; callers decide whether the policy
    is enabled and use this only for the lifecycle-state check.
    """
    if not skill_name:
        return False
    rec = get_record(skill_name)
    return rec.get("state") == STATE_STALE and not bool(rec.get("pinned"))


def forget(skill_name: str) -> None:
    """Drop a skill's usage entry entirely. Called when the skill is deleted."""
    if not skill_name:
        return
    try:
        data = load_usage()
        if skill_name in data:
            del data[skill_name]
            save_usage(data)
    except Exception as e:
        logger.debug("skill_usage.forget(%s) failed: %s", skill_name, e, exc_info=True)


# ---------------------------------------------------------------------------
# Archive / restore
# ---------------------------------------------------------------------------

def archive_skill(skill_name: str) -> Tuple[bool, str]:
    """Move an agent-created skill directory to ~/.hermes/skills/.archive/.

    Returns (ok, message). Never archives bundled or hub skills — callers are
    responsible for checking provenance, but we double-check here as a safety net.
    """
    if not is_agent_created(skill_name):
        return False, f"skill '{skill_name}' is bundled or hub-installed; never archive"

    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None:
        return False, f"skill '{skill_name}' not found"

    archive_root = _archive_dir()
    try:
        archive_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"failed to create archive dir: {e}"

    # Flatten any category nesting into a single ".archive/<skill>/" so restores
    # are simple. If a collision exists, append a timestamp.
    dest = archive_root / skill_dir.name
    if dest.exists():
        dest = archive_root / f"{skill_dir.name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    try:
        skill_dir.rename(dest)
    except OSError as e:
        # Cross-device — fall back to shutil.move
        import shutil
        try:
            shutil.move(str(skill_dir), str(dest))
        except Exception as e2:
            return False, f"failed to archive: {e2}"

    set_state(skill_name, STATE_ARCHIVED)
    return True, f"archived to {dest}"


def restore_skill(skill_name: str) -> Tuple[bool, str]:
    """Move an archived skill back to ~/.hermes/skills/. Restores to the flat
    top-level layout; original category nesting is NOT reconstructed.

    Refuses to restore under a name that now collides with a bundled or
    hub-installed skill — that would shadow the upstream version.
    """
    # If a bundled or hub skill has since been installed under the same
    # name, refuse to restore rather than shadow it.
    if not is_agent_created(skill_name):
        return False, (
            f"skill '{skill_name}' is now bundled or hub-installed; "
            "restore would shadow the upstream version"
        )
    archive_root = _archive_dir()
    if not archive_root.exists():
        return False, "no archive directory"

    # Try exact name match first, then any prefix match (for timestamped dupes)
    candidates = [p for p in archive_root.iterdir() if p.is_dir() and p.name == skill_name]
    if not candidates:
        candidates = sorted(
            [p for p in archive_root.iterdir()
             if p.is_dir() and p.name.startswith(f"{skill_name}-")],
            reverse=True,
        )
    if not candidates:
        return False, f"skill '{skill_name}' not found in archive"

    src = candidates[0]
    dest = _skills_dir() / skill_name
    if dest.exists():
        return False, f"destination already exists: {dest}"

    try:
        src.rename(dest)
    except OSError:
        import shutil
        try:
            shutil.move(str(src), str(dest))
        except Exception as e:
            return False, f"failed to restore: {e}"

    set_state(skill_name, STATE_ACTIVE)
    return True, f"restored to {dest}"


def _find_skill_dir(skill_name: str) -> Optional[Path]:
    """Locate the directory for a skill by its frontmatter `name:` field.

    Handles both flat (~/.hermes/skills/<skill>/SKILL.md) and category-nested
    (~/.hermes/skills/<category>/<skill>/SKILL.md) layouts.
    """
    base = _skills_dir()
    if not base.exists():
        return None
    for skill_md in base.rglob("SKILL.md"):
        try:
            rel = skill_md.relative_to(base)
        except ValueError:
            continue
        if rel.parts and rel.parts[0].startswith("."):
            continue
        if _read_skill_name(skill_md, fallback=skill_md.parent.name) == skill_name:
            return skill_md.parent
    return None


# ---------------------------------------------------------------------------
# Reporting — for the curator CLI / slash command
# ---------------------------------------------------------------------------

def agent_created_report() -> List[Dict[str, Any]]:
    """Return a list of {name, state, pinned, last_used_at, use_count, ...}
    records for every agent-created skill. Missing usage records are backfilled
    with defaults so callers can always index fields."""
    data = load_usage()
    rows: List[Dict[str, Any]] = []
    for name in list_agent_created_skill_names():
        rec = data.get(name)
        if not isinstance(rec, dict):
            rec = _empty_record()
        base = _empty_record()
        for k, v in base.items():
            rec.setdefault(k, v)
        rows.append({"name": name, **rec})
    return rows
