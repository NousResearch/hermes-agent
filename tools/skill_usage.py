"""Skill usage telemetry + provenance tracking for the Curator feature.

Tracks per-skill usage metadata in a sidecar JSON file (~/.hermes/skills/.usage.json)
keyed by canonical skill name (including bundle namespace when known). Counters
are bumped by the existing skill tools (skill_view, skill_manage); the curator
orchestrator reads the derived activity timestamp to
decide lifecycle transitions.

Design notes:
  - Sidecar, not frontmatter. Keeps operational telemetry out of user-authored
    SKILL.md content and avoids conflict pressure for bundled/hub skills.
  - Atomic writes via tempfile + os.replace (same pattern as .bundled_manifest).
  - All counter bumps are best-effort: failures log at DEBUG and return silently.
    A broken sidecar never breaks the underlying tool call.
  - Provenance filter: curator-managed skills are explicitly marked when
    created through skill_manage. Bundled / hub-installed skills stay
    off-limits, and manually authored skills are not inferred from location.

Lifecycle states:
    active    -> default
    stale     -> unused > stale_after_days (config)
    archived  -> unused > archive_after_days (config); moved to .archive/
    pinned    -> opt-out from auto transitions (boolean flag, orthogonal to state)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home
from agent.skill_utils import is_excluded_skill_path

logger = logging.getLogger(__name__)

# fcntl is Unix-only; on Windows use msvcrt for file locking.
msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover - platform-specific fallback
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass


STATE_ACTIVE = "active"
STATE_STALE = "stale"
STATE_ARCHIVED = "archived"
_VALID_STATES = {STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED}


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def _usage_file() -> Path:
    return _skills_dir() / ".usage.json"


@contextmanager
def _usage_file_lock():
    """Serialize .usage.json read-modify-write cycles across processes."""
    lock_path = _usage_file().with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if fcntl is None and msvcrt is None:
        yield
        return

    if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        lock_path.write_text(" ", encoding="utf-8")

    fd = open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8")
    try:
        if fcntl:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            fd.seek(0)
            msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        if fcntl:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except (OSError, IOError):
                pass
        elif msvcrt:
            try:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, IOError):
                pass
        fd.close()


def _archive_dir() -> Path:
    return _skills_dir() / ".archive"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO timestamp defensively for activity comparisons."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def latest_activity_at(record: Dict[str, Any]) -> Optional[str]:
    """Return the newest actual activity timestamp for a usage record.

    "Activity" means a skill was used, viewed, or patched. Creation time is
    intentionally excluded so callers can still distinguish never-active skills;
    lifecycle code can fall back to ``created_at`` as its own anchor.
    """
    latest_dt: Optional[datetime] = None
    latest_raw: Optional[str] = None
    for key in ("last_used_at", "last_viewed_at", "last_patched_at"):
        raw = record.get(key)
        dt = _parse_iso_timestamp(raw)
        if dt is None:
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt = dt
            latest_raw = str(raw)
    return latest_raw


def activity_count(record: Dict[str, Any]) -> int:
    """Return the total observed activity count across use/view/patch events."""
    total = 0
    for key in ("use_count", "view_count", "patch_count"):
        try:
            total += int(record.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return total


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
    return _expand_known_skill_aliases(names)


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
                names = {str(k) for k in installed.keys()}
                skills_dir = _skills_dir()
                for entry in installed.values():
                    if not isinstance(entry, dict):
                        continue
                    install_path = entry.get("install_path")
                    if not isinstance(install_path, str) or not install_path.strip():
                        continue
                    skill_dir = Path(install_path)
                    if not skill_dir.is_absolute():
                        skill_dir = skills_dir / skill_dir
                    try:
                        resolved = skill_dir.resolve()
                        resolved.relative_to(skills_dir.resolve())
                    except (OSError, ValueError):
                        continue
                    skill_md = resolved / "SKILL.md"
                    if skill_md.exists():
                        actual_name = _read_skill_name(skill_md, fallback=resolved.name)
                        names.add(actual_name)
                        names.add(skill_usage_key_for_path(
                            skill_md,
                            root=skills_dir,
                            skill_name=actual_name,
                        ).get("usage_key") or actual_name)
                        try:
                            names.add(str(resolved.relative_to(skills_dir.resolve())).strip("/"))
                        except (OSError, ValueError):
                            pass
                return names
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read hub lock file: %s", e)
    return set()


def _iter_agent_created_skill_entries() -> List[Dict[str, Any]]:
    """Enumerate curator-managed skills with canonical identity metadata."""
    base = _skills_dir()
    if not base.exists():
        return []
    off_limits = _read_bundled_manifest_names() | _read_hub_installed_names()
    usage = load_usage()
    entries: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()

    for skill_md in base.rglob("SKILL.md"):
        if is_excluded_skill_path(skill_md):
            continue
        try:
            skill_md.relative_to(base)
        except ValueError:
            continue
        frontmatter = _read_skill_frontmatter(skill_md)
        name = _clean_str(frontmatter.get("name")) or skill_md.parent.name
        provenance = skill_usage_key_for_path(
            skill_md,
            root=base,
            skill_name=name,
            frontmatter=frontmatter,
        )
        usage_key = str(provenance.get("usage_key") or name)
        try:
            rel_name = str(skill_md.parent.resolve().relative_to(base.resolve())).strip("/")
        except (OSError, ValueError):
            rel_name = skill_md.parent.name
        if {str(name), usage_key, rel_name} & off_limits:
            continue
        rec = _record_for_usage_key(usage, usage_key, str(name))
        if not _is_curator_managed_record(rec):
            continue
        if usage_key in seen_keys:
            continue
        seen_keys.add(usage_key)
        entries.append({
            "name": str(name),
            "usage_key": usage_key,
            "bundle_id": provenance.get("bundle_id"),
            "source_repo": provenance.get("source_repo"),
            "source_ref": provenance.get("source_ref"),
            "source_commit": provenance.get("source_commit"),
            "skill_md": skill_md,
        })
    return sorted(entries, key=lambda e: (str(e.get("usage_key") or ""), str(e.get("name") or "")))


def list_agent_created_skill_names() -> List[str]:
    """Enumerate skills explicitly authored by the agent.

    The curator operates exclusively on this set. Skills are only eligible
    after ``skill_manage(action="create")`` marks them in ``.usage.json``;
    manually authored skills must not be inferred from filesystem location.
    Bundled / hub skills are maintained by their upstream sources and must
    never be pruned here.
    """
    return sorted({entry["name"] for entry in _iter_agent_created_skill_entries()})


def list_archived_skill_names() -> List[str]:
    """Enumerate skills in ``~/.hermes/skills/.archive/``.

    Archive layout is flat (``.archive/<skill>/``) as set by ``archive_skill``,
    so the directory name is the skill name. Used by ``hermes curator
    list-archived`` to help users pass a name to ``hermes curator restore``.
    """
    archive_root = _archive_dir()
    if not archive_root.exists():
        return []
    return sorted({p.name for p in archive_root.iterdir() if p.is_dir()})


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


def _clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_skill_frontmatter(skill_md: Path) -> Dict[str, Any]:
    """Read SKILL.md frontmatter with nested metadata support."""
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    try:
        from agent.skill_utils import parse_frontmatter

        frontmatter, _ = parse_frontmatter(text)
    except Exception:
        return {}
    return frontmatter if isinstance(frontmatter, dict) else {}



def _containing_skill_root(skill_md: Path) -> Optional[Path]:
    """Return the configured skills root that contains *skill_md*."""
    try:
        from agent.skill_utils import get_all_skills_dirs

        roots = get_all_skills_dirs()
    except Exception:
        roots = [_skills_dir()]
    for root in roots:
        try:
            skill_md.resolve().relative_to(root.resolve())
            return root
        except (OSError, ValueError):
            continue
    return None


def skill_usage_key_for_path(
    skill_md: Path,
    *,
    root: Optional[Path] = None,
    skill_name: Optional[str] = None,
    frontmatter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return provenance and the canonical sidecar key for a SKILL.md path.

    ``metadata.hermes.bundle_id`` wins. When absent, a nested path such as
    ``superpowers-zh/using-superpowers/SKILL.md`` is treated as a package-like
    namespace and produces ``superpowers-zh/using-superpowers``.
    """
    if frontmatter is None:
        frontmatter = _read_skill_frontmatter(skill_md)
    if skill_name is None:
        skill_name = _clean_str(frontmatter.get("name")) or skill_md.parent.name

    metadata = frontmatter.get("metadata") if isinstance(frontmatter, dict) else None
    hermes_meta = metadata.get("hermes") if isinstance(metadata, dict) else None
    if not isinstance(hermes_meta, dict):
        hermes_meta = {}

    bundle_id = _clean_str(hermes_meta.get("bundle_id"))
    source_repo = _clean_str(hermes_meta.get("source_repo"))
    source_ref = _clean_str(hermes_meta.get("source_ref"))
    source_commit = _clean_str(hermes_meta.get("source_commit"))

    if root is None:
        root = _containing_skill_root(skill_md)

    rel_parts: Tuple[str, ...] = ()
    if root is not None:
        try:
            rel_parts = skill_md.resolve().relative_to(root.resolve()).parts
        except (OSError, ValueError):
            try:
                rel_parts = skill_md.relative_to(root).parts
            except ValueError:
                rel_parts = ()

    if not bundle_id and len(rel_parts) >= 3 and rel_parts[-1] == "SKILL.md":
        bundle_id = rel_parts[0]

    usage_key = str(skill_name)
    if bundle_id:
        if not usage_key.startswith(f"{bundle_id}/") and not usage_key.startswith(f"{bundle_id}:"):
            usage_key = f"{bundle_id}/{usage_key}"

    return {
        "skill_name": str(skill_name),
        "usage_key": usage_key,
        "bundle_id": bundle_id,
        "source_repo": source_repo,
        "source_ref": source_ref,
        "source_commit": source_commit,
    }


def _iter_skill_files_under(root: Path):
    try:
        from agent.skill_utils import iter_skill_index_files

        yield from iter_skill_index_files(root, "SKILL.md")
        return
    except Exception:
        pass
    if root.exists():
        for skill_md in sorted(root.rglob("SKILL.md")):
            if not is_excluded_skill_path(skill_md):
                yield skill_md


def _find_skill_file_candidates(skill_name: str) -> List[Path]:
    """Find SKILL.md files matching a bare or canonical skill identifier."""
    if not skill_name:
        return []
    bare = skill_name.rsplit("/", 1)[-1]
    candidates: List[Path] = []
    seen: Set[Path] = set()
    try:
        from agent.skill_utils import get_all_skills_dirs

        roots = get_all_skills_dirs()
    except Exception:
        roots = [_skills_dir()]
    for root in roots:
        if not root.exists():
            continue
        for skill_md in _iter_skill_files_under(root):
            frontmatter = _read_skill_frontmatter(skill_md)
            actual = _clean_str(frontmatter.get("name")) or skill_md.parent.name
            provenance = skill_usage_key_for_path(
                skill_md,
                root=root,
                skill_name=actual,
                frontmatter=frontmatter,
            )
            try:
                rel_parent = str(skill_md.parent.resolve().relative_to(root.resolve()))
            except (OSError, ValueError):
                rel_parent = str(skill_md.parent.name)
            identifiers = {
                str(actual),
                skill_md.parent.name,
                provenance.get("usage_key") or str(actual),
                rel_parent.strip("/"),
            }
            if skill_name in identifiers or ("/" not in skill_name and bare in identifiers):
                try:
                    key = skill_md.resolve()
                except OSError:
                    key = skill_md
                if key not in seen:
                    seen.add(key)
                    candidates.append(skill_md)
    return candidates


def canonicalize_usage_key(skill_name: str) -> str:
    """Canonicalize a telemetry key, adding a bundle namespace when unique."""
    skill_name = str(skill_name or "").strip()
    if not skill_name or ":" in skill_name:
        return skill_name
    if "/" in skill_name:
        return skill_name.strip("/")
    candidates = _find_skill_file_candidates(skill_name)
    if len(candidates) != 1:
        return skill_name
    actual = skill_name if candidates[0].parent.name == skill_name else _read_skill_name(
        candidates[0], fallback=candidates[0].parent.name
    )
    return skill_usage_key_for_path(candidates[0], skill_name=actual).get("usage_key") or skill_name


def _known_key_variants(skill_name: str) -> Set[str]:
    raw = str(skill_name or "").strip().strip("/")
    if not raw:
        return set()
    variants = {raw}
    if "/" in raw:
        variants.add(raw.rsplit("/", 1)[-1])
    canonical = canonicalize_usage_key(raw)
    if canonical:
        variants.add(canonical)
    return variants


def _existing_usage_keys_for_name(
    skill_name: str,
    data: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Find existing sidecar keys that refer to *skill_name*.

    Bare names may use a unique suffix fallback (``agent-made`` ->
    ``devops/agent-made``) for post-move/delete cleanup. Explicit qualified
    keys are exact/canonical only; ``devops/agent-made`` must never match a
    different namespace such as ``qa/agent-made``.
    """
    raw = str(skill_name or "").strip().strip("/")
    if not raw:
        return []
    candidates: List[str] = []

    def _add(key: Optional[str]) -> None:
        if key and key in data and key not in candidates:
            candidates.append(key)

    _add(raw)
    _add(canonicalize_usage_key(raw))
    if "/" in raw:
        return candidates
    bare = raw.rsplit("/", 1)[-1]
    if bare:
        for key in data.keys():
            if key.rsplit("/", 1)[-1] == bare and key not in candidates:
                candidates.append(key)
    return candidates


def _select_usage_key_for_mutation(
    skill_name: str,
    data: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    raw = str(skill_name or "").strip().strip("/")
    if not raw:
        return None
    existing = _existing_usage_keys_for_name(raw, data)
    if len(existing) == 1:
        return existing[0]
    canonical = canonicalize_usage_key(raw)
    if canonical and (canonical != raw or not existing or canonical in data):
        return canonical
    if raw in data or not existing:
        return raw
    # Ambiguous suffix-only matches; avoid creating or mutating the wrong record.
    return None


def _usage_keys_to_forget(
    skill_name: str,
    data: Dict[str, Dict[str, Any]],
) -> Set[str]:
    raw = str(skill_name or "").strip().strip("/")
    if not raw:
        return set()
    keys: Set[str] = set()
    canonical = canonicalize_usage_key(raw)
    for key in {raw, canonical}:
        if key in data:
            keys.add(key)
    if "/" in raw:
        return keys
    existing = _existing_usage_keys_for_name(raw, data)
    if len(existing) == 1:
        keys.add(existing[0])
    return keys


def _expand_known_skill_aliases(names: Set[str]) -> Set[str]:
    """Add canonical bundle/path aliases for known off-limit skill names."""
    out = {str(n) for n in names if str(n).strip()}
    if not out:
        return out
    base = _skills_dir()
    if not base.exists():
        return out
    for skill_md in _iter_skill_files_under(base):
        actual = _read_skill_name(skill_md, fallback=skill_md.parent.name)
        if actual not in out and skill_md.parent.name not in out:
            continue
        provenance = skill_usage_key_for_path(
            skill_md,
            root=base,
            skill_name=actual,
        )
        key = provenance.get("usage_key")
        if key:
            out.add(str(key))
        try:
            out.add(str(skill_md.parent.resolve().relative_to(base.resolve())).strip("/"))
        except (OSError, ValueError):
            pass
    return out


def is_agent_created(skill_name: str) -> bool:
    """Whether *skill_name* is neither bundled nor hub-installed."""
    off_limits = _read_bundled_manifest_names() | _read_hub_installed_names()
    return _known_key_variants(skill_name).isdisjoint(off_limits)


def _is_curator_managed_record(record: Any) -> bool:
    """Return True when a usage record opts a skill into curator management."""
    if not isinstance(record, dict):
        return False
    return record.get("created_by") == "agent" or record.get("agent_created") is True


# ---------------------------------------------------------------------------
# Sidecar I/O
# ---------------------------------------------------------------------------

def _empty_record() -> Dict[str, Any]:
    return {
        "created_by": None,
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
    key = canonicalize_usage_key(skill_name)
    data = load_usage()
    rec = data.get(key)
    if not isinstance(rec, dict) and key != skill_name:
        # Legacy sidecars may have been keyed by bare frontmatter name before
        # bundle-aware canonicalization existed.
        rec = data.get(skill_name)
    if not isinstance(rec, dict):
        existing = _existing_usage_keys_for_name(skill_name, data)
        if len(existing) == 1:
            rec = data.get(existing[0])
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
    Local manual skills may still accrue usage telemetry, but they only
    become curator-managed when ``created_by`` is explicitly marked.
    """
    if not skill_name:
        return
    try:
        if not is_agent_created(skill_name):
            return
        with _usage_file_lock():
            data = load_usage()
            skill_key = _select_usage_key_for_mutation(skill_name, data)
            if not skill_key:
                return
            rec = data.get(skill_key)
            if not isinstance(rec, dict):
                canonical = canonicalize_usage_key(skill_name)
                if canonical != skill_key and isinstance(data.get(canonical), dict):
                    rec = data.pop(canonical)
                elif skill_key != skill_name and isinstance(data.get(skill_name), dict):
                    rec = data.pop(skill_name)
            if not isinstance(rec, dict):
                rec = _empty_record()
            mutator(rec)
            data[skill_key] = rec
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


def mark_agent_created(skill_name: str) -> None:
    """Opt a skill created by skill_manage into curator management.

    Viewing or invoking a manually authored skill may still create telemetry,
    but only this explicit marker makes it eligible for automatic curation.
    """
    def _apply(rec: Dict[str, Any]) -> None:
        rec["created_by"] = "agent"
    _mutate(skill_name, _apply)


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


def forget(skill_name: str) -> None:
    """Drop a skill's usage entry entirely. Called when the skill is deleted."""
    if not skill_name:
        return
    try:
        with _usage_file_lock():
            data = load_usage()
            keys = _usage_keys_to_forget(skill_name, data)
            removed = False
            for key in keys:
                if key in data:
                    del data[key]
                    removed = True
            if removed:
                save_usage(data)
    except Exception as e:
        logger.debug("skill_usage.forget(%s) failed: %s", skill_name, e, exc_info=True)


# ---------------------------------------------------------------------------
# Archive / restore
# ---------------------------------------------------------------------------

def _archive_usage_key_file(skill_dir: Path) -> Path:
    return skill_dir / ".usage_key"


def _read_archive_usage_key(skill_dir: Path) -> Optional[str]:
    marker = _archive_usage_key_file(skill_dir)
    try:
        value = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _write_archive_usage_key(skill_dir: Path, usage_key: str) -> None:
    if not usage_key:
        return
    try:
        _archive_usage_key_file(skill_dir).write_text(usage_key + "\n", encoding="utf-8")
    except OSError as e:
        logger.debug("Failed to write archive usage key for %s: %s", skill_dir, e)


def _remove_archive_usage_key(skill_dir: Path) -> None:
    try:
        _archive_usage_key_file(skill_dir).unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.debug("Failed to remove archive usage key for %s: %s", skill_dir, e)


def _mutate_usage_key_exact(skill_key: str, mutator) -> None:
    """Mutate an exact sidecar key without canonical/suffix fallback."""
    if not skill_key:
        return
    try:
        with _usage_file_lock():
            data = load_usage()
            rec = data.get(skill_key)
            if not isinstance(rec, dict):
                rec = _empty_record()
            mutator(rec)
            data[skill_key] = rec
            save_usage(data)
    except Exception as e:
        logger.debug(
            "skill_usage._mutate_usage_key_exact(%s) failed: %s",
            skill_key,
            e,
            exc_info=True,
        )


def _migrate_archived_usage_key(old_key: Optional[str], target_key: str) -> None:
    """Move a trusted archived usage key to *target_key* after restore."""
    if not old_key or not target_key or old_key == target_key:
        return
    try:
        with _usage_file_lock():
            data = load_usage()
            rec = data.pop(old_key, None)
            if isinstance(rec, dict):
                data[target_key] = rec
                save_usage(data)
    except Exception as e:
        logger.debug(
            "skill_usage._migrate_archived_usage_key(%s -> %s) failed: %s",
            old_key,
            target_key,
            e,
            exc_info=True,
        )


def archive_skill(skill_name: str) -> Tuple[bool, str]:
    """Move an agent-created skill directory to ~/.hermes/skills/.archive/.

    Returns (ok, message). Never archives bundled or hub skills — callers are
    responsible for checking provenance, but we double-check here as a safety net.
    """
    if not is_agent_created(skill_name):
        return False, f"skill '{skill_name}' is bundled or hub-installed; never archive"

    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None:
        return False, f"skill '{skill_name}' not found or ambiguous; use usage_key for same-name namespaces"
    skill_md = skill_dir / "SKILL.md"
    archived_usage_key = skill_usage_key_for_path(
        skill_md,
        root=_skills_dir(),
        skill_name=_read_skill_name(skill_md, fallback=skill_dir.name),
    ).get("usage_key") or skill_name

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

    _write_archive_usage_key(dest, str(archived_usage_key))
    def _mark_archived(rec: Dict[str, Any]) -> None:
        rec["state"] = STATE_ARCHIVED
        rec["archived_at"] = _now_iso()
    _mutate_usage_key_exact(str(archived_usage_key), _mark_archived)
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

    # Try exact name match first, then any prefix match (for timestamped dupes).
    # Recursive walk handles nested archive layouts (e.g. .archive/<category>/<skill>/)
    # left behind by older archive paths or external imports.
    candidates = [p for p in archive_root.rglob("*") if p.is_dir() and p.name == skill_name]
    if not candidates:
        candidates = sorted(
            [p for p in archive_root.rglob("*")
             if p.is_dir() and p.name.startswith(f"{skill_name}-")],
            reverse=True,
        )
    if not candidates:
        return False, f"skill '{skill_name}' not found in archive"

    src = candidates[0]
    archived_usage_key = _read_archive_usage_key(src)
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

    target_key = canonicalize_usage_key(skill_name)
    _migrate_archived_usage_key(archived_usage_key, target_key)
    _remove_archive_usage_key(dest)
    def _mark_active(rec: Dict[str, Any]) -> None:
        rec["state"] = STATE_ACTIVE
        rec["archived_at"] = None
    _mutate_usage_key_exact(target_key, _mark_active)
    return True, f"restored to {dest}"


def _find_skill_dir(skill_name: str) -> Optional[Path]:
    """Locate a skill directory by usage key, relative path, or bare name.

    Qualified identities (``bundle/skill``) are matched exactly against the
    canonical usage key or directory-relative path. Bare names keep the legacy
    frontmatter-name lookup but refuse ambiguous same-name namespaces.
    """
    base = _skills_dir()
    if not base.exists():
        return None
    raw = str(skill_name or "").strip().strip("/")
    if not raw:
        return None
    candidates: List[Path] = []
    for skill_md in base.rglob("SKILL.md"):
        if is_excluded_skill_path(skill_md):
            continue
        frontmatter = _read_skill_frontmatter(skill_md)
        name = _clean_str(frontmatter.get("name")) or skill_md.parent.name
        provenance = skill_usage_key_for_path(
            skill_md,
            root=base,
            skill_name=name,
            frontmatter=frontmatter,
        )
        usage_key = str(provenance.get("usage_key") or name)
        try:
            rel_name = str(skill_md.parent.resolve().relative_to(base.resolve())).strip("/")
        except (OSError, ValueError):
            rel_name = skill_md.parent.name
        if "/" in raw:
            matched = raw in {usage_key, rel_name}
        else:
            matched = raw in {str(name), skill_md.parent.name}
        if matched:
            candidates.append(skill_md.parent)
    if len(candidates) == 1:
        return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Reporting — for the curator CLI / slash command
# ---------------------------------------------------------------------------

def agent_created_report() -> List[Dict[str, Any]]:
    """Return curator-managed skill records keyed by canonical ``usage_key``.

    ``name`` is a display label and may collide across namespaces; callers that
    mutate lifecycle state should use ``usage_key`` when present.
    """
    data = load_usage()
    rows: List[Dict[str, Any]] = []
    for entry in _iter_agent_created_skill_entries():
        name = str(entry["name"])
        key = str(entry["usage_key"])
        rec = _record_for_usage_key(data, key, name)
        if not isinstance(rec, dict):
            rec = _empty_record()
        base = _empty_record()
        for k, v in base.items():
            rec.setdefault(k, v)
        row = {
            "name": name,
            "usage_key": key,
            "bundle_id": entry.get("bundle_id"),
            "source_repo": entry.get("source_repo"),
            "source_ref": entry.get("source_ref"),
            "source_commit": entry.get("source_commit"),
            **rec,
        }
        row["last_activity_at"] = latest_activity_at(row)
        row["activity_count"] = activity_count(row)
        rows.append(row)
    return rows


def _record_for_usage_key(
    usage: Dict[str, Dict[str, Any]],
    usage_key: str,
    legacy_name: str,
) -> Optional[Dict[str, Any]]:
    """Return exact canonical record or exact legacy bare-name record only.

    Reporting paths must not suffix-match ``*/legacy_name``: a record for
    ``qa/agent-made`` does not belong to ``devops/agent-made``.
    """
    rec = usage.get(usage_key)
    if isinstance(rec, dict):
        return rec
    rec = usage.get(legacy_name)
    if isinstance(rec, dict):
        return rec
    return None


def bundle_usage_report() -> List[Dict[str, Any]]:
    """Aggregate usage telemetry by skill package / bundle namespace.

    Groups are discovered from ``metadata.hermes.bundle_id`` when present, or
    from the first path segment for nested skill packages such as
    ``superpowers-zh/<skill>/SKILL.md``. Counts include legacy bare-name usage
    records so older sidecars are still reflected in package totals.
    """
    usage = load_usage()
    bundles: Dict[str, Dict[str, Any]] = {}
    try:
        from agent.skill_utils import get_all_skills_dirs

        roots = get_all_skills_dirs()
    except Exception:
        roots = [_skills_dir()]

    seen_files: Set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for skill_md in _iter_skill_files_under(root):
            try:
                file_key = skill_md.resolve()
            except OSError:
                file_key = skill_md
            if file_key in seen_files:
                continue
            seen_files.add(file_key)

            frontmatter = _read_skill_frontmatter(skill_md)
            skill_name = _clean_str(frontmatter.get("name")) or skill_md.parent.name
            provenance = skill_usage_key_for_path(
                skill_md,
                root=root,
                skill_name=skill_name,
                frontmatter=frontmatter,
            )
            bundle_id = provenance.get("bundle_id")
            if not bundle_id:
                continue

            row = bundles.setdefault(
                str(bundle_id),
                {
                    "bundle_id": str(bundle_id),
                    "source_repo": None,
                    "source_ref": None,
                    "source_commit": None,
                    "skill_count": 0,
                    "recorded_skill_count": 0,
                    "use_count": 0,
                    "view_count": 0,
                    "patch_count": 0,
                    "activity_count": 0,
                    "last_activity_at": None,
                    "skills": [],
                },
            )
            row["skill_count"] += 1
            row["skills"].append(provenance.get("usage_key") or str(skill_name))
            for field in ("source_repo", "source_ref", "source_commit"):
                if not row.get(field) and provenance.get(field):
                    row[field] = provenance.get(field)

            rec = _record_for_usage_key(
                usage,
                str(provenance.get("usage_key") or skill_name),
                str(skill_name),
            )
            if not isinstance(rec, dict):
                continue
            row["recorded_skill_count"] += 1
            base = _empty_record()
            merged = {**base, **rec}
            for field in ("use_count", "view_count", "patch_count"):
                try:
                    row[field] += int(merged.get(field) or 0)
                except (TypeError, ValueError):
                    pass
            row["activity_count"] += activity_count(merged)
            last = latest_activity_at(merged)
            if last:
                if not row["last_activity_at"]:
                    row["last_activity_at"] = last
                else:
                    current = _parse_iso_timestamp(row["last_activity_at"])
                    candidate = _parse_iso_timestamp(last)
                    if candidate and (current is None or candidate > current):
                        row["last_activity_at"] = last

    for row in bundles.values():
        row["skills"] = sorted(set(row["skills"]))
    return sorted(
        bundles.values(),
        key=lambda r: (-(r.get("activity_count") or 0), str(r.get("bundle_id") or "")),
    )
