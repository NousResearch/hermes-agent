"""Skill usage telemetry + provenance for the Curator: a sidecar ``~/.hermes/skills/.usage.json`` keyed by
skill name (never frontmatter — keeps telemetry out of user-authored SKILL.md and off bundled/hub skills).
Counter bumps are best-effort (DEBUG-logged failures never break the tool call); writes are atomic under a
cross-process lock. Curator management is an explicit ``created_by: agent`` marker written by skill_manage —
never inferred from location. Lifecycle: active -> stale -> archived (moved to .archive/); ``pinned`` opts
out of auto transitions, orthogonal to state."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home
from agent.skill_utils import is_excluded_skill_path, is_external_skill_path
from utils import atomic_write_text

logger = logging.getLogger(__name__)

# fcntl is Unix-only; on Windows use msvcrt for file locking.
msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover - platform-specific fallback
    fcntl = None
    with suppress(ImportError):
        import msvcrt


STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED = "active", "stale", "archived"
_VALID_STATES = {STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED}

# Load-bearing built-ins (by frontmatter ``name``) the curator must NEVER archive/consolidate regardless of
# ``curator.prune_builtins``, pins or LLM judgment — archiving one breaks its slash command. Keep tiny.
PROTECTED_BUILTIN_SKILLS: Set[str] = set()


def is_protected_builtin(skill_name: str) -> bool:
    return skill_name in PROTECTED_BUILTIN_SKILLS


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def _usage_file() -> Path:
    return _skills_dir() / ".usage.json"


def _archive_dir() -> Path:
    return _skills_dir() / ".archive"


def _flock(fd, lock: bool) -> None:
    if fcntl:
        return fcntl.flock(fd, fcntl.LOCK_EX if lock else fcntl.LOCK_UN)
    fd.seek(0)
    msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK if lock else msvcrt.LK_UNLCK, 1)


@contextmanager
def _usage_file_lock():
    """Serialize .usage.json read-modify-write cycles across processes."""
    lock_path = _usage_file().with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None and msvcrt is None:
        yield
        return
    if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
        lock_path.write_text(" ", encoding="utf-8")  # msvcrt needs a non-empty byte range to lock
    with open(lock_path, "r+" if msvcrt else "a+", encoding="utf-8") as fd:
        _flock(fd, True)
        try:
            yield
        finally:
            with suppress(OSError, IOError):
                _flock(fd, False)


def _read_lines(path: Path, fail_log: str) -> List[str]:
    """Stripped, non-empty lines of a small metadata file ([] if missing/unreadable)."""
    if not path.exists():
        return []
    try:
        return [s for s in (line.strip() for line in path.read_text(encoding="utf-8").splitlines()) if s]
    except OSError as e:
        logger.debug(fail_log, e)
        return []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value)) if value else None
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed and parsed.tzinfo is None else parsed


def latest_activity_at(record: Dict[str, Any]) -> Optional[str]:
    """Newest use/view/patch timestamp; ``created_at`` is excluded so never-active skills stay distinguishable."""
    stamps = [(dt, str(raw)) for raw in (record.get(k) for k in ("last_used_at", "last_viewed_at", "last_patched_at"))
              if (dt := _parse_iso_timestamp(raw)) is not None]
    return max(stamps, key=lambda t: t[0])[1] if stamps else None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _non_negative_int(value: Any) -> int:
    return 0 if isinstance(value, bool) else max(0, _int_or_zero(value))


def activity_count(record: Dict[str, Any]) -> int:
    """Total observed use+view+patch events."""
    return sum(_int_or_zero(record.get(key)) for key in ("use_count", "view_count", "patch_count"))


# --- Provenance — which skills are agent-created (and thus eligible for curation) ---
def _read_bundled_manifest_names() -> Set[str]:
    """Names from ``.bundled_manifest`` ("name:hash" per line); empty if missing/unreadable."""
    lines = _read_lines(_skills_dir() / ".bundled_manifest", "Failed to read bundled manifest: %s")
    return {n for n in (line.split(":", 1)[0].strip() for line in lines) if n}


def _read_hub_installed_names() -> Set[str]:
    """Hub-installed names (``.hub/lock.json``) plus the frontmatter name of each in-tree ``install_path``."""
    skills_dir = _skills_dir()
    lock_path = skills_dir / ".hub" / "lock.json"
    if not lock_path.exists():
        return set()
    # The whole walk sits under one handler (BASE semantics): an OSError anywhere — including the
    # per-skill SKILL.md read — logs and yields an empty set rather than a partial one.
    try:
        # errors="replace": hub descriptions can carry Windows-1252 high bytes; a strict read raises
        # UnicodeDecodeError (a ValueError, not caught below) and would 500 the whole /api/skills endpoint.
        data = json.loads(lock_path.read_text(encoding="utf-8", errors="replace"))
        installed = (data.get("installed") or {}) if isinstance(data, dict) else None
        if not isinstance(installed, dict):
            return set()
        names = {str(k) for k in installed}
        paths = (e.get("install_path") for e in installed.values() if isinstance(e, dict))
        for install_path in (p for p in paths if isinstance(p, str) and p.strip()):
            try:  # ValueError: install_path escapes the skills dir
                resolved = (skills_dir / install_path).resolve()
                resolved.relative_to(skills_dir.resolve())
            except (OSError, ValueError):
                continue
            if (resolved / "SKILL.md").exists():
                names.add(_read_skill_name(resolved / "SKILL.md", fallback=resolved.name))
        return names
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read hub lock file: %s", e)
    return set()


def _prune_builtins_enabled() -> bool:
    """``curator.prune_builtins`` (default True); lazy config import keeps this module importable during update/sync."""
    try:
        from hermes_cli.config import load_config
        cur = load_config().get("curator")
        return bool(cur.get("prune_builtins", True)) if isinstance(cur, dict) else True
    except Exception as e:  # pragma: no cover — best-effort config read
        logger.debug("Failed to read curator.prune_builtins: %s", e)
        return True


def read_suppressed_names() -> Set[str]:
    """Built-ins the curator pruned (``.curator_suppressed``); the update-time re-seeder must leave these archived."""
    lines = _read_lines(_skills_dir() / ".curator_suppressed", "Failed to read curator suppression list: %s")
    return {line for line in lines if not line.startswith("#")}


def _toggle_suppressed_name(skill_name: str, *, add: bool) -> None:
    """Add (built-in pruned) or drop (restored) *skill_name* in the suppression list; no-op when unchanged."""
    if not skill_name or (skill_name in (names := read_suppressed_names())) == add:
        return
    (names.add if add else names.discard)(skill_name)
    try:
        atomic_write_text(_skills_dir() / ".curator_suppressed", "\n".join(sorted(names)) + ("\n" if names else ""),
                          tmp_prefix=".curator_suppressed_")
    except Exception as e:
        logger.debug("Failed to write curator suppression list: %s", e, exc_info=True)


def _iter_skill_mds(base: Path, *, local_only: bool) -> Iterator[Tuple[str, Path]]:
    """``(frontmatter name, SKILL.md)`` under *base* minus metadata/VCS/venv/cache dirs; *local_only* also skips
    external skill dirs mounted below the tree (curation must not touch them)."""
    for skill_md in base.rglob("SKILL.md"):
        if not (is_excluded_skill_path(skill_md) or (local_only and is_external_skill_path(skill_md))):
            yield _read_skill_name(skill_md, fallback=skill_md.parent.name), skill_md


def _scan_local_skills(keep: Callable[[str, Path, Set[str], Dict[str, Any]], bool]) -> List[str]:
    """Sorted local skill names passing *keep(name, skill_md, bundled, usage)*; hub/protected names never reach it."""
    if not (base := _skills_dir()).exists():
        return []
    hub, bundled, usage = _read_hub_installed_names(), _read_bundled_manifest_names(), load_usage()
    return sorted({name for name, skill_md in _iter_skill_mds(base, local_only=True)
                   if name not in hub and not is_protected_builtin(name) and keep(name, skill_md, bundled, usage)})


def list_agent_created_skill_names() -> List[str]:
    """Curator-manageable skills: ``created_by: agent`` records plus, with ``curator.prune_builtins``, bundled
    built-ins (which never carry a managed record, so the record gate applies only to local skills). Never hub."""
    prune_builtins = _prune_builtins_enabled()  # read once, before the walk
    return _scan_local_skills(
        lambda name, _md, bundled, usage: prune_builtins if name in bundled else _is_curator_managed_record(usage.get(name)))


def list_archived_skill_names() -> List[str]:
    """Skills in ``.archive/`` — flat layout (``archive_skill`` flattens), so dir name == skill name."""
    root = _archive_dir()
    return sorted({p.name for p in root.iterdir() if p.is_dir()}) if root.exists() else []


def _read_skill_name(skill_md: Path, fallback: str) -> str:
    """The frontmatter ``name:`` field of a SKILL.md (first 4000 chars), else *fallback*."""
    try:
        lines = [line.strip() for line in skill_md.read_text(encoding="utf-8", errors="replace")[:4000].split("\n")]
    except OSError:
        return fallback
    if "---" not in lines:
        return fallback
    block = lines[lines.index("---") + 1:]  # frontmatter runs to the closing --- or (truncated) end of text
    block = block[:block.index("---")] if "---" in block else block
    values = (line.split(":", 1)[1].strip().strip("\"'") for line in block if line.startswith("name:"))
    return next((v for v in values if v), fallback)


def is_agent_created(skill_name: str) -> bool:
    """Neither bundled nor hub-installed (and not only present in an external dir)."""
    return not (is_bundled(skill_name) or is_hub_installed(skill_name)) and (
        _find_skill_dir(skill_name) is not None or _find_external_skill_dir(skill_name) is None)


def is_hub_installed(skill_name: str) -> bool:
    return skill_name in _read_hub_installed_names()


def is_bundled(skill_name: str) -> bool:
    return skill_name in _read_bundled_manifest_names()


def _external_read_only_message(skill_name: str) -> str:
    return f"skill '{skill_name}' lives in skills.external_dirs; external skills are read-only to the curator"


def is_curation_eligible(skill_name: str, skill_path: Optional[Path] = None) -> bool:
    """Agent-created: yes. Bundled: only with ``curator.prune_builtins``. Hub / external-dir / protected built-ins:
    never (external owner). Org-shared skills are eligible here but protected from ARCHIVE/DELETE elsewhere."""
    if ((skill_path is not None and is_external_skill_path(skill_path)) or is_protected_builtin(skill_name)
            or is_hub_installed(skill_name)):
        return False
    if is_bundled(skill_name):
        return _prune_builtins_enabled()
    local_dir = _find_skill_dir(skill_name)
    return not is_external_skill_path(local_dir) if local_dir else _find_external_skill_dir(skill_name) is None


def _is_curator_managed_record(record: Any) -> bool:
    """``created_by`` is a curator-management OPT-IN flag, not proof of authorship (``curator adopt`` flips it);
    the key name is kept because it lives in every user's ``.usage.json``.

    NAMING (issue #67140): the on-disk field is ``created_by``, which reads like provenance but is consumed
    as a **curator-management opt-in policy flag**. The two are not the same question:
    """
    return isinstance(record, dict) and (record.get("created_by") == "agent" or record.get("agent_created") is True)


def is_curator_managed(skill_name: str) -> bool:
    return _is_curator_managed_record(load_usage().get(skill_name))


def list_unmanaged_skill_names() -> List[str]:
    """Curation-ELIGIBLE skills without a provenance marker (pre-``created_by`` records, or foreground creates that
    belong to the user). Invisible to ``curated_report()`` and auto transitions; only ``curator adopt`` hands
    them over — provenance is declared, never inferred from activity."""
    return _scan_local_skills(
        lambda name, md, bundled, usage: name not in bundled and not _is_curator_managed_record(usage.get(name))
        and is_curation_eligible(name, md))


def unmanaged_report() -> List[Dict[str, Any]]:
    """Rows for :func:`list_unmanaged_skill_names`; ``has_provenance_key`` (False = pre-dates ``created_by``) explains
    WHY, it is not a signal to adopt on."""
    usage = load_usage()
    return [_report_row(n, usage.get(n), has_provenance_key="created_by" in usage.get(n, {}), has_record=n in usage)
            for n in list_unmanaged_skill_names()]


def adopt_skill(skill_name: str) -> Tuple[bool, str]:
    """User-declared handover: writes the ``created_by: agent`` marker (inactivity clock NOT reset). Refuses hub,
    external, bundled and protected skills. Returns (ok, message)."""
    if not skill_name:
        return False, "no skill name given"
    if is_protected_builtin(skill_name):
        return False, f"'{skill_name}' is a protected built-in; the curator never manages it"
    if is_hub_installed(skill_name):
        return False, f"'{skill_name}' is hub-installed; its upstream owns it"
    if is_bundled(skill_name):  # governed by prune_builtins; stamping created_by=agent would change nothing
        return False, f"'{skill_name}' is a bundled built-in — it is governed by curator.prune_builtins, not by adoption"
    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None:
        if _find_external_skill_dir(skill_name) is not None:
            return False, f"'{skill_name}' lives in skills.external_dirs and is read-only to the curator"
        return False, f"skill '{skill_name}' not found"
    if is_external_skill_path(skill_dir):
        return False, _external_read_only_message(skill_name)
    if is_curator_managed(skill_name):
        return True, f"'{skill_name}' is already curator-managed"
    mark_agent_created(skill_name)
    if is_curator_managed(skill_name):
        return True, f"adopted '{skill_name}' into curator management"
    return False, f"could not mark '{skill_name}' as curator-managed"


# --- Sidecar I/O ---
def _empty_record() -> Dict[str, Any]:
    return {"created_by": None, "use_count": 0, "view_count": 0, "last_used_at": None, "last_viewed_at": None,
            "patch_count": 0, "patch_generation": 0, "last_reused_patch_generation": 0, "last_patched_at": None,
            "created_at": _now_iso(), "state": STATE_ACTIVE, "pinned": False, "archived_at": None}


def _backfilled(rec: Any) -> Dict[str, Any]:
    """*rec* with every missing default key appended (a fresh record when not a dict)."""
    if not isinstance(rec, dict):
        return _empty_record()
    return {**rec, **{k: v for k, v in _empty_record().items() if k not in rec}}


def _report_row(name: str, raw: Any, **extra: Any) -> Dict[str, Any]:
    row = {"name": name, **_backfilled(raw), **extra}
    row.update(last_activity_at=latest_activity_at(row), activity_count=activity_count(row))
    return row


def load_usage() -> Dict[str, Dict[str, Any]]:
    """The whole .usage.json map (non-dict values dropped); {} on missing/corrupt."""
    path = _usage_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Failed to read %s: %s", path, e)
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def save_usage(data: Dict[str, Dict[str, Any]]) -> bool:
    """Write the usage map atomically; True when it committed."""
    path = _usage_file()
    try:
        atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), tmp_prefix=".usage_")
        return True
    except Exception as e:
        logger.debug("Failed to write %s: %s", path, e, exc_info=True)
        return False


def get_record(skill_name: str) -> Dict[str, Any]:
    """The (backfilled) record for *skill_name*; fresh defaults if missing."""
    return _backfilled(load_usage().get(skill_name))


def _locked_update(skill_name: str, op: Callable[[Dict[str, Dict[str, Any]]], Tuple[Any, bool]], fail_log: str,
                   guard: Optional[Callable[[], bool]] = None) -> Any:
    """*op(data) -> (result, dirty)* under the file lock, saving only when dirty; *guard* runs before locking.
    None when the guard failed, the save did not land, or anything raised (DEBUG-logged via *fail_log*)."""
    try:
        if guard is not None and not guard():
            return None
        with _usage_file_lock():
            data = load_usage()
            result, dirty = op(data)
            return None if dirty and not save_usage(data) else result
    except Exception as e:
        logger.debug(fail_log, skill_name, e, exc_info=True)
        return None


def seed_record_if_missing(skill_name: str) -> None:
    """Baseline record for a curation-eligible skill so its inactivity clock starts at first sight, not epoch."""
    if skill_name and is_curation_eligible(skill_name):
        # load_usage() already dropped non-dict values, so "missing" == key absent; dirty only when inserted.
        def _seed(data):
            return None, skill_name not in data and data.setdefault(skill_name, _empty_record()) is not None
        _locked_update(skill_name, _seed, "skill_usage.seed_record_if_missing(%s) failed: %s")


def _mutate(skill_name: str, mutator, *, require_curation_eligible: bool = False) -> Any:
    """Load, apply *mutator(record)* in place, save; the mutator result (None if nothing landed). Telemetry is
    recorded for ANY skill; lifecycle mutators pass ``require_curation_eligible=True`` (never write onto unmanaged)."""
    if not skill_name:
        return None
    return _locked_update(skill_name, lambda data: (mutator(data.setdefault(skill_name, _empty_record())), True),
                          "skill_usage._mutate(%s) failed: %s",
                          (lambda: is_curation_eligible(skill_name)) if require_curation_eligible else None)


def _set_field(skill_name: str, key: str, value: Any) -> bool:
    """Curation-gated single-field write; True only when the write landed."""
    return bool(_mutate(skill_name, lambda rec: rec.update({key: value}) or True, require_curation_eligible=True))


def _bump(rec: Dict[str, Any], count_key: str, ts_key: str) -> None:
    rec[count_key] = _non_negative_int(rec.get(count_key)) + 1
    rec[ts_key] = _now_iso()


def telemetry_provenance(skill_name: str, record: Optional[Dict[str, Any]] = None) -> str:
    """Bounded provenance label for shared skill metrics."""
    if is_hub_installed(skill_name) or is_bundled(skill_name):
        return "installed"
    if ":" in skill_name:
        with suppress(Exception):
            from hermes_cli.plugins import get_plugin_manager
            if get_plugin_manager().find_plugin_skill(skill_name) is not None:
                return "installed"
    if label := {"installed": "installed", "agent": "agent_created"}.get(
            record.get("created_by") if isinstance(record, dict) else None):
        return label
    if _find_external_skill_dir(skill_name) is not None:
        return "external"
    return "local" if _find_skill_dir(skill_name) is not None or isinstance(record, dict) else "unknown"


def _emit_skill_lifecycle(skill_name: str, action: str, *, record: Optional[Dict[str, Any]] = None,
                          task_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
    """Best-effort lifecycle hook after an authoritative state change; facts absent from *record* go as None."""
    facts = record or {}
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook
        if has_hook("on_skill_lifecycle"):
            invoke_hook("on_skill_lifecycle", action=action, skill_name=skill_name,
                        provenance=telemetry_provenance(skill_name, record), task_id=task_id or "",
                        session_id=session_id or "", use_count=facts.get("use_count"), reused=facts.get("reused"),
                        reuse_after_patch=facts.get("reuse_after_patch"))
    except Exception:
        logger.debug("skill_usage lifecycle hook failed for %s/%s", skill_name, action, exc_info=True)


def _mutate_and_emit(skill_name: str, action: str, mutator: Callable[[Dict[str, Any]], Dict[str, Any]],
                     **hook_kwargs: Any) -> None:
    """``_mutate`` then emit *action* with the mutator's facts as the record — only if the write landed."""
    if isinstance(facts := _mutate(skill_name, mutator), dict):
        _emit_skill_lifecycle(skill_name, action, record=facts, **hook_kwargs)


# --- Counter bumps — telemetry for ALL skills regardless of provenance (observability only) ---
def bump_view(skill_name: str) -> None:
    _mutate(skill_name, lambda rec: _bump(rec, "view_count", "last_viewed_at"))


def bump_use(skill_name: str, *, task_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
    """Skill actively used (loaded into the prompt path / referenced from an assistant turn)."""
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        uses = _non_negative_int(rec.get("use_count"))
        gen = _non_negative_int(rec.get("patch_generation"))
        last_reused = min(_non_negative_int(rec.get("last_reused_patch_generation")), gen)
        reuse_after_patch = uses > 0 and gen > last_reused
        rec.update(use_count=uses + 1, last_used_at=_now_iso(), patch_generation=gen,
                   last_reused_patch_generation=gen if reuse_after_patch else last_reused)
        return {"created_by": rec.get("created_by"), "use_count": uses + 1, "reused": uses > 0,
                "reuse_after_patch": reuse_after_patch}
    _mutate_and_emit(skill_name, "loaded", _apply, task_id=task_id, session_id=session_id)


def bump_patch(skill_name: str, *, action: str = "patch", task_id: Optional[str] = None,
               session_id: Optional[str] = None) -> None:
    """Called from skill_manage (patch/edit)."""
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        _bump(rec, "patch_count", "last_patched_at")
        rec["patch_generation"] = _non_negative_int(rec.get("patch_generation")) + 1
        return {"created_by": rec.get("created_by")}
    _mutate_and_emit(skill_name, "patched" if action == "patch" else "edited", _apply, task_id=task_id,
                     session_id=session_id)


def record_created(skill_name: str, *, agent_created: bool, task_id: Optional[str] = None,
                   session_id: Optional[str] = None) -> None:
    """Persist creation provenance and emit a create fact; the record is reset (a create is a new logical skill).

    Foreground creates (``agent_created=False`` — e.g. ``/learn`` at the user's request) are stamped
    ``created_by="learn"``: a learning-signal marker, NOT the curator-management opt-in (``"agent"``),
    so /journey can show user-taught skills without handing them to autonomous curation.
    """
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        rec.clear()
        rec.update(_empty_record(), created_by="agent" if agent_created else "learn")
        return {"created_by": rec["created_by"]}
    _mutate_and_emit(skill_name, "created", _apply, task_id=task_id, session_id=session_id)


def record_installed(skill_name: str) -> None:
    """Record a successful Skills Hub install without exporting its name."""
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        rec.update(created_by="installed", state=STATE_ACTIVE, archived_at=None)
        return {"created_by": "installed"}
    _mutate_and_emit(skill_name, "installed", _apply)


def mark_agent_created(skill_name: str) -> None:
    """Opt a skill into curator management — the only thing that makes it eligible for automatic curation."""
    _set_field(skill_name, "created_by", "agent")


def set_state(skill_name: str, state: str) -> None:
    """Set lifecycle state (no-op if invalid / unmanageable). Emits archived/stale/restored; active<-stale is silent."""
    if state not in _VALID_STATES:
        logger.debug("set_state: invalid state %r for %s", state, skill_name)
        return

    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        previous = rec.get("state")
        if previous != state:
            rec["state"] = state
            if state != STATE_STALE:
                rec["archived_at"] = _now_iso() if state == STATE_ARCHIVED else None
        return {"changed": previous != state, "created_by": rec.get("created_by"), "previous_state": previous}
    facts = _mutate(skill_name, _apply, require_curation_eligible=True)
    if isinstance(facts, dict) and facts["changed"]:
        restored = state == STATE_ACTIVE and facts["previous_state"] == STATE_ARCHIVED
        action = "restored" if restored else {STATE_ARCHIVED: "archived", STATE_STALE: "stale"}.get(state)
        if action is not None:
            _emit_skill_lifecycle(skill_name, action, record=facts)


def set_pinned(skill_name: str, pinned: bool) -> bool:
    """False when the write did not land (not curation-eligible).

    (skill not curation-eligible), True on success — so callers can report failure instead of a false
    success (issue #92993).
    """
    return _set_field(skill_name, "pinned", bool(pinned))


def set_sync(skill_name: str, sync: bool) -> None:
    """Opt-in ``sync`` flag (read by ``skills_sync_client``); curation-gated so bundled/hub/external can't be marked."""
    _set_field(skill_name, "sync", bool(sync))


def is_sync_enabled(skill_name: str) -> bool:
    return get_record(skill_name).get("sync") is True


def forget(skill_name: str) -> None:
    if skill_name:
        _locked_update(skill_name, lambda d: (None, d.pop(skill_name, None) is not None), "skill_usage.forget(%s) failed: %s")


# --- Archive / restore ---
def _relocate(src: Path, dest: Path, skill_name: str, action: str, **capture_kwargs: Any) -> Tuple[bool, str]:
    """Move *src* to *dest* for *action* ("archive" | "restore") inside a best-effort audit-ledger entry, then apply
    suppression + state side effects; rename falls back to shutil.move across devices."""
    try:
        from tools import skill_ledger as _ledger
        _ledger_before = _ledger.capture_before(src, **capture_kwargs)
    except Exception:
        _ledger = _ledger_before = None  # type: ignore[assignment]
    try:
        src.rename(dest)
    except OSError:
        import shutil
        try:
            shutil.move(str(src), str(dest))
        except Exception as e:
            return False, f"failed to {action}: {e}"
    archiving = action == "archive"
    if not archiving or is_bundled(skill_name):  # pruning a built-in only sticks if the re-seeder skips it
        _toggle_suppressed_name(skill_name, add=archiving)
    set_state(skill_name, STATE_ARCHIVED if archiving else STATE_ACTIVE)
    with suppress(Exception):
        if _ledger is not None:
            _ledger.record_mutation(action, skill_name, before=_ledger_before or [], after_root=dest)
    return True, f"{action}d to {dest}"


def archive_skill(skill_name: str) -> Tuple[bool, str]:
    """Move a curator-eligible skill dir to ``.archive/`` (flattened; timestamp suffix on collision). Never hub;
    bundled built-ins only with ``curator.prune_builtins`` (and then suppressed from re-seeding)."""
    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None and _find_external_skill_dir(skill_name) is not None:
        return False, _external_read_only_message(skill_name)
    if not is_curation_eligible(skill_name, skill_dir):
        if is_protected_builtin(skill_name):
            return False, f"skill '{skill_name}' is a protected built-in; it backs load-bearing UX and is never archived or consolidated"
        if is_hub_installed(skill_name):
            return False, f"skill '{skill_name}' is hub-installed; never archive"
        return False, f"skill '{skill_name}' is a bundled built-in; enable curator.prune_builtins to allow pruning it"
    if skill_dir is None:
        return False, f"skill '{skill_name}' not found"
    if is_external_skill_path(skill_dir):
        return False, _external_read_only_message(skill_name)
    dest = _archive_dir() / skill_dir.name
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"failed to create archive dir: {e}"
    if dest.exists():
        dest = dest.with_name(f"{skill_dir.name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    # complete_package: consolidation may have re-homed support files first, so a disk-only capture can come
    # back hollow; the fill from the newest curator backup keeps rollback restorable.
    return _relocate(skill_dir, dest, skill_name, "archive", complete_package=True, skill=skill_name)


def restore_skill(skill_name: str) -> Tuple[bool, str]:
    """Move an archived skill back to the flat layout (nesting NOT reconstructed). Refuses a name now colliding with
    a hub skill, or a bundled built-in unless ``curator.prune_builtins`` is on (restoring lifts a prune)."""
    if is_hub_installed(skill_name):
        return False, f"skill '{skill_name}' is now hub-installed; restore would shadow the upstream version"
    if is_bundled(skill_name) and not _prune_builtins_enabled():
        return False, f"skill '{skill_name}' is now bundled; restore would shadow the upstream version"
    archive_root = _archive_dir()
    if not archive_root.exists():
        return False, "no archive directory"
    # Exact name first (recursive: older archives left nested layouts), then the timestamped duplicate. Only
    # "<skill>-YYYYMMDDHHMMSS" counts — a bare startswith("<skill>-") would let restoring "git" steal "git-helpers".
    dirs = [p for p in archive_root.rglob("*") if p.is_dir()]
    prefix = f"{skill_name}-"
    candidates = [p for p in dirs if p.name == skill_name] or sorted(
        (p for p in dirs if p.name.startswith(prefix) and len(p.name) - len(prefix) == 14
         and p.name[len(prefix):].isdigit()), reverse=True)
    if not candidates:
        return False, f"skill '{skill_name}' not found in archive"
    if (dest := _skills_dir() / skill_name).exists():
        return False, f"destination already exists: {dest}"
    return _relocate(candidates[0], dest, skill_name, "restore")


def _match_skill_dir(skill_mds: Iterable[Path], skill_name: str) -> Optional[Path]:
    return next((p.parent for p in skill_mds if _read_skill_name(p, fallback=p.parent.name) == skill_name), None)


def _find_skill_dir(skill_name: str) -> Optional[Path]:
    """Skill dir by frontmatter ``name`` (flat or nested); the gated index iterator sees only the active org mirror."""
    from agent.skill_utils import iter_skill_index_files
    base = _skills_dir()
    return _match_skill_dir((p for p in iter_skill_index_files(base, "SKILL.md") if not is_external_skill_path(p)),
                            skill_name) if base.exists() else None


def _find_external_skill_dir(skill_name: str) -> Optional[Path]:
    """Skill dir under configured external dirs by frontmatter name."""
    from agent.skill_utils import get_all_skills_dirs
    return next((found for base in get_all_skills_dirs()[1:] if base.exists()
                 if (found := _match_skill_dir((p for p in base.rglob("SKILL.md") if not is_excluded_skill_path(p)),
                                               skill_name)) is not None), None)


# --- Reporting — for the curator CLI / slash command ---
def curated_report() -> List[Dict[str, Any]]:
    """One backfilled row per curator-managed skill with ``provenance`` and ``_persisted`` (real record exists; fresh
    backfills get their inactivity clock seeded instead of counting as ancient)."""
    data = load_usage()
    names = set(list_agent_created_skill_names())
    # Pinned-but-unmanaged skills stay visible or their pin silently vanishes from `curator status`; the local-dir
    # guard keeps stale records for deleted dirs from rendering as ghost rows.
    names.update(name for name, rec in data.items()
                 if rec.get("pinned") and is_curation_eligible(name) and _find_skill_dir(name) is not None)
    return [_report_row(n, data.get(n), _persisted=n in data, provenance=provenance(n)) for n in sorted(names)]


def provenance(skill_name: str) -> str:
    """'hub' | 'bundled' | 'agent' (the latter also covers local manually-authored skills)."""
    return "hub" if is_hub_installed(skill_name) else "bundled" if is_bundled(skill_name) else "agent"


def usage_report() -> List[Dict[str, Any]]:
    """Usage rows for EVERY skill on disk (built-ins and hub included); ``curated_report()`` is the managed subset."""
    if not (base := _skills_dir()).exists():
        return []
    data = load_usage()
    return [_report_row(n, data.get(n), provenance=provenance(n), _persisted=n in data)
            for n in sorted({name for name, _md in _iter_skill_mds(base, local_only=False)})]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import tempfile  # noqa: F401,E402
import os  # noqa: F401,E402
import os  # noqa: F401,E402
import tempfile  # noqa: F401,E402

def _suppressed_file() -> Path:
    return _skills_dir() / ".curator_suppressed"

def _write_suppressed_names(names: Set[str]) -> None:
    path = _suppressed_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = "\n".join(sorted(names)) + ("\n" if names else "")
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".curator_suppressed_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("Failed to write curator suppression list: %s", e, exc_info=True)

def add_suppressed_name(skill_name: str) -> None:
    """Record that a built-in skill was pruned, so sync won't restore it."""
    if not skill_name:
        return
    names = read_suppressed_names()
    if skill_name not in names:
        names.add(skill_name)
        _write_suppressed_names(names)


def remove_suppressed_name(skill_name: str) -> None:
    """Clear a built-in's suppression entry (e.g. on restore)."""
    if not skill_name:
        return
    names = read_suppressed_names()
    if skill_name in names:
        names.discard(skill_name)
        _write_suppressed_names(names)


def list_agent_created_skill_names() -> List[str]:
    """Enumerate skills the curator may manage.

    Always includes agent-authored skills (those marked in ``.usage.json`` via
    ``skill_manage(action="create")``). When ``curator.prune_builtins`` is
    enabled, bundled built-in skills are ALSO included even though they have no
    agent-created usage record — their inactivity clock is anchored on first
    sight (see ``apply_automatic_transitions``). Hub-installed skills are never
    included; manually authored skills are not inferred from filesystem
    location.
    """
    base = _skills_dir()
    if not base.exists():
        return []
    hub = _read_hub_installed_names()
    bundled = _read_bundled_manifest_names()
    prune_builtins = _prune_builtins_enabled()
    usage = load_usage()

    names: List[str] = []
    # Top-level SKILL.md files (flat layout) AND nested category/skill/SKILL.md
    for skill_md in base.rglob("SKILL.md"):
        # Skip Hermes metadata, VCS, virtualenv/dependency, and cache dirs
        if is_excluded_skill_path(skill_md):
            continue
        # External skill dirs can be mounted below the local skills tree.
        # Discovery may see them, but autonomous lifecycle curation must not.
        if is_external_skill_path(skill_md):
            continue
        try:
            skill_md.relative_to(base)
        except ValueError:
            continue
        name = _read_skill_name(skill_md, fallback=skill_md.parent.name)
        # Hub-installed skills are always off-limits.
        if name in hub:
            continue
        # Protected built-ins are never curation candidates — exempt from the
        # automatic transition walk AND the LLM consolidation pass.
        if is_protected_builtin(name):
            continue
        if name in bundled:
            # Built-ins are only candidates when pruning is enabled. They never
            # carry a curator-managed record, so the record gate is skipped.
            if not prune_builtins:
                continue
            names.append(name)
            continue
        # Agent-authored (or local-manual) skills must opt in via their record.
        if not _is_curator_managed_record(usage.get(name)):
            continue
        names.append(name)
    return sorted(set(names))


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


def is_agent_created(skill_name: str) -> bool:
    """Whether *skill_name* is neither bundled nor hub-installed."""
    off_limits = _read_bundled_manifest_names() | _read_hub_installed_names()
    if skill_name in off_limits:
        return False
    return not (
        _find_skill_dir(skill_name) is None
        and _find_external_skill_dir(skill_name) is not None
    )


def is_hub_installed(skill_name: str) -> bool:
    """Whether *skill_name* was installed via the Skills Hub."""
    return skill_name in _read_hub_installed_names()


def is_bundled(skill_name: str) -> bool:
    """Whether *skill_name* was seeded from the bundled repo skills."""
    return skill_name in _read_bundled_manifest_names()


def _external_read_only_message(skill_name: str) -> str:
    return (
        f"skill '{skill_name}' lives in skills.external_dirs; "
        "external skills are read-only to the curator"
    )


def is_curation_eligible(skill_name: str, skill_path: Optional[Path] = None) -> bool:
    """Whether the curator may track/archive *skill_name*.

    Agent-created skills are always eligible. Bundled built-ins become eligible
    only when ``curator.prune_builtins`` is enabled. Hub-installed and external
    skill-dir skills are NEVER eligible — they have an external upstream owner.
    Org-shared skills ARE eligible for improvement (the curator may patch them
    like any other skill; edits stay local until proposed) but are protected
    from ARCHIVE/DELETE elsewhere — removing a shared skill is an org-admin
    action, not a local curation decision.
    Protected built-ins (``PROTECTED_BUILTIN_SKILLS``) are NEVER eligible
    regardless of any flag — they back load-bearing UX and must never be
    archived or consolidated.
    """
    if skill_path is not None and is_external_skill_path(skill_path):
        return False
    if is_protected_builtin(skill_name):
        return False
    if is_hub_installed(skill_name):
        return False
    if is_bundled(skill_name):
        return _prune_builtins_enabled()
    local_dir = _find_skill_dir(skill_name)
    if local_dir is not None:
        return not is_external_skill_path(local_dir)
    if _find_external_skill_dir(skill_name) is not None:
        return False
    return True


def _is_curator_managed_record(record: Any) -> bool:
    """Return True when a usage record opts a skill into curator management.

    NAMING (issue #67140): the on-disk field is ``created_by``, which reads
    like provenance but is consumed as a **curator-management opt-in policy
    flag**. The two are not the same question:

    * provenance = "who authored this file" — historical fact, and for records
      written before the marker existed it is simply unrecoverable.
    * management = "may autonomous curation mutate/archive this" — a policy
      decision the user can change at any time via ``hermes curator adopt``.

    ``created_by: "agent"`` therefore means "curator-managed", NOT "proof the
    agent wrote it". The field name is retained because it is already on disk
    in every user's ``.usage.json``; renaming it would strand those records.
    Read it as policy, and prefer ``is_curator_managed()`` at call sites so the
    intent is unambiguous.
    """
    if not isinstance(record, dict):
        return False
    return record.get("created_by") == "agent" or record.get("agent_created") is True


def is_curator_managed(skill_name: str) -> bool:
    """Whether *skill_name* is opted into curator management.

    Policy-intent alias for the ``created_by``-marker check, so call sites read
    as the question they are actually asking (see ``_is_curator_managed_record``
    for why the stored field name says "created_by").
    """
    return _is_curator_managed_record(load_usage().get(skill_name))


def list_unmanaged_skill_names() -> List[str]:
    """Enumerate curation-ELIGIBLE skills that carry no provenance marker.

    These are skills the curator *could* manage (they are not hub-installed,
    not external, not protected built-ins) but never will, because nothing
    ever wrote ``created_by: agent`` onto their usage record. Two ways a skill
    lands here:

    * It predates the provenance mechanism entirely — records written before
      ``created_by`` existed carry no key at all, so their authorship is
      unknowable from the record alone.
    * It was created by a FOREGROUND ``skill_manage(action="create")`` call,
      which deliberately does not mark provenance (skills a user asks for
      belong to the user).

    Either way the skill is invisible to ``curated_report()`` and therefore to
    every automatic transition. ``hermes curator status`` surfaces this count
    so the blind spot is legible instead of silent, and ``hermes curator
    adopt`` lets the user hand specific skills over explicitly.

    Provenance is a DECLARATION, never an inference: this function only
    reports, and callers must not auto-adopt what it returns. Heavy patch or
    use counts are evidence of maintenance, not of authorship — the agent
    edits user-authored skills on the user's behalf routinely.
    """
    base = _skills_dir()
    if not base.exists():
        return []
    hub = _read_hub_installed_names()
    bundled = _read_bundled_manifest_names()
    usage = load_usage()

    names: List[str] = []
    for skill_md in base.rglob("SKILL.md"):
        if is_excluded_skill_path(skill_md) or is_external_skill_path(skill_md):
            continue
        try:
            skill_md.relative_to(base)
        except ValueError:
            continue
        name = _read_skill_name(skill_md, fallback=skill_md.parent.name)
        # Anything with an external owner or a bundled/protected identity is
        # outside the adoption question entirely.
        if name in hub or name in bundled or is_protected_builtin(name):
            continue
        if _is_curator_managed_record(usage.get(name)):
            continue
        if not is_curation_eligible(name, skill_md):
            continue
        names.append(name)
    return sorted(set(names))


def unmanaged_report() -> List[Dict[str, Any]]:
    """Rows for every skill :func:`list_unmanaged_skill_names` returns.

    Each row carries the usual activity fields plus ``has_provenance_key``:
    False when the record has no ``created_by`` key at all (pre-dates the
    mechanism), True when the key is present but unset (a foreground create
    under the current policy). The distinction matters for explaining WHY a
    skill is unmanaged; it is not a signal to adopt on.
    """
    usage = load_usage()
    rows: List[Dict[str, Any]] = []
    for name in list_unmanaged_skill_names():
        raw = usage.get(name)
        rec: Dict[str, Any] = dict(raw) if isinstance(raw, dict) else _empty_record()
        for k, v in _empty_record().items():
            rec.setdefault(k, v)
        row = {"name": name, **rec}
        row["has_provenance_key"] = isinstance(raw, dict) and "created_by" in raw
        row["has_record"] = isinstance(raw, dict)
        row["last_activity_at"] = latest_activity_at(row)
        row["activity_count"] = activity_count(row)
        rows.append(row)
    return rows


def adopt_skill(skill_name: str) -> Tuple[bool, str]:
    """Hand *skill_name* to the curator by user declaration.

    Writes the same ``created_by: agent`` marker the background review fork
    writes, so the skill joins ``curated_report()`` and the automatic
    transition walk. The inactivity clock is NOT reset: the skill's existing
    ``last_activity_at`` still governs staleness, so adopting something idle
    for months does not buy it a fresh window (nor does it archive it on the
    spot — the state machine decides on the next pass).

    Returns (ok, message). Refuses hub-installed, external, and protected
    built-in skills, which have an owner other than the user.
    """
    if not skill_name:
        return False, "no skill name given"
    if is_protected_builtin(skill_name):
        return False, f"'{skill_name}' is a protected built-in; the curator never manages it"
    if is_hub_installed(skill_name):
        return False, f"'{skill_name}' is hub-installed; its upstream owns it"
    if is_bundled(skill_name):
        # Bundled skills already fall under the curator via
        # ``curator.prune_builtins``; stamping created_by=agent on one would
        # claim Hermes' own shipped skill was agent-authored and change nothing
        # about its eligibility.
        return False, (
            f"'{skill_name}' is a bundled built-in — it is governed by "
            "curator.prune_builtins, not by adoption"
        )
    skill_dir = _find_skill_dir(skill_name)
    if skill_dir is None:
        if _find_external_skill_dir(skill_name) is not None:
            return False, f"'{skill_name}' lives in skills.external_dirs and is read-only to the curator"
        return False, f"skill '{skill_name}' not found"
    if is_external_skill_path(skill_dir):
        return False, _external_read_only_message(skill_name)
    usage = load_usage()
    if _is_curator_managed_record(usage.get(skill_name)):
        return True, f"'{skill_name}' is already curator-managed"
    mark_agent_created(skill_name)
    if not _is_curator_managed_record(load_usage().get(skill_name)):
        return False, f"could not mark '{skill_name}' as curator-managed"
    return True, f"adopted '{skill_name}' into curator management"


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
        "patch_generation": 0,
        "last_reused_patch_generation": 0,
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


def save_usage(data: Dict[str, Dict[str, Any]]) -> bool:
    """Write the usage map atomically and report whether it committed."""
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
            return True
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("Failed to write %s: %s", path, e, exc_info=True)
        return False


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


def seed_record_if_missing(skill_name: str) -> None:
    """Persist a baseline usage record for a curation-eligible skill.

    Built-ins carry no usage record until something touches them, which leaves
    their inactivity clock with no anchor. Seeding a record here fixes
    ``created_at`` to the moment the curator first sees the skill, so the
    archive/stale clock measures non-use FROM THEN — not from epoch. No-op when
    a record already exists or the skill isn't curation-eligible.
    """
    if not skill_name or not is_curation_eligible(skill_name):
        return
    try:
        with _usage_file_lock():
            data = load_usage()
            if isinstance(data.get(skill_name), dict):
                return
            data[skill_name] = _empty_record()
            save_usage(data)
    except Exception as e:
        logger.debug("skill_usage.seed_record_if_missing(%s) failed: %s", skill_name, e, exc_info=True)


def _mutate(skill_name: str, mutator, *, require_curation_eligible: bool = False) -> Any:
    """Load, apply *mutator(record)* in place, save. Best-effort.

    By default this records telemetry for ANY skill — bundled, hub-installed,
    or agent-created — because usage tracking is pure observability and is
    orthogonal to whether a skill is ever curated. Lifecycle mutators
    (``set_state``, ``set_pinned``, ``mark_agent_created``) pass
    ``require_curation_eligible=True`` so they never write meaningless state
    onto a skill the curator can't manage (e.g. an ``archived`` flag on a
    hub-installed skill).
    """
    if not skill_name:
        return None
    try:
        if require_curation_eligible and not is_curation_eligible(skill_name):
            return None
        with _usage_file_lock():
            data = load_usage()
            rec = data.get(skill_name)
            if not isinstance(rec, dict):
                rec = _empty_record()
            result = mutator(rec)
            data[skill_name] = rec
            if not save_usage(data):
                return None
            return result
    except Exception as e:
        logger.debug("skill_usage._mutate(%s) failed: %s", skill_name, e, exc_info=True)
        return None


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def telemetry_provenance(
    skill_name: str,
    record: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the bounded provenance used by shared skill metrics."""
    if is_hub_installed(skill_name) or is_bundled(skill_name):
        return "installed"
    if ":" in skill_name:
        try:
            from hermes_cli.plugins import get_plugin_manager

            if get_plugin_manager().find_plugin_skill(skill_name) is not None:
                return "installed"
        except Exception:
            pass
    if isinstance(record, dict):
        created_by = record.get("created_by")
        if created_by == "installed":
            return "installed"
        if created_by == "agent":
            return "agent_created"
    if _find_external_skill_dir(skill_name) is not None:
        return "external"
    if _find_skill_dir(skill_name) is not None or isinstance(record, dict):
        return "local"
    return "unknown"


def _emit_skill_lifecycle(
    skill_name: str,
    action: str,
    *,
    record: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    use_count: Optional[int] = None,
    reused: Optional[bool] = None,
    reuse_after_patch: Optional[bool] = None,
) -> None:
    """Emit one best-effort lifecycle fact after authoritative state changes."""
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook

        if not has_hook("on_skill_lifecycle"):
            return
        invoke_hook(
            "on_skill_lifecycle",
            action=action,
            skill_name=skill_name,
            provenance=telemetry_provenance(skill_name, record),
            task_id=task_id or "",
            session_id=session_id or "",
            use_count=use_count,
            reused=reused,
            reuse_after_patch=reuse_after_patch,
        )
    except Exception:
        logger.debug(
            "skill_usage lifecycle hook failed for %s/%s",
            skill_name,
            action,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Public counter-bump helpers — telemetry for ALL skills (observability only)
# ---------------------------------------------------------------------------

def bump_view(skill_name: str) -> None:
    """Bump view_count and last_viewed_at. Called from skill_view().

    Tracks every skill regardless of provenance — built-ins and hub skills
    included. Usage telemetry is observability, not a curation signal.
    """
    def _apply(rec: Dict[str, Any]) -> None:
        rec["view_count"] = _non_negative_int(rec.get("view_count")) + 1
        rec["last_viewed_at"] = _now_iso()
    _mutate(skill_name, _apply)


def bump_use(
    skill_name: str,
    *,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Bump use_count and last_used_at. Called when a skill is actively used
    (e.g. loaded into the prompt path or referenced from an assistant turn).

    Tracks every skill regardless of provenance.
    """
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        previous_use_count = _non_negative_int(rec.get("use_count"))
        patch_generation = _non_negative_int(rec.get("patch_generation"))
        last_reused_generation = min(
            _non_negative_int(rec.get("last_reused_patch_generation")),
            patch_generation,
        )
        reused = previous_use_count > 0
        reuse_after_patch = reused and patch_generation > last_reused_generation
        rec["use_count"] = previous_use_count + 1
        rec["last_used_at"] = _now_iso()
        rec["patch_generation"] = patch_generation
        rec["last_reused_patch_generation"] = last_reused_generation
        if reuse_after_patch:
            rec["last_reused_patch_generation"] = patch_generation
        return {
            "created_by": rec.get("created_by"),
            "use_count": rec["use_count"],
            "reused": reused,
            "reuse_after_patch": reuse_after_patch,
        }

    facts = _mutate(skill_name, _apply)
    if isinstance(facts, dict):
        _emit_skill_lifecycle(
            skill_name,
            "loaded",
            record=facts,
            task_id=task_id,
            session_id=session_id,
            use_count=facts["use_count"],
            reused=facts["reused"],
            reuse_after_patch=facts["reuse_after_patch"],
        )


def bump_patch(
    skill_name: str,
    *,
    action: str = "patch",
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Bump patch_count and last_patched_at. Called from skill_manage (patch/edit).

    Tracks every skill regardless of provenance.
    """
    lifecycle_action = "patched" if action == "patch" else "edited"

    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        rec["patch_count"] = _non_negative_int(rec.get("patch_count")) + 1
        rec["patch_generation"] = _non_negative_int(rec.get("patch_generation")) + 1
        rec["last_patched_at"] = _now_iso()
        return {"created_by": rec.get("created_by")}

    facts = _mutate(skill_name, _apply)
    if isinstance(facts, dict):
        _emit_skill_lifecycle(
            skill_name,
            lifecycle_action,
            record=facts,
            task_id=task_id,
            session_id=session_id,
        )


def record_created(
    skill_name: str,
    *,
    agent_created: bool,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Persist explicit creation provenance and emit a successful create fact."""
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        # A successful create is a new logical skill even if stale sidecar
        # state survived an earlier deletion or manual filesystem change.
        rec.clear()
        rec.update(_empty_record())
        if agent_created:
            rec["created_by"] = "agent"
        return {"created_by": rec["created_by"]}

    facts = _mutate(skill_name, _apply)
    if isinstance(facts, dict):
        _emit_skill_lifecycle(
            skill_name,
            "created",
            record=facts,
            task_id=task_id,
            session_id=session_id,
        )


def record_installed(skill_name: str) -> None:
    """Record a successful Skills Hub install without exporting its name."""
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        rec["created_by"] = "installed"
        rec["state"] = STATE_ACTIVE
        rec["archived_at"] = None
        return {"created_by": rec["created_by"]}

    facts = _mutate(skill_name, _apply)
    if isinstance(facts, dict):
        _emit_skill_lifecycle(skill_name, "installed", record=facts)


def mark_agent_created(skill_name: str) -> None:
    """Opt a skill created by skill_manage into curator management.

    Viewing or invoking a manually authored skill may still create telemetry,
    but only this explicit marker makes it eligible for automatic curation.
    """
    def _apply(rec: Dict[str, Any]) -> None:
        rec["created_by"] = "agent"
    _mutate(skill_name, _apply, require_curation_eligible=True)


def set_state(skill_name: str, state: str) -> None:
    """Set lifecycle state. No-op if *state* is invalid or the skill isn't
    curator-manageable (hub skills, or built-ins with pruning disabled)."""
    if state not in _VALID_STATES:
        logger.debug("set_state: invalid state %r for %s", state, skill_name)
        return
    def _apply(rec: Dict[str, Any]) -> Dict[str, Any]:
        previous_state = rec.get("state")
        if previous_state == state:
            return {"changed": False, "created_by": rec.get("created_by")}
        rec["state"] = state
        if state == STATE_ARCHIVED:
            rec["archived_at"] = _now_iso()
        elif state == STATE_ACTIVE:
            rec["archived_at"] = None
        return {
            "changed": True,
            "created_by": rec.get("created_by"),
            "previous_state": previous_state,
        }

    facts = _mutate(skill_name, _apply, require_curation_eligible=True)
    if not isinstance(facts, dict) or not facts.get("changed"):
        return
    action = {
        STATE_ARCHIVED: "archived",
        STATE_STALE: "stale",
    }.get(state)
    if state == STATE_ACTIVE and facts.get("previous_state") == STATE_ARCHIVED:
        action = "restored"
    if action is not None:
        _emit_skill_lifecycle(skill_name, action, record=facts)


def set_pinned(skill_name: str, pinned: bool) -> bool:
    """Set/clear the pin flag. Returns False when the write did not land
    (skill not curation-eligible), True on success — so callers can report
    failure instead of a false success (issue #92993)."""
    def _apply(rec: Dict[str, Any]) -> Any:
        rec["pinned"] = bool(pinned)
        return True  # non-None sentinel: _mutate propagates the mutator result
    return bool(_mutate(skill_name, _apply, require_curation_eligible=True))


def set_sync(skill_name: str, sync: bool) -> None:
    """Set the sync opt-in flag on a skill's usage record.

    Sync is OPT-IN: nothing propagates to the sync plane unless the user marks
    a skill with ``sync: true`` here. Sits alongside ``pinned``/``created_by``
    on the ``.usage.json`` sidecar and is read by
    ``tools.skills_sync_client.list_synced_skill_names``. Gated on curation
    eligibility so bundled/hub/external skills (which never sync) can't be
    marked. Provisional per the M1-D default.
    """
    def _apply(rec: Dict[str, Any]) -> None:
        rec["sync"] = bool(sync)
    _mutate(skill_name, _apply, require_curation_eligible=True)


def is_sync_enabled(skill_name: str) -> bool:
    """Whether a skill is opted into sync (``sync: true`` in its record)."""
    return get_record(skill_name).get("sync") is True


def forget(skill_name: str) -> None:
    """Drop a skill's usage entry entirely. Called when the skill is deleted."""
    if not skill_name:
        return
    try:
        with _usage_file_lock():
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
    """Move a curator-eligible skill directory to ~/.hermes/skills/.archive/.

    Returns (ok, message). Never archives hub-installed skills. Bundled
    built-ins are only archivable when ``curator.prune_builtins`` is enabled;
    when one is archived, its name is added to the suppression list so the
    update-time re-seeder leaves it archived instead of restoring it.
    """
    local_skill_dir = _find_skill_dir(skill_name)
    if local_skill_dir is None and _find_external_skill_dir(skill_name) is not None:
        return False, _external_read_only_message(skill_name)

    if not is_curation_eligible(skill_name, local_skill_dir):
        if is_protected_builtin(skill_name):
            return False, (
                f"skill '{skill_name}' is a protected built-in; it backs "
                "load-bearing UX and is never archived or consolidated"
            )
        if is_hub_installed(skill_name):
            return False, f"skill '{skill_name}' is hub-installed; never archive"
        return False, (
            f"skill '{skill_name}' is a bundled built-in; enable "
            "curator.prune_builtins to allow pruning it"
        )

    skill_dir = local_skill_dir
    if skill_dir is None:
        return False, f"skill '{skill_name}' not found"
    if is_external_skill_path(skill_dir):
        return False, _external_read_only_message(skill_name)

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

    # Audit ledger pre-capture (best-effort; never blocks the archive).
    _ledger_before = None
    try:
        from tools import skill_ledger as _ledger
        _ledger_before = _ledger.capture_before(skill_dir)
    except Exception:
        _ledger = None  # type: ignore[assignment]

    try:
        skill_dir.rename(dest)
    except OSError:
        # Cross-device — fall back to shutil.move
        import shutil
        try:
            shutil.move(str(skill_dir), str(dest))
        except Exception as e2:
            return False, f"failed to archive: {e2}"

    # Pruning a built-in only sticks if the re-seeder is told to leave it alone.
    if is_bundled(skill_name):
        add_suppressed_name(skill_name)

    set_state(skill_name, STATE_ARCHIVED)
    try:
        if _ledger is not None:
            _ledger.record_mutation(
                "archive",
                skill_name,
                before=_ledger_before if _ledger_before is not None else [],
                after_root=dest,
            )
    except Exception:
        pass
    return True, f"archived to {dest}"


def restore_skill(skill_name: str) -> Tuple[bool, str]:
    """Move an archived skill back to ~/.hermes/skills/. Restores to the flat
    top-level layout; original category nesting is NOT reconstructed.

    Refuses to restore under a name that now collides with a hub-installed
    skill — that would shadow the upstream version. Also refuses to restore
    over a bundled built-in UNLESS ``curator.prune_builtins`` is enabled (in
    which case built-ins are curator-managed and restoring is the documented
    way to lift a prune). Restoring clears any suppression entry so future
    updates may re-seed the built-in again.
    """
    # Hub skills always have an external upstream owner — never shadow them.
    if is_hub_installed(skill_name):
        return False, (
            f"skill '{skill_name}' is now hub-installed; "
            "restore would shadow the upstream version"
        )
    # A bundled built-in is upstream-owned UNLESS prune_builtins is on. With the
    # flag off, restoring over it would shadow the bundled version.
    if is_bundled(skill_name) and not _prune_builtins_enabled():
        return False, (
            f"skill '{skill_name}' is now bundled; "
            "restore would shadow the upstream version"
        )
    archive_root = _archive_dir()
    if not archive_root.exists():
        return False, "no archive directory"

    # Try exact name match first, then the timestamped-duplicate fallback.
    # Recursive walk handles nested archive layouts (e.g. .archive/<category>/<skill>/)
    # left behind by older archive paths or external imports.
    candidates = [p for p in archive_root.rglob("*") if p.is_dir() and p.name == skill_name]
    if not candidates:
        # A name collision makes archive_skill() disambiguate by appending its
        # UTC timestamp ("<skill>-YYYYMMDDHHMMSS", a 14-digit suffix), so only
        # that exact shape is another copy of THIS skill. A bare
        # startswith(f"{skill_name}-") also swallows unrelated sibling skills —
        # restoring "git" would otherwise pull an archived "git-helpers" out of
        # the archive and rename it to "git", destroying the sibling's only
        # copy. Require the suffix to be the timestamp archive_skill writes.
        prefix = f"{skill_name}-"
        candidates = sorted(
            [
                p for p in archive_root.rglob("*")
                if p.is_dir()
                and p.name.startswith(prefix)
                and len(p.name) - len(prefix) == 14
                and p.name[len(prefix):].isdigit()
            ],
            reverse=True,
        )
    if not candidates:
        return False, f"skill '{skill_name}' not found in archive"

    src = candidates[0]
    dest = _skills_dir() / skill_name
    if dest.exists():
        return False, f"destination already exists: {dest}"

    # Audit ledger pre-capture (best-effort; never blocks the restore).
    _ledger_before = None
    try:
        from tools import skill_ledger as _ledger
        _ledger_before = _ledger.capture_before(src)
    except Exception:
        _ledger = None  # type: ignore[assignment]

    try:
        src.rename(dest)
    except OSError:
        import shutil
        try:
            shutil.move(str(src), str(dest))
        except Exception as e:
            return False, f"failed to restore: {e}"

    # Restoring a pruned built-in lifts its suppression so updates can manage it.
    remove_suppressed_name(skill_name)

    set_state(skill_name, STATE_ACTIVE)
    try:
        if _ledger is not None:
            _ledger.record_mutation(
                "restore",
                skill_name,
                before=_ledger_before if _ledger_before is not None else [],
                after_root=dest,
            )
    except Exception:
        pass
    return True, f"restored to {dest}"


def _find_skill_dir(skill_name: str) -> Optional[Path]:
    """Locate the directory for a skill by its frontmatter `name:` field.

    Handles both flat (~/.hermes/skills/<skill>/SKILL.md) and category-nested
    (~/.hermes/skills/<category>/<skill>/SKILL.md) layouts. Uses the gated
    index iterator so M2 org mirrors resolve ONLY for the active org
    (stale ``_org/<other>/`` trees never match).
    """
    base = _skills_dir()
    if not base.exists():
        return None
    from agent.skill_utils import iter_skill_index_files

    for skill_md in iter_skill_index_files(base, "SKILL.md"):
        if is_external_skill_path(skill_md):
            continue
        if _read_skill_name(skill_md, fallback=skill_md.parent.name) == skill_name:
            return skill_md.parent
    return None


def _find_external_skill_dir(skill_name: str) -> Optional[Path]:
    """Locate a skill under configured external dirs by frontmatter name."""
    from agent.skill_utils import get_all_skills_dirs

    for base in get_all_skills_dirs()[1:]:
        if not base.exists():
            continue
        for skill_md in base.rglob("SKILL.md"):
            if is_excluded_skill_path(skill_md):
                continue
            if _read_skill_name(skill_md, fallback=skill_md.parent.name) == skill_name:
                return skill_md.parent
    return None


# ---------------------------------------------------------------------------
# Reporting — for the curator CLI / slash command
# ---------------------------------------------------------------------------

def curated_report() -> List[Dict[str, Any]]:
    """Return a list of {name, provenance, state, pinned, last_activity_at, ...}
    records for every curator-managed skill. Missing usage records are
    backfilled with defaults so callers can always index fields.

    ``provenance`` is 'agent', 'bundled', or 'hub' (see :func:`provenance`).
    Bundled skills are only included when ``curator.prune_builtins`` is enabled.
    Hub-installed skills are never included.

    Each row carries ``_persisted``: True when a real record exists in
    ``.usage.json``, False when the row is a fresh backfill (e.g. a built-in
    seen for the first time). The curator uses this to seed the inactivity
    clock instead of treating an unrecorded skill as ancient.
    """
    data = load_usage()
    rows: List[Dict[str, Any]] = []
    names = set(list_agent_created_skill_names())
    # Issue #92993: a successfully pinned skill must be visible in the report
    # even when it lacks the created_by marker (eligible-but-unmanaged), or
    # its pin silently vanishes from `curator status`. The local-dir guard
    # keeps stale records for deleted skill dirs from rendering as ghost rows;
    # `curator unpin` is the cleanup path for those.
    for name, rec in data.items():
        if (
            isinstance(rec, dict)
            and rec.get("pinned")
            and is_curation_eligible(name)
            and _find_skill_dir(name) is not None
        ):
            names.add(name)
    for name in sorted(names):
        raw = data.get(name)
        persisted = isinstance(raw, dict)
        rec: Dict[str, Any] = raw if isinstance(raw, dict) else _empty_record()
        base = _empty_record()
        for k, v in base.items():
            rec.setdefault(k, v)
        row = {"name": name, **rec, "_persisted": persisted}
        row["last_activity_at"] = latest_activity_at(row)
        row["activity_count"] = activity_count(row)
        row["provenance"] = provenance(name)
        rows.append(row)
    return rows


def agent_created_report() -> List[Dict[str, Any]]:
    """DEPRECATED — use :func:`curated_report` instead.

    Used to return everything :func:`curated_report` returns (including bundled
    skills when ``curator.prune_builtins`` is enabled), which made the
    "agent-created" name misleading. Kept as a compatibility alias for
    external callers; new code should call ``curated_report()``.
    """
    return curated_report()

def remove_suppressed_name(skill_name: str) -> None:
    """Clear a built-in's suppression entry (e.g. on restore)."""
    if not skill_name:
        return
    names = read_suppressed_names()
    if skill_name in names:
        names.discard(skill_name)
        _write_suppressed_names(names)
# ---- END PLUGIN-COMPAT ----
