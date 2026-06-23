"""Version history and rollback for autonomous skill edits."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

_HISTORY_DIR = ".skill_history"
_FILES_DIR = "files"
_VALID_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def snapshot_autonomous_edit(
    *,
    name: str,
    action: str,
    skill_dir: Path | None,
    details: dict[str, Any] | None = None,
) -> str | None:
    """Snapshot a skill before an autonomous edit.

    Returns the snapshot id, or None when the snapshot could not be written.
    Snapshot failures are intentionally non-fatal; the caller records the write
    result only when a snapshot exists.
    """
    skill_name = _normalize_skill_name(name)
    if not skill_name:
        return None

    history_dir = _skill_history_dir(skill_name)
    snapshot_id = _unique_snapshot_id(history_dir)
    snapshot_dir = history_dir / snapshot_id
    files_dir = snapshot_dir / _FILES_DIR

    try:
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        existed = bool(skill_dir and skill_dir.exists() and skill_dir.is_dir())
        if existed and skill_dir is not None:
            shutil.copytree(
                skill_dir,
                files_dir,
                symlinks=True,
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".hg",
                    ".svn",
                    "__pycache__",
                    ".pytest_cache",
                    ".mypy_cache",
                    ".ruff_cache",
                    ".venv",
                    "venv",
                    ".skill_history",
                ),
            )

        manifest = {
            "id": snapshot_id,
            "skill": skill_name,
            "action": action,
            "created_at": _now_iso(),
            "existed": existed,
            "original_path": str(skill_dir) if skill_dir else None,
            "details": details or {},
        }
        _write_json(snapshot_dir / "manifest.json", manifest)
        return snapshot_id
    except Exception:
        try:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
        except Exception:
            pass
        return None


def record_autonomous_edit(
    *,
    name: str,
    action: str,
    snapshot_id: str | None,
    result: dict[str, Any],
) -> None:
    """Append an audit record for an autonomous skill write."""
    skill_name = _normalize_skill_name(name)
    if not skill_name or not snapshot_id:
        return

    record = {
        "timestamp": _now_iso(),
        "skill": skill_name,
        "action": action,
        "snapshot_id": snapshot_id,
        "success": bool(result.get("success")),
        "message": result.get("message"),
        "error": result.get("error"),
        "path": result.get("path") or result.get("skill_md"),
    }
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    for audit_path in (_audit_log_path(), _skill_history_dir(skill_name) / "audit.jsonl"):
        try:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            continue


def list_skill_history(name: str) -> list[dict[str, Any]]:
    """Return snapshots for a skill, newest first."""
    skill_name = _normalize_skill_name(name)
    if not skill_name:
        return []

    history_dir = _skill_history_dir(skill_name)
    if not history_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    for child in sorted(history_dir.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        manifest = _read_json(child / "manifest.json")
        if not manifest:
            continue
        manifest.setdefault("id", child.name)
        manifest.setdefault("path", str(child))
        rows.append(manifest)
    return rows


def rollback_skill(name: str, snapshot_id: str | None = None) -> tuple[bool, str, Path | None]:
    """Restore a skill to a prior autonomous-edit snapshot."""
    skill_name = _normalize_skill_name(name)
    if not skill_name:
        return False, "skill name is required", None

    snapshot_dir = _resolve_snapshot(skill_name, snapshot_id)
    if snapshot_dir is None:
        return False, _missing_snapshot_message(skill_name, snapshot_id), None

    manifest = _read_json(snapshot_dir / "manifest.json")
    existed = bool(manifest.get("existed"))
    original_path = manifest.get("original_path")
    target = _resolve_target_path(skill_name, original_path)
    if target is None:
        return False, f"could not resolve rollback target for skill '{skill_name}'", None

    safety_id = snapshot_autonomous_edit(
        name=skill_name,
        action="pre-rollback",
        skill_dir=target if target.exists() else None,
        details={"rollback_to": snapshot_dir.name},
    )

    try:
        if existed:
            files_dir = snapshot_dir / _FILES_DIR
            if not files_dir.exists():
                return False, f"snapshot {snapshot_dir.name} has no files payload", None
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(files_dir, target, symlinks=True)
        else:
            if target.exists():
                shutil.rmtree(target)
    except Exception as exc:
        return False, f"rollback failed: {exc}", None

    record_autonomous_edit(
        name=skill_name,
        action="rollback",
        snapshot_id=safety_id,
        result={
            "success": True,
            "message": f"rolled back to {snapshot_dir.name}",
            "path": str(target),
        },
    )
    return True, f"Skill '{skill_name}' rolled back to {snapshot_dir.name}.", target


def _history_root() -> Path:
    return get_hermes_home() / "skills" / _HISTORY_DIR


def _skill_history_dir(skill_name: str) -> Path:
    return _history_root() / skill_name


def _audit_log_path() -> Path:
    return _history_root() / "audit.jsonl"


def _unique_snapshot_id(history_dir: Path) -> str:
    base = _utc_id()
    candidate = base
    idx = 1
    while (history_dir / candidate).exists():
        candidate = f"{base}-{idx:02d}"
        idx += 1
    return candidate


def _utc_id() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z").replace(":", "-")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_snapshot(skill_name: str, snapshot_id: str | None) -> Path | None:
    history_dir = _skill_history_dir(skill_name)
    if not history_dir.exists():
        return None
    if snapshot_id:
        candidate = history_dir / snapshot_id
        if candidate.is_dir() and (candidate / "manifest.json").exists():
            return candidate
        return None
    rows = list_skill_history(skill_name)
    if not rows:
        return None
    return history_dir / rows[0]["id"]


def _resolve_target_path(skill_name: str, original_path: Any) -> Path | None:
    if isinstance(original_path, str) and original_path.strip():
        path = Path(original_path)
        if _is_under_known_skills_root(path):
            return path

    try:
        from tools.skill_manager_tool import SKILLS_DIR, _find_skill

        existing = _find_skill(skill_name)
        if existing and isinstance(existing.get("path"), Path):
            return existing["path"]
        return SKILLS_DIR / skill_name
    except Exception:
        return get_hermes_home() / "skills" / skill_name


def _is_under_known_skills_root(path: Path) -> bool:
    try:
        from agent.skill_utils import get_all_skills_dirs

        resolved = path.resolve(strict=False)
        for root in get_all_skills_dirs():
            try:
                resolved.relative_to(root.resolve())
                return True
            except (OSError, ValueError):
                continue
    except Exception:
        return False
    return False


def _missing_snapshot_message(skill_name: str, snapshot_id: str | None) -> str:
    if snapshot_id:
        return f"no autonomous-edit snapshot '{snapshot_id}' found for skill '{skill_name}'"
    return f"no autonomous-edit snapshots found for skill '{skill_name}'"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_skill_name(name: str) -> str:
    skill_name = str(name or "").strip()
    if not _VALID_SKILL_NAME_RE.match(skill_name):
        return ""
    return skill_name
