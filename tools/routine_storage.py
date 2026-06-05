"""Storage layer for routines (scheduled workflow triggers).

A *routine* is a named, cron-scheduled trigger that fires a workflow.
When the cron expression fires, the routine emits a
``schedule_triggered`` event which the bound workflow (matched by
``workflow_id``) consumes. Routines are the cron-half of the
workflow + routine pair operators author from chat.

Routines live as one YAML file per routine under
``<HERMES_HOME>/routines/<id>.yaml``. Companion visibility mirrors (when configured) are applied by the adapter layer.
The Hermes cron module is the actual scheduler:
``tools/workflow_routine_tools.py`` calls ``cron.jobs.create_job`` with
``workflow_ids=[workflow_id]`` so existing cron infrastructure (PID,
state file, scheduler) handles execution.

Routine record shape::

    {
        "id": str,
        "version": "1",
        "workflow_id": str,
        "name_he": str,
        "cron_schedule": str,            # e.g. "0 * * * *"
        "natural_language_schedule_he": str,
        "cron_job_id": str | None,       # set after create_job succeeds
    }
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hermes_constants import get_hermes_home


_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


def routines_dir() -> Path:
    root = get_hermes_home() / "routines"
    root.mkdir(parents=True, exist_ok=True)
    return root


def bound_routines_dir() -> Optional[Path]:
    """Deprecated stub (mirror logic in adapter)."""
    return None


def _routine_path(root: Path, routine_id: str) -> Path:
    if not isinstance(routine_id, str) or not _SAFE_ID_RE.match(routine_id):
        raise ValueError("routine id must match [A-Za-z0-9_-]{1,128}")
    return root / f"{routine_id}.yaml"


def _validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("routine must be a JSON/YAML object")

    routine_id = record.get("id")
    if not isinstance(routine_id, str) or not routine_id.strip():
        raise ValueError("routine field 'id' is required (non-empty string)")
    routine_id = routine_id.strip()
    if not _SAFE_ID_RE.match(routine_id):
        raise ValueError("routine id must match [A-Za-z0-9_-]{1,128}")

    workflow_id = record.get("workflow_id")
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        raise ValueError("routine field 'workflow_id' is required (non-empty string)")

    name_he = record.get("name_he")
    if not isinstance(name_he, str) or not name_he.strip():
        raise ValueError("routine field 'name_he' is required (non-empty string)")

    cron_schedule = record.get("cron_schedule")
    if not isinstance(cron_schedule, str) or not cron_schedule.strip():
        raise ValueError("routine field 'cron_schedule' is required (non-empty string)")

    natural = record.get("natural_language_schedule_he", "")
    if natural is None:
        natural = ""
    if not isinstance(natural, str):
        raise ValueError("'natural_language_schedule_he' must be a string")

    cron_job_id = record.get("cron_job_id")
    if cron_job_id is not None and not isinstance(cron_job_id, str):
        raise ValueError("'cron_job_id' must be a string or null")

    return {
        "id": routine_id,
        "version": "1",
        "workflow_id": workflow_id.strip(),
        "name_he": name_he,
        "cron_schedule": cron_schedule.strip(),
        "natural_language_schedule_he": natural,
        "cron_job_id": cron_job_id,
    }


def _atomic_write_yaml(target: Path, data: Dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".yaml.tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(
                data,
                fh,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_routine(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _validate_record(record)
    canonical = _routine_path(routines_dir(), normalized["id"])
    _atomic_write_yaml(canonical, normalized)
    # Companion mirror (if any) applied via adapter patch on this function.
    return normalized


def load_routine(routine_id: str) -> Optional[Dict[str, Any]]:
    path = _routine_path(routines_dir(), routine_id)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else None


def list_routines() -> List[Dict[str, Any]]:
    root = routines_dir()
    out: List[Dict[str, Any]] = []
    for entry in sorted(root.glob("*.yaml")):
        if entry.name.startswith("."):
            continue
        try:
            with entry.open("r", encoding="utf-8") as fh:
                record = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(record, dict):
            out.append(record)
    return out
