"""Storage layer for workflows (the office automation primitive).

A *workflow* is an ordered, declarative description of work that fires
when a trigger event is observed. Hermes is the canonical writer; UI
surfaces read the on-disk YAML files as the source of truth.

Workflows live as one YAML file per workflow under
``<HERMES_HOME>/workflows/<id>.yaml``. Companion integrations may mirror
copies for UI surfaces (best-effort, see adapter layer).

Workflow record shape::

    {
        "id": str,                       # stable id [A-Za-z0-9_-]{1,128}
        "version": "1",
        "name_he": str,
        "description_he": str,
        "trigger": {
            "kind": "schedule_triggered" | "wa_incoming_message" | "manual",
        },
        "triage": {                      # optional, when triage_keywords_he set
            "keywords_he": [str, ...],
        },
        "actions": [                     # ordered downstream actions
            {"kind": str, ...action-specific fields...},
        ],
    }

This module replaces the per-rule JSON store in
``tools/workflow_rules_storage.py`` (now deprecated — see
``hermes-agent-202606-028``).
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from hermes_constants import get_hermes_home


_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")

_ALLOWED_TRIGGER_KINDS = frozenset(
    {"schedule_triggered", "wa_incoming_message", "manual"}
)


def workflows_dir() -> Path:
    """Resolve and (lazily) create the workflows directory under HERMES_HOME."""
    root = get_hermes_home() / "workflows"
    root.mkdir(parents=True, exist_ok=True)
    return root


def bound_workflows_dir() -> Optional[Path]:
    """Deprecated stub (mirror logic lives in adapter for isolation).
    Returns None; companion writes are applied via post-save wrapper.
    """
    return None


def _workflow_path(root: Path, workflow_id: str) -> Path:
    if not isinstance(workflow_id, str) or not _SAFE_ID_RE.match(workflow_id):
        raise ValueError("workflow id must match [A-Za-z0-9_-]{1,128}")
    return root / f"{workflow_id}.yaml"


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError("expected a list of strings")
    out: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("expected a list of strings")
        item = item.strip()
        if item:
            out.append(item)
    return out


def _coerce_actions(actions: Any) -> List[Dict[str, Any]]:
    if actions is None:
        return []
    if not isinstance(actions, list):
        raise ValueError("actions must be a list of objects")
    normalized: List[Dict[str, Any]] = []
    for entry in actions:
        if not isinstance(entry, dict):
            raise ValueError("each action must be an object")
        kind = entry.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("each action must include a non-empty 'kind' string")
        clone = dict(entry)
        clone["kind"] = kind.strip()
        normalized.append(clone)
    return normalized


def _validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("workflow must be a JSON/YAML object")

    workflow_id = record.get("id")
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        raise ValueError("workflow field 'id' is required (non-empty string)")
    workflow_id = workflow_id.strip()
    if not _SAFE_ID_RE.match(workflow_id):
        raise ValueError("workflow id must match [A-Za-z0-9_-]{1,128}")

    name_he = record.get("name_he")
    if not isinstance(name_he, str) or not name_he.strip():
        raise ValueError("workflow field 'name_he' is required (non-empty string)")

    description_he = record.get("description_he", "")
    if description_he is None:
        description_he = ""
    if not isinstance(description_he, str):
        raise ValueError("'description_he' must be a string")

    trigger_kind = record.get("trigger_kind")
    if not isinstance(trigger_kind, str) or trigger_kind.strip() not in _ALLOWED_TRIGGER_KINDS:
        raise ValueError(
            "trigger_kind must be one of: "
            + ", ".join(sorted(_ALLOWED_TRIGGER_KINDS))
        )
    trigger_kind = trigger_kind.strip()

    keywords = _coerce_str_list(record.get("triage_keywords_he"))
    actions = _coerce_actions(record.get("actions"))

    normalized: Dict[str, Any] = {
        "id": workflow_id,
        "version": "1",
        "name_he": name_he,
        "description_he": description_he,
        "trigger": {"kind": trigger_kind},
        "actions": actions,
    }
    if keywords:
        normalized["triage"] = {"keywords_he": keywords}
    return normalized


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


def save_workflow(record: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a workflow atomically. Returns the normalized record.

    Always writes the canonical copy under ``<HERMES_HOME>/workflows/``.
    Companion mirroring (when configured) is applied by the adapter layer
    after this returns (best-effort).
    """
    normalized = _validate_record(record)
    canonical = _workflow_path(workflows_dir(), normalized["id"])
    _atomic_write_yaml(canonical, normalized)
    return normalized


def load_workflow(workflow_id: str) -> Optional[Dict[str, Any]]:
    path = _workflow_path(workflows_dir(), workflow_id)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else None


def list_workflows() -> List[Dict[str, Any]]:
    """List all workflows under HERMES_HOME/workflows/, sorted by id."""
    root = workflows_dir()
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
