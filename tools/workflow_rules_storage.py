"""Storage layer for workflow rules.

Workflow rules are per-user automation rules that match incoming connector
messages and decide whether to spawn a ticket. Hermes is the canonical
writer for these rules — they live as one JSON record per rule under
``<HERMES_HOME>/workflow-rules/<id>.json`` and any other surface (e.g.
the companion PGLite cache) is a read-side projection that Hermes
owns.

Rule record shape::

    {
        "id": str,                          # stable UUID-like id
        "connector_id": str,                # connector this rule binds to
        "rule_natural_language": str,       # the user's own words (Hebrew OK)
        "matcher_pattern": str,             # opaque matcher description
        "target_ticket_template": str,      # template id / inline body
        "enabled": bool,                    # default True
        "created_by_session_id": str | None # provenance
    }
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home


# Restrict ids to a safe filesystem charset so the <id>.json path can't
# escape the workflow-rules directory.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


def workflow_rules_dir() -> Path:
    """Resolve and (lazily) create the workflow-rules directory."""
    root = get_hermes_home() / "workflow-rules"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _rule_path(rule_id: str) -> Path:
    if not isinstance(rule_id, str) or not _SAFE_ID_RE.match(rule_id):
        raise ValueError(
            "rule id must match [A-Za-z0-9_-]{1,128}"
        )
    return workflow_rules_dir() / f"{rule_id}.json"


def _validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("workflow rule must be a JSON object")

    required = ("id", "connector_id", "rule_natural_language")
    for field in required:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"workflow rule field '{field}' is required and must be a non-empty string")

    # Normalize optional fields.
    normalized: Dict[str, Any] = {
        "id": record["id"].strip(),
        "connector_id": record["connector_id"].strip(),
        "rule_natural_language": record["rule_natural_language"],
        "matcher_pattern": str(record.get("matcher_pattern") or ""),
        "target_ticket_template": str(record.get("target_ticket_template") or ""),
        "enabled": bool(record.get("enabled", True)),
        "created_by_session_id": record.get("created_by_session_id"),
    }
    if normalized["created_by_session_id"] is not None and not isinstance(
        normalized["created_by_session_id"], str
    ):
        raise ValueError("created_by_session_id must be a string or null")

    if not _SAFE_ID_RE.match(normalized["id"]):
        raise ValueError("rule id must match [A-Za-z0-9_-]{1,128}")
    return normalized


def save_rule(record: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a rule atomically. Returns the normalized record."""
    normalized = _validate_record(record)
    target = _rule_path(normalized["id"])
    target.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same directory, then rename for atomicity.
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{normalized['id']}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(normalized, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except Exception:
        # Best-effort cleanup if the rename never happened.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return normalized


def load_rule(rule_id: str) -> Optional[Dict[str, Any]]:
    path = _rule_path(rule_id)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def list_rules(connector_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """List rules, optionally filtered by connector_id. Sorted by id for
    deterministic output."""
    root = workflow_rules_dir()
    out: List[Dict[str, Any]] = []
    for entry in sorted(root.glob("*.json")):
        # Skip our own dotfile temp files defensively.
        if entry.name.startswith("."):
            continue
        try:
            with entry.open("r", encoding="utf-8") as fh:
                record = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        if connector_id and record.get("connector_id") != connector_id:
            continue
        out.append(record)
    return out


def delete_rule(rule_id: str) -> bool:
    path = _rule_path(rule_id)
    if not path.is_file():
        return False
    path.unlink()
    return True
