"""Persistent state for deliverable approval boxes.

Unlike shell-command approvals, deliverable approvals do not unblock a waiting
agent thread. They are an audit trail and a mobile-friendly review surface: an
agent posts a draft/artifact with Approve / Needs Work / Reject buttons, and the
platform callback records the decision under ``HERMES_HOME/approval_boxes`` while
updating the original message.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home

VALID_STATUSES = {"pending", "approved", "needs_work", "rejected"}


def _state_dir() -> Path:
    path = Path(get_hermes_home()) / "approval_boxes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path(approval_id: str) -> Path:
    safe = "".join(ch for ch in approval_id if ch.isalnum() or ch in "-_")
    return _state_dir() / f"{safe}.json"


def new_id() -> str:
    return f"appr_{uuid.uuid4().hex[:16]}"


def create_record(
    *,
    platform: str,
    target: str,
    title: str,
    body: str,
    drive_link: str = "",
    artifact_path: str = "",
    message_id: str = "",
    status: str = "pending",
) -> Dict[str, Any]:
    approval_id = new_id()
    now = time.time()
    record: Dict[str, Any] = {
        "approval_id": approval_id,
        "platform": platform,
        "target": target,
        "title": title,
        "body": body,
        "drive_link": drive_link,
        "artifact_path": artifact_path,
        "message_id": message_id,
        "status": status,
        "created_at": now,
        "updated_at": now,
        "decision": None,
        "decided_by": None,
    }
    save_record(record)
    return record


def load_record(approval_id: str) -> Optional[Dict[str, Any]]:
    path = _state_path(approval_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_record(record: Dict[str, Any]) -> None:
    approval_id = str(record.get("approval_id") or "")
    if not approval_id:
        raise ValueError("approval record missing approval_id")
    tmp = _state_path(approval_id).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, _state_path(approval_id))


def resolve_record(approval_id: str, status: str, decided_by: str = "") -> Optional[Dict[str, Any]]:
    if status not in VALID_STATUSES - {"pending"}:
        raise ValueError(f"invalid approval status: {status}")
    record = load_record(approval_id)
    if not record:
        return None
    # First click wins. Keep the original decision if already resolved.
    if record.get("status") != "pending":
        return record
    record["status"] = status
    record["decision"] = status
    record["decided_by"] = decided_by
    record["updated_at"] = time.time()
    save_record(record)
    return record


def decision_label(status: str, decided_by: str = "") -> str:
    who = f" by {decided_by}" if decided_by else ""
    labels = {
        "approved": f"✅ Approved{who}",
        "needs_work": f"🛠️ Needs work{who}",
        "rejected": f"❌ Rejected{who}",
        "pending": "⏳ Pending approval",
    }
    return labels.get(status, f"Resolved{who}")
