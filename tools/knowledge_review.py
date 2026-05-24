"""Knowledge review queue tool — manages the promote/reject/defer workflow.

Tool name: review_knowledge
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry


def _get_queue_path() -> Path:
    """Get the review queue file path."""
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")
    if vault_path:
        vault = Path(vault_path)
    else:
        vault = Path.home() / "ObsidianVault" / "HermesAgent"
    return vault / "domains" / ".review_queue.json"


def _load_queue() -> List[Dict[str, Any]]:
    """Load the review queue from disk."""
    queue_path = _get_queue_path()
    if not queue_path.exists():
        return []
    try:
        text = queue_path.read_text(encoding="utf-8")
        return json.loads(text)
    except (json.JSONDecodeError, Exception):
        return []


def _save_queue(queue: List[Dict[str, Any]]) -> None:
    """Save the review queue to disk."""
    queue_path = _get_queue_path()
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")


def review_knowledge(
    action: str,
    knowledge_id: Optional[str] = None,
    title: Optional[str] = None,
    content: Optional[str] = None,
    source_project: Optional[str] = None,
    target_domain: Optional[str] = None,
    summary: Optional[str] = None,
    reason: Optional[str] = None,
    task_id: str = None,
) -> str:
    """Manage the knowledge review queue.

    Args:
        action: One of 'list', 'add', 'approve', 'reject', 'defer', 'delete'
        knowledge_id: ID of the knowledge entry (for approve/reject/defer/delete)
        title: Title of knowledge (for add)
        content: Content of knowledge (for add)
        source_project: Source project slug (for add)
        target_domain: Target domain (for add)
        summary: Optional summary (for add)
        reason: Reason for rejection/deferral (for reject/defer)
        task_id: Optional task identifier

    Returns:
        JSON string with result
    """
    if action == "list":
        queue = _load_queue()
        return json.dumps({
            "success": True,
            "action": "list",
            "count": len(queue),
            "items": queue,
        })

    elif action == "add":
        if not all([title, content, source_project, target_domain]):
            return json.dumps({
                "success": False,
                "error": "add requires title, content, source_project, target_domain",
            })
        queue = _load_queue()
        entry_id = str(uuid.uuid4())[:8]
        entry = {
            "id": entry_id,
            "title": title,
            "content": content,
            "source_project": source_project,
            "target_domain": target_domain,
            "summary": summary or "",
            "status": "pending",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        queue.append(entry)
        _save_queue(queue)
        return json.dumps({
            "success": True,
            "action": "add",
            "knowledge_id": entry_id,
            "status": "pending",
        })

    elif action in ("approve", "reject", "defer"):
        if not knowledge_id:
            return json.dumps({
                "success": False,
                "error": f"{action} requires knowledge_id",
            })
        queue = _load_queue()
        for entry in queue:
            if entry["id"] == knowledge_id:
                entry["status"] = "approved" if action == "approve" else action
                entry["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if reason:
                    entry["reason"] = reason
                _save_queue(queue)

                # If approved, actually promote the knowledge
                if action == "approve":
                    from tools.knowledge_promote import promote_knowledge
                    promote_result = promote_knowledge(
                        title=entry["title"],
                        content=entry["content"],
                        source_project=entry["source_project"],
                        target_domain=entry["target_domain"],
                        summary=entry.get("summary"),
                    )
                    return json.dumps({
                        "success": True,
                        "action": action,
                        "knowledge_id": knowledge_id,
                        "promote_result": json.loads(promote_result),
                    })

                return json.dumps({
                    "success": True,
                    "action": action,
                    "knowledge_id": knowledge_id,
                })
        return json.dumps({
            "success": False,
            "error": f"Knowledge ID {knowledge_id} not found in queue",
        })

    elif action == "delete":
        if not knowledge_id:
            return json.dumps({
                "success": False,
                "error": "delete requires knowledge_id",
            })
        queue = _load_queue()
        new_queue = [e for e in queue if e["id"] != knowledge_id]
        if len(new_queue) == len(queue):
            return json.dumps({
                "success": False,
                "error": f"Knowledge ID {knowledge_id} not found",
            })
        _save_queue(new_queue)
        return json.dumps({
            "success": True,
            "action": "delete",
            "knowledge_id": knowledge_id,
        })

    else:
        return json.dumps({
            "success": False,
            "error": f"Invalid action: {action}. Valid: list, add, approve, reject, defer, delete",
        })


def check_requirements() -> bool:
    """Check if vault path exists."""
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")
    if vault_path:
        return Path(vault_path).exists()
    return (Path.home() / "ObsidianVault" / "HermesAgent").exists()


# Register the tool
registry.register(
    name="review_knowledge",
    toolset="knowledge",
    schema={
        "name": "review_knowledge",
        "description": (
            "Manage the knowledge review queue. Actions: list, add, approve, reject, defer, delete. "
            "Approved entries are automatically promoted to the domain KB."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "add", "approve", "reject", "defer", "delete"],
                    "description": "Action to perform",
                },
                "knowledge_id": {"type": "string", "description": "Knowledge entry ID"},
                "title": {"type": "string", "description": "Title (for add)"},
                "content": {"type": "string", "description": "Content (for add)"},
                "source_project": {"type": "string", "description": "Source project (for add)"},
                "target_domain": {"type": "string", "description": "Target domain (for add)"},
                "summary": {"type": "string", "description": "Summary (for add)"},
                "reason": {"type": "string", "description": "Reason (for reject/defer)"},
            },
            "required": ["action"],
        },
    },
    handler=lambda args, **kw: review_knowledge(
        action=args.get("action", "list"),
        knowledge_id=args.get("knowledge_id"),
        title=args.get("title"),
        content=args.get("content"),
        source_project=args.get("source_project"),
        target_domain=args.get("target_domain"),
        summary=args.get("summary"),
        reason=args.get("reason"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
)
