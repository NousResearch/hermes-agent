"""Knowledge review queue tool — manages the promote/reject/defer workflow.

Now backed by Card Store (agent/card_store.py) and ReviewInbox (agent/review_inbox.py).
The old JSON-based queue is migrated on first load.

Tool name: review_knowledge
Actions: list, add, approve, reject, defer, delete, mark_duplicate, request_revision
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.review_inbox import ReviewInbox
from tools.registry import registry


def _get_vault_path() -> Path:
    """Get the Obsidian vault path."""
    vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")
    if vault_path:
        return Path(vault_path)
    return Path.home() / "ObsidianVault" / "HermesAgent"


def _migrate_old_queue() -> None:
    """Migrate old JSON queue entries to Card Store on first use."""
    vault = _get_vault_path()
    queue_path = vault / "domains" / ".review_queue.json"
    if not queue_path.exists():
        return
    try:
        text = queue_path.read_text(encoding="utf-8")
        old_entries = json.loads(text)
        if not old_entries:
            return
        inbox = ReviewInbox(vault_path=vault)
        migrated = 0
        for entry in old_entries:
            if entry.get("status") == "pending":
                inbox.queue_knowledge(
                    title=entry.get("title", ""),
                    body=entry.get("content", ""),
                    source=entry.get("source_project", ""),
                    domains=[entry.get("target_domain", "")] if entry.get("target_domain") else None,
                    origin_project=entry.get("source_project", ""),
                    summary=entry.get("summary", ""),
                )
                migrated += 1
        if migrated:
            # Rename old queue to mark as migrated
            queue_path.rename(queue_path.with_suffix(".json.migrated"))
    except Exception:
        pass  # Non-fatal — old queue stays, Card Store used going forward


def review_knowledge(
    action: str,
    knowledge_id: Optional[str] = None,
    title: Optional[str] = None,
    content: Optional[str] = None,
    source_project: Optional[str] = None,
    target_domain: Optional[str] = None,
    summary: Optional[str] = None,
    reason: Optional[str] = None,
    duplicate_of: Optional[str] = None,
    feedback: Optional[str] = None,
    task_id: str = None,
) -> str:
    """Manage the knowledge review queue via Card Store.

    Actions: list, add, approve, reject, defer, delete, mark_duplicate, request_revision
    """
    # Migrate old JSON queue on first use
    _migrate_old_queue()

    vault = _get_vault_path()
    inbox = ReviewInbox(vault_path=vault)

    if action == "list":
        items = inbox.list_pending()
        return json.dumps({
            "success": True,
            "action": "list",
            "count": len(items),
            "items": items,
        })

    elif action == "add":
        if not all([title, content, source_project, target_domain]):
            return json.dumps({
                "success": False,
                "error": "add requires title, content, source_project, target_domain",
            })
        result = inbox.queue_knowledge(
            title=title,
            body=content,
            source=source_project,
            domains=[target_domain],
            origin_project=source_project,
            summary=summary or "",
        )
        # Normalize Card Store response to match legacy API
        return json.dumps({
            "success": result.get("success", True),
            "action": "add",
            "knowledge_id": result.get("card_id", ""),
            "status": "pending",
        })

    elif action == "approve":
        if not knowledge_id:
            return json.dumps({"success": False, "error": "approve requires knowledge_id"})
        result = inbox.approve(knowledge_id)
        # Also promote the knowledge to domain KB
        if result.get("success"):
            card = inbox.store.get_knowledge_card(knowledge_id)
            if card:
                from tools.knowledge_promote import promote_knowledge
                promote_result = promote_knowledge(
                    title=card["title"],
                    content=card["body"],
                    source_project=card.get("origin_project", ""),
                    target_domain=(json.loads(card.get("domains", "[]"))[0] if card.get("domains") else "backend"),
                    summary=summary,
                )
                result["promote_result"] = json.loads(promote_result)
        result.setdefault("action", "approve")
        return json.dumps(result)

    elif action == "reject":
        if not knowledge_id:
            return json.dumps({"success": False, "error": "reject requires knowledge_id"})
        result = inbox.reject(knowledge_id, reason=reason or "")
        result.setdefault("action", "reject")
        return json.dumps(result)

    elif action == "defer":
        if not knowledge_id:
            return json.dumps({"success": False, "error": "defer requires knowledge_id"})
        result = inbox.defer(knowledge_id, reason=reason or "")
        result.setdefault("action", "defer")
        return json.dumps(result)

    elif action == "mark_duplicate":
        if not knowledge_id:
            return json.dumps({"success": False, "error": "mark_duplicate requires knowledge_id"})
        if not duplicate_of:
            return json.dumps({"success": False, "error": "mark_duplicate requires duplicate_of"})
        result = inbox.mark_duplicate(knowledge_id, duplicate_of)
        result.setdefault("action", "mark_duplicate")
        return json.dumps(result)

    elif action == "request_revision":
        if not knowledge_id:
            return json.dumps({"success": False, "error": "request_revision requires knowledge_id"})
        result = inbox.request_revision(knowledge_id, feedback=feedback or "")
        result.setdefault("action", "request_revision")
        return json.dumps(result)

    elif action == "delete":
        if not knowledge_id:
            return json.dumps({"success": False, "error": "delete requires knowledge_id"})
        ok = inbox.store.delete_knowledge_card(knowledge_id)
        return json.dumps({"success": ok, "action": "delete", "knowledge_id": knowledge_id})

    else:
        return json.dumps({
            "success": False,
            "error": (
                f"Invalid action: {action}. "
                "Valid: list, add, approve, reject, defer, delete, mark_duplicate, request_revision"
            ),
        })


def check_requirements() -> bool:
    """Check if vault path exists."""
    return _get_vault_path().exists()


# Register the tool
registry.register(
    name="review_knowledge",
    toolset="knowledge",
    schema={
        "name": "review_knowledge",
        "description": (
            "Manage the knowledge review queue via Card Store. "
            "Actions: list, add, approve, reject, defer, delete, mark_duplicate, request_revision. "
            "Approved entries are automatically promoted to the domain KB. "
            "Rejected entries are archived to domains/.rejected/."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "add", "approve", "reject", "defer", "delete", "mark_duplicate", "request_revision"],
                    "description": "Action to perform",
                },
                "knowledge_id": {"type": "string", "description": "Knowledge card ID"},
                "title": {"type": "string", "description": "Title (for add)"},
                "content": {"type": "string", "description": "Content (for add)"},
                "source_project": {"type": "string", "description": "Source project (for add)"},
                "target_domain": {"type": "string", "description": "Target domain (for add)"},
                "summary": {"type": "string", "description": "Summary (for add)"},
                "reason": {"type": "string", "description": "Reason (for reject/defer)"},
                "duplicate_of": {"type": "string", "description": "Card ID this duplicates (for mark_duplicate)"},
                "feedback": {"type": "string", "description": "Feedback (for request_revision)"},
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
        duplicate_of=args.get("duplicate_of"),
        feedback=args.get("feedback"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
)
