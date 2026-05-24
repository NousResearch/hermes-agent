"""Knowledge card management tool.

Tool name: manage_knowledge_card

Actions: create, get, update, delete, search, list, review, find_duplicates
"""

from __future__ import annotations

import json
from typing import Optional

from agent.card_store import CardStore
from tools.registry import registry


def manage_knowledge_card(
    action: str,
    card_id: Optional[str] = None,
    title: Optional[str] = None,
    body: Optional[str] = None,
    source: Optional[str] = None,
    evidence: Optional[str] = None,
    truth_level: Optional[str] = None,
    project_fit: Optional[float] = None,
    status: Optional[str] = None,
    domains: Optional[str] = None,
    review_status: Optional[str] = None,
    duplicate_of: Optional[str] = None,
    origin_project: Optional[str] = None,
    promoted: Optional[bool] = None,
    query: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    task_id: str = None,
) -> str:
    """Manage knowledge cards.

    Actions:
      create          — create a new knowledge card
      get             — get a card by ID
      update          — update an existing card
      delete          — delete a card
      search          — full-text search across all cards
      list            — list cards with filters
      review          — approve/reject/defer/duplicate/revision_requested
      find_duplicates — find existing cards similar to proposed content
    """
    store = CardStore()
    action = (action or "").strip().lower()

    if action == "create":
        if not all([title, body]):
            return json.dumps({
                "success": False,
                "error": "create requires title, body",
            })
        try:
            cid = store.create_knowledge_card(
                title=title,
                body=body,
                source=source or "",
                evidence=json.loads(evidence) if evidence else None,
                truth_level=truth_level or "probable",
                project_fit=project_fit if project_fit is not None else 0.5,
                status=status or "draft",
                domains=json.loads(domains) if domains else None,
                review_status=review_status or "pending_review",
                duplicate_of=duplicate_of or "",
                origin_project=origin_project or "",
                promoted=bool(promoted),
            )
            return json.dumps({"success": True, "action": "create", "card_id": cid})
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)})

    elif action == "get":
        if not card_id:
            return json.dumps({"success": False, "error": "get requires card_id"})
        card = store.get_knowledge_card(card_id)
        if card:
            return json.dumps({"success": True, "action": "get", "card": card})
        return json.dumps({"success": False, "error": f"Card {card_id} not found"})

    elif action == "update":
        if not card_id:
            return json.dumps({"success": False, "error": "update requires card_id"})
        updates = {}
        if title is not None: updates["title"] = title
        if body is not None: updates["body"] = body
        if source is not None: updates["source"] = source
        if evidence is not None: updates["evidence"] = json.loads(evidence)
        if truth_level is not None: updates["truth_level"] = truth_level
        if project_fit is not None: updates["project_fit"] = project_fit
        if status is not None: updates["status"] = status
        if domains is not None: updates["domains"] = json.loads(domains)
        if review_status is not None: updates["review_status"] = review_status
        if duplicate_of is not None: updates["duplicate_of"] = duplicate_of
        if origin_project is not None: updates["origin_project"] = origin_project
        if promoted is not None: updates["promoted"] = promoted
        if not updates:
            return json.dumps({"success": False, "error": "no fields to update"})
        ok = store.update_knowledge_card(card_id, **updates)
        return json.dumps({"success": ok, "action": "update", "card_id": card_id})

    elif action == "delete":
        if not card_id:
            return json.dumps({"success": False, "error": "delete requires card_id"})
        ok = store.delete_knowledge_card(card_id)
        return json.dumps({"success": ok, "action": "delete", "card_id": card_id})

    elif action == "search":
        if not query:
            return json.dumps({"success": False, "error": "search requires query"})
        domain_list = json.loads(domains) if domains else None
        results = store.search_cards(
            query=query,
            card_type="knowledge",
            domains=domain_list,
            review_status=review_status,
            limit=limit,
        )
        return json.dumps({"success": True, "action": "search", "count": len(results), "cards": results})

    elif action == "list":
        domain_list = json.loads(domains) if domains else None
        cards, total = store.list_knowledge_cards(
            review_status=review_status,
            truth_level=truth_level,
            domains=domain_list,
            limit=limit,
            offset=offset,
        )
        return json.dumps({
            "success": True, "action": "list",
            "count": len(cards), "total": total, "cards": cards,
        })

    elif action in ("approve", "reject", "defer", "mark_duplicate", "request_revision"):
        if not card_id:
            return json.dumps({"success": False, "error": f"{action} requires card_id"})
        review_map = {
            "approve": "approved",
            "reject": "rejected",
            "defer": "deferred",
            "mark_duplicate": "duplicate",
            "request_revision": "revision_requested",
        }
        ok = store.update_knowledge_card(card_id, review_status=review_map[action])
        return json.dumps({"success": ok, "action": action, "card_id": card_id})

    elif action == "find_duplicates":
        if not all([title, body]):
            return json.dumps({"success": False, "error": "find_duplicates requires title, body"})
        domain_list = json.loads(domains) if domains else None
        candidates = store.find_duplicate_knowledge_cards(title, body, domains=domain_list)
        return json.dumps({
            "success": True, "action": "find_duplicates",
            "count": len(candidates), "candidates": candidates,
        })

    else:
        return json.dumps({
            "success": False,
            "error": f"Invalid action: {action}. Valid: create, get, update, delete, search, list, approve, reject, defer, mark_duplicate, request_revision, find_duplicates",
        })


def check_requirements() -> bool:
    return True


registry.register(
    name="manage_knowledge_card",
    toolset="knowledge",
    schema={
        "name": "manage_knowledge_card",
        "description": (
            "Manage knowledge cards with source, evidence, truth level, project fit, and review status. "
            "Actions: create, get, update, delete, search, list, approve, reject, defer, mark_duplicate, "
            "request_revision, find_duplicates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "get", "update", "delete", "search", "list",
                             "approve", "reject", "defer", "mark_duplicate", "request_revision",
                             "find_duplicates"],
                },
                "card_id": {"type": "string", "description": "Card ID (for get/update/delete/review)"},
                "title": {"type": "string", "description": "Card title"},
                "body": {"type": "string", "description": "Card body/content"},
                "source": {"type": "string", "description": "Source of the knowledge"},
                "evidence": {"type": "string", "description": "JSON array of evidence strings"},
                "truth_level": {
                    "type": "string",
                    "enum": ["verified", "probable", "speculative", "disproven"],
                    "description": "Truth level of the knowledge",
                },
                "project_fit": {"type": "number", "description": "Project fit score 0-1"},
                "status": {"type": "string", "description": "General status"},
                "domains": {"type": "string", "description": "JSON array of domain slugs"},
                "review_status": {
                    "type": "string",
                    "enum": ["pending_review", "approved", "rejected", "deferred", "duplicate", "revision_requested"],
                    "description": "Review status",
                },
                "duplicate_of": {"type": "string", "description": "ID of the card this duplicates"},
                "origin_project": {"type": "string", "description": "Project where knowledge originated"},
                "promoted": {"type": "boolean", "description": "Whether promoted to domain KB"},
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Result limit (default 50)"},
                "offset": {"type": "integer", "description": "Pagination offset"},
            },
            "required": ["action"],
        },
    },
    handler=lambda args, **kw: manage_knowledge_card(
        action=args.get("action", "list"),
        card_id=args.get("card_id"),
        title=args.get("title"),
        body=args.get("body"),
        source=args.get("source"),
        evidence=args.get("evidence"),
        truth_level=args.get("truth_level"),
        project_fit=args.get("project_fit"),
        status=args.get("status"),
        domains=args.get("domains"),
        review_status=args.get("review_status"),
        duplicate_of=args.get("duplicate_of"),
        origin_project=args.get("origin_project"),
        promoted=args.get("promoted"),
        query=args.get("query"),
        limit=args.get("limit", 50),
        offset=args.get("offset", 0),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
)
