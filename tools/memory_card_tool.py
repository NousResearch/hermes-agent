"""Memory card management tool.

Tool name: manage_memory_card

Actions: create, get, update, delete, search, list, find_duplicates, pin, unpin
"""

from __future__ import annotations

import json
from typing import Optional

from agent.card_store import CardStore
from tools.registry import registry


def manage_memory_card(
    action: str,
    card_id: Optional[str] = None,
    card_type: Optional[str] = None,
    title: Optional[str] = None,
    body: Optional[str] = None,
    tags: Optional[str] = None,
    source: Optional[str] = None,
    project: Optional[str] = None,
    context: Optional[str] = None,
    confidence: Optional[float] = None,
    pinned: Optional[bool] = None,
    query: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    task_id: str = None,
) -> str:
    """Manage memory cards (decision, rule, incident, preference, active_context).

    Actions:
      create          — create a new memory card
      get             — get a card by ID
      update          — update an existing card
      delete          — delete a card
      search          — full-text search across all cards
      list            — list cards with filters
      find_duplicates — find existing cards similar to proposed content
      pin             — pin a card
      unpin           — unpin a card
    """
    store = CardStore()
    action = (action or "").strip().lower()

    if action == "create":
        if not all([card_type, title, body]):
            return json.dumps({
                "success": False,
                "error": "create requires card_type, title, body",
                "valid_types": ["decision", "rule", "incident", "preference", "active_context"],
            })
        try:
            cid = store.create_memory_card(
                card_type=card_type,
                title=title,
                body=body,
                tags=json.loads(tags) if tags else None,
                source=source or "",
                project=project or "",
                context=context or "",
                confidence=confidence if confidence is not None else 0.5,
                pinned=bool(pinned),
            )
            return json.dumps({"success": True, "action": "create", "card_id": cid})
        except ValueError as e:
            return json.dumps({"success": False, "error": str(e)})

    elif action == "get":
        if not card_id:
            return json.dumps({"success": False, "error": "get requires card_id"})
        card = store.get_memory_card(card_id)
        if card:
            return json.dumps({"success": True, "action": "get", "card": card})
        return json.dumps({"success": False, "error": f"Card {card_id} not found"})

    elif action == "update":
        if not card_id:
            return json.dumps({"success": False, "error": "update requires card_id"})
        updates = {}
        if title is not None: updates["title"] = title
        if body is not None: updates["body"] = body
        if tags is not None: updates["tags"] = json.loads(tags)
        if source is not None: updates["source"] = source
        if project is not None: updates["project"] = project
        if context is not None: updates["context"] = context
        if confidence is not None: updates["confidence"] = confidence
        if pinned is not None: updates["pinned"] = pinned
        if not updates:
            return json.dumps({"success": False, "error": "no fields to update"})
        ok = store.update_memory_card(card_id, **updates)
        return json.dumps({"success": ok, "action": "update", "card_id": card_id})

    elif action == "delete":
        if not card_id:
            return json.dumps({"success": False, "error": "delete requires card_id"})
        ok = store.delete_memory_card(card_id)
        return json.dumps({"success": ok, "action": "delete", "card_id": card_id})

    elif action == "search":
        if not query:
            return json.dumps({"success": False, "error": "search requires query"})
        results = store.search_cards(
            query=query,
            card_type=card_type,
            limit=limit,
        )
        return json.dumps({"success": True, "action": "search", "count": len(results), "cards": results})

    elif action == "list":
        cards, total = store.list_memory_cards(
            card_type=card_type,
            project=project,
            limit=limit,
            offset=offset,
        )
        return json.dumps({
            "success": True, "action": "list",
            "count": len(cards), "total": total, "cards": cards,
        })

    elif action == "find_duplicates":
        if not all([card_type, title, body]):
            return json.dumps({"success": False, "error": "find_duplicates requires card_type, title, body"})
        candidates = store.find_duplicate_memory_cards(card_type, title, body)
        return json.dumps({
            "success": True, "action": "find_duplicates",
            "count": len(candidates), "candidates": candidates,
        })

    elif action == "pin":
        if not card_id:
            return json.dumps({"success": False, "error": "pin requires card_id"})
        ok = store.update_memory_card(card_id, pinned=True)
        return json.dumps({"success": ok, "action": "pin", "card_id": card_id})

    elif action == "unpin":
        if not card_id:
            return json.dumps({"success": False, "error": "unpin requires card_id"})
        ok = store.update_memory_card(card_id, pinned=False)
        return json.dumps({"success": ok, "action": "unpin", "card_id": card_id})

    else:
        return json.dumps({
            "success": False,
            "error": f"Invalid action: {action}. Valid: create, get, update, delete, search, list, find_duplicates, pin, unpin",
        })


def check_requirements() -> bool:
    return True


registry.register(
    name="manage_memory_card",
    toolset="memory",
    schema={
        "name": "manage_memory_card",
        "description": (
            "Manage memory cards: decision, rule, incident, preference, active_context. "
            "Actions: create, get, update, delete, search, list, find_duplicates, pin, unpin. "
            "Cards are persisted in SQLite and full-text searchable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "get", "update", "delete", "search", "list", "find_duplicates", "pin", "unpin"],
                },
                "card_id": {"type": "string", "description": "Card ID (for get/update/delete/pin/unpin)"},
                "card_type": {
                    "type": "string",
                    "enum": ["decision", "rule", "incident", "preference", "active_context"],
                    "description": "Memory card type",
                },
                "title": {"type": "string", "description": "Card title"},
                "body": {"type": "string", "description": "Card body/content"},
                "tags": {"type": "string", "description": "JSON array of tags (for create/update)"},
                "source": {"type": "string", "description": "Source of the knowledge"},
                "project": {"type": "string", "description": "Project slug"},
                "context": {"type": "string", "description": "Context/conversation context"},
                "confidence": {"type": "number", "description": "Confidence score 0-1"},
                "pinned": {"type": "boolean", "description": "Whether the card is pinned"},
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Result limit (default 50)"},
                "offset": {"type": "integer", "description": "Pagination offset"},
            },
            "required": ["action"],
        },
    },
    handler=lambda args, **kw: manage_memory_card(
        action=args.get("action", "list"),
        card_id=args.get("card_id"),
        card_type=args.get("card_type"),
        title=args.get("title"),
        body=args.get("body"),
        tags=args.get("tags"),
        source=args.get("source"),
        project=args.get("project"),
        context=args.get("context"),
        confidence=args.get("confidence"),
        pinned=args.get("pinned"),
        query=args.get("query"),
        limit=args.get("limit", 50),
        offset=args.get("offset", 0),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
)
