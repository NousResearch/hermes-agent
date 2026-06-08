"""Trello boards, lists, and cards management tool.

Provides the agent with the ability to interact with Trello workspaces
via the Trello REST API v1. Uses only the Python standard library
(urllib) — no third-party dependencies.

Credentials are read from environment variables:
    TRELLO_API_KEY   — your Trello Power-Up / API key
    TRELLO_API_TOKEN — your user OAuth token (read/write access)

Both can be obtained from https://trello.com/power-ups/admin/ and following the instructions

The tool is only included in the agent's schema when both credentials
are present (gated via check_fn), so it has zero cost for users
who have not configured Trello.

Available actions
-----------------
  list_boards()                        — list all boards for the authenticated user
  get_board(board_id)                  — board details (name, desc, url, prefs)
  list_lists(board_id)                 — lists on a board with card counts
  list_cards(list_id | board_id)       — cards in a list or across a whole board
  get_card(card_id)                    — full card details (desc, labels, due, members)
  create_card(list_id, name)           — create a card; optional desc/due/label_ids
  update_card(card_id)                 — update name, desc, due, or closed state
  move_card(card_id, list_id)          — move card to a different list
  archive_card(card_id)                — archive (close) a card
  add_comment(card_id, text)           — post a comment on a card
  list_members(board_id)               — members of a board
  list_labels(board_id)                — labels defined on a board
  search(query)                        — search boards, cards, and orgs by text
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import registry

logger = logging.getLogger(__name__)

TRELLO_API_BASE = "https://api.trello.com/1"

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def _get_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Return (api_key, api_token) from environment, or (None, None)."""
    key = os.getenv("TRELLO_API_KEY", "").strip() or None
    token = os.getenv("TRELLO_API_TOKEN", "").strip() or None
    return key, token


def _auth_params() -> Dict[str, str]:
    """Build the query-param auth dict for Trello API calls."""
    key, token = _get_credentials()
    return {"key": key or "", "token": token or ""}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _trello_request(
    method: str,
    path: str,
    params: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
) -> Any:
    """Make an authenticated request to the Trello REST API.

    Returns the parsed JSON response, or None for 204/empty bodies.
    Raises :class:`TrelloAPIError` on HTTP errors.
    """
    combined_params = {**_auth_params(), **(params or {})}
    url = f"{TRELLO_API_BASE}{path}?{urllib.parse.urlencode(combined_params)}"

    data = None
    headers: Dict[str, str] = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, method=method, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 204:
                return None
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else None
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise TrelloAPIError(e.code, error_body) from e


class TrelloAPIError(Exception):
    """Raised when a Trello API call fails with an HTTP error."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"Trello API error {status}: {body}")


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------

def _list_boards(**_kwargs: Any) -> str:
    """List all boards for the authenticated user."""
    boards = _trello_request(
        "GET",
        "/members/me/boards",
        params={"fields": "id,name,desc,url,closed,idOrganization,prefs"},
    )
    result = []
    for b in boards or []:
        result.append({
            "id": b["id"],
            "name": b["name"],
            "desc": b.get("desc", ""),
            "url": b.get("url", ""),
            "closed": b.get("closed", False),
            "organization_id": b.get("idOrganization"),
        })
    return json.dumps({"boards": result, "count": len(result)})


def _get_board(board_id: str, **_kwargs: Any) -> str:
    """Get detailed information about a specific board."""
    b = _trello_request(
        "GET",
        f"/boards/{board_id}",
        params={"fields": "id,name,desc,url,closed,idOrganization,prefs,dateLastActivity"},
    )
    return json.dumps({
        "id": b["id"],
        "name": b["name"],
        "desc": b.get("desc", ""),
        "url": b.get("url", ""),
        "closed": b.get("closed", False),
        "organization_id": b.get("idOrganization"),
        "date_last_activity": b.get("dateLastActivity"),
        "prefs": {
            "background": b.get("prefs", {}).get("background"),
            "visibility": b.get("prefs", {}).get("permissionLevel"),
        },
    })


def _list_lists(board_id: str, **_kwargs: Any) -> str:
    """List all lists on a board."""
    lists = _trello_request(
        "GET",
        f"/boards/{board_id}/lists",
        params={"fields": "id,name,closed,pos", "cards": "none"},
    )
    result = []
    for lst in lists or []:
        result.append({
            "id": lst["id"],
            "name": lst["name"],
            "closed": lst.get("closed", False),
            "position": lst.get("pos", 0),
        })
    return json.dumps({"lists": result, "count": len(result)})


def _list_cards(
    list_id: str = "",
    board_id: str = "",
    **_kwargs: Any,
) -> str:
    """List cards in a specific list, or all open cards on a board."""
    fields = "id,name,desc,due,dueComplete,closed,idList,idBoard,labels,idMembers,url,shortUrl,pos"
    if list_id:
        cards = _trello_request(
            "GET",
            f"/lists/{list_id}/cards",
            params={"fields": fields},
        )
    elif board_id:
        cards = _trello_request(
            "GET",
            f"/boards/{board_id}/cards/open",
            params={"fields": fields},
        )
    else:
        return json.dumps({"error": "Provide list_id or board_id for list_cards."})

    result = []
    for c in cards or []:
        result.append({
            "id": c["id"],
            "name": c["name"],
            "desc": c.get("desc", ""),
            "due": c.get("due"),
            "due_complete": c.get("dueComplete", False),
            "closed": c.get("closed", False),
            "list_id": c.get("idList"),
            "board_id": c.get("idBoard"),
            "labels": [{"id": lb["id"], "name": lb.get("name", ""), "color": lb.get("color")} for lb in c.get("labels", [])],
            "member_ids": c.get("idMembers", []),
            "url": c.get("shortUrl", c.get("url", "")),
        })
    return json.dumps({"cards": result, "count": len(result)})


def _get_card(card_id: str, **_kwargs: Any) -> str:
    """Get full details of a card including checklists and attachments."""
    c = _trello_request(
        "GET",
        f"/cards/{card_id}",
        params={
            "fields": "id,name,desc,due,dueComplete,closed,idList,idBoard,labels,idMembers,url,shortUrl,pos,dateLastActivity",
            "checklists": "all",
            "members": "true",
            "attachments": "true",
        },
    )
    checklists = []
    for cl in c.get("checklists", []):
        checklists.append({
            "id": cl["id"],
            "name": cl["name"],
            "items": [
                {"name": item["name"], "complete": item["state"] == "complete"}
                for item in cl.get("checkItems", [])
            ],
        })

    members = []
    for m in c.get("members", []):
        members.append({
            "id": m["id"],
            "full_name": m.get("fullName", ""),
            "username": m.get("username", ""),
        })

    attachments = []
    for att in c.get("attachments", []):
        attachments.append({
            "id": att["id"],
            "name": att.get("name", ""),
            "url": att.get("url", ""),
            "mimeType": att.get("mimeType", ""),
        })

    return json.dumps({
        "id": c["id"],
        "name": c["name"],
        "desc": c.get("desc", ""),
        "due": c.get("due"),
        "due_complete": c.get("dueComplete", False),
        "closed": c.get("closed", False),
        "list_id": c.get("idList"),
        "board_id": c.get("idBoard"),
        "labels": [{"id": lb["id"], "name": lb.get("name", ""), "color": lb.get("color")} for lb in c.get("labels", [])],
        "members": members,
        "checklists": checklists,
        "attachments": attachments,
        "url": c.get("shortUrl", c.get("url", "")),
        "date_last_activity": c.get("dateLastActivity"),
    })


def _create_card(
    list_id: str,
    name: str,
    desc: str = "",
    due: str = "",
    label_ids: str = "",
    **_kwargs: Any,
) -> str:
    """Create a new card in a list."""
    body: Dict[str, Any] = {"idList": list_id, "name": name}
    if desc:
        body["desc"] = desc
    if due:
        body["due"] = due
    if label_ids:
        # Accept comma-separated label IDs
        ids = [lid.strip() for lid in label_ids.split(",") if lid.strip()]
        if ids:
            body["idLabels"] = ids

    card = _trello_request("POST", "/cards", body=body)
    return json.dumps({
        "success": True,
        "card_id": card["id"],
        "name": card["name"],
        "url": card.get("shortUrl", card.get("url", "")),
        "list_id": card.get("idList"),
        "board_id": card.get("idBoard"),
    })


def _update_card(
    card_id: str,
    name: str = "",
    desc: str = "",
    due: str = "",
    closed: str = "",
    **_kwargs: Any,
) -> str:
    """Update a card's name, description, due date, or archived state."""
    body: Dict[str, Any] = {}
    if name:
        body["name"] = name
    if desc:
        body["desc"] = desc
    if due:
        body["due"] = due
    if closed in ("true", "false"):
        body["closed"] = closed == "true"

    if not body:
        return json.dumps({"error": "No fields to update. Provide name, desc, due, or closed."})

    card = _trello_request("PUT", f"/cards/{card_id}", body=body)
    return json.dumps({
        "success": True,
        "card_id": card["id"],
        "name": card["name"],
        "closed": card.get("closed", False),
        "list_id": card.get("idList"),
    })


def _move_card(card_id: str, list_id: str, **_kwargs: Any) -> str:
    """Move a card to a different list (optionally a different board)."""
    card = _trello_request("PUT", f"/cards/{card_id}", body={"idList": list_id})
    return json.dumps({
        "success": True,
        "card_id": card["id"],
        "name": card["name"],
        "new_list_id": card.get("idList"),
        "board_id": card.get("idBoard"),
    })


def _archive_card(card_id: str, **_kwargs: Any) -> str:
    """Archive (close) a card."""
    card = _trello_request("PUT", f"/cards/{card_id}", body={"closed": True})
    return json.dumps({
        "success": True,
        "card_id": card["id"],
        "name": card["name"],
        "closed": card.get("closed", True),
    })


def _add_comment(card_id: str, text: str, **_kwargs: Any) -> str:
    """Add a comment to a card."""
    comment = _trello_request(
        "POST",
        f"/cards/{card_id}/actions/comments",
        body={"text": text},
    )
    return json.dumps({
        "success": True,
        "comment_id": comment.get("id"),
        "card_id": card_id,
        "text": text,
    })


def _list_members(board_id: str, **_kwargs: Any) -> str:
    """List all members of a board."""
    members = _trello_request(
        "GET",
        f"/boards/{board_id}/members",
        params={"fields": "id,fullName,username,avatarUrl"},
    )
    result = []
    for m in members or []:
        result.append({
            "id": m["id"],
            "full_name": m.get("fullName", ""),
            "username": m.get("username", ""),
        })
    return json.dumps({"members": result, "count": len(result)})


def _list_labels(board_id: str, **_kwargs: Any) -> str:
    """List all labels on a board."""
    labels = _trello_request(
        "GET",
        f"/boards/{board_id}/labels",
        params={"fields": "id,name,color"},
    )
    result = []
    for lb in labels or []:
        result.append({
            "id": lb["id"],
            "name": lb.get("name", ""),
            "color": lb.get("color"),
        })
    return json.dumps({"labels": result, "count": len(result)})


def _search(query: str, **_kwargs: Any) -> str:
    """Search Trello for boards, cards, and organizations matching a query."""
    results = _trello_request(
        "GET",
        "/search",
        params={
            "query": query,
            "modelTypes": "boards,cards",
            "board_fields": "id,name,url,closed",
            "card_fields": "id,name,url,idBoard,idList",
            "boards_limit": "10",
            "cards_limit": "20",
            "partial": "true",
        },
    )

    boards = []
    for b in (results or {}).get("boards", []):
        boards.append({
            "id": b["id"],
            "name": b["name"],
            "url": b.get("url", ""),
            "closed": b.get("closed", False),
        })

    cards = []
    for c in (results or {}).get("cards", []):
        cards.append({
            "id": c["id"],
            "name": c["name"],
            "url": c.get("url", ""),
            "board_id": c.get("idBoard"),
            "list_id": c.get("idList"),
        })

    return json.dumps({
        "query": query,
        "boards": boards,
        "cards": cards,
        "total_boards": len(boards),
        "total_cards": len(cards),
    })


# ---------------------------------------------------------------------------
# Action dispatch table
# ---------------------------------------------------------------------------

_ACTIONS = {
    "list_boards": _list_boards,
    "get_board": _get_board,
    "list_lists": _list_lists,
    "list_cards": _list_cards,
    "get_card": _get_card,
    "create_card": _create_card,
    "update_card": _update_card,
    "move_card": _move_card,
    "archive_card": _archive_card,
    "add_comment": _add_comment,
    "list_members": _list_members,
    "list_labels": _list_labels,
    "search": _search,
}

# Per-action required params for runtime validation.
_REQUIRED_PARAMS: Dict[str, List[str]] = {
    "get_board": ["board_id"],
    "list_lists": ["board_id"],
    "list_cards": [],            # needs list_id OR board_id — checked in handler
    "get_card": ["card_id"],
    "create_card": ["list_id", "name"],
    "update_card": ["card_id"],
    "move_card": ["card_id", "list_id"],
    "archive_card": ["card_id"],
    "add_comment": ["card_id", "text"],
    "list_members": ["board_id"],
    "list_labels": ["board_id"],
    "search": ["query"],
}

# Single-source-of-truth manifest: action → (signature, one-line description).
_ACTION_MANIFEST: List[Tuple[str, str, str]] = [
    ("list_boards",  "()",                             "list all boards for the authenticated user"),
    ("get_board",    "(board_id)",                     "board details (name, desc, url, prefs)"),
    ("list_lists",   "(board_id)",                     "lists on a board with positions"),
    ("list_cards",   "(list_id | board_id)",           "cards in a list or across a whole board"),
    ("get_card",     "(card_id)",                      "full card details including checklists and attachments"),
    ("create_card",  "(list_id, name)",                "create a card; optional desc/due/label_ids"),
    ("update_card",  "(card_id)",                      "update name, desc, due, or closed state"),
    ("move_card",    "(card_id, list_id)",              "move card to a different list"),
    ("archive_card", "(card_id)",                      "archive (close) a card"),
    ("add_comment",  "(card_id, text)",                "post a comment on a card"),
    ("list_members", "(board_id)",                     "members of a board"),
    ("list_labels",  "(board_id)",                     "labels defined on a board"),
    ("search",       "(query)",                        "search boards and cards by text"),
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _build_schema() -> Dict[str, Any]:
    """Build the static tool schema exposed to the model."""
    manifest_lines = [
        f"  {name}{sig}  — {desc}"
        for name, sig, desc in _ACTION_MANIFEST
    ]
    manifest_block = "\n".join(manifest_lines)

    description = (
        "Manage Trello boards, lists, and cards via the Trello REST API.\n\n"
        "Available actions:\n"
        f"{manifest_block}\n\n"
        "Workflow: call list_boards to discover board_ids, then list_lists for "
        "list_ids, then list_cards to see cards in a list. Use get_card for full "
        "details including checklists. IDs are opaque strings — always pass them "
        "exactly as returned.\n\n"
        "Dates use ISO 8601 format (e.g. '2024-12-31T00:00:00.000Z'). "
        "Credentials are read from TRELLO_API_KEY and TRELLO_API_TOKEN env vars. "
        "Get them at https://trello.com/power-ups/admin/"
    )

    properties: Dict[str, Any] = {
        "action": {
            "type": "string",
            "enum": list(_ACTIONS.keys()),
            "description": "The Trello operation to perform.",
        },
        "board_id": {
            "type": "string",
            "description": "Trello board ID.",
        },
        "list_id": {
            "type": "string",
            "description": "Trello list ID.",
        },
        "card_id": {
            "type": "string",
            "description": "Trello card ID.",
        },
        "name": {
            "type": "string",
            "description": "Card name (create_card, update_card).",
        },
        "desc": {
            "type": "string",
            "description": "Card description / body text (create_card, update_card).",
        },
        "due": {
            "type": "string",
            "description": "Due date in ISO 8601 format, e.g. '2024-12-31T00:00:00.000Z' (create_card, update_card).",
        },
        "label_ids": {
            "type": "string",
            "description": "Comma-separated label IDs to attach when creating a card (create_card).",
        },
        "closed": {
            "type": "string",
            "enum": ["true", "false"],
            "description": "Set to 'true' to archive or 'false' to unarchive a card (update_card).",
        },
        "text": {
            "type": "string",
            "description": "Comment text (add_comment).",
        },
        "query": {
            "type": "string",
            "description": "Search query string (search).",
        },
    }

    return {
        "name": "trello",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["action"],
        },
    }


_SCHEMA = _build_schema()


# ---------------------------------------------------------------------------
# Check function
# ---------------------------------------------------------------------------

def check_trello_requirements() -> bool:
    """Tool is available only when both TRELLO_API_KEY and TRELLO_API_TOKEN are set."""
    key, token = _get_credentials()
    return bool(key and token)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

_HANDLER_DEFAULTS: Dict[str, Any] = {
    "action": "",
    "board_id": "",
    "list_id": "",
    "card_id": "",
    "name": "",
    "desc": "",
    "due": "",
    "label_ids": "",
    "closed": "",
    "text": "",
    "query": "",
}


def trello_handler(
    action: str,
    board_id: str = "",
    list_id: str = "",
    card_id: str = "",
    name: str = "",
    desc: str = "",
    due: str = "",
    label_ids: str = "",
    closed: str = "",
    text: str = "",
    query: str = "",
    **_kwargs: Any,
) -> str:
    """Dispatch a Trello action to the appropriate implementation."""
    key, token = _get_credentials()
    if not key or not token:
        return json.dumps({
            "error": (
                "Trello credentials not configured. "
                "Set TRELLO_API_KEY and TRELLO_API_TOKEN in ~/.hermes/.env. "
                "Get them at https://trello.com/app-key"
            )
        })

    action_fn = _ACTIONS.get(action)
    if not action_fn:
        return json.dumps({
            "error": f"Unknown action: '{action}'",
            "available_actions": list(_ACTIONS.keys()),
        })

    # Runtime required-param validation
    local_vars = {
        "board_id": board_id,
        "list_id": list_id,
        "card_id": card_id,
        "name": name,
        "text": text,
        "query": query,
    }
    missing = [p for p in _REQUIRED_PARAMS.get(action, []) if not local_vars.get(p)]
    if missing:
        return json.dumps({
            "error": f"Missing required parameters for '{action}': {', '.join(missing)}",
        })

    # Special case: list_cards needs list_id OR board_id
    if action == "list_cards" and not list_id and not board_id:
        return json.dumps({
            "error": "list_cards requires either list_id or board_id.",
        })

    try:
        return action_fn(
            board_id=board_id,
            list_id=list_id,
            card_id=card_id,
            name=name,
            desc=desc,
            due=due,
            label_ids=label_ids,
            closed=closed,
            text=text,
            query=query,
        )
    except TrelloAPIError as e:
        logger.warning("Trello API error in action '%s': %s", action, e)
        if e.status == 401:
            return json.dumps({
                "error": (
                    f"Trello API 401 (unauthorized) on '{action}'. "
                    "Check that TRELLO_API_KEY and TRELLO_API_TOKEN are correct "
                    "and the token has read/write scope. "
                    f"(Raw: {e.body})"
                )
            })
        if e.status == 403:
            return json.dumps({
                "error": (
                    f"Trello API 403 (forbidden) on '{action}'. "
                    "The authenticated user may not have permission to perform this action. "
                    f"(Raw: {e.body})"
                )
            })
        if e.status == 404:
            return json.dumps({
                "error": (
                    f"Trello API 404 (not found) on '{action}'. "
                    "The requested resource (board, list, or card) does not exist or "
                    "is not accessible with the current credentials. "
                    f"(Raw: {e.body})"
                )
            })
        return json.dumps({"error": f"Trello API error {e.status}: {e.body}"})
    except Exception as e:
        logger.exception("Unexpected error in Trello action '%s'", action)
        return json.dumps({"error": f"Unexpected error: {e}"})


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

registry.register(
    name="trello",
    toolset="trello",
    schema=_SCHEMA,
    handler=lambda args, **kw: trello_handler(
        **{k: args.get(k, v) for k, v in _HANDLER_DEFAULTS.items()}
    ),
    check_fn=check_trello_requirements,
    requires_env=["TRELLO_API_KEY", "TRELLO_API_TOKEN"],
)
