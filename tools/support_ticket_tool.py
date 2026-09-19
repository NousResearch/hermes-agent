#!/usr/bin/env python3
"""
Support Ticket Tool — Creates and manages support tickets with tracking numbers.

Tickets are stored in Elasticsearch (index: byrd-it-support-tickets).
Each ticket gets a tracking number like BYRD-20260721-0001.

The support agent uses these tools to:
- Create a ticket when a customer reports an issue
- Update the ticket with diagnostic findings and actions
- Mark tickets as resolved when the customer confirms
- Look up tickets by tracking number or customer

Tickets should NEVER be closed by the agent — only Brandon closes tickets.
"""

import json
import logging
import time
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path

import urllib.request
import urllib.error
import ssl

logger = logging.getLogger(__name__)

# Elasticsearch configuration. No hardcoded key default — an API key must come
# from the environment (sourced from the secrets vault by the service unit,
# same pattern as every other Byrd-IT ES credential). t_890d8d8c: a prior
# draft of this file shipped a live-looking key as a hardcoded fallback
# default, which would have leaked into git history on commit; removed.
ES_URL = os.getenv("ES_URL", "https://es1.byrd-it.com:443")
ES_API_KEY = os.getenv("ES_API_KEY", "")
ES_INDEX = "byrd-it-support-tickets"
ES_SSL_SKIP = os.getenv("ES_SSL_SKIP_VERIFY", "true").lower() in ("true", "1", "yes")

_SSL_CTX = ssl.create_default_context()
if ES_SSL_SKIP:
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode = ssl.CERT_NONE

# In-memory counter cache for tracking numbers (resets on restart, but ES has the real count)
_last_ticket_count = 0
_last_count_check = 0


def _es_request(method: str, path: str, body: Optional[dict] = None) -> Dict[str, Any]:
    """Make an HTTP request to Elasticsearch."""
    url = f"{ES_URL}{path}"
    headers = {
        "Authorization": f"ApiKey {ES_API_KEY}",
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, context=_SSL_CTX, timeout=15) as response:
            response_data = response.read().decode("utf-8")
            return json.loads(response_data) if response_data else {}
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        return {"error": f"HTTP {e.code}: {e.reason}", "details": error_body[:500]}
    except Exception as e:
        return {"error": str(e)}


def _ensure_index_exists():
    """Create the tickets index if it doesn't exist."""
    mapping = {
        "mappings": {
            "properties": {
                "tracking_number": {"type": "keyword"},
                "status": {"type": "keyword"},
                "priority": {"type": "keyword"},
                "category": {"type": "keyword"},
                "site_name": {"type": "keyword"},
                "customer_name": {"type": "text"},
                "customer_email": {"type": "keyword"},
                "customer_telegram_id": {"type": "keyword"},
                "subject": {"type": "text"},
                "description": {"type": "text"},
                "diagnostics": {"type": "text"},
                "actions_taken": {"type": "text"},
                "resolution": {"type": "text"},
                "escalated": {"type": "boolean"},
                "escalated_to": {"type": "keyword"},
                "channel": {"type": "keyword"},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
                "resolved_at": {"type": "date"},
                "closed_at": {"type": "date"},
                "ticket_history": {"type": "text"},
            }
        }
    }
    result = _es_request("PUT", f"/{ES_INDEX}", mapping)
    if "error" in result and "resource_already_exists" not in str(result.get("details", "")):
        logger.warning("[Tickets] Index creation issue: %s", result.get("error"))
    return result


def _generate_tracking_number() -> str:
    """Generate a tracking number like BYRD-20260721-0001."""
    global _last_ticket_count, _last_count_check
    now = time.time()

    # Refresh count every 60 seconds
    if now - _last_count_check > 60:
        _last_count_check = now
        count_result = _es_request("GET", f"/{ES_INDEX}/_count")
        if "count" in count_result:
            _last_ticket_count = count_result["count"]
        else:
            # Index might not exist yet
            _ensure_index_exists()
            _last_ticket_count = 0

    _last_ticket_count += 1
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"BYRD-{date_str}-{_last_ticket_count:04d}"


# ============================================================
# Tool Handlers
# ============================================================


def support_create_ticket(
    customer_name: str,
    customer_email: str = "",
    customer_telegram_id: str = "",
    site_name: str = "",
    subject: str = "",
    description: str = "",
    category: str = "general",
    priority: str = "normal",
    channel: str = "email",
    **kwargs,
) -> Dict[str, Any]:
    """
    Create a new support ticket with a tracking number.

    Returns the ticket with its tracking number.
    """
    _ensure_index_exists()

    tracking_number = _generate_tracking_number()
    now = datetime.now(timezone.utc).isoformat()

    ticket = {
        "tracking_number": tracking_number,
        "status": "open",
        "priority": priority,
        "category": category,
        "site_name": site_name,
        "customer_name": customer_name,
        "customer_email": customer_email,
        "customer_telegram_id": customer_telegram_id,
        "subject": subject,
        "description": description,
        "diagnostics": "",
        "actions_taken": "",
        "resolution": "",
        "escalated": False,
        "escalated_to": "",
        "channel": channel,
        "created_at": now,
        "updated_at": now,
        "resolved_at": None,
        "closed_at": None,
        "ticket_history": f"[{now}] Ticket created via {channel}. Subject: {subject}",
    }

    # Index the ticket
    result = _es_request("POST", f"/{ES_INDEX}/_doc", ticket)

    if "error" in result:
        return {"code": "error", "error": "Failed to create ticket", "details": result.get("error")}

    return {
        "code": "SUCCESS",
        "tracking_number": tracking_number,
        "status": "open",
        "subject": subject,
        "created_at": now,
        "message": f"Ticket {tracking_number} created successfully.",
    }


def support_update_ticket(
    tracking_number: str,
    status: Optional[str] = None,
    diagnostics: Optional[str] = None,
    actions_taken: Optional[str] = None,
    resolution: Optional[str] = None,
    escalated: Optional[bool] = None,
    escalated_to: Optional[str] = None,
    add_history: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Update an existing support ticket.

    Can update: status, diagnostics, actions_taken, resolution, escalation.
    Adds a timestamped entry to ticket_history for each update.
    """
    # First, find the ticket by tracking number
    search_body = {
        "query": {"term": {"tracking_number": tracking_number}},
        "size": 1,
    }
    search_result = _es_request("POST", f"/{ES_INDEX}/_search", search_body)

    if "error" in search_result:
        return {"code": "error", "error": "Failed to search for ticket", "details": search_result.get("error")}

    hits = search_result.get("hits", {}).get("hits", [])
    if not hits:
        return {"code": "error", "error": f"Ticket {tracking_number} not found"}

    doc_id = hits[0]["_id"]
    existing = hits[0]["_source"]
    now = datetime.now(timezone.utc).isoformat()

    # Build the update body
    update_fields = {"updated_at": now}

    if status is not None:
        # Validate status — agent can set: open, in_progress, waiting_customer, resolved
        # Agent CANNOT set "closed" — only Brandon does that
        valid_statuses = ["open", "in_progress", "waiting_customer", "resolved", "reopened"]
        if status == "closed":
            return {
                "code": "error",
                "error": "Tickets can only be closed by Brandon. Set status to 'resolved' when the customer confirms the issue is fixed.",
            }
        if status not in valid_statuses:
            return {"code": "error", "error": f"Invalid status '{status}'. Valid: {valid_statuses}"}
        update_fields["status"] = status
        if status == "resolved":
            update_fields["resolved_at"] = now

    if diagnostics is not None:
        # Append to existing diagnostics
        existing_diag = existing.get("diagnostics", "")
        update_fields["diagnostics"] = f"{existing_diag}\n[{now}] {diagnostics}".strip() if existing_diag else f"[{now}] {diagnostics}"

    if actions_taken is not None:
        existing_actions = existing.get("actions_taken", "")
        update_fields["actions_taken"] = f"{existing_actions}\n[{now}] {actions_taken}".strip() if existing_actions else f"[{now}] {actions_taken}"

    if resolution is not None:
        update_fields["resolution"] = resolution

    if escalated is not None:
        update_fields["escalated"] = escalated
        if escalated_to:
            update_fields["escalated_to"] = escalated_to

    # Add to ticket history
    existing_history = existing.get("ticket_history", "")
    history_entry = f"[{now}]"
    if add_history:
        history_entry += f" {add_history}"
    if status:
        history_entry += f" Status changed to: {status}"
    if escalated:
        history_entry += f" Escalated to: {escalated_to or 'Brandon'}"
    update_fields["ticket_history"] = f"{existing_history}\n{history_entry}".strip()

    update_body = {"doc": update_fields}
    result = _es_request("POST", f"/{ES_INDEX}/_update/{doc_id}", update_body)

    if "error" in result:
        return {"code": "error", "error": "Failed to update ticket", "details": result.get("error")}

    return {
        "code": "SUCCESS",
        "tracking_number": tracking_number,
        "updated_fields": list(update_fields.keys()),
        "message": f"Ticket {tracking_number} updated successfully.",
    }


def support_get_ticket(tracking_number: str, **kwargs) -> Dict[str, Any]:
    """Get full details of a ticket by tracking number."""
    search_body = {
        "query": {"term": {"tracking_number": tracking_number}},
        "size": 1,
    }
    result = _es_request("POST", f"/{ES_INDEX}/_search", search_body)

    if "error" in result:
        return {"code": "error", "error": "Failed to search for ticket", "details": result.get("error")}

    hits = result.get("hits", {}).get("hits", [])
    if not hits:
        return {"code": "error", "error": f"Ticket {tracking_number} not found"}

    return {
        "code": "SUCCESS",
        "ticket": hits[0]["_source"],
    }


def support_search_tickets(
    customer_email: Optional[str] = None,
    customer_telegram_id: Optional[str] = None,
    site_name: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 10,
    **kwargs,
) -> Dict[str, Any]:
    """
    Search for tickets by customer, site, status, or category.
    Returns matching tickets sorted by most recent first.
    """
    must_clauses = []
    if customer_email:
        must_clauses.append({"term": {"customer_email": customer_email}})
    if customer_telegram_id:
        must_clauses.append({"term": {"customer_telegram_id": customer_telegram_id}})
    if site_name:
        must_clauses.append({"term": {"site_name": site_name}})
    if status:
        must_clauses.append({"term": {"status": status}})
    if category:
        must_clauses.append({"term": {"category": category}})

    search_body = {
        "query": {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}},
        "size": min(limit, 50),
        "sort": [{"created_at": {"order": "desc"}}],
    }

    result = _es_request("POST", f"/{ES_INDEX}/_search", search_body)

    if "error" in result:
        return {"code": "error", "error": "Failed to search tickets", "details": result.get("error")}

    hits = result.get("hits", {}).get("hits", [])
    tickets = []
    for hit in hits:
        src = hit["_source"]
        tickets.append({
            "tracking_number": src.get("tracking_number"),
            "status": src.get("status"),
            "subject": src.get("subject"),
            "site_name": src.get("site_name"),
            "customer_name": src.get("customer_name"),
            "created_at": src.get("created_at"),
            "priority": src.get("priority"),
        })

    return {
        "code": "SUCCESS",
        "count": len(tickets),
        "tickets": tickets,
    }


def support_list_open_tickets(limit: int = 20, **kwargs) -> Dict[str, Any]:
    """List all open/in-progress/waiting tickets, most recent first."""
    search_body = {
        "query": {
            "bool": {
                "should": [
                    {"term": {"status": "open"}},
                    {"term": {"status": "in_progress"}},
                    {"term": {"status": "waiting_customer"}},
                    {"term": {"status": "reopened"}},
                ]
            }
        },
        "size": min(limit, 50),
        "sort": [{"updated_at": {"order": "desc"}}],
    }
    result = _es_request("POST", f"/{ES_INDEX}/_search", search_body)

    if "error" in result:
        return {"code": "error", "error": "Failed to list tickets", "details": result.get("error")}

    hits = result.get("hits", {}).get("hits", [])
    tickets = []
    for hit in hits:
        src = hit["_source"]
        tickets.append({
            "tracking_number": src.get("tracking_number"),
            "status": src.get("status"),
            "subject": src.get("subject"),
            "site_name": src.get("site_name"),
            "customer_name": src.get("customer_name"),
            "updated_at": src.get("updated_at"),
            "priority": src.get("priority"),
            "escalated": src.get("escalated", False),
        })

    return {
        "code": "SUCCESS",
        "count": len(tickets),
        "tickets": tickets,
    }


# ============================================================
# Tool Definitions (Schemas)
# ============================================================

from tools.registry import registry


def _check_ticket_requirements() -> bool:
    """Check if ticket tools are available."""
    return bool(ES_URL) and bool(ES_API_KEY)


_CREATE_TICKET_SCHEMA = {
    "name": "support_create_ticket",
    "description": (
        "Create a new support ticket with a tracking number. "
        "Call this when a customer reports an issue, question, or request. "
        "Returns a tracking number like BYRD-20260721-0001 that you should "
        "share with the customer so they can reference it later. "
        "NEVER close tickets — only set status to 'resolved' when the customer "
        "confirms the issue is fixed. Brandon closes tickets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "customer_name": {"type": "string", "description": "Customer's full name"},
            "customer_email": {"type": "string", "description": "Customer's email address (if known)"},
            "customer_telegram_id": {"type": "string", "description": "Customer's Telegram user ID (if known)"},
            "site_name": {"type": "string", "description": "UniFi site name if applicable (e.g., 'Byrd_Ranch_Gate')"},
            "subject": {"type": "string", "description": "Short summary of the issue (e.g., 'Main gate won't open')"},
            "description": {"type": "string", "description": "Detailed description of what the customer reported"},
            "category": {
                "type": "string",
                "enum": ["network", "protect", "access", "computer", "general", "question"],
                "description": "Category of the support request",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high", "urgent"],
                "description": "Priority level (default: normal)",
            },
            "channel": {
                "type": "string",
                "enum": ["email", "telegram", "web", "phone"],
                "description": "How the customer contacted us (default: email)",
            },
        },
        "required": ["customer_name", "subject", "description"],
    },
}

_UPDATE_TICKET_SCHEMA = {
    "name": "support_update_ticket",
    "description": (
        "Update an existing support ticket. Add diagnostics, actions taken, "
        "change status, or mark as resolved. "
        "Valid statuses: open, in_progress, waiting_customer, resolved, reopened. "
        "NEVER use status 'closed' — only Brandon closes tickets. "
        "Use add_history to log any significant event."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tracking_number": {"type": "string", "description": "The ticket tracking number (e.g., BYRD-20260721-0001)"},
            "status": {
                "type": "string",
                "enum": ["open", "in_progress", "waiting_customer", "resolved", "reopened"],
                "description": "New status for the ticket (optional)",
            },
            "diagnostics": {"type": "string", "description": "Diagnostic findings to add to the ticket (optional)"},
            "actions_taken": {"type": "string", "description": "Actions taken to resolve the issue (optional)"},
            "resolution": {"type": "string", "description": "How the issue was resolved (optional, set when marking resolved)"},
            "escalated": {"type": "boolean", "description": "Whether this ticket was escalated to Brandon (optional)"},
            "escalated_to": {"type": "string", "description": "Who it was escalated to (e.g., 'Brandon') (optional)"},
            "add_history": {"type": "string", "description": "Free-form entry to add to the ticket history (optional)"},
        },
        "required": ["tracking_number"],
    },
}

_GET_TICKET_SCHEMA = {
    "name": "support_get_ticket",
    "description": "Get full details of a support ticket by tracking number. Includes all diagnostics, actions, and history.",
    "parameters": {
        "type": "object",
        "properties": {
            "tracking_number": {"type": "string", "description": "The ticket tracking number"},
        },
        "required": ["tracking_number"],
    },
}

_SEARCH_TICKETS_SCHEMA = {
    "name": "support_search_tickets",
    "description": (
        "Search for tickets by customer email, Telegram ID, site name, status, or category. "
        "Use to find a customer's ticket history or check if they have an existing open ticket "
        "before creating a new one."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "customer_email": {"type": "string", "description": "Search by customer email (optional)"},
            "customer_telegram_id": {"type": "string", "description": "Search by Telegram ID (optional)"},
            "site_name": {"type": "string", "description": "Search by site name (optional)"},
            "status": {
                "type": "string",
                "enum": ["open", "in_progress", "waiting_customer", "resolved", "reopened", "closed"],
                "description": "Filter by status (optional)",
            },
            "category": {
                "type": "string",
                "enum": ["network", "protect", "access", "computer", "general", "question"],
                "description": "Filter by category (optional)",
            },
            "limit": {"type": "integer", "description": "Max results (default 10, max 50)"},
        },
        "required": [],
    },
}

_LIST_OPEN_TICKETS_SCHEMA = {
    "name": "support_list_open_tickets",
    "description": "List all open and in-progress support tickets, most recently updated first. Use to see the current workload.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max results (default 20, max 50)"},
        },
        "required": [],
    },
}


# ============================================================
# Registry — self-registering tools
# ============================================================

from toolsets import create_custom_toolset

create_custom_toolset(
    name="support-tickets",
    description="Support ticket tracking tools — create, update, search, and list tickets with tracking numbers.",
    tools=[
        "support_create_ticket",
        "support_update_ticket",
        "support_get_ticket",
        "support_search_tickets",
        "support_list_open_tickets",
    ],
)


registry.register(
    name="support_create_ticket",
    toolset="support-tickets",
    schema=_CREATE_TICKET_SCHEMA,
    handler=lambda args, **kw: support_create_ticket(
        customer_name=args["customer_name"],
        customer_email=args.get("customer_email", ""),
        customer_telegram_id=args.get("customer_telegram_id", ""),
        site_name=args.get("site_name", ""),
        subject=args["subject"],
        description=args["description"],
        category=args.get("category", "general"),
        priority=args.get("priority", "normal"),
        channel=args.get("channel", "email"),
    ),
    check_fn=_check_ticket_requirements,
    emoji="🎫",
)

registry.register(
    name="support_update_ticket",
    toolset="support-tickets",
    schema=_UPDATE_TICKET_SCHEMA,
    handler=lambda args, **kw: support_update_ticket(
        tracking_number=args["tracking_number"],
        status=args.get("status"),
        diagnostics=args.get("diagnostics"),
        actions_taken=args.get("actions_taken"),
        resolution=args.get("resolution"),
        escalated=args.get("escalated"),
        escalated_to=args.get("escalated_to"),
        add_history=args.get("add_history"),
    ),
    check_fn=_check_ticket_requirements,
    emoji="🎫",
)

registry.register(
    name="support_get_ticket",
    toolset="support-tickets",
    schema=_GET_TICKET_SCHEMA,
    handler=lambda args, **kw: support_get_ticket(tracking_number=args["tracking_number"]),
    check_fn=_check_ticket_requirements,
    emoji="🎫",
)

registry.register(
    name="support_search_tickets",
    toolset="support-tickets",
    schema=_SEARCH_TICKETS_SCHEMA,
    handler=lambda args, **kw: support_search_tickets(
        customer_email=args.get("customer_email"),
        customer_telegram_id=args.get("customer_telegram_id"),
        site_name=args.get("site_name"),
        status=args.get("status"),
        category=args.get("category"),
        limit=args.get("limit", 10),
    ),
    check_fn=_check_ticket_requirements,
    emoji="🔍",
)

registry.register(
    name="support_list_open_tickets",
    toolset="support-tickets",
    schema=_LIST_OPEN_TICKETS_SCHEMA,
    handler=lambda args, **kw: support_list_open_tickets(limit=args.get("limit", 20)),
    check_fn=_check_ticket_requirements,
    emoji="📋",
)