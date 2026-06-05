"""Customer intent escalation tools for constrained customer-service channels.

Sophie/front-office agents should use ``customer_intent_raise`` when a customer
asks for an action the customer-service route must not execute directly. Zeus or
a supervised cron then inspects pending intents, performs the real action with
privileged tools, verifies delivery, and updates the intent status.
"""
from __future__ import annotations

import json
from typing import Any

from hermes_cli import agent_core_sql as sql
from tools.registry import registry, tool_error

INTENT_METADATA_DESCRIPTION = (
    "Optional JSON metadata. Keep it generic and tenant-neutral: business_id, "
    "service_type, source_channel, client_id, external_ref, labels, notes."
)

SAFE_INTENT_TYPES = [
    "send_email",
    "formal_quote",
    "document",
    "calendar_request",
    "follow_up",
    "escalation",
    "other",
]

STATUSES = ["pending", "processing", "completed", "blocked", "cancelled"]


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields}, ensure_ascii=False, sort_keys=True)


def _err(exc: Exception | str) -> str:
    return tool_error(str(exc))


def _check_customer_intents() -> bool:
    try:
        if not sql.enabled():
            return False
        sql.psql("SELECT 1 FROM crm.customer_intents LIMIT 1;", user=_user())
        return True
    except Exception:
        return False


def _user() -> str:
    return sql.runtime_env().get("CRM_DB_RUNTIME_USER", "crm_runtime")


def _q(value: Any) -> str:
    return sql.quote_literal(value)


def _j(value: Any) -> str:
    return sql.quote_jsonb(value)


def _limit(value: Any, default: int = 20, maximum: int = 100) -> int:
    try:
        n = int(value or default)
    except Exception:
        n = default
    return max(1, min(maximum, n))


def _slug(value: str) -> str:
    return sql.slugify(value or "customer-intent")


def _intent_id(args: dict[str, Any]) -> str:
    explicit = str(args.get("intent_id") or "").strip()
    if explicit:
        return explicit
    source = str(args.get("source_ref") or args.get("conversation_ref") or args.get("summary") or "customer-intent")
    return f"intent-{_slug(source)}"


def _normalize_intent_type(value: Any) -> str:
    text = str(value or "other").strip().lower().replace("-", "_")
    return text or "other"


def _normalize_status(value: Any, *, default: str = "pending") -> str:
    text = str(value or default).strip().lower()
    if text not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    return text


def _handle_customer_intent_raise(args: dict[str, Any], **_kwargs) -> str:
    """Create or update a pending customer intent from a restricted channel."""
    try:
        raw = str(args.get("customer_request_raw") or "").strip()
        summary = str(args.get("summary") or "").strip()
        if not raw:
            raise ValueError("customer_request_raw is required")
        if not summary:
            raise ValueError("summary is required")
        intent_id = _intent_id(args)
        intent_type = _normalize_intent_type(args.get("intent_type"))
        raw_metadata = args.get("metadata")
        metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
        metadata = {
            **metadata,
            "raised_by": metadata.get("raised_by") or "customer_service",
            "sophie_acknowledgement_required": True,
        }
        row = sql.statement_one(f"""
          INSERT INTO crm.customer_intents (
            intent_id, organization_id, contact_id, opportunity_id, interaction_id,
            channel, conversation_ref, source_ref, intent_type, customer_request_raw,
            summary, required_action, priority, status, assigned_to, due_at, metadata
          ) VALUES (
            {_q(intent_id)}, {_q(args.get('organization_id'))}, {_q(args.get('contact_id'))},
            {_q(args.get('opportunity_id'))}, {_q(args.get('interaction_id'))},
            {_q(args.get('channel') or 'whatsapp')}, {_q(args.get('conversation_ref'))}, {_q(args.get('source_ref'))},
            {_q(intent_type)}, {_q(raw)}, {_q(summary)}, {_q(args.get('required_action'))},
            {_q(args.get('priority') or 'normal')}, 'pending', {_q(args.get('assigned_to') or 'zeus')},
            {_q(args.get('due_at'))}, {_j(metadata)}
          )
          ON CONFLICT (intent_id) DO UPDATE SET
            organization_id=COALESCE(EXCLUDED.organization_id, crm.customer_intents.organization_id),
            contact_id=COALESCE(EXCLUDED.contact_id, crm.customer_intents.contact_id),
            opportunity_id=COALESCE(EXCLUDED.opportunity_id, crm.customer_intents.opportunity_id),
            interaction_id=COALESCE(EXCLUDED.interaction_id, crm.customer_intents.interaction_id),
            channel=EXCLUDED.channel,
            conversation_ref=COALESCE(EXCLUDED.conversation_ref, crm.customer_intents.conversation_ref),
            source_ref=COALESCE(EXCLUDED.source_ref, crm.customer_intents.source_ref),
            intent_type=EXCLUDED.intent_type,
            customer_request_raw=EXCLUDED.customer_request_raw,
            summary=EXCLUDED.summary,
            required_action=EXCLUDED.required_action,
            priority=EXCLUDED.priority,
            status='pending',
            assigned_to=EXCLUDED.assigned_to,
            due_at=EXCLUDED.due_at,
            metadata=crm.customer_intents.metadata || EXCLUDED.metadata,
            updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(intent=row, acknowledgement="Solicitud registrada y escalada para revisión de Zeus/SitioUno.")
    except Exception as exc:
        return _err(exc)


def _handle_customer_intent_list(args: dict[str, Any], **_kwargs) -> str:
    try:
        status = _normalize_status(args.get("status"), default="pending") if args.get("status") else "pending"
        limit = _limit(args.get("limit"), default=20)
        clauses = [f"ci.status = {_q(status)}"]
        if args.get("contact_id"):
            clauses.append(f"ci.contact_id = {_q(args.get('contact_id'))}")
        if args.get("conversation_ref"):
            clauses.append(f"ci.conversation_ref = {_q(args.get('conversation_ref'))}")
        where = " AND ".join(clauses)
        rows = sql.rows(f"""
          SELECT ci.*, c.full_name AS contact_name, c.email AS contact_email, c.phone AS contact_phone,
                 o.name AS organization_name, opp.title AS opportunity_title
          FROM crm.customer_intents ci
          LEFT JOIN crm.contacts c ON c.contact_id = ci.contact_id
          LEFT JOIN crm.organizations o ON o.organization_id = ci.organization_id
          LEFT JOIN crm.opportunities opp ON opp.opportunity_id = ci.opportunity_id
          WHERE {where}
          ORDER BY ci.created_at ASC
          LIMIT {limit}
        """, user=_user())
        return _ok(count=len(rows), intents=rows)
    except Exception as exc:
        return _err(exc)


def _handle_customer_intent_update(args: dict[str, Any], **_kwargs) -> str:
    try:
        intent_id = str(args.get("intent_id") or "").strip()
        if not intent_id:
            raise ValueError("intent_id is required")
        status = _normalize_status(args.get("status")) if args.get("status") else None
        updates: list[str] = ["updated_at=now()"]
        if status:
            updates.append(f"status={_q(status)}")
            if status == "processing":
                updates.append("claimed_at=COALESCE(claimed_at, now())")
                updates.append(f"claimed_by={_q(args.get('claimed_by') or 'zeus')}")
            if status in {"completed", "blocked", "cancelled"}:
                updates.append("processed_at=COALESCE(processed_at, now())")
        if args.get("result_summary") is not None:
            updates.append(f"result_summary={_q(args.get('result_summary'))}")
        if isinstance(args.get("metadata"), dict):
            updates.append(f"metadata=metadata || {_j(args.get('metadata'))}")
        row = sql.statement_one(f"""
          UPDATE crm.customer_intents
          SET {', '.join(updates)}
          WHERE intent_id = {_q(intent_id)}
          RETURNING *
        """, user=_user())
        if not row:
            raise ValueError(f"customer intent not found: {intent_id}")
        return _ok(intent=row)
    except Exception as exc:
        return _err(exc)


def _schema(name: str, description: str, props: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required or []}}}


def _meta_props() -> dict[str, Any]:
    return {"metadata": {"type": "object", "description": INTENT_METADATA_DESCRIPTION}}


_common_props = {
    "intent_id": {"type": "string"},
    "organization_id": {"type": "string"},
    "contact_id": {"type": "string"},
    "opportunity_id": {"type": "string"},
    "interaction_id": {"type": "integer"},
    "channel": {"type": "string", "description": "whatsapp, email, voice, sms, telegram, etc."},
    "conversation_ref": {"type": "string"},
    "source_ref": {"type": "string", "description": "Provider message/call/thread id."},
    "intent_type": {"type": "string", "enum": SAFE_INTENT_TYPES},
    "customer_request_raw": {"type": "string"},
    "summary": {"type": "string"},
    "required_action": {"type": "string"},
    "priority": {"type": "string"},
    "assigned_to": {"type": "string"},
    "due_at": {"type": "string"},
    **_meta_props(),
}

registry.register(
    name="customer_intent_raise",
    toolset="customer_service",
    schema=_schema(
        "customer_intent_raise",
        "Raise a structured customer intent/escalation for Zeus/SitioUno to process asynchronously. Use this when Sophie cannot execute the requested action directly.",
        _common_props,
        ["customer_request_raw", "summary"],
    ),
    handler=_handle_customer_intent_raise,
    check_fn=_check_customer_intents,
    emoji="📌",
)

registry.register(
    name="customer_intent_list",
    toolset="customer_intents",
    schema=_schema(
        "customer_intent_list",
        "List pending or status-filtered customer intents for Zeus/supervisor processing.",
        {"status": {"type": "string", "enum": STATUSES}, "contact_id": {"type": "string"}, "conversation_ref": {"type": "string"}, "limit": {"type": "integer"}},
    ),
    handler=_handle_customer_intent_list,
    check_fn=_check_customer_intents,
    emoji="📌",
)

registry.register(
    name="customer_intent_update",
    toolset="customer_intents",
    schema=_schema(
        "customer_intent_update",
        "Update a customer intent status/result after Zeus/supervisor has processed it.",
        {"intent_id": {"type": "string"}, "status": {"type": "string", "enum": STATUSES}, "claimed_by": {"type": "string"}, "result_summary": {"type": "string"}, **_meta_props()},
        ["intent_id"],
    ),
    handler=_handle_customer_intent_update,
    check_fn=_check_customer_intents,
    emoji="📌",
)
