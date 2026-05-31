"""Agent-native CRM tools backed by Agent Core DB."""
from __future__ import annotations

import json
from typing import Any

from hermes_cli import agent_core_sql as sql
from tools.registry import registry, tool_error


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields}, ensure_ascii=False, sort_keys=True)


def _check_crm() -> bool:
    try:
        if not sql.enabled():
            return False
        sql.psql("SELECT 1;", user=sql.runtime_env().get("CRM_DB_RUNTIME_USER", "crm_runtime"))
        return True
    except Exception:
        return False


def _user() -> str:
    return sql.runtime_env().get("CRM_DB_RUNTIME_USER", "crm_runtime")


def _q(v: Any) -> str:
    return sql.quote_literal(v)


def _j(v: Any) -> str:
    return sql.quote_jsonb(v)


def _handle_org_upsert(args: dict, **_kwargs) -> str:
    try:
        name = str(args.get("name") or "").strip()
        if not name:
            raise ValueError("name is required")
        oid = args.get("organization_id") or f"org-{sql.slugify(name)}"
        row = sql.statement_one(f"""
          INSERT INTO crm.organizations (organization_id, name, domain, phone, email, website, status, metadata, created_at, updated_at)
          VALUES ({_q(oid)}, {_q(name)}, {_q(args.get('domain'))}, {_q(args.get('phone'))}, {_q(args.get('email'))}, {_q(args.get('website'))}, {_q(args.get('status') or 'active')}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (organization_id) DO UPDATE SET name=EXCLUDED.name, domain=EXCLUDED.domain, phone=EXCLUDED.phone, email=EXCLUDED.email, website=EXCLUDED.website, status=EXCLUDED.status, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(organization=row)
    except Exception as exc:
        return tool_error(str(exc))


def _handle_contact_upsert(args: dict, **_kwargs) -> str:
    try:
        full_name = str(args.get("full_name") or "").strip()
        if not full_name:
            raise ValueError("full_name is required")
        cid = args.get("contact_id") or f"contact-{sql.slugify(full_name)}"
        row = sql.statement_one(f"""
          INSERT INTO crm.contacts (contact_id, organization_id, full_name, email, phone, title, status, source, metadata, created_at, updated_at)
          VALUES ({_q(cid)}, {_q(args.get('organization_id'))}, {_q(full_name)}, {_q(args.get('email'))}, {_q(args.get('phone'))}, {_q(args.get('title'))}, {_q(args.get('status') or 'active')}, {_q(args.get('source'))}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (contact_id) DO UPDATE SET organization_id=EXCLUDED.organization_id, full_name=EXCLUDED.full_name, email=EXCLUDED.email, phone=EXCLUDED.phone, title=EXCLUDED.title, status=EXCLUDED.status, source=EXCLUDED.source, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(contact=row)
    except Exception as exc:
        return tool_error(str(exc))


def _handle_interaction_record(args: dict, **_kwargs) -> str:
    try:
        summary = str(args.get("summary") or "").strip()
        channel = str(args.get("channel") or "note").strip()
        if not summary:
            raise ValueError("summary is required")
        row = sql.statement_one(f"""
          INSERT INTO crm.interactions (organization_id, contact_id, opportunity_id, channel, direction, summary, occurred_at, actor, metadata)
          VALUES ({_q(args.get('organization_id'))}, {_q(args.get('contact_id'))}, {_q(args.get('opportunity_id'))}, {_q(channel)}, {_q(args.get('direction') or 'note')}, {_q(summary)}, COALESCE({_q(args.get('occurred_at'))}::timestamptz, now()), {_q(args.get('actor'))}, {_j(args.get('metadata') or {})})
          RETURNING *
        """, user=_user())
        return _ok(interaction=row)
    except Exception as exc:
        return tool_error(str(exc))


def _handle_crm_search(args: dict, **_kwargs) -> str:
    try:
        query = str(args.get("query") or "").strip()
        limit = int(args.get("limit") or 20)
        pattern = f"%{query}%"
        contacts = sql.rows(f"SELECT * FROM crm.contacts WHERE ({_q(query)} = '' OR full_name ILIKE {_q(pattern)} OR email ILIKE {_q(pattern)} OR phone ILIKE {_q(pattern)}) ORDER BY updated_at DESC LIMIT {limit}", user=_user())
        orgs = sql.rows(f"SELECT * FROM crm.organizations WHERE ({_q(query)} = '' OR name ILIKE {_q(pattern)} OR domain ILIKE {_q(pattern)} OR email ILIKE {_q(pattern)}) ORDER BY updated_at DESC LIMIT {limit}", user=_user())
        return _ok(contacts=contacts, organizations=orgs)
    except Exception as exc:
        return tool_error(str(exc))


def _handle_crm_status(args: dict, **_kwargs) -> str:
    try:
        counts = sql.one("SELECT (SELECT count(*) FROM crm.organizations) AS organizations, (SELECT count(*) FROM crm.contacts) AS contacts, (SELECT count(*) FROM crm.opportunities) AS opportunities, (SELECT count(*) FROM crm.interactions) AS interactions", user=_user())
        return _ok(db_backend="agent_core_postgres", counts=counts)
    except Exception as exc:
        return tool_error(str(exc))


def _schema(name: str, description: str, props: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required or []}}}


registry.register(name="crm_organization_upsert", toolset="crm", schema=_schema("crm_organization_upsert", "Create or update a CRM organization in Agent Core DB.", {"organization_id": {"type": "string"}, "name": {"type": "string"}, "domain": {"type": "string"}, "phone": {"type": "string"}, "email": {"type": "string"}, "website": {"type": "string"}, "status": {"type": "string"}, "metadata": {"type": "object"}}, ["name"]), handler=_handle_org_upsert, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_contact_upsert", toolset="crm", schema=_schema("crm_contact_upsert", "Create or update a CRM contact in Agent Core DB.", {"contact_id": {"type": "string"}, "organization_id": {"type": "string"}, "full_name": {"type": "string"}, "email": {"type": "string"}, "phone": {"type": "string"}, "title": {"type": "string"}, "status": {"type": "string"}, "source": {"type": "string"}, "metadata": {"type": "object"}}, ["full_name"]), handler=_handle_contact_upsert, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_interaction_record", toolset="crm", schema=_schema("crm_interaction_record", "Record a CRM interaction/note/call/message.", {"organization_id": {"type": "string"}, "contact_id": {"type": "string"}, "opportunity_id": {"type": "string"}, "channel": {"type": "string"}, "direction": {"type": "string"}, "summary": {"type": "string"}, "occurred_at": {"type": "string"}, "actor": {"type": "string"}, "metadata": {"type": "object"}}, ["summary"]), handler=_handle_interaction_record, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_search", toolset="crm", schema=_schema("crm_search", "Search CRM contacts and organizations.", {"query": {"type": "string"}, "limit": {"type": "integer"}}), handler=_handle_crm_search, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_status", toolset="crm", schema=_schema("crm_status", "Return CRM module row counts and DB backend.", {}), handler=_handle_crm_status, check_fn=_check_crm, emoji="👥")
