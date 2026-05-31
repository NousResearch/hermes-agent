"""Agent-native CRM tools backed by Agent Core DB, with optional Twenty adapter.

CRM Core is the canonical local source for Zeus-style agents.  External CRMs
(Twenty first, Odoo/Lago/etc. later) are adapters linked through
``crm.external_links``; agents should use these generic tools instead of binding
conversation logic directly to vendor object models.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from hermes_cli import agent_core_sql as sql
from tools.registry import registry, tool_error


CRM_METADATA_DESCRIPTION = (
    "Optional JSON metadata. Keep it generic and tenant-neutral: business_id, "
    "owner_id, source_channel, external_ref, labels, notes."
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields}, ensure_ascii=False, sort_keys=True)


def _err(exc: Exception | str) -> str:
    return tool_error(str(exc))


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


def _slug(prefix: str, value: str) -> str:
    return f"{prefix}-{sql.slugify(value)}"


def _num(v: Any, default: str = "NULL") -> str:
    if v is None or v == "":
        return default
    try:
        # Normalize to a PostgreSQL numeric literal and reject arbitrary text so
        # user-provided numbers cannot become SQL fragments.
        return repr(float(v))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value: {v!r}")


def _limit(v: Any, default: int = 20, maximum: int = 100) -> int:
    try:
        n = int(v or default)
    except Exception:
        n = default
    return max(1, min(maximum, n))


def _status_clause(status: str | None, alias: str = "") -> str:
    if not status:
        return "TRUE"
    prefix = f"{alias}." if alias else ""
    return f"{prefix}status = {_q(status)}"


def _link_external(local_type: str, local_id: str, provider: str, external_type: str,
                   external_id: str, external_url: str | None = None,
                   metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    return sql.statement_one(f"""
      INSERT INTO crm.external_links (local_type, local_id, provider, external_type, external_id, external_url, sync_status, last_synced_at, metadata)
      VALUES ({_q(local_type)}, {_q(local_id)}, {_q(provider)}, {_q(external_type)}, {_q(external_id)}, {_q(external_url)}, 'linked', now(), {_j(metadata or {})})
      ON CONFLICT (local_type, local_id, provider, external_type)
      DO UPDATE SET external_id=EXCLUDED.external_id, external_url=EXCLUDED.external_url, sync_status='linked', last_synced_at=now(), metadata=EXCLUDED.metadata, updated_at=now()
      RETURNING *
    """, user=_user())


# ---------------------------------------------------------------------------
# Twenty adapter helpers
# ---------------------------------------------------------------------------

def _twenty_base_url() -> str:
    env = sql.runtime_env()
    return (env.get("TWENTY_BASE_URL") or env.get("CRM_TWENTY_BASE_URL") or os.getenv("TWENTY_BASE_URL") or "").rstrip("/")


def _twenty_api_key() -> str:
    env = sql.runtime_env()
    return env.get("TWENTY_API_KEY") or env.get("CRM_TWENTY_API_KEY") or os.getenv("TWENTY_API_KEY") or ""


def _twenty_enabled() -> bool:
    return bool(_twenty_base_url() and _twenty_api_key())


def _twenty_request(method: str, path: str, body: dict[str, Any] | None = None,
                    query: dict[str, Any] | None = None) -> dict[str, Any]:
    base = _twenty_base_url()
    key = _twenty_api_key()
    if not base or not key:
        return {"ok": False, "configured": False, "error": "Twenty adapter is not configured. Set TWENTY_BASE_URL and TWENTY_API_KEY via Infisical/runtime env."}

    url = f"{base}/{path.lstrip('/')}"
    if query:
        clean = {k: v for k, v in query.items() if v is not None}
        if clean:
            url += "?" + urllib.parse.urlencode(clean)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"}
    req = urllib.request.Request(url=url, method=method.upper(), data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            payload = json.loads(raw) if raw else None
            return {"ok": True, "status": resp.status, "url": url, "data": payload}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload: Any = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = raw
        return {"ok": False, "status": exc.code, "url": url, "error": payload}
    except Exception as exc:
        return {"ok": False, "url": url, "error": str(exc)}


def _twenty_record_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("id", "recordId"):
            if payload.get(key):
                return str(payload[key])
        for key in ("data", "createCompany", "updateCompany", "createPerson", "updatePerson", "createOpportunity", "updateOpportunity"):
            if key in payload:
                found = _twenty_record_id(payload[key])
                if found:
                    return found
        for value in payload.values():
            if isinstance(value, dict):
                found = _twenty_record_id(value)
                if found:
                    return found
    return None

def _twenty_link(value: Any) -> dict[str, Any] | None:
    if not value:
        return None
    url = str(value).strip()
    if not url:
        return None
    if "://" not in url:
        url = f"https://{url}"
    return {"primaryLinkLabel": "", "primaryLinkUrl": url, "secondaryLinks": []}


def _sync_twenty(local_type: str, local_id: str, external_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a record in Twenty and store the external link.

    Twenty generates schema-per-tenant REST endpoints; the object path is passed
    in as plural lower-case names such as ``companies`` or ``people``.
    """
    existing = sql.one(
        f"SELECT * FROM crm.external_links WHERE local_type={_q(local_type)} AND local_id={_q(local_id)} AND provider='twenty' AND external_type={_q(external_type)}",
        user=_user(),
    )
    if existing and existing.get("external_id"):
        result = _twenty_request("PATCH", f"/rest/{external_type}/{existing['external_id']}", payload)
        external_id = existing["external_id"]
    else:
        result = _twenty_request("POST", f"/rest/{external_type}", payload)
        external_id = _twenty_record_id(result.get("data"))

    if result.get("ok") and external_id:
        _link_external(local_type, local_id, "twenty", external_type, external_id, metadata={"payload": payload})
    return {"adapter": "twenty", "external_type": external_type, "external_id": external_id, "result": result}


# ---------------------------------------------------------------------------
# Core CRM handlers
# ---------------------------------------------------------------------------

def _handle_org_upsert(args: dict, **_kwargs) -> str:
    try:
        name = str(args.get("name") or "").strip()
        if not name:
            raise ValueError("name is required")
        oid = args.get("organization_id") or _slug("org", args.get("domain") or name)
        row = sql.statement_one(f"""
          INSERT INTO crm.organizations (organization_id, name, domain, phone, email, website, status, metadata, created_at, updated_at)
          VALUES ({_q(oid)}, {_q(name)}, {_q(args.get('domain'))}, {_q(args.get('phone'))}, {_q(args.get('email'))}, {_q(args.get('website'))}, {_q(args.get('status') or 'active')}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (organization_id) DO UPDATE SET name=EXCLUDED.name, domain=EXCLUDED.domain, phone=EXCLUDED.phone, email=EXCLUDED.email, website=EXCLUDED.website, status=EXCLUDED.status, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        sync = None
        if args.get("sync_twenty"):
            sync = _sync_twenty("organization", oid, "companies", {
                "name": name,
                "domainName": _twenty_link(args.get("website") or args.get("domain")),
                "address": args.get("metadata", {}).get("address") if isinstance(args.get("metadata"), dict) else None,
            })
        return _ok(organization=row, twenty=sync)
    except Exception as exc:
        return _err(exc)


def _handle_contact_upsert(args: dict, **_kwargs) -> str:
    try:
        full_name = str(args.get("full_name") or "").strip()
        if not full_name:
            raise ValueError("full_name is required")
        cid = args.get("contact_id") or _slug("contact", args.get("email") or full_name)
        row = sql.statement_one(f"""
          INSERT INTO crm.contacts (contact_id, organization_id, full_name, email, phone, title, status, source, metadata, created_at, updated_at)
          VALUES ({_q(cid)}, {_q(args.get('organization_id'))}, {_q(full_name)}, {_q(args.get('email'))}, {_q(args.get('phone'))}, {_q(args.get('title'))}, {_q(args.get('status') or 'active')}, {_q(args.get('source'))}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (contact_id) DO UPDATE SET organization_id=EXCLUDED.organization_id, full_name=EXCLUDED.full_name, email=EXCLUDED.email, phone=EXCLUDED.phone, title=EXCLUDED.title, status=EXCLUDED.status, source=EXCLUDED.source, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        sync = None
        if args.get("sync_twenty"):
            first, _, last = full_name.partition(" ")
            sync = _sync_twenty("contact", cid, "people", {
                "name": {"firstName": first, "lastName": last},
                "emails": {"primaryEmail": args.get("email")} if args.get("email") else None,
                "phones": {"primaryPhoneNumber": args.get("phone")} if args.get("phone") else None,
                "jobTitle": args.get("title"),
            })
        return _ok(contact=row, twenty=sync)
    except Exception as exc:
        return _err(exc)


def _handle_opportunity_upsert(args: dict, **_kwargs) -> str:
    try:
        title = str(args.get("title") or "").strip()
        if not title:
            raise ValueError("title is required")
        oid = args.get("opportunity_id") or _slug("opp", f"{args.get('organization_id') or args.get('contact_id') or ''}-{title}")
        row = sql.statement_one(f"""
          INSERT INTO crm.opportunities (opportunity_id, organization_id, contact_id, title, stage, value_amount, currency, expected_close_date, status, metadata, created_at, updated_at)
          VALUES ({_q(oid)}, {_q(args.get('organization_id'))}, {_q(args.get('contact_id'))}, {_q(title)}, {_q(args.get('stage') or 'lead')}, {_num(args.get('value_amount'))}, {_q(args.get('currency') or 'USD')}, {_q(args.get('expected_close_date'))}::date, {_q(args.get('status') or 'open')}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (opportunity_id) DO UPDATE SET organization_id=EXCLUDED.organization_id, contact_id=EXCLUDED.contact_id, title=EXCLUDED.title, stage=EXCLUDED.stage, value_amount=EXCLUDED.value_amount, currency=EXCLUDED.currency, expected_close_date=EXCLUDED.expected_close_date, status=EXCLUDED.status, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        sync = None
        if args.get("sync_twenty"):
            local_stage = str(args.get("stage") or "lead").strip().lower()
            twenty_stage = {
                "lead": "NEW",
                "new": "NEW",
                "qualified": "SCREENING",
                "screening": "SCREENING",
                "meeting": "MEETING",
                "proposal": "PROPOSAL",
                "negotiation": "PROPOSAL",
                "won": "CUSTOMER",
                "customer": "CUSTOMER",
            }.get(local_stage, local_stage.upper())
            sync = _sync_twenty("opportunity", oid, "opportunities", {
                "name": title,
                "amount": {"amountMicros": int(float(args.get("value_amount") or 0) * 1_000_000), "currencyCode": args.get("currency") or "USD"},
                "stage": twenty_stage,
            })
        return _ok(opportunity=row, twenty=sync)
    except Exception as exc:
        return _err(exc)


def _handle_product_upsert(args: dict, **_kwargs) -> str:
    try:
        name = str(args.get("name") or "").strip()
        if not name:
            raise ValueError("name is required")
        pid = args.get("product_id") or _slug("product", args.get("sku") or name)
        row = sql.statement_one(f"""
          INSERT INTO crm.products (product_id, sku, name, description, unit_price, currency, status, metadata, created_at, updated_at)
          VALUES ({_q(pid)}, {_q(args.get('sku'))}, {_q(name)}, {_q(args.get('description'))}, {_num(args.get('unit_price'))}, {_q(args.get('currency') or 'USD')}, {_q(args.get('status') or 'active')}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (product_id) DO UPDATE SET sku=EXCLUDED.sku, name=EXCLUDED.name, description=EXCLUDED.description, unit_price=EXCLUDED.unit_price, currency=EXCLUDED.currency, status=EXCLUDED.status, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(product=row)
    except Exception as exc:
        return _err(exc)


def _quote_totals(items: list[dict[str, Any]]) -> tuple[float, float, float]:
    subtotal = 0.0
    tax_amount = 0.0
    for item in items:
        qty = float(item.get("quantity") or 1)
        price = float(item.get("unit_price") or 0)
        tax_rate = float(item.get("tax_rate") or 0)
        line = qty * price
        subtotal += line
        tax_amount += line * tax_rate
    return subtotal, tax_amount, subtotal + tax_amount


def _handle_quote_create(args: dict, **_kwargs) -> str:
    try:
        title = str(args.get("title") or "").strip()
        if not title:
            raise ValueError("title is required")
        items = args.get("items") or []
        if not isinstance(items, list):
            raise ValueError("items must be a list")
        qid = args.get("quote_id") or _slug("quote", f"{args.get('organization_id') or args.get('contact_id') or ''}-{title}")
        subtotal, tax_amount, total = _quote_totals(items)
        quote = sql.statement_one(f"""
          INSERT INTO crm.quotes (quote_id, organization_id, contact_id, opportunity_id, title, status, valid_until, currency, subtotal, tax_amount, total, metadata, created_at, updated_at)
          VALUES ({_q(qid)}, {_q(args.get('organization_id'))}, {_q(args.get('contact_id'))}, {_q(args.get('opportunity_id'))}, {_q(title)}, {_q(args.get('status') or 'draft')}, {_q(args.get('valid_until'))}::date, {_q(args.get('currency') or 'USD')}, {subtotal}, {tax_amount}, {total}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (quote_id) DO UPDATE SET organization_id=EXCLUDED.organization_id, contact_id=EXCLUDED.contact_id, opportunity_id=EXCLUDED.opportunity_id, title=EXCLUDED.title, status=EXCLUDED.status, valid_until=EXCLUDED.valid_until, currency=EXCLUDED.currency, subtotal=EXCLUDED.subtotal, tax_amount=EXCLUDED.tax_amount, total=EXCLUDED.total, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        sql.psql(f"DELETE FROM crm.quote_items WHERE quote_id={_q(qid)};", user=_user())
        saved_items = []
        for item in items:
            qty = float(item.get("quantity") or 1)
            price = float(item.get("unit_price") or 0)
            tax_rate = float(item.get("tax_rate") or 0)
            line_total = qty * price * (1 + tax_rate)
            saved_items.append(sql.statement_one(f"""
              INSERT INTO crm.quote_items (quote_id, product_id, description, quantity, unit_price, tax_rate, line_total, metadata)
              VALUES ({_q(qid)}, {_q(item.get('product_id'))}, {_q(item.get('description') or item.get('name') or 'Item')}, {qty}, {price}, {tax_rate}, {line_total}, {_j(item.get('metadata') or {})})
              RETURNING *
            """, user=_user()))
        return _ok(quote=quote, items=saved_items)
    except Exception as exc:
        return _err(exc)


def _handle_invoice_create(args: dict, **_kwargs) -> str:
    try:
        title = str(args.get("title") or "").strip()
        if not title:
            quote_id = args.get("quote_id")
            if quote_id:
                qrow = sql.one(f"SELECT * FROM crm.quotes WHERE quote_id={_q(quote_id)}", user=_user())
                title = qrow.get("title") if qrow else "Invoice"
            else:
                title = "Invoice"
        iid = args.get("invoice_id") or _slug("invoice", f"{args.get('quote_id') or args.get('organization_id') or ''}-{title}")
        quote = sql.one(f"SELECT * FROM crm.quotes WHERE quote_id={_q(args.get('quote_id'))}", user=_user()) if args.get("quote_id") else None
        organization_id = args.get("organization_id") or (quote or {}).get("organization_id")
        contact_id = args.get("contact_id") or (quote or {}).get("contact_id")
        subtotal = args.get("subtotal", (quote or {}).get("subtotal", 0))
        tax_amount = args.get("tax_amount", (quote or {}).get("tax_amount", 0))
        total = args.get("total", (quote or {}).get("total", 0))
        row = sql.statement_one(f"""
          INSERT INTO crm.invoices (invoice_id, quote_id, organization_id, contact_id, title, status, issue_date, due_date, currency, subtotal, tax_amount, total, metadata, created_at, updated_at)
          VALUES ({_q(iid)}, {_q(args.get('quote_id'))}, {_q(organization_id)}, {_q(contact_id)}, {_q(title)}, {_q(args.get('status') or 'draft')}, COALESCE({_q(args.get('issue_date'))}::date, CURRENT_DATE), {_q(args.get('due_date'))}::date, {_q(args.get('currency') or (quote or {}).get('currency') or 'USD')}, {_num(subtotal, '0')}, {_num(tax_amount, '0')}, {_num(total, '0')}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (invoice_id) DO UPDATE SET quote_id=EXCLUDED.quote_id, organization_id=EXCLUDED.organization_id, contact_id=EXCLUDED.contact_id, title=EXCLUDED.title, status=EXCLUDED.status, issue_date=EXCLUDED.issue_date, due_date=EXCLUDED.due_date, currency=EXCLUDED.currency, subtotal=EXCLUDED.subtotal, tax_amount=EXCLUDED.tax_amount, total=EXCLUDED.total, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(invoice=row)
    except Exception as exc:
        return _err(exc)


def _handle_relationship_upsert(args: dict, **_kwargs) -> str:
    try:
        for field in ("source_type", "source_id", "target_type", "target_id", "relationship_type"):
            if not args.get(field):
                raise ValueError(f"{field} is required")
        rid = args.get("relationship_id") or _slug("rel", f"{args['source_type']}-{args['source_id']}-{args['relationship_type']}-{args['target_type']}-{args['target_id']}")
        row = sql.statement_one(f"""
          INSERT INTO crm.relationships (relationship_id, source_type, source_id, target_type, target_id, relationship_type, strength, status, notes, metadata, created_at, updated_at)
          VALUES ({_q(rid)}, {_q(args.get('source_type'))}, {_q(args.get('source_id'))}, {_q(args.get('target_type'))}, {_q(args.get('target_id'))}, {_q(args.get('relationship_type'))}, {_num(args.get('strength'))}, {_q(args.get('status') or 'active')}, {_q(args.get('notes'))}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (source_type, source_id, target_type, target_id, relationship_type) DO UPDATE SET strength=EXCLUDED.strength, status=EXCLUDED.status, notes=EXCLUDED.notes, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(relationship=row)
    except Exception as exc:
        return _err(exc)


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
        follow_up = None
        if args.get("follow_up_at") and args.get("follow_up_summary"):
            follow_up = sql.statement_one(f"""
              INSERT INTO crm.follow_ups (organization_id, contact_id, opportunity_id, due_at, summary, priority, assignee, metadata)
              VALUES ({_q(args.get('organization_id'))}, {_q(args.get('contact_id'))}, {_q(args.get('opportunity_id'))}, {_q(args.get('follow_up_at'))}::timestamptz, {_q(args.get('follow_up_summary'))}, {_q(args.get('follow_up_priority') or 'normal')}, {_q(args.get('actor'))}, {_j({'source_interaction_id': row.get('interaction_id') if row else None})})
              RETURNING *
            """, user=_user())
        return _ok(interaction=row, follow_up=follow_up)
    except Exception as exc:
        return _err(exc)


def _handle_follow_up_create(args: dict, **_kwargs) -> str:
    try:
        if not args.get("due_at") or not args.get("summary"):
            raise ValueError("due_at and summary are required")
        row = sql.statement_one(f"""
          INSERT INTO crm.follow_ups (organization_id, contact_id, opportunity_id, due_at, summary, status, priority, assignee, metadata, created_at, updated_at)
          VALUES ({_q(args.get('organization_id'))}, {_q(args.get('contact_id'))}, {_q(args.get('opportunity_id'))}, {_q(args.get('due_at'))}::timestamptz, {_q(args.get('summary'))}, {_q(args.get('status') or 'open')}, {_q(args.get('priority') or 'normal')}, {_q(args.get('assignee'))}, {_j(args.get('metadata') or {})}, now(), now())
          RETURNING *
        """, user=_user())
        return _ok(follow_up=row)
    except Exception as exc:
        return _err(exc)


def _handle_crm_search(args: dict, **_kwargs) -> str:
    try:
        query = str(args.get("query") or "").strip()
        limit = _limit(args.get("limit"))
        pattern = f"%{query}%"
        contacts = sql.rows(f"SELECT * FROM crm.contacts WHERE ({_q(query)} = '' OR full_name ILIKE {_q(pattern)} OR email ILIKE {_q(pattern)} OR phone ILIKE {_q(pattern)}) ORDER BY updated_at DESC LIMIT {limit}", user=_user())
        orgs = sql.rows(f"SELECT * FROM crm.organizations WHERE ({_q(query)} = '' OR name ILIKE {_q(pattern)} OR domain ILIKE {_q(pattern)} OR email ILIKE {_q(pattern)}) ORDER BY updated_at DESC LIMIT {limit}", user=_user())
        opportunities = sql.rows(f"SELECT * FROM crm.opportunities WHERE ({_q(query)} = '' OR title ILIKE {_q(pattern)}) ORDER BY updated_at DESC LIMIT {limit}", user=_user())
        products = sql.rows(f"SELECT * FROM crm.products WHERE ({_q(query)} = '' OR name ILIKE {_q(pattern)} OR sku ILIKE {_q(pattern)}) ORDER BY updated_at DESC LIMIT {limit}", user=_user())
        return _ok(contacts=contacts, organizations=orgs, opportunities=opportunities, products=products)
    except Exception as exc:
        return _err(exc)


def _handle_customer_timeline(args: dict, **_kwargs) -> str:
    try:
        org = args.get("organization_id")
        contact = args.get("contact_id")
        opportunity = args.get("opportunity_id")
        if not any([org, contact, opportunity]):
            raise ValueError("organization_id, contact_id, or opportunity_id is required")
        limit = _limit(args.get("limit"), default=50, maximum=200)
        where = []
        if org:
            where.append(f"organization_id={_q(org)}")
        if contact:
            where.append(f"contact_id={_q(contact)}")
        if opportunity:
            where.append(f"opportunity_id={_q(opportunity)}")
        condition = " OR ".join(where)
        interactions = sql.rows(f"SELECT * FROM crm.interactions WHERE {condition} ORDER BY occurred_at DESC LIMIT {limit}", user=_user())
        opportunities = sql.rows(f"SELECT * FROM crm.opportunities WHERE {condition} ORDER BY updated_at DESC LIMIT {limit}", user=_user())
        quotes = sql.rows(f"SELECT * FROM crm.quotes WHERE {condition} ORDER BY updated_at DESC LIMIT {limit}", user=_user())
        invoices = sql.rows(f"SELECT * FROM crm.invoices WHERE ({' OR '.join([w for w in where if 'opportunity_id' not in w]) or 'FALSE'}) ORDER BY updated_at DESC LIMIT {limit}", user=_user())
        follow_ups = sql.rows(f"SELECT * FROM crm.follow_ups WHERE {condition} ORDER BY due_at ASC LIMIT {limit}", user=_user())
        relationships = sql.rows(f"SELECT * FROM crm.relationships WHERE (source_id IN ({_q(org)}, {_q(contact)}, {_q(opportunity)}) OR target_id IN ({_q(org)}, {_q(contact)}, {_q(opportunity)})) ORDER BY updated_at DESC LIMIT {limit}", user=_user())
        return _ok(interactions=interactions, opportunities=opportunities, quotes=quotes, invoices=invoices, follow_ups=follow_ups, relationships=relationships)
    except Exception as exc:
        return _err(exc)


def _handle_crm_status(args: dict, **_kwargs) -> str:
    try:
        counts = sql.one("""
          SELECT
            (SELECT count(*) FROM crm.organizations) AS organizations,
            (SELECT count(*) FROM crm.contacts) AS contacts,
            (SELECT count(*) FROM crm.opportunities) AS opportunities,
            (SELECT count(*) FROM crm.interactions) AS interactions,
            (SELECT count(*) FROM crm.relationships) AS relationships,
            (SELECT count(*) FROM crm.products) AS products,
            (SELECT count(*) FROM crm.quotes) AS quotes,
            (SELECT count(*) FROM crm.invoices) AS invoices,
            (SELECT count(*) FROM crm.follow_ups WHERE status='open') AS open_follow_ups,
            (SELECT count(*) FROM crm.external_links) AS external_links
        """, user=_user())
        return _ok(db_backend="agent_core_postgres", counts=counts, adapters={"twenty": {"configured": _twenty_enabled(), "base_url": _twenty_base_url() or None}})
    except Exception as exc:
        return _err(exc)


def _handle_twenty_sync(args: dict, **_kwargs) -> str:
    try:
        local_type = str(args.get("local_type") or "")
        local_id = str(args.get("local_id") or "")
        if not local_id:
            raise ValueError("local_id is required")
        if local_type == "organization":
            row = sql.one(f"SELECT * FROM crm.organizations WHERE organization_id={_q(local_id)}", user=_user())
            if not row:
                raise ValueError("organization not found")
            sync = _sync_twenty("organization", local_id, "companies", {"name": row["name"], "domainName": row.get("domain") or row.get("website")})
        elif local_type == "contact":
            row = sql.one(f"SELECT * FROM crm.contacts WHERE contact_id={_q(local_id)}", user=_user())
            if not row:
                raise ValueError("contact not found")
            first, _, last = row["full_name"].partition(" ")
            sync = _sync_twenty("contact", local_id, "people", {"name": {"firstName": first, "lastName": last}, "emails": {"primaryEmail": row.get("email")} if row.get("email") else None, "phones": {"primaryPhoneNumber": row.get("phone")} if row.get("phone") else None, "jobTitle": row.get("title")})
        elif local_type == "opportunity":
            row = sql.one(f"SELECT * FROM crm.opportunities WHERE opportunity_id={_q(local_id)}", user=_user())
            if not row:
                raise ValueError("opportunity not found")
            sync = _sync_twenty("opportunity", local_id, "opportunities", {"name": row["title"], "stage": row.get("stage")})
        else:
            raise ValueError("local_type must be organization, contact, or opportunity")
        return _ok(sync=sync)
    except Exception as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Tool schemas / registration
# ---------------------------------------------------------------------------

def _schema(name: str, description: str, props: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required or []}}}


def _meta_props() -> dict[str, Any]:
    return {"metadata": {"type": "object", "description": CRM_METADATA_DESCRIPTION}}


registry.register(name="crm_status", toolset="crm", schema=_schema("crm_status", "Return CRM Core row counts, DB backend, and adapter configuration status.", {}), handler=_handle_crm_status, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_organization_upsert", toolset="crm", schema=_schema("crm_organization_upsert", "Create or update a CRM organization/company in Agent Core DB. Optionally sync to Twenty.", {"organization_id": {"type": "string"}, "name": {"type": "string"}, "domain": {"type": "string"}, "phone": {"type": "string"}, "email": {"type": "string"}, "website": {"type": "string"}, "status": {"type": "string"}, "sync_twenty": {"type": "boolean"}, **_meta_props()}, ["name"]), handler=_handle_org_upsert, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_contact_upsert", toolset="crm", schema=_schema("crm_contact_upsert", "Create or update a CRM contact/person. Use organization_id to attach it to a company. Optionally sync to Twenty.", {"contact_id": {"type": "string"}, "organization_id": {"type": "string"}, "full_name": {"type": "string"}, "email": {"type": "string"}, "phone": {"type": "string"}, "title": {"type": "string"}, "status": {"type": "string"}, "source": {"type": "string"}, "sync_twenty": {"type": "boolean"}, **_meta_props()}, ["full_name"]), handler=_handle_contact_upsert, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_opportunity_upsert", toolset="crm", schema=_schema("crm_opportunity_upsert", "Create or update a sales opportunity/pipeline record.", {"opportunity_id": {"type": "string"}, "organization_id": {"type": "string"}, "contact_id": {"type": "string"}, "title": {"type": "string"}, "stage": {"type": "string"}, "value_amount": {"type": "number"}, "currency": {"type": "string"}, "expected_close_date": {"type": "string", "description": "YYYY-MM-DD"}, "status": {"type": "string"}, "sync_twenty": {"type": "boolean"}, **_meta_props()}, ["title"]), handler=_handle_opportunity_upsert, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_product_upsert", toolset="crm", schema=_schema("crm_product_upsert", "Create or update a product/service catalog item for quotes and invoices.", {"product_id": {"type": "string"}, "sku": {"type": "string"}, "name": {"type": "string"}, "description": {"type": "string"}, "unit_price": {"type": "number"}, "currency": {"type": "string"}, "status": {"type": "string"}, **_meta_props()}, ["name"]), handler=_handle_product_upsert, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_quote_create", toolset="crm", schema=_schema("crm_quote_create", "Create or replace a quote with line items. Totals are computed from items.", {"quote_id": {"type": "string"}, "organization_id": {"type": "string"}, "contact_id": {"type": "string"}, "opportunity_id": {"type": "string"}, "title": {"type": "string"}, "status": {"type": "string"}, "valid_until": {"type": "string", "description": "YYYY-MM-DD"}, "currency": {"type": "string"}, "items": {"type": "array", "items": {"type": "object"}}, **_meta_props()}, ["title"]), handler=_handle_quote_create, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_invoice_create", toolset="crm", schema=_schema("crm_invoice_create", "Create or update an invoice, optionally from an existing quote.", {"invoice_id": {"type": "string"}, "quote_id": {"type": "string"}, "organization_id": {"type": "string"}, "contact_id": {"type": "string"}, "title": {"type": "string"}, "status": {"type": "string"}, "issue_date": {"type": "string", "description": "YYYY-MM-DD"}, "due_date": {"type": "string", "description": "YYYY-MM-DD"}, "currency": {"type": "string"}, "subtotal": {"type": "number"}, "tax_amount": {"type": "number"}, "total": {"type": "number"}, **_meta_props()}), handler=_handle_invoice_create, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_relationship_upsert", toolset="crm", schema=_schema("crm_relationship_upsert", "Record a relationship edge between CRM objects (e.g. partner, decision-maker, competitor, owner, referral).", {"relationship_id": {"type": "string"}, "source_type": {"type": "string"}, "source_id": {"type": "string"}, "target_type": {"type": "string"}, "target_id": {"type": "string"}, "relationship_type": {"type": "string"}, "strength": {"type": "number"}, "status": {"type": "string"}, "notes": {"type": "string"}, **_meta_props()}, ["source_type", "source_id", "target_type", "target_id", "relationship_type"]), handler=_handle_relationship_upsert, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_interaction_record", toolset="crm", schema=_schema("crm_interaction_record", "Record a CRM interaction/note/call/message and optionally schedule a follow-up.", {"organization_id": {"type": "string"}, "contact_id": {"type": "string"}, "opportunity_id": {"type": "string"}, "channel": {"type": "string"}, "direction": {"type": "string"}, "summary": {"type": "string"}, "occurred_at": {"type": "string"}, "actor": {"type": "string"}, "follow_up_at": {"type": "string"}, "follow_up_summary": {"type": "string"}, "follow_up_priority": {"type": "string"}, **_meta_props()}, ["summary"]), handler=_handle_interaction_record, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_follow_up_create", toolset="crm", schema=_schema("crm_follow_up_create", "Create a follow-up task tied to a contact, organization, or opportunity.", {"organization_id": {"type": "string"}, "contact_id": {"type": "string"}, "opportunity_id": {"type": "string"}, "due_at": {"type": "string"}, "summary": {"type": "string"}, "status": {"type": "string"}, "priority": {"type": "string"}, "assignee": {"type": "string"}, **_meta_props()}, ["due_at", "summary"]), handler=_handle_follow_up_create, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_customer_timeline", toolset="crm", schema=_schema("crm_customer_timeline", "Return the relationship timeline for a customer/contact/opportunity: interactions, opportunities, quotes, invoices, follow-ups, relationships.", {"organization_id": {"type": "string"}, "contact_id": {"type": "string"}, "opportunity_id": {"type": "string"}, "limit": {"type": "integer"}}), handler=_handle_customer_timeline, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_search", toolset="crm", schema=_schema("crm_search", "Search CRM contacts, organizations, opportunities, and products.", {"query": {"type": "string"}, "limit": {"type": "integer"}}), handler=_handle_crm_search, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_twenty_sync", toolset="crm", schema=_schema("crm_twenty_sync", "Sync a local organization/contact/opportunity to the configured Twenty workspace and store the external link.", {"local_type": {"type": "string", "enum": ["organization", "contact", "opportunity"]}, "local_id": {"type": "string"}}, ["local_type", "local_id"]), handler=_handle_twenty_sync, check_fn=_check_crm, emoji="👥")
registry.register(name="crm_twenty_raw_request", toolset="crm", schema=_schema("crm_twenty_raw_request", "Advanced escape hatch for Twenty REST endpoints. Use only after preferring canonical CRM tools.", {"method": {"type": "string", "enum": ["GET", "POST", "PATCH", "DELETE"]}, "path": {"type": "string"}, "body": {"type": "object"}, "query": {"type": "object"}}, ["method", "path"]), handler=lambda args, **_kw: json.dumps(_twenty_request(args["method"], args["path"], args.get("body"), args.get("query")), ensure_ascii=False, sort_keys=True), check_fn=_check_crm, emoji="👥")
