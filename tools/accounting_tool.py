"""Agent-native Accounting Lite Core tools backed by Agent Core DB.

Accounting Lite is the local canonical module for small-business payment
receipts, income/expense ledger entries, bank/cash account tracking, and exports
for the accountant. ERP/accounting suites are adapters; the local Agent Core DB
keeps the operational source of truth for Zeus-style single-tenant agents.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from hermes_cli import agent_core_sql as sql
from tools.registry import registry, tool_error

ACCOUNTING_METADATA_DESCRIPTION = (
    "Optional JSON metadata. Keep it generic and tenant-neutral: business_id, "
    "owner_id, source_channel, external_ref, labels, notes."
)
EXPORT_DIR = Path.home() / ".hermes" / "output" / "accounting_exports"


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields}, ensure_ascii=False, sort_keys=True)


def _err(exc: Exception | str) -> str:
    return tool_error(str(exc))


def _user() -> str:
    return sql.runtime_env().get("ACCOUNTING_DB_RUNTIME_USER", "accounting_runtime")


def _check_accounting() -> bool:
    try:
        if not sql.enabled():
            return False
        sql.psql("SELECT 1;", user=_user())
        return True
    except Exception:
        return False


def _q(v: Any) -> str:
    return sql.quote_literal(v)


def _j(v: Any) -> str:
    return sql.quote_jsonb(v)


def _slug(prefix: str, value: str) -> str:
    return f"{prefix}-{sql.slugify(value)}"


def _money(v: Any) -> float:
    return round(float(v or 0), 6)


def _num(v: Any, default: str = "0") -> str:
    if v is None or v == "":
        return default
    try:
        return repr(float(v))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value: {v!r}")


def _handle_accounting_status(args: dict, **_kwargs) -> str:
    try:
        counts = sql.one("""
          SELECT
            (SELECT count(*) FROM accounting.accounts) AS accounts,
            (SELECT count(*) FROM accounting.receipts) AS receipts,
            (SELECT count(*) FROM accounting.receipt_events) AS receipt_events,
            (SELECT count(*) FROM accounting.journal_entries) AS journal_entries,
            (SELECT count(*) FROM accounting.journal_lines) AS journal_lines,
            (SELECT count(*) FROM accounting.export_runs) AS export_runs
        """, user=_user())
        return _ok(db_backend="agent_core_postgres", counts=counts)
    except Exception as exc:
        return _err(exc)


def _handle_account_upsert(args: dict, **_kwargs) -> str:
    try:
        name = str(args.get("name") or "").strip()
        account_type = str(args.get("account_type") or "").strip()
        if not name or not account_type:
            raise ValueError("name and account_type are required")
        account_id = args.get("account_id") or _slug("acct", f"{args.get('business_id') or 'business'}-{name}")
        row = sql.statement_one(f"""
          INSERT INTO accounting.accounts (account_id, business_id, name, account_type, subtype, currency, institution, account_number_last4, status, metadata, created_at, updated_at)
          VALUES ({_q(account_id)}, {_q(args.get('business_id'))}, {_q(name)}, {_q(account_type)}, {_q(args.get('subtype'))}, {_q(args.get('currency') or 'USD')}, {_q(args.get('institution'))}, {_q(args.get('account_number_last4'))}, {_q(args.get('status') or 'active')}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (account_id) DO UPDATE SET business_id=EXCLUDED.business_id, name=EXCLUDED.name, account_type=EXCLUDED.account_type, subtype=EXCLUDED.subtype, currency=EXCLUDED.currency, institution=EXCLUDED.institution, account_number_last4=EXCLUDED.account_number_last4, status=EXCLUDED.status, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(account=row)
    except Exception as exc:
        return _err(exc)


def _handle_receipt_create(args: dict, **_kwargs) -> str:
    try:
        concept = str(args.get("concept") or "").strip()
        amount = _money(args.get("amount"))
        direction = args.get("direction") or "outgoing"
        if not concept or amount <= 0:
            raise ValueError("concept and positive amount are required")
        receipt_id = args.get("receipt_id") or _slug("receipt", f"{args.get('business_id') or 'business'}-{args.get('receipt_number') or concept}")
        metadata = args.get("metadata") or {}
        row = sql.statement_one(f"""
          INSERT INTO accounting.receipts (receipt_id, receipt_number, business_id, payer_organization_id, payer_contact_id, payee_organization_id, payee_contact_id, direction, status, issue_date, payment_date, concept, payment_method, payment_reference, amount, currency, source_account_id, destination_account_id, public_token, public_url, pdf_url, metadata, created_at, updated_at)
          VALUES ({_q(receipt_id)}, {_q(args.get('receipt_number'))}, {_q(args.get('business_id'))}, {_q(args.get('payer_organization_id'))}, {_q(args.get('payer_contact_id'))}, {_q(args.get('payee_organization_id'))}, {_q(args.get('payee_contact_id'))}, {_q(direction)}, {_q(args.get('status') or 'draft')}, {_q(args.get('issue_date'))}::date, {_q(args.get('payment_date'))}::date, {_q(concept)}, {_q(args.get('payment_method'))}, {_q(args.get('payment_reference'))}, {_num(amount)}, {_q(args.get('currency') or 'USD')}, {_q(args.get('source_account_id'))}, {_q(args.get('destination_account_id'))}, {_q(args.get('public_token'))}, {_q(args.get('public_url'))}, {_q(args.get('pdf_url'))}, {_j(metadata)}, now(), now())
          ON CONFLICT (receipt_id) DO UPDATE SET receipt_number=EXCLUDED.receipt_number, business_id=EXCLUDED.business_id, payer_organization_id=EXCLUDED.payer_organization_id, payer_contact_id=EXCLUDED.payer_contact_id, payee_organization_id=EXCLUDED.payee_organization_id, payee_contact_id=EXCLUDED.payee_contact_id, direction=EXCLUDED.direction, status=EXCLUDED.status, issue_date=EXCLUDED.issue_date, payment_date=EXCLUDED.payment_date, concept=EXCLUDED.concept, payment_method=EXCLUDED.payment_method, payment_reference=EXCLUDED.payment_reference, amount=EXCLUDED.amount, currency=EXCLUDED.currency, source_account_id=EXCLUDED.source_account_id, destination_account_id=EXCLUDED.destination_account_id, public_token=EXCLUDED.public_token, public_url=EXCLUDED.public_url, pdf_url=EXCLUDED.pdf_url, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        event = sql.statement_one(f"""
          INSERT INTO accounting.receipt_events (receipt_id, event_type, actor_type, actor_ref, comment, metadata)
          VALUES ({_q(receipt_id)}, 'created', {_q(args.get('actor_type') or 'agent')}, {_q(args.get('actor_ref') or 'agent')}, {_q(args.get('comment') or 'Recibo creado')}, {_j(metadata)})
          RETURNING *
        """, user=_user())
        return _ok(receipt=row, event=event)
    except Exception as exc:
        return _err(exc)


def _journal_totals(lines: list[dict[str, Any]]) -> tuple[float, float]:
    debit = sum(_money(line.get("debit")) for line in lines)
    credit = sum(_money(line.get("credit")) for line in lines)
    return _money(debit), _money(credit)


def _handle_journal_entry_create(args: dict, **_kwargs) -> str:
    try:
        description = str(args.get("description") or "").strip()
        lines = args.get("lines") or []
        if not description or not isinstance(lines, list) or len(lines) < 2:
            raise ValueError("description and at least two journal lines are required")
        total_debit, total_credit = _journal_totals(lines)
        if total_debit <= 0 or total_debit != total_credit:
            raise ValueError(f"journal entry must balance; debit={total_debit} credit={total_credit}")
        entry_id = args.get("journal_entry_id") or _slug("journal", f"{args.get('business_id') or 'business'}-{args.get('entry_date') or 'today'}-{description}")
        entry = sql.statement_one(f"""
          INSERT INTO accounting.journal_entries (journal_entry_id, business_id, entry_date, description, source_type, source_id, status, currency, total_debit, total_credit, metadata, created_at, updated_at)
          VALUES ({_q(entry_id)}, {_q(args.get('business_id'))}, COALESCE({_q(args.get('entry_date'))}::date, CURRENT_DATE), {_q(description)}, {_q(args.get('source_type'))}, {_q(args.get('source_id'))}, {_q(args.get('status') or 'posted')}, {_q(args.get('currency') or 'USD')}, {_num(total_debit)}, {_num(total_credit)}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (journal_entry_id) DO UPDATE SET business_id=EXCLUDED.business_id, entry_date=EXCLUDED.entry_date, description=EXCLUDED.description, source_type=EXCLUDED.source_type, source_id=EXCLUDED.source_id, status=EXCLUDED.status, currency=EXCLUDED.currency, total_debit=EXCLUDED.total_debit, total_credit=EXCLUDED.total_credit, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        sql.psql(f"DELETE FROM accounting.journal_lines WHERE journal_entry_id={_q(entry_id)};", user=_user())
        saved = []
        for idx, line in enumerate(lines, start=1):
            saved.append(sql.statement_one(f"""
              INSERT INTO accounting.journal_lines (journal_entry_id, account_id, line_no, description, debit, credit, contact_id, organization_id, metadata)
              VALUES ({_q(entry_id)}, {_q(line.get('account_id'))}, {int(line.get('line_no') or idx)}, {_q(line.get('description'))}, {_num(line.get('debit'))}, {_num(line.get('credit'))}, {_q(line.get('contact_id'))}, {_q(line.get('organization_id'))}, {_j(line.get('metadata') or {})})
              RETURNING *
            """, user=_user()))
        return _ok(journal_entry=entry, lines=saved)
    except Exception as exc:
        return _err(exc)


def _handle_receipt_get(args: dict, **_kwargs) -> str:
    try:
        receipt_id = str(args.get("receipt_id") or "").strip()
        if not receipt_id:
            raise ValueError("receipt_id is required")
        receipt = sql.one(f"SELECT * FROM accounting.receipts WHERE receipt_id={_q(receipt_id)}", user=_user())
        if not receipt:
            raise ValueError("receipt not found")
        events = sql.rows(f"SELECT * FROM accounting.receipt_events WHERE receipt_id={_q(receipt_id)} ORDER BY occurred_at, receipt_event_id", user=_user())
        journals = sql.rows(f"SELECT * FROM accounting.journal_entries WHERE source_type='receipt' AND source_id={_q(receipt_id)} ORDER BY entry_date, created_at", user=_user())
        return _ok(receipt=receipt, events=events, journal_entries=journals)
    except Exception as exc:
        return _err(exc)


def _handle_export_create(args: dict, **_kwargs) -> str:
    try:
        business_id = args.get("business_id") or "business"
        start_date = args.get("start_date")
        end_date = args.get("end_date")
        export_id = args.get("export_id") or _slug("acct-export", f"{business_id}-{start_date or 'start'}-{end_date or 'end'}")
        rows = sql.rows(f"""
          SELECT e.entry_date, e.journal_entry_id, e.description AS entry_description, e.source_type, e.source_id, l.line_no, a.name AS account_name, a.account_type, a.subtype, l.description AS line_description, l.debit, l.credit, e.currency
          FROM accounting.journal_entries e
          JOIN accounting.journal_lines l ON l.journal_entry_id=e.journal_entry_id
          JOIN accounting.accounts a ON a.account_id=l.account_id
          WHERE ({_q(business_id)} IS NULL OR e.business_id={_q(business_id)})
            AND ({_q(start_date)} IS NULL OR e.entry_date >= {_q(start_date)}::date)
            AND ({_q(end_date)} IS NULL OR e.entry_date <= {_q(end_date)}::date)
          ORDER BY e.entry_date, e.journal_entry_id, l.line_no
        """, user=_user())
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        file_path = EXPORT_DIR / f"{export_id}.csv"
        fieldnames = ["entry_date", "journal_entry_id", "entry_description", "source_type", "source_id", "line_no", "account_name", "account_type", "subtype", "line_description", "debit", "credit", "currency"]
        with file_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        export = sql.statement_one(f"""
          INSERT INTO accounting.export_runs (export_id, business_id, start_date, end_date, format, status, file_path, row_count, metadata)
          VALUES ({_q(export_id)}, {_q(business_id)}, {_q(start_date)}::date, {_q(end_date)}::date, 'csv', 'created', {_q(str(file_path))}, {len(rows)}, {_j(args.get('metadata') or {})})
          ON CONFLICT (export_id) DO UPDATE SET status=EXCLUDED.status, file_path=EXCLUDED.file_path, row_count=EXCLUDED.row_count, metadata=EXCLUDED.metadata
          RETURNING *
        """, user=_user())
        return _ok(export=export, file_path=str(file_path), row_count=len(rows))
    except Exception as exc:
        return _err(exc)


def _schema(name: str, description: str, props: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required or []}}}


def _meta_props() -> dict[str, Any]:
    return {"metadata": {"type": "object", "description": ACCOUNTING_METADATA_DESCRIPTION}}


registry.register(name="accounting_status", toolset="accounting", schema=_schema("accounting_status", "Return Accounting Lite Core row counts and DB backend.", {}), handler=_handle_accounting_status, check_fn=_check_accounting, emoji="🧾")
registry.register(name="accounting_account_upsert", toolset="accounting", schema=_schema("accounting_account_upsert", "Create/update an accounting account such as Bank of America cash, income, or expense.", {"account_id": {"type": "string"}, "business_id": {"type": "string"}, "name": {"type": "string"}, "account_type": {"type": "string", "enum": ["asset", "liability", "equity", "income", "expense"]}, "subtype": {"type": "string"}, "currency": {"type": "string"}, "institution": {"type": "string"}, "account_number_last4": {"type": "string"}, "status": {"type": "string"}, **_meta_props()}, ["name", "account_type"]), handler=_handle_account_upsert, check_fn=_check_accounting, emoji="🧾")
registry.register(name="accounting_receipt_create", toolset="accounting", schema=_schema("accounting_receipt_create", "Create/update a payment receipt record for incoming or outgoing payments.", {"receipt_id": {"type": "string"}, "receipt_number": {"type": "string"}, "business_id": {"type": "string"}, "payer_organization_id": {"type": "string"}, "payer_contact_id": {"type": "string"}, "payee_organization_id": {"type": "string"}, "payee_contact_id": {"type": "string"}, "direction": {"type": "string", "enum": ["incoming", "outgoing"]}, "status": {"type": "string"}, "issue_date": {"type": "string"}, "payment_date": {"type": "string"}, "concept": {"type": "string"}, "payment_method": {"type": "string"}, "payment_reference": {"type": "string"}, "amount": {"type": "number"}, "currency": {"type": "string"}, "source_account_id": {"type": "string"}, "destination_account_id": {"type": "string"}, "public_token": {"type": "string"}, "public_url": {"type": "string"}, "pdf_url": {"type": "string"}, "actor_ref": {"type": "string"}, "comment": {"type": "string"}, **_meta_props()}, ["concept", "amount"]), handler=_handle_receipt_create, check_fn=_check_accounting, emoji="🧾")
registry.register(name="accounting_receipt_get", toolset="accounting", schema=_schema("accounting_receipt_get", "Read a receipt with events and linked journal entries.", {"receipt_id": {"type": "string"}}, ["receipt_id"]), handler=_handle_receipt_get, check_fn=_check_accounting, emoji="🧾")
registry.register(name="accounting_journal_entry_create", toolset="accounting", schema=_schema("accounting_journal_entry_create", "Create a balanced double-entry journal entry with debit/credit lines.", {"journal_entry_id": {"type": "string"}, "business_id": {"type": "string"}, "entry_date": {"type": "string"}, "description": {"type": "string"}, "source_type": {"type": "string"}, "source_id": {"type": "string"}, "status": {"type": "string"}, "currency": {"type": "string"}, "lines": {"type": "array", "items": {"type": "object"}}, **_meta_props()}, ["description", "lines"]), handler=_handle_journal_entry_create, check_fn=_check_accounting, emoji="🧾")
registry.register(name="accounting_export_create", toolset="accounting", schema=_schema("accounting_export_create", "Export accounting journal lines to a CSV file for an accountant.", {"export_id": {"type": "string"}, "business_id": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}, **_meta_props()}), handler=_handle_export_create, check_fn=_check_accounting, emoji="🧾")
