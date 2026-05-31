"""Agent-native Sales Core tools backed by Agent Core DB.

Sales Core is the operational commercial layer for SMB/freelancer agents:
products/catalog, simple inventory, quotes, orders, operational invoices, and
payment requests. External payment/ERP systems are adapters; the local core
remains usable without them.
"""
from __future__ import annotations

import json
import os
from typing import Any

from hermes_cli import agent_core_sql as sql
from tools.registry import registry, tool_error

SALES_METADATA_DESCRIPTION = (
    "Optional JSON metadata. Keep it generic and tenant-neutral: business_id, "
    "owner_id, source_channel, external_ref, labels, notes."
)


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields}, ensure_ascii=False, sort_keys=True)


def _err(exc: Exception | str) -> str:
    return tool_error(str(exc))


def _user() -> str:
    return sql.runtime_env().get("SALES_DB_RUNTIME_USER", "sales_runtime")


def _check_sales() -> bool:
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


def _num(v: Any, default: str = "NULL") -> str:
    if v is None or v == "":
        return default
    try:
        return repr(float(v))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value: {v!r}")


def _money(v: Any) -> float:
    return round(float(v or 0), 6)


def _line_amounts(item: dict[str, Any]) -> dict[str, float]:
    qty = float(item.get("quantity") or 1)
    price = float(item.get("unit_price") or 0)
    subtotal = qty * price
    if item.get("discount_amount") not in (None, ""):
        discount = float(item.get("discount_amount") or 0)
    else:
        discount = subtotal * float(item.get("discount_rate") or 0)
    taxable = max(0.0, subtotal - discount)
    tax = taxable * float(item.get("tax_rate") or 0)
    return {
        "quantity": qty,
        "unit_price": price,
        "line_subtotal": _money(subtotal),
        "line_discount": _money(discount),
        "line_tax": _money(tax),
        "line_total": _money(taxable + tax),
    }


def _quote_totals(items: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    subtotal = 0.0
    discount_amount = 0.0
    tax_amount = 0.0
    total = 0.0
    for item in items:
        amounts = _line_amounts(item)
        subtotal += amounts["line_subtotal"]
        discount_amount += amounts["line_discount"]
        tax_amount += amounts["line_tax"]
        total += amounts["line_total"]
    return _money(subtotal), _money(discount_amount), _money(tax_amount), _money(total)


def _payment_adapter_request(payload: dict[str, Any]) -> dict[str, Any]:
    env = sql.runtime_env()
    adapter = (
        env.get("SALES_PAYMENT_ADAPTER")
        or env.get("PAYMENT_ADAPTER")
        or os.getenv("SALES_PAYMENT_ADAPTER")
        or os.getenv("PAYMENT_ADAPTER")
        or ""
    ).strip()
    if not adapter:
        return {
            "ok": False,
            "configured": False,
            "status": "unavailable",
            "error": "Payment adapter is not configured. Set PAYMENT_ADAPTER or SALES_PAYMENT_ADAPTER via Infisical/runtime env.",
        }
    return {
        "ok": False,
        "configured": True,
        "status": "unsupported",
        "adapter": adapter,
        "error": f"Payment adapter {adapter!r} is declared but not implemented in Sales Core Sprint 1.",
        "request": payload,
    }


def _handle_product_upsert(args: dict, **_kwargs) -> str:
    try:
        name = str(args.get("name") or "").strip()
        if not name:
            raise ValueError("name is required")
        product_id = args.get("product_id") or _slug("sales-product", args.get("sku") or name)
        row = sql.statement_one(f"""
          INSERT INTO sales.products (product_id, sku, name, description, unit_price, currency, status, metadata, created_at, updated_at)
          VALUES ({_q(product_id)}, {_q(args.get('sku'))}, {_q(name)}, {_q(args.get('description'))}, {_num(args.get('unit_price'))}, {_q(args.get('currency') or 'USD')}, {_q(args.get('status') or 'active')}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (product_id) DO UPDATE SET sku=EXCLUDED.sku, name=EXCLUDED.name, description=EXCLUDED.description, unit_price=EXCLUDED.unit_price, currency=EXCLUDED.currency, status=EXCLUDED.status, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(product=row)
    except Exception as exc:
        return _err(exc)


def _handle_inventory_adjust(args: dict, **_kwargs) -> str:
    try:
        product_id = str(args.get("product_id") or "").strip()
        if not product_id:
            raise ValueError("product_id is required")
        quantity_delta = _num(args.get("quantity_delta"), "0")
        movement = sql.statement_one(f"""
          INSERT INTO sales.inventory_movements (product_id, quantity_delta, reason, reference_type, reference_id, occurred_at, metadata)
          VALUES ({_q(product_id)}, {quantity_delta}, {_q(args.get('reason') or 'adjustment')}, {_q(args.get('reference_type'))}, {_q(args.get('reference_id'))}, COALESCE({_q(args.get('occurred_at'))}::timestamptz, now()), {_j(args.get('metadata') or {})})
          RETURNING *
        """, user=_user())
        balance = sql.statement_one(f"""
          INSERT INTO sales.inventory_balances (product_id, quantity_on_hand, updated_at)
          VALUES ({_q(product_id)}, {quantity_delta}, now())
          ON CONFLICT (product_id) DO UPDATE SET quantity_on_hand=sales.inventory_balances.quantity_on_hand + EXCLUDED.quantity_on_hand, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(movement=movement, balance=balance)
    except Exception as exc:
        return _err(exc)


def _handle_quote_create(args: dict, **_kwargs) -> str:
    try:
        title = str(args.get("title") or "").strip()
        if not title:
            raise ValueError("title is required")
        items = args.get("items") or []
        if not isinstance(items, list):
            raise ValueError("items must be a list")
        quote_id = args.get("quote_id") or _slug("sales-quote", f"{args.get('organization_id') or args.get('contact_id') or ''}-{title}")
        subtotal, discount_amount, tax_amount, total = _quote_totals(items)
        quote = sql.statement_one(f"""
          INSERT INTO sales.quotes (quote_id, organization_id, contact_id, opportunity_id, title, status, valid_until, currency, subtotal, discount_amount, tax_amount, total, metadata, created_at, updated_at)
          VALUES ({_q(quote_id)}, {_q(args.get('organization_id'))}, {_q(args.get('contact_id'))}, {_q(args.get('opportunity_id'))}, {_q(title)}, {_q(args.get('status') or 'draft')}, {_q(args.get('valid_until'))}::date, {_q(args.get('currency') or 'USD')}, {subtotal}, {discount_amount}, {tax_amount}, {total}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (quote_id) DO UPDATE SET organization_id=EXCLUDED.organization_id, contact_id=EXCLUDED.contact_id, opportunity_id=EXCLUDED.opportunity_id, title=EXCLUDED.title, status=EXCLUDED.status, valid_until=EXCLUDED.valid_until, currency=EXCLUDED.currency, subtotal=EXCLUDED.subtotal, discount_amount=EXCLUDED.discount_amount, tax_amount=EXCLUDED.tax_amount, total=EXCLUDED.total, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        sql.psql(f"DELETE FROM sales.quote_items WHERE quote_id={_q(quote_id)};", user=_user())
        saved_items = []
        for item in items:
            amounts = _line_amounts(item)
            saved_items.append(sql.statement_one(f"""
              INSERT INTO sales.quote_items (quote_id, product_id, description, quantity, unit_price, discount_rate, discount_amount, tax_rate, line_subtotal, line_discount, line_tax, line_total, metadata)
              VALUES ({_q(quote_id)}, {_q(item.get('product_id'))}, {_q(item.get('description') or item.get('name') or 'Item')}, {amounts['quantity']}, {amounts['unit_price']}, {_num(item.get('discount_rate'), '0')}, {_num(item.get('discount_amount'), '0')}, {_num(item.get('tax_rate'), '0')}, {amounts['line_subtotal']}, {amounts['line_discount']}, {amounts['line_tax']}, {amounts['line_total']}, {_j(item.get('metadata') or {})})
              RETURNING *
            """, user=_user()))
        return _ok(quote=quote, items=saved_items)
    except Exception as exc:
        return _err(exc)


def _handle_order_create(args: dict, **_kwargs) -> str:
    try:
        quote_id = args.get("quote_id")
        quote = sql.one(f"SELECT * FROM sales.quotes WHERE quote_id={_q(quote_id)}", user=_user()) if quote_id else None
        if quote_id and not quote:
            raise ValueError("quote not found")
        title = args.get("title") or (quote or {}).get("title") or "Order"
        order_id = args.get("order_id") or _slug("sales-order", f"{quote_id or args.get('organization_id') or ''}-{title}")
        organization_id = args.get("organization_id") or (quote or {}).get("organization_id")
        contact_id = args.get("contact_id") or (quote or {}).get("contact_id")
        opportunity_id = args.get("opportunity_id") or (quote or {}).get("opportunity_id")
        subtotal = args.get("subtotal", (quote or {}).get("subtotal", 0))
        discount_amount = args.get("discount_amount", (quote or {}).get("discount_amount", 0))
        tax_amount = args.get("tax_amount", (quote or {}).get("tax_amount", 0))
        total = args.get("total", (quote or {}).get("total", 0))
        order = sql.statement_one(f"""
          INSERT INTO sales.orders (order_id, quote_id, organization_id, contact_id, opportunity_id, title, status, currency, subtotal, discount_amount, tax_amount, total, metadata, created_at, updated_at)
          VALUES ({_q(order_id)}, {_q(quote_id)}, {_q(organization_id)}, {_q(contact_id)}, {_q(opportunity_id)}, {_q(title)}, {_q(args.get('status') or 'confirmed')}, {_q(args.get('currency') or (quote or {}).get('currency') or 'USD')}, {_num(subtotal, '0')}, {_num(discount_amount, '0')}, {_num(tax_amount, '0')}, {_num(total, '0')}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (order_id) DO UPDATE SET quote_id=EXCLUDED.quote_id, organization_id=EXCLUDED.organization_id, contact_id=EXCLUDED.contact_id, opportunity_id=EXCLUDED.opportunity_id, title=EXCLUDED.title, status=EXCLUDED.status, currency=EXCLUDED.currency, subtotal=EXCLUDED.subtotal, discount_amount=EXCLUDED.discount_amount, tax_amount=EXCLUDED.tax_amount, total=EXCLUDED.total, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        sql.psql(f"DELETE FROM sales.order_items WHERE order_id={_q(order_id)};", user=_user())
        source_items = args.get("items") or []
        if not source_items and quote_id:
            source_items = sql.rows(f"SELECT * FROM sales.quote_items WHERE quote_id={_q(quote_id)} ORDER BY quote_item_id", user=_user())
        saved_items = []
        for item in source_items:
            amounts = _line_amounts(item) if "line_total" not in item else {
                "quantity": float(item.get("quantity") or 1),
                "unit_price": float(item.get("unit_price") or 0),
                "line_subtotal": float(item.get("line_subtotal") or 0),
                "line_discount": float(item.get("line_discount") or item.get("discount_amount") or 0),
                "line_tax": float(item.get("line_tax") or 0),
                "line_total": float(item.get("line_total") or 0),
            }
            saved_items.append(sql.statement_one(f"""
              INSERT INTO sales.order_items (order_id, product_id, description, quantity, unit_price, discount_amount, tax_rate, line_subtotal, line_discount, line_tax, line_total, metadata)
              VALUES ({_q(order_id)}, {_q(item.get('product_id'))}, {_q(item.get('description') or item.get('name') or 'Item')}, {amounts['quantity']}, {amounts['unit_price']}, {amounts['line_discount']}, {_num(item.get('tax_rate'), '0')}, {amounts['line_subtotal']}, {amounts['line_discount']}, {amounts['line_tax']}, {amounts['line_total']}, {_j(item.get('metadata') or {})})
              RETURNING *
            """, user=_user()))
        if quote_id:
            sql.statement_one(f"UPDATE sales.quotes SET status='accepted', updated_at=now() WHERE quote_id={_q(quote_id)} RETURNING *", user=_user())
        return _ok(order=order, items=saved_items)
    except Exception as exc:
        return _err(exc)


def _handle_invoice_create(args: dict, **_kwargs) -> str:
    try:
        order_id = args.get("order_id")
        order = sql.one(f"SELECT * FROM sales.orders WHERE order_id={_q(order_id)}", user=_user()) if order_id else None
        if order_id and not order:
            raise ValueError("order not found")
        title = args.get("title") or (order or {}).get("title") or "Invoice"
        invoice_id = args.get("invoice_id") or _slug("sales-invoice", f"{order_id or args.get('organization_id') or ''}-{title}")
        organization_id = args.get("organization_id") or (order or {}).get("organization_id")
        contact_id = args.get("contact_id") or (order or {}).get("contact_id")
        subtotal = args.get("subtotal", (order or {}).get("subtotal", 0))
        discount_amount = args.get("discount_amount", (order or {}).get("discount_amount", 0))
        tax_amount = args.get("tax_amount", (order or {}).get("tax_amount", 0))
        total = args.get("total", (order or {}).get("total", 0))
        invoice = sql.statement_one(f"""
          INSERT INTO sales.invoices (invoice_id, order_id, organization_id, contact_id, title, status, issue_date, due_date, currency, subtotal, discount_amount, tax_amount, total, metadata, created_at, updated_at)
          VALUES ({_q(invoice_id)}, {_q(order_id)}, {_q(organization_id)}, {_q(contact_id)}, {_q(title)}, {_q(args.get('status') or 'draft')}, COALESCE({_q(args.get('issue_date'))}::date, CURRENT_DATE), {_q(args.get('due_date'))}::date, {_q(args.get('currency') or (order or {}).get('currency') or 'USD')}, {_num(subtotal, '0')}, {_num(discount_amount, '0')}, {_num(tax_amount, '0')}, {_num(total, '0')}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (invoice_id) DO UPDATE SET order_id=EXCLUDED.order_id, organization_id=EXCLUDED.organization_id, contact_id=EXCLUDED.contact_id, title=EXCLUDED.title, status=EXCLUDED.status, issue_date=EXCLUDED.issue_date, due_date=EXCLUDED.due_date, currency=EXCLUDED.currency, subtotal=EXCLUDED.subtotal, discount_amount=EXCLUDED.discount_amount, tax_amount=EXCLUDED.tax_amount, total=EXCLUDED.total, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        sql.psql(f"DELETE FROM sales.invoice_items WHERE invoice_id={_q(invoice_id)};", user=_user())
        source_items = args.get("items") or []
        if not source_items and order_id:
            source_items = sql.rows(f"SELECT * FROM sales.order_items WHERE order_id={_q(order_id)} ORDER BY order_item_id", user=_user())
        saved_items = []
        for item in source_items:
            amounts = _line_amounts(item) if "line_total" not in item else {
                "quantity": float(item.get("quantity") or 1),
                "unit_price": float(item.get("unit_price") or 0),
                "line_subtotal": float(item.get("line_subtotal") or 0),
                "line_discount": float(item.get("line_discount") or item.get("discount_amount") or 0),
                "line_tax": float(item.get("line_tax") or 0),
                "line_total": float(item.get("line_total") or 0),
            }
            saved_items.append(sql.statement_one(f"""
              INSERT INTO sales.invoice_items (invoice_id, product_id, description, quantity, unit_price, discount_amount, tax_rate, line_subtotal, line_discount, line_tax, line_total, metadata)
              VALUES ({_q(invoice_id)}, {_q(item.get('product_id'))}, {_q(item.get('description') or item.get('name') or 'Item')}, {amounts['quantity']}, {amounts['unit_price']}, {amounts['line_discount']}, {_num(item.get('tax_rate'), '0')}, {amounts['line_subtotal']}, {amounts['line_discount']}, {amounts['line_tax']}, {amounts['line_total']}, {_j(item.get('metadata') or {})})
              RETURNING *
            """, user=_user()))
        return _ok(invoice=invoice, items=saved_items)
    except Exception as exc:
        return _err(exc)


def _handle_payment_request_create(args: dict, **_kwargs) -> str:
    try:
        invoice = sql.one(f"SELECT * FROM sales.invoices WHERE invoice_id={_q(args.get('invoice_id'))}", user=_user()) if args.get("invoice_id") else None
        amount = args.get("amount", (invoice or {}).get("total", 0))
        currency = args.get("currency") or (invoice or {}).get("currency") or "USD"
        request_id = args.get("payment_request_id") or _slug("sales-pay", f"{args.get('invoice_id') or args.get('organization_id') or ''}-{amount}-{currency}")
        adapter_result = _payment_adapter_request({"invoice_id": args.get("invoice_id"), "amount": amount, "currency": currency, "metadata": args.get("metadata") or {}})
        status = "pending" if adapter_result.get("ok") else "unavailable"
        row = sql.statement_one(f"""
          INSERT INTO sales.payment_requests (payment_request_id, invoice_id, organization_id, contact_id, amount, currency, status, adapter, payment_url, adapter_response, metadata, created_at, updated_at)
          VALUES ({_q(request_id)}, {_q(args.get('invoice_id'))}, {_q(args.get('organization_id') or (invoice or {}).get('organization_id'))}, {_q(args.get('contact_id') or (invoice or {}).get('contact_id'))}, {_num(amount, '0')}, {_q(currency)}, {_q(args.get('status') or status)}, {_q(adapter_result.get('adapter'))}, {_q(adapter_result.get('payment_url'))}, {_j(adapter_result)}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (payment_request_id) DO UPDATE SET invoice_id=EXCLUDED.invoice_id, organization_id=EXCLUDED.organization_id, contact_id=EXCLUDED.contact_id, amount=EXCLUDED.amount, currency=EXCLUDED.currency, status=EXCLUDED.status, adapter=EXCLUDED.adapter, payment_url=EXCLUDED.payment_url, adapter_response=EXCLUDED.adapter_response, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(payment_request=row, adapter_result=adapter_result)
    except Exception as exc:
        return _err(exc)


def _handle_sales_status(args: dict, **_kwargs) -> str:
    try:
        counts = sql.one("""
          SELECT
            (SELECT count(*) FROM sales.products) AS products,
            (SELECT count(*) FROM sales.quotes) AS quotes,
            (SELECT count(*) FROM sales.orders) AS orders,
            (SELECT count(*) FROM sales.invoices) AS invoices,
            (SELECT count(*) FROM sales.payment_requests) AS payment_requests
        """, user=_user())
        return _ok(db_backend="agent_core_postgres", counts=counts)
    except Exception as exc:
        return _err(exc)


def _schema(name: str, description: str, props: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required or []}}}


def _meta_props() -> dict[str, Any]:
    return {"metadata": {"type": "object", "description": SALES_METADATA_DESCRIPTION}}


_COMMON_ENTITY_PROPS = {
    "organization_id": {"type": "string"},
    "contact_id": {"type": "string"},
    "opportunity_id": {"type": "string"},
}

registry.register(name="sales_status", toolset="sales", schema=_schema("sales_status", "Return Sales Core row counts and DB backend.", {}), handler=_handle_sales_status, check_fn=_check_sales, emoji="🧾")
registry.register(name="sales_product_upsert", toolset="sales", schema=_schema("sales_product_upsert", "Create or update a Sales Core product/service catalog item.", {"product_id": {"type": "string"}, "sku": {"type": "string"}, "name": {"type": "string"}, "description": {"type": "string"}, "unit_price": {"type": "number"}, "currency": {"type": "string"}, "status": {"type": "string"}, **_meta_props()}, ["name"]), handler=_handle_product_upsert, check_fn=_check_sales, emoji="🧾")
registry.register(name="sales_inventory_adjust", toolset="sales", schema=_schema("sales_inventory_adjust", "Adjust simple inventory quantity for a product and record a movement.", {"product_id": {"type": "string"}, "quantity_delta": {"type": "number"}, "reason": {"type": "string"}, "reference_type": {"type": "string"}, "reference_id": {"type": "string"}, "occurred_at": {"type": "string"}, **_meta_props()}, ["product_id", "quantity_delta"]), handler=_handle_inventory_adjust, check_fn=_check_sales, emoji="🧾")
registry.register(name="sales_quote_create", toolset="sales", schema=_schema("sales_quote_create", "Create or replace an operational quote with line items and computed totals.", {"quote_id": {"type": "string"}, **_COMMON_ENTITY_PROPS, "title": {"type": "string"}, "status": {"type": "string"}, "valid_until": {"type": "string"}, "currency": {"type": "string"}, "items": {"type": "array", "items": {"type": "object"}}, **_meta_props()}, ["title"]), handler=_handle_quote_create, check_fn=_check_sales, emoji="🧾")
registry.register(name="sales_order_create", toolset="sales", schema=_schema("sales_order_create", "Create or update an operational order, optionally from a Sales Core quote.", {"order_id": {"type": "string"}, "quote_id": {"type": "string"}, **_COMMON_ENTITY_PROPS, "title": {"type": "string"}, "status": {"type": "string"}, "currency": {"type": "string"}, "items": {"type": "array", "items": {"type": "object"}}, **_meta_props()}), handler=_handle_order_create, check_fn=_check_sales, emoji="🧾")
registry.register(name="sales_invoice_create", toolset="sales", schema=_schema("sales_invoice_create", "Create or update an operational invoice, optionally from an order. Not fiscal unless an adapter is configured.", {"invoice_id": {"type": "string"}, "order_id": {"type": "string"}, **_COMMON_ENTITY_PROPS, "title": {"type": "string"}, "status": {"type": "string"}, "issue_date": {"type": "string"}, "due_date": {"type": "string"}, "currency": {"type": "string"}, "subtotal": {"type": "number"}, "discount_amount": {"type": "number"}, "tax_amount": {"type": "number"}, "total": {"type": "number"}, **_meta_props()}), handler=_handle_invoice_create, check_fn=_check_sales, emoji="🧾")
registry.register(name="sales_payment_request_create", toolset="sales", schema=_schema("sales_payment_request_create", "Create a payment request for an invoice. Returns graceful unavailable status if no payment adapter is configured.", {"payment_request_id": {"type": "string"}, "invoice_id": {"type": "string"}, "organization_id": {"type": "string"}, "contact_id": {"type": "string"}, "amount": {"type": "number"}, "currency": {"type": "string"}, "status": {"type": "string"}, **_meta_props()}), handler=_handle_payment_request_create, check_fn=_check_sales, emoji="🧾")
