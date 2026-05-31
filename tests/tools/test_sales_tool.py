import json

from tools import sales_tool


def test_quote_totals_supports_discounts_and_tax():
    subtotal, discount_amount, tax_amount, total = sales_tool._quote_totals([
        {"quantity": 2, "unit_price": 100, "discount_rate": 0.1, "tax_rate": 0.16},
        {"quantity": 1, "unit_price": 50, "discount_amount": 5, "tax_rate": 0},
    ])

    assert subtotal == 250
    assert discount_amount == 25
    assert tax_amount == 28.8
    assert total == 253.8


def test_sales_num_rejects_sql_fragments():
    try:
        sales_tool._num("1; DROP TABLE sales.products")
    except ValueError as exc:
        assert "Invalid numeric value" in str(exc)
    else:
        raise AssertionError("expected numeric validation failure")


def test_inventory_adjust_rejects_invalid_quantity_before_db(monkeypatch):
    called = False

    def fake_statement_one(*_args, **_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(sales_tool.sql, "statement_one", fake_statement_one)

    payload = json.loads(sales_tool._handle_inventory_adjust({
        "product_id": "product-demo",
        "quantity_delta": "1; DROP TABLE sales.inventory_movements",
        "reason": "correction",
    }))

    assert called is False
    assert "Invalid numeric value" in payload["error"]


def test_order_from_quote_preserves_items_and_status(monkeypatch):
    statements = []

    def fake_one(query, **_kwargs):
        if "FROM sales.quotes" in query:
            return {
                "quote_id": "quote-1",
                "organization_id": "org-1",
                "contact_id": "contact-1",
                "opportunity_id": "opp-1",
                "currency": "USD",
                "subtotal": 100,
                "discount_amount": 10,
                "tax_amount": 14.4,
                "total": 104.4,
            }
        return None

    def fake_rows(query, **_kwargs):
        if "FROM sales.quote_items" in query:
            return [{
                "product_id": "product-1",
                "description": "Consulting",
                "quantity": 1,
                "unit_price": 100,
                "discount_rate": 0.1,
                "discount_amount": 10,
                "tax_rate": 0.16,
                "line_subtotal": 100,
                "line_discount": 10,
                "line_tax": 14.4,
                "line_total": 104.4,
                "metadata": {"source": "test"},
            }]
        return []

    def fake_statement_one(statement, **_kwargs):
        statements.append(statement)
        if "INSERT INTO sales.orders" in statement:
            return {"order_id": "order-1", "status": "confirmed", "total": 104.4}
        if "INSERT INTO sales.order_items" in statement:
            return {"order_item_id": 1, "line_total": 104.4}
        return {}

    monkeypatch.setattr(sales_tool.sql, "one", fake_one)
    monkeypatch.setattr(sales_tool.sql, "rows", fake_rows)
    monkeypatch.setattr(sales_tool.sql, "statement_one", fake_statement_one)
    monkeypatch.setattr(sales_tool.sql, "psql", lambda *_args, **_kwargs: None)

    payload = json.loads(sales_tool._handle_order_create({
        "quote_id": "quote-1",
        "order_id": "order-1",
        "status": "confirmed",
    }))

    assert payload["ok"] is True
    assert payload["order"]["status"] == "confirmed"
    assert payload["items"][0]["line_total"] == 104.4
    assert any("status='accepted'" in statement for statement in statements)


def test_invoice_from_order_preserves_items(monkeypatch):
    def fake_one(query, **_kwargs):
        if "FROM sales.orders" in query:
            return {
                "order_id": "order-1",
                "organization_id": "org-1",
                "contact_id": "contact-1",
                "currency": "USD",
                "subtotal": 100,
                "discount_amount": 10,
                "tax_amount": 14.4,
                "total": 104.4,
                "title": "Order title",
            }
        return None

    def fake_rows(query, **_kwargs):
        if "FROM sales.order_items" in query:
            return [{
                "product_id": "product-1",
                "description": "Consulting",
                "quantity": 1,
                "unit_price": 100,
                "discount_amount": 10,
                "tax_rate": 0.16,
                "line_subtotal": 100,
                "line_discount": 10,
                "line_tax": 14.4,
                "line_total": 104.4,
                "metadata": {"source": "test"},
            }]
        return []

    def fake_statement_one(statement, **_kwargs):
        if "INSERT INTO sales.invoices" in statement:
            return {"invoice_id": "invoice-1", "total": 104.4}
        if "INSERT INTO sales.invoice_items" in statement:
            return {"invoice_item_id": 1, "line_total": 104.4}
        return {}

    monkeypatch.setattr(sales_tool.sql, "one", fake_one)
    monkeypatch.setattr(sales_tool.sql, "rows", fake_rows)
    monkeypatch.setattr(sales_tool.sql, "statement_one", fake_statement_one)
    monkeypatch.setattr(sales_tool.sql, "psql", lambda *_args, **_kwargs: None)

    payload = json.loads(sales_tool._handle_invoice_create({
        "order_id": "order-1",
        "invoice_id": "invoice-1",
    }))

    assert payload["ok"] is True
    assert payload["items"][0]["line_total"] == 104.4



def test_payment_request_without_adapter_is_gracefully_unavailable(monkeypatch):
    monkeypatch.setattr(sales_tool.sql, "runtime_env", lambda: {})
    monkeypatch.delenv("PAYMENT_ADAPTER", raising=False)
    monkeypatch.delenv("SALES_PAYMENT_ADAPTER", raising=False)

    result = sales_tool._payment_adapter_request({"amount": 10, "currency": "USD"})

    assert result["ok"] is False
    assert result["configured"] is False
    assert result["status"] == "unavailable"
    assert "PAYMENT_ADAPTER" in result["error"]


def test_toolset_exports_sales_tools():
    import toolsets

    sales_tools = set(toolsets.TOOLSETS["sales"]["tools"])

    assert "sales_product_upsert" in sales_tools
    assert "sales_inventory_adjust" in sales_tools
    assert "sales_quote_create" in sales_tools
    assert "sales_order_create" in sales_tools
    assert "sales_invoice_create" in sales_tools
    assert "sales_payment_request_create" in sales_tools
