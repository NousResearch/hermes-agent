import json

from tools import crm_tool


def test_quote_totals_computes_subtotal_tax_and_total():
    subtotal, tax_amount, total = crm_tool._quote_totals([
        {"quantity": 2, "unit_price": 100, "tax_rate": 0.1},
        {"quantity": 1, "unit_price": 50, "tax_rate": 0},
    ])

    assert subtotal == 250
    assert tax_amount == 20
    assert total == 270


def test_twenty_request_reports_unconfigured_without_network(monkeypatch):
    monkeypatch.setattr(crm_tool.sql, "runtime_env", lambda: {})
    monkeypatch.delenv("TWENTY_BASE_URL", raising=False)
    monkeypatch.delenv("TWENTY_API_KEY", raising=False)

    result = crm_tool._twenty_request("GET", "/rest/companies")

    assert result["ok"] is False
    assert result["configured"] is False
    assert "TWENTY_BASE_URL" in result["error"]


def test_twenty_sync_validates_local_id_before_db_query():
    payload = json.loads(crm_tool._handle_twenty_sync({"local_type": "organization"}))

    assert payload["error"] == "local_id is required"


def test_num_rejects_sql_fragments():
    try:
        crm_tool._num("1; DROP TABLE crm.contacts")
    except ValueError as exc:
        assert "Invalid numeric value" in str(exc)
    else:
        raise AssertionError("expected numeric validation failure")


def test_toolset_exports_expanded_crm_tools():
    import toolsets

    crm_tools = set(toolsets.TOOLSETS["crm"]["tools"])

    assert "crm_opportunity_upsert" in crm_tools
    assert "crm_product_upsert" in crm_tools
    assert "crm_quote_create" in crm_tools
    assert "crm_invoice_create" in crm_tools
    assert "crm_relationship_upsert" in crm_tools
    assert "crm_customer_timeline" in crm_tools
    assert "crm_twenty_sync" in crm_tools
