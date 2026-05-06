from __future__ import annotations

import json

import tools.evotraders_tool as mod


def test_route_dry_run_enforces_quote_first_for_single_stock_analysis():
    raw = mod.evotraders_route_dry_run(query="请分析 600519 的走势，给出交易计划")
    obj = json.loads(raw)
    assert obj["decision"]["selected_tool"] == "evotraders_tq_call"
    assert obj["matched_rule"]["rule_id"] == "stock-analysis-evidence-core"
    assert obj["decision"]["decision_args"]["stock_code"].startswith("600519")


def test_route_and_call_enforces_quote_first_with_stubbed_backend(monkeypatch):
    monkeypatch.setattr(mod, "_has_base", lambda: True)

    def fake_call_json(fn, **kwargs):
        if fn is mod.evotraders_tq_call and kwargs.get("method") == "get_market_snapshot":
            return {"ok": True, "data": {"price": 123.45}}
        if fn is mod.evotraders_tq_call and kwargs.get("method") == "get_more_info":
            return {"ok": True, "data": {"turnover": 1.23}}
        return {"ok": True}

    monkeypatch.setattr(mod, "_call_json", fake_call_json)

    raw = mod.evotraders_route_and_call(query="帮我分析 SH600000 走势")
    obj = json.loads(raw)
    assert obj["matched_rule_id"] == "stock-analysis-evidence-core"
    assert obj["selected_tool"] == "evotraders_tq_call"
    assert obj["result"]["stock_code"] == "600000.SH"
    assert len(obj["result"]["steps"]) >= 2


def test_buy_word_without_explicit_order_stays_in_analysis_route():
    raw = mod.evotraders_route_dry_run(query="帮我看 SH600000 现在适不适合买入")
    obj = json.loads(raw)
    assert obj["decision"]["selected_tool"] == "evotraders_tq_call"
    assert obj["matched_rule"]["rule_id"] == "stock-analysis-evidence-core"


def test_explicit_order_intent_routes_to_trade_buy():
    raw = mod.evotraders_route_dry_run(
        query="请下单买入 SH600000",
        stock_code="600000.SH",
        price=10.5,
        order_volume=1000,
        confirm=True,
    )
    obj = json.loads(raw)
    assert obj["decision"]["selected_tool"] == "evotraders_trade_buy"

