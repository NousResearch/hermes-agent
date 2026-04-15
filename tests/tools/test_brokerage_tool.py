"""Tests for Hermes brokerage tool wrappers."""

from __future__ import annotations

import json

import httpx

from model_tools import get_tool_definitions

import tools.brokerage_tool as brokerage_tool


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)
        self.request = httpx.Request("POST", "http://127.0.0.1:8787")

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("request failed", request=self.request, response=self)


class _FakeClient:
    def __init__(self, calls: list[dict], response: _FakeResponse):
        self._calls = calls
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def request(self, method, url, *, json=None, headers=None):
        self._calls.append({
            "method": method,
            "url": url,
            "json": json,
            "headers": headers,
        })
        return self._response


def test_brokerage_tools_register_under_brokerage_toolset(monkeypatch):
    monkeypatch.setattr(
        brokerage_tool,
        "_load_brokerage_config",
        lambda: {"enabled": True, "service_url": "http://127.0.0.1:8787", "service_token": "secret"},
    )

    tools = get_tool_definitions(enabled_toolsets=["brokerage"], quiet_mode=True)
    names = {tool["function"]["name"] for tool in tools}

    assert names == {
        "create_trade_intent",
        "confirm_trade_intent",
        "cancel_trade_intent",
        "get_trade_intent_status",
    }


def test_create_trade_intent_sends_expected_payload_and_auth_header(monkeypatch):
    calls = []
    monkeypatch.setattr(
        brokerage_tool,
        "_load_brokerage_config",
        lambda: {"enabled": True, "service_url": "http://127.0.0.1:8787", "service_token": "secret"},
    )
    monkeypatch.setattr(
        brokerage_tool.httpx,
        "Client",
        lambda timeout: _FakeClient(
            calls,
            _FakeResponse(
                201,
                {
                    "intent_id": "ti_123",
                    "status": "pending_confirmation",
                    "confirmation_code": "T-82K4",
                },
            ),
        ),
    )

    result = json.loads(
        brokerage_tool.create_trade_intent_tool(
            {
                "account_mode": "paper",
                "symbol": "aapl",
                "side": "buy",
                "quantity": 10,
                "order_type": "market",
                "asset_class": "stock",
                "raw_user_text": "buy 10 shares of aapl at market in paper",
            }
        )
    )

    assert result["intent_id"] == "ti_123"
    assert calls == [
        {
            "method": "POST",
            "url": "http://127.0.0.1:8787/trade-intents",
            "json": {
                "account_mode": "paper",
                "symbol": "aapl",
                "side": "buy",
                "quantity": 10,
                "order_type": "market",
                "asset_class": "stock",
                "raw_request_text": "buy 10 shares of aapl at market in paper",
            },
            "headers": {"Authorization": "Bearer secret"},
        }
    ]


def test_confirm_trade_intent_forwards_intent_id_and_confirmation_text(monkeypatch):
    calls = []
    monkeypatch.setattr(
        brokerage_tool,
        "_load_brokerage_config",
        lambda: {"enabled": True, "service_url": "http://127.0.0.1:8787", "service_token": None},
    )
    monkeypatch.setattr(
        brokerage_tool.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, _FakeResponse(200, {"status": "submitted"})),
    )

    result = json.loads(
        brokerage_tool.confirm_trade_intent_tool(
            {"intent_id": "ti_123", "confirmation_text": "CONFIRM T-82K4"}
        )
    )

    assert result == {"status": "submitted"}
    assert calls == [
        {
            "method": "POST",
            "url": "http://127.0.0.1:8787/trade-intents/ti_123/confirm",
            "json": {"confirmation_text": "CONFIRM T-82K4"},
            "headers": {},
        }
    ]


def test_cancel_trade_intent_calls_cancel_endpoint(monkeypatch):
    calls = []
    monkeypatch.setattr(
        brokerage_tool,
        "_load_brokerage_config",
        lambda: {"enabled": True, "service_url": "http://127.0.0.1:8787", "service_token": None},
    )
    monkeypatch.setattr(
        brokerage_tool.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, _FakeResponse(200, {"status": "cancelled"})),
    )

    result = json.loads(brokerage_tool.cancel_trade_intent_tool({"intent_id": "ti_123"}))

    assert result == {"status": "cancelled"}
    assert calls == [
        {
            "method": "POST",
            "url": "http://127.0.0.1:8787/trade-intents/ti_123/cancel",
            "json": None,
            "headers": {},
        }
    ]


def test_service_http_errors_become_json_errors(monkeypatch):
    monkeypatch.setattr(
        brokerage_tool,
        "_load_brokerage_config",
        lambda: {"enabled": True, "service_url": "http://127.0.0.1:8787", "service_token": "secret"},
    )
    monkeypatch.setattr(
        brokerage_tool.httpx,
        "Client",
        lambda timeout: _FakeClient(calls=[], response=_FakeResponse(400, {"detail": "bad confirmation"})),
    )

    result = json.loads(
        brokerage_tool.confirm_trade_intent_tool(
            {"intent_id": "ti_123", "confirmation_text": "CONFIRM T-WRONG"}
        )
    )

    assert result == {"error": "Brokerage service error (400): bad confirmation"}


def test_brokerage_toolset_is_unavailable_when_disabled(monkeypatch):
    monkeypatch.setattr(
        brokerage_tool,
        "_load_brokerage_config",
        lambda: {"enabled": False, "service_url": "http://127.0.0.1:8787", "service_token": "secret"},
    )

    tools = get_tool_definitions(enabled_toolsets=["brokerage"], quiet_mode=True)

    assert tools == []
