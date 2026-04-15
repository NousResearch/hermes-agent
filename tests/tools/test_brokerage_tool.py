"""Tests for Hermes brokerage tool wrappers."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from brokerage.app import create_app
from brokerage.brokers.base import BrokerAdapter
from brokerage.config import BrokerageSettings
from brokerage.models import BrokerSubmissionResult, TradeIntent
from brokerage.policy import BrokeragePolicy
from brokerage.service import BrokerageService
from brokerage.storage import SQLiteBrokerageStore
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


class _IntegrationFakeBroker(BrokerAdapter):
    def __init__(self, result: BrokerSubmissionResult):
        self.result = result
        self.submitted: list[TradeIntent] = []

    def submit_order(self, intent: TradeIntent) -> BrokerSubmissionResult:
        self.submitted.append(intent)
        return self.result

    def get_order_status(self, order_id: str):
        return None

    def cancel_order(self, order_id: str):
        return None


def _make_integration_client(tmp_path: Path) -> tuple[TestClient, _IntegrationFakeBroker]:
    settings = BrokerageSettings(enabled=True, service_token="test-token")
    store = SQLiteBrokerageStore(tmp_path / "brokerage.db")
    policy = BrokeragePolicy(settings)
    broker = _IntegrationFakeBroker(
        BrokerSubmissionResult(
            accepted=True,
            broker_order_id="ib-int-123",
            broker_status="Submitted",
        )
    )
    service = BrokerageService(settings, store, policy, broker)
    app = create_app(service=service, auth_token="test-token")
    return TestClient(app), broker


def _patch_integration_transport(monkeypatch, client: TestClient) -> None:
    class _BridgeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, method, url, *, json=None, headers=None):
            response = client.request(method, url, json=json, headers=headers)
            return httpx.Response(
                status_code=response.status_code,
                json=response.json(),
                request=httpx.Request(method, url),
            )

    monkeypatch.setattr(brokerage_tool.httpx, "Client", _BridgeClient)
    monkeypatch.setattr(
        brokerage_tool,
        "_load_brokerage_config",
        lambda: {
            "enabled": True,
            "service_url": "http://testserver",
            "service_token": "test-token",
        },
    )


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
        "brokerage_health",
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


def test_tool_flow_can_reach_real_service_and_submit_trade(tmp_path, monkeypatch):
    client, broker = _make_integration_client(tmp_path)
    _patch_integration_transport(monkeypatch, client)

    created = json.loads(
        brokerage_tool.create_trade_intent_tool(
            {
                "account_mode": "paper",
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10,
                "order_type": "MARKET",
                "asset_class": "stock",
            }
        )
    )
    confirmed = json.loads(
        brokerage_tool.confirm_trade_intent_tool(
            {
                "intent_id": created["intent_id"],
                "confirmation_text": f"CONFIRM {created['confirmation_code']}",
            }
        )
    )
    status = json.loads(
        brokerage_tool.get_trade_intent_status_tool({"intent_id": created["intent_id"]})
    )

    assert created["status"] == "pending_confirmation"
    assert confirmed["status"] == "submitted"
    assert confirmed["broker_order_id"] == "ib-int-123"
    assert status["status"] == "submitted"
    assert [intent.symbol for intent in broker.submitted] == ["AAPL"]
