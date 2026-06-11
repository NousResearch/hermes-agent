"""Tests for Vapi/textbee.dev telephony tools."""

import json

from tools import telephony_tools as tt


def test_vapi_available_requires_api_key(monkeypatch):
    monkeypatch.delenv("VAPI_API_KEY", raising=False)
    assert tt._check_vapi_available() is False
    monkeypatch.setenv("VAPI_API_KEY", "test-key")
    assert tt._check_vapi_available() is True


def test_textbee_available_requires_api_key(monkeypatch):
    monkeypatch.delenv("TEXTBEE_API_KEY", raising=False)
    assert tt._check_textbee_available() is False
    monkeypatch.setenv("TEXTBEE_API_KEY", "test-key")
    assert tt._check_textbee_available() is True


def test_textbee_send_sms_validation(monkeypatch):
    monkeypatch.setenv("TEXTBEE_API_KEY", "test-key")
    monkeypatch.setenv("TEXTBEE_DEVICE_ID", "device-1")

    result = json.loads(tt._handle_textbee_send_sms({"recipients": ["555"], "message": "hi"}))

    assert result["success"] is False
    assert "E.164" in result["error"]


def test_textbee_send_sms_posts_expected_payload(monkeypatch):
    calls = []

    def fake_http(method, url, *, headers, payload=None, timeout=30):
        calls.append({"method": method, "url": url, "headers": headers, "payload": payload})
        return {"ok": True, "status_code": 200, "data": {"messageId": "m1"}}

    monkeypatch.setenv("TEXTBEE_API_KEY", "test-key")
    monkeypatch.setenv("TEXTBEE_DEVICE_ID", "device-1")
    monkeypatch.setattr(tt, "_http_json", fake_http)

    result = json.loads(tt._handle_textbee_send_sms({
        "recipients": ["+12345678900"],
        "message": "Hello",
        "sim_subscription_id": 1,
    }))

    assert result["success"] is True
    assert result["recipients_count"] == 1
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"].endswith("/api/v1/gateway/devices/device-1/send-sms")
    assert calls[0]["headers"]["x-api-key"] == "test-key"
    assert calls[0]["payload"] == {
        "recipients": ["+12345678900"],
        "message": "Hello",
        "simSubscriptionId": 1,
    }


def test_vapi_create_call_requires_defaults_or_args(monkeypatch):
    monkeypatch.setenv("VAPI_API_KEY", "test-key")
    monkeypatch.delenv("VAPI_ASSISTANT_ID", raising=False)
    monkeypatch.delenv("VAPI_PHONE_NUMBER_ID", raising=False)

    result = json.loads(tt._handle_vapi_create_call({"customer_number": "+12345678900"}))

    assert result["success"] is False
    assert "phone_number_id" in result["error"]


def test_vapi_create_call_posts_expected_payload(monkeypatch):
    calls = []

    def fake_http(method, url, *, headers, payload=None, timeout=30):
        calls.append({"method": method, "url": url, "headers": headers, "payload": payload})
        return {"ok": True, "status_code": 201, "data": {"id": "call-1", "status": "scheduled", "transcript": "hidden"}}

    monkeypatch.setenv("VAPI_API_KEY", "test-key")
    monkeypatch.setenv("VAPI_ASSISTANT_ID", "asst-1")
    monkeypatch.setenv("VAPI_PHONE_NUMBER_ID", "pn-1")
    monkeypatch.setattr(tt, "_http_json", fake_http)

    result = json.loads(tt._handle_vapi_create_call({
        "customer_number": "+12345678900",
        "customer_name": "Jane",
        "metadata": {"source": "test"},
    }))

    assert result["success"] is True
    assert result["call"]["id"] == "call-1"
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"].endswith("/call")
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert calls[0]["payload"] == {
        "phoneNumberId": "pn-1",
        "assistantId": "asst-1",
        "customer": {"number": "+12345678900", "name": "Jane"},
        "metadata": {"source": "test"},
    }


def test_vapi_get_call_quotes_call_id(monkeypatch):
    calls = []

    def fake_http(method, url, *, headers, payload=None, timeout=20):
        calls.append({"method": method, "url": url, "headers": headers})
        return {"ok": True, "status_code": 200, "data": {"id": "call/1", "status": "ended"}}

    monkeypatch.setenv("VAPI_API_KEY", "test-key")
    monkeypatch.setattr(tt, "_http_json", fake_http)

    result = json.loads(tt._handle_vapi_get_call({"call_id": "call/1"}))

    assert result["success"] is True
    assert calls[0]["method"] == "GET"
    assert calls[0]["url"].endswith("/call/call%2F1")
