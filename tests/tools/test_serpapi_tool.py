"""Tests for SerpApi read-only credential/capability probe."""

from __future__ import annotations

import json

import requests


class _FakeResponse:
    def __init__(self, payload, *, status_code=200, text=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        return self._payload


def test_serpapi_probe_missing_secret_is_actionable(monkeypatch):
    from tools.serpapi_tool import serpapi_read_only_probe

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    result = json.loads(serpapi_read_only_probe())

    assert result["success"] is False
    assert result["missing_secret"] == "SERPAPI_API_KEY"
    assert "SERPAPI_API_KEY" in result["error"]


def test_serpapi_probe_success_redacts_secret(monkeypatch):
    from tools.serpapi_tool import serpapi_read_only_probe

    captured = {}

    def _fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "account_id": "acct_123",
                "api_key": "secret-must-not-leak",
                "account_email": "demo@example.com",
                "plan_name": "Developer",
                "total_searches_left": 42,
                "this_month_usage": 1,
            }
        )

    monkeypatch.setenv("SERPAPI_API_KEY", "secret-must-not-leak")
    monkeypatch.setattr("requests.get", _fake_get)

    result_text = serpapi_read_only_probe()
    result = json.loads(result_text)

    assert captured["url"] == "https://serpapi.com/account.json"
    assert captured["params"] == {"api_key": "secret-must-not-leak"}
    assert captured["timeout"] == 20
    assert result["success"] is True
    assert result["capability"] == "SerpApi account/read-only search capability confirmed"
    assert result["account"]["account_id"] == "acct_123"
    assert result["account"]["total_searches_left"] == 42
    assert "api_key" not in result["account"]
    assert "secret-must-not-leak" not in result_text


def test_serpapi_probe_invalid_key_failure_does_not_leak_secret(monkeypatch):
    from tools.serpapi_tool import serpapi_read_only_probe

    def _fake_get(url, params=None, timeout=None):
        return _FakeResponse({"error": "Invalid API key"}, status_code=401)

    monkeypatch.setenv("SERPAPI_API_KEY", "bad-secret")
    monkeypatch.setattr("requests.get", _fake_get)

    result_text = serpapi_read_only_probe()
    result = json.loads(result_text)

    assert result["success"] is False
    assert result["status_code"] == 401
    assert result["error"] == "Invalid API key"
    assert "bad-secret" not in result_text


def test_serpapi_probe_handles_timeout(monkeypatch):
    from tools.serpapi_tool import serpapi_read_only_probe

    def _fake_get(url, params=None, timeout=None):
        raise requests.Timeout("slow")

    monkeypatch.setenv("SERPAPI_API_KEY", "secret")
    monkeypatch.setattr("requests.get", _fake_get)

    result = json.loads(serpapi_read_only_probe())

    assert result["success"] is False
    assert result["error_type"] == "Timeout"
    assert "secret" not in json.dumps(result)


def test_serpapi_requirements_check(monkeypatch):
    from tools.serpapi_tool import check_serpapi_requirements

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    assert check_serpapi_requirements() is False

    monkeypatch.setenv("SERPAPI_API_KEY", "secret")
    assert check_serpapi_requirements() is True
