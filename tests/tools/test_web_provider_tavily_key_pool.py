"""Tests for Tavily multi-key rotation/failover."""

from unittest.mock import MagicMock

import httpx


def test_get_tavily_api_keys_pool_dedupes(monkeypatch):
    from plugins.web.tavily import provider

    monkeypatch.setenv("TAVILY_API_KEYS", "key-a, key-b\nkey-a")
    monkeypatch.setenv("TAVILY_API_KEY", "key-c")
    monkeypatch.setenv("TAVILY_API_KEY_BACKUP", "key-b")
    monkeypatch.setenv("TAVILY_API_KEY_2", "key-d")

    assert provider._get_tavily_api_keys() == ["key-a", "key-b", "key-c", "key-d"]


def test_tavily_request_rotates_and_falls_back_on_quota(monkeypatch):
    from plugins.web.tavily import provider

    monkeypatch.setenv("TAVILY_API_KEYS", "quota-key,good-key")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(provider, "_TAVILY_KEY_CURSOR", 0)

    quota_response = MagicMock()
    quota_response.status_code = 432
    quota_response.text = "usage limit exceeded"
    quota_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "432 Usage Limit Exceeded",
        request=MagicMock(),
        response=quota_response,
    )

    ok_response = MagicMock()
    ok_response.raise_for_status = MagicMock()
    ok_response.json.return_value = {"results": [{"title": "ok"}]}

    calls = []

    def fake_post(url, *, json, headers, timeout):  # noqa: A002 - match httpx kw
        calls.append(json["api_key"])
        return quota_response if json["api_key"] == "quota-key" else ok_response

    monkeypatch.setattr(provider.httpx, "post", fake_post)

    result = provider._tavily_request("search", {"query": "hello"})

    assert result == {"results": [{"title": "ok"}]}
    assert calls == ["quota-key", "good-key"]


def test_tavily_request_round_robins_start_key(monkeypatch):
    from plugins.web.tavily import provider

    monkeypatch.setenv("TAVILY_API_KEYS", "key-a,key-b")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(provider, "_TAVILY_KEY_CURSOR", 0)

    used = []

    def fake_post(url, *, json, headers, timeout):  # noqa: A002 - match httpx kw
        used.append(json["api_key"])
        ok_response = MagicMock()
        ok_response.raise_for_status = MagicMock()
        ok_response.json.return_value = {"results": []}
        return ok_response

    monkeypatch.setattr(provider.httpx, "post", fake_post)

    provider._tavily_request("search", {"query": "one"})
    provider._tavily_request("search", {"query": "two"})

    assert used == ["key-a", "key-b"]
