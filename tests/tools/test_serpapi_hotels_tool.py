"""Tests for SerpApi Google Hotels search wrapper."""

from __future__ import annotations

import json

import requests


class _FakeResponse:
    def __init__(self, payload, *, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def test_google_hotels_missing_secret_is_actionable(monkeypatch):
    from tools.serpapi_tool import serpapi_google_hotels_search

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    result = json.loads(serpapi_google_hotels_search("Amsterdam", "2026-06-01", "2026-06-03"))

    assert result["success"] is False
    assert result["tool"] == "serpapi_google_hotels_search"
    assert result["missing_secret"] == "SERPAPI_API_KEY"
    assert "SERPAPI_API_KEY" in result["error"]


def test_google_hotels_success_normalizes_results_and_redacts_secret(monkeypatch):
    from tools.serpapi_tool import serpapi_google_hotels_search

    captured = {}

    def _fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "search_metadata": {"id": "abc", "status": "Success"},
                "search_parameters": {"engine": "google_hotels", "api_key": "secret-must-not-leak"},
                "properties": [
                    {
                        "name": "Canal Hotel",
                        "rate_per_night": {"lowest": "$123", "extracted_lowest": 123},
                        "source": "Booking.com",
                        "link": "https://example.invalid/hotel",
                        "address": "Amsterdam Centrum",
                        "overall_rating": 4.5,
                        "reviews": 321,
                        "hotel_class": "4-star hotel",
                        "property_token": "prop-token",
                        "api_key": "secret-must-not-leak",
                    }
                ],
            }
        )

    monkeypatch.setenv("SERPAPI_API_KEY", "secret-must-not-leak")
    monkeypatch.setattr("requests.get", _fake_get)

    result_text = serpapi_google_hotels_search(
        "Amsterdam",
        "2026-06-01",
        "2026-06-03",
        adults=2,
        rooms=1,
        currency="EUR",
        gl="nl",
        hl="en",
        filters={"sort_by": "3", "api_key": "ignored"},
    )
    result = json.loads(result_text)

    assert captured["url"] == "https://serpapi.com/search.json"
    assert captured["params"]["engine"] == "google_hotels"
    assert captured["params"]["q"] == "Amsterdam"
    assert captured["params"]["api_key"] == "secret-must-not-leak"
    assert captured["params"]["sort_by"] == "3"
    assert captured["timeout"] == 20
    assert result["success"] is True
    assert result["results_count"] == 1
    assert result["hotels"][0]["name"] == "Canal Hotel"
    assert result["hotels"][0]["price_rate"]["lowest"] == "$123"
    assert result["hotels"][0]["dates"] == {"check_in": "2026-06-01", "check_out": "2026-06-03"}
    assert result["hotels"][0]["provider_source"] == "Booking.com"
    assert result["hotels"][0]["location"] == "Amsterdam Centrum"
    assert result["hotels"][0]["rating"] == 4.5
    assert "rechecked" in result["caveat"]
    assert "secret-must-not-leak" not in result_text
    assert "api_key" not in result["query"]
    assert "api_key" not in result["search_parameters"]


def test_google_hotels_serpapi_error_is_clean_and_redacted(monkeypatch):
    from tools.serpapi_tool import serpapi_google_hotels_search

    def _fake_get(url, params=None, timeout=None):
        return _FakeResponse({"error": "Invalid API key"}, status_code=401)

    monkeypatch.setenv("SERPAPI_API_KEY", "bad-secret")
    monkeypatch.setattr("requests.get", _fake_get)

    result_text = serpapi_google_hotels_search("Amsterdam", "2026-06-01", "2026-06-03")
    result = json.loads(result_text)

    assert result["success"] is False
    assert result["status_code"] == 401
    assert result["error"] == "Invalid API key"
    assert "bad-secret" not in result_text
    assert "api_key" not in result["query"]


def test_google_hotels_handles_timeout(monkeypatch):
    from tools.serpapi_tool import serpapi_google_hotels_search

    def _fake_get(url, params=None, timeout=None):
        raise requests.Timeout("slow")

    monkeypatch.setenv("SERPAPI_API_KEY", "secret")
    monkeypatch.setattr("requests.get", _fake_get)

    result = json.loads(serpapi_google_hotels_search("Amsterdam", "2026-06-01", "2026-06-03"))

    assert result["success"] is False
    assert result["error_type"] == "Timeout"
    assert "secret" not in json.dumps(result)
    assert "api_key" not in result["query"]
