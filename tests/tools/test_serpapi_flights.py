"""Tests for SerpApi Google Flights search wrapper."""

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


def test_google_flights_missing_secret_is_actionable(monkeypatch):
    from tools.serpapi_tool import serpapi_google_flights_search

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    result = json.loads(
        serpapi_google_flights_search(
            origin="AMS",
            destination="JFK",
            departure_date="2026-10-25",
        )
    )

    assert result["success"] is False
    assert result["missing_secret"] == "SERPAPI_API_KEY"
    assert "SERPAPI_API_KEY" in result["error"]


def test_google_flights_builds_search_params_and_normalizes_best_flights(monkeypatch):
    from tools.serpapi_tool import serpapi_google_flights_search

    captured = {}

    def _fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "search_metadata": {
                    "id": "search-123",
                    "status": "Success",
                    "json_endpoint": "https://serpapi.com/searches/search-123.json",
                },
                "best_flights": [
                    {
                        "price": 432,
                        "type": "Round trip",
                        "airline_logo": "https://example.test/logo.png",
                        "booking_token": "opaque-token",
                        "flights": [
                            {
                                "departure_airport": {
                                    "name": "Amsterdam Airport Schiphol",
                                    "id": "AMS",
                                    "time": "2026-10-25 10:20",
                                },
                                "arrival_airport": {
                                    "name": "John F. Kennedy International Airport",
                                    "id": "JFK",
                                    "time": "2026-10-25 12:50",
                                },
                                "airline": "KLM",
                                "flight_number": "KL641",
                                "duration": 510,
                                "airplane": "Boeing 787",
                                "travel_class": "Economy",
                            }
                        ],
                        "total_duration": 510,
                    }
                ],
                "other_flights": [],
            }
        )

    monkeypatch.setenv("SERPAPI_API_KEY", "secret-must-not-leak")
    monkeypatch.setattr("requests.get", _fake_get)

    result_text = serpapi_google_flights_search(
        origin="AMS",
        destination="JFK",
        departure_date="2026-10-25",
        return_date="2026-11-01",
        adults=2,
        children=1,
        infants_in_seat=1,
        infants_on_lap=0,
        cabin_class="premium_economy",
        currency="EUR",
        limit=3,
    )
    result = json.loads(result_text)

    assert captured["url"] == "https://serpapi.com/search.json"
    assert captured["timeout"] == 30
    assert captured["params"] == {
        "engine": "google_flights",
        "api_key": "secret-must-not-leak",
        "departure_id": "AMS",
        "arrival_id": "JFK",
        "outbound_date": "2026-10-25",
        "return_date": "2026-11-01",
        "type": "1",
        "adults": 2,
        "children": 1,
        "infants_in_seat": 1,
        "infants_on_lap": 0,
        "travel_class": "2",
        "currency": "EUR",
        "hl": "en",
    }
    assert result["success"] is True
    assert result["provider"] == "serpapi"
    assert result["tool"] == "serpapi_google_flights_search"
    assert result["query"]["origin"] == "AMS"
    assert result["query"]["destination"] == "JFK"
    assert result["query"]["cabin_class"] == "premium_economy"
    assert "volatile" in result["caveat"].lower()
    assert result["results"][0]["price"] == {"amount": 432, "currency": "EUR"}
    assert result["results"][0]["airlines"] == ["KLM"]
    assert result["results"][0]["provider"] == "Google Flights via SerpApi"
    assert result["results"][0]["booking_link"] is None
    assert result["results"][0]["booking_token_available"] is True
    assert result["results"][0]["segments"][0]["flight_number"] == "KL641"
    assert result["source_metadata"]["search_id"] == "search-123"
    assert "secret-must-not-leak" not in result_text
    assert "opaque-token" not in result_text


def test_google_flights_one_way_omits_return_date_and_uses_one_way_type(monkeypatch):
    from tools.serpapi_tool import serpapi_google_flights_search

    captured = {}

    def _fake_get(url, params=None, timeout=None):
        captured["params"] = params
        return _FakeResponse({"search_metadata": {"status": "Success"}, "best_flights": []})

    monkeypatch.setenv("SERPAPI_API_KEY", "secret")
    monkeypatch.setattr("requests.get", _fake_get)

    result = json.loads(
        serpapi_google_flights_search(
            origin="LHR",
            destination="NRT",
            departure_date="2026-01-23",
            trip_type="one_way",
        )
    )

    assert captured["params"]["type"] == "2"
    assert "return_date" not in captured["params"]
    assert result["success"] is True


def test_google_flights_surfaces_serpapi_errors_without_leaking_secret(monkeypatch):
    from tools.serpapi_tool import serpapi_google_flights_search

    def _fake_get(url, params=None, timeout=None):
        return _FakeResponse({"error": "Invalid API key"}, status_code=401)

    monkeypatch.setenv("SERPAPI_API_KEY", "bad-secret")
    monkeypatch.setattr("requests.get", _fake_get)

    result_text = serpapi_google_flights_search(
        origin="AMS",
        destination="JFK",
        departure_date="2026-10-25",
    )
    result = json.loads(result_text)

    assert result["success"] is False
    assert result["status_code"] == 401
    assert result["error"] == "Invalid API key"
    assert "bad-secret" not in result_text


def test_google_flights_validates_required_inputs_before_network(monkeypatch):
    from tools.serpapi_tool import serpapi_google_flights_search

    def _fake_get(url, params=None, timeout=None):
        raise AssertionError("network should not be called for invalid input")

    monkeypatch.setenv("SERPAPI_API_KEY", "secret")
    monkeypatch.setattr("requests.get", _fake_get)

    result = json.loads(
        serpapi_google_flights_search(
            origin="AMS",
            destination="JFK",
            departure_date="not-a-date",
        )
    )

    assert result["success"] is False
    assert result["error_type"] == "validation_error"
    assert "departure_date" in result["error"]


if __name__ == "__main__":
    from unittest.mock import patch

    class _MonkeyPatch:
        def __init__(self):
            self._patches = []

        def setenv(self, key, value):
            patcher = patch.dict("os.environ", {key: value})
            patcher.start()
            self._patches.append(patcher)

        def delenv(self, key, raising=True):
            old = dict(__import__("os").environ)
            patcher = patch.dict("os.environ", old, clear=True)
            patcher.start()
            self._patches.append(patcher)
            __import__("os").environ.pop(key, None)

        def setattr(self, target, value):
            patcher = patch(target, value)
            patcher.start()
            self._patches.append(patcher)

        def undo(self):
            for patcher in reversed(self._patches):
                patcher.stop()

    tests = [
        test_google_flights_missing_secret_is_actionable,
        test_google_flights_builds_search_params_and_normalizes_best_flights,
        test_google_flights_one_way_omits_return_date_and_uses_one_way_type,
        test_google_flights_surfaces_serpapi_errors_without_leaking_secret,
        test_google_flights_validates_required_inputs_before_network,
    ]
    for test in tests:
        monkeypatch = _MonkeyPatch()
        try:
            test(monkeypatch)
        finally:
            monkeypatch.undo()
    print(f"passed {len(tests)} manual tests")
