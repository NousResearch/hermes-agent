"""Tests for the web_recent helper and tool wiring."""

import json

import tools.web_tools as web_tools
from tools.web_tools import get_last_web_results


def test_get_last_web_results_returns_error_when_empty_cache():
    # Ensure cache is empty for this test
    web_tools._last_web_results = []

    data = json.loads(get_last_web_results())

    assert data["results"] == []
    assert "error" in data


def test_get_last_web_results_returns_recent_metadata():
    web_tools._last_web_results = [
        {"url": "https://example.com/1", "title": "First", "preview": "one", "error": None},
        {"url": "https://example.com/2", "title": "Second", "preview": "two", "error": None},
    ]

    data = json.loads(get_last_web_results())

    assert len(data["results"]) == 2
    assert data["results"][0]["url"] == "https://example.com/1"
    assert data["results"][1]["title"] == "Second"

