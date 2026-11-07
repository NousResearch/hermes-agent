"""Regression tests for web_search argument compatibility."""

from unittest.mock import patch
from tools.registry import registry
import pytest

from tools.web_tools import (
    _normalize_web_search_query,
    _web_search_handler,
)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"query": "Dramione"}, "Dramione"),
        ({"query": "  Dramione  "}, "Dramione"),
        ({"queries": ["Dramione"]}, "Dramione"),
        ({"queries": '["Dramione"]'}, "Dramione"),
        ({"queries": ["A", "B"]}, None),
        ({"queries": "invalid json"}, None),
        ({"queries": []}, None),
        ({"queries": [""]}, None),
        ({}, None),
        ({"query": 123}, None),
        (
            {"query": "Primary", "queries": ["Secondary"]},
            "Primary",
        ),
    ],
)


def test_normalize_web_search_query(args, expected):
    assert _normalize_web_search_query(args) == expected


def test_web_search_handler_passes_normalized_query():
    with patch(
        "tools.web_tools.web_search_tool",
        return_value="search results",
    ) as search:
        result = _web_search_handler(
            {"queries": '["Dramione"]', "limit": 3}
        )

    assert result == "search results"
    search.assert_called_once_with("Dramione", limit=3)


def test_web_search_handler_rejects_multiple_queries():
    with patch("tools.web_tools.web_search_tool") as search:
        result = _web_search_handler(
            {"queries": ["Dramione", "Harry Potter"]}
        )

    assert result.startswith("Invalid web_search arguments:")
    search.assert_not_called()


def test_registry_dispatch_accepts_single_queries_argument():
    """Registry dispatch forwards compatible arguments to web_search."""
    with patch(
        "tools.web_tools.web_search_tool",
        return_value="search results",
    ) as search:
        result = registry.dispatch(
            "web_search",
            {"queries": '["Dramione"]', "limit": 3},
        )

    assert result == "search results"
    search.assert_called_once_with("Dramione", limit=3)


def test_registry_dispatch_rejects_multiple_queries():
    """Invalid arguments must not reach the search backend."""
    with patch("tools.web_tools.web_search_tool") as search:
        result = registry.dispatch(
            "web_search",
            {"queries": ["Dramione", "Harry Potter"]},
        )

    assert "Invalid web_search arguments:" in result
    search.assert_not_called()
