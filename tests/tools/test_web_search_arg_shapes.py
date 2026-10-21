"""web_search argument-shape contracts.

Guards two behaviours that a bare ``args.get("query", "")`` handler broke:
an empty query was forwarded to a paid backend (burning a request plus a
paid keyless-rescue retry) and returned a provider error that never said the
parameter was missing, and the ``queries`` shape models reach for silently
became an empty search instead of the search the model asked for.
"""
import json

import pytest


@pytest.fixture(autouse=True)
def _registry():
    import model_tools
    model_tools.discover_builtin_tools()


def _dispatch(args):
    from tools.registry import registry
    result = registry.dispatch("web_search", args)
    return result if isinstance(result, dict) else json.loads(result)


def test_missing_query_fails_without_calling_a_backend(monkeypatch):
    """An unusable query is rejected locally, so no paid request is sent."""
    import tools.web_tools as web_tools

    called = []
    monkeypatch.setattr(
        web_tools, "_memoized_search",
        lambda *a, **kw: called.append(a) or {"success": True, "data": {"web": []}},
    )

    for args in ({}, {"query": ""}, {"query": "   "}, {"query": None}):
        result = _dispatch(args)
        assert result["success"] is False, args
        # The message must name the parameter; a provider's own error does not.
        assert "query" in result["error"]

    assert called == [], "an unusable query must not reach a search backend"


def test_plural_queries_resolves_to_a_query(monkeypatch):
    """`queries` (list or JSON-encoded list) searches instead of sending empty."""
    import tools.web_tools as web_tools

    seen = []

    def _fake(provider, query, limit):
        seen.append(query)
        return {"success": True, "data": {"web": []}}

    monkeypatch.setattr(web_tools, "_memoized_search", _fake)

    for args in (
        {"queries": ["standards based grading"]},
        {"queries": '["standards based grading", "second"]'},
        {"queries": "standards based grading"},
        {"query": ["standards based grading"]},
    ):
        seen.clear()
        result = _dispatch(args)
        assert result["success"] is True, args
        assert seen and seen[0].strip(), f"{args} produced an empty query"


def test_singular_query_is_passed_through_unchanged(monkeypatch):
    """The documented shape keeps working, operators and all."""
    import tools.web_tools as web_tools

    seen = []
    monkeypatch.setattr(
        web_tools, "_memoized_search",
        lambda provider, query, limit: seen.append((query, limit)) or {"success": True, "data": {"web": []}},
    )

    assert _dispatch({"query": 'site:example.com "exact phrase"', "limit": 3})["success"] is True
    assert seen[0][0] == 'site:example.com "exact phrase"'
    assert seen[0][1] == 3


def test_schema_documents_the_singular_parameter():
    """The schema must steer models away from the plural form."""
    from tools.web_tools import WEB_SEARCH_SCHEMA

    assert "query" in WEB_SEARCH_SCHEMA["parameters"]["properties"]
    assert WEB_SEARCH_SCHEMA["parameters"]["required"] == ["query"]
    assert "queries" not in WEB_SEARCH_SCHEMA["parameters"]["properties"]
    assert "queries" in WEB_SEARCH_SCHEMA["description"]
