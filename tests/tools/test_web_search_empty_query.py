"""Regression tests for #72270 — an empty web_search query must be rejected
before it reaches any backend.

Models (especially small local ones) occasionally call web_search with
query=''. Without a guard the empty string is forwarded verbatim: SearXNG
receives ``/search?q=`` and answers HTTP 400, so the model only sees an
opaque "SearXNG returned HTTP 400" and tends to retry the same empty call.
The tool must fail fast with actionable guidance instead, for every backend,
without any plugin discovery or network dispatch.
"""

import json

import pytest

from tools import web_tools


def _forbid_backend_dispatch(monkeypatch):
    def _boom():
        raise AssertionError(
            "backend dispatch must not happen for an empty web_search query"
        )

    # _ensure_web_plugins_loaded is the first step of backend dispatch inside
    # web_search_tool, so reaching it means the guard did not fire.
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", _boom)


@pytest.mark.parametrize(
    "query",
    ["", "   ", "\n\t ", None, 5],
    ids=["empty", "spaces", "other-whitespace", "none", "non-string"],
)
def test_blank_or_invalid_query_is_rejected_before_dispatch(monkeypatch, query):
    _forbid_backend_dispatch(monkeypatch)

    payload = json.loads(web_tools.web_search_tool(query))

    assert payload["success"] is False
    assert "query" in payload["error"]


def test_valid_query_is_stripped_and_still_dispatched(monkeypatch):
    import agent.web_search_registry as web_search_registry

    seen = {}

    class _FakeProvider:
        name = "fake-backend"

        def supports_search(self):
            return True

        def search(self, query, limit):
            seen["query"] = query
            seen["limit"] = limit
            return {"success": True, "data": {"web": []}}

    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_get_search_backend", lambda: "fake-backend")
    monkeypatch.setattr(
        web_search_registry, "get_provider", lambda name: _FakeProvider()
    )

    payload = json.loads(
        web_tools.web_search_tool("  python packaging \n", limit=3)
    )

    assert payload["success"] is True
    assert seen == {"query": "python packaging", "limit": 3}
