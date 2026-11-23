"""A search-only direct credential beats the autodetected managed gateway for web_search only.

With no stored web selection, BRAVE_SEARCH_API_KEY / SEARXNG_URL must keep web_search on the user's own
key even when an entitled Nous account makes the Tool Gateway ready; extract (which those backends
cannot serve) stays on the gateway's firecrawl. Stored selections and paid keys behave as before.
"""
from __future__ import annotations

import pytest

_KEYS = (
    "TAVILY_API_KEY", "PERPLEXITY_API_KEY", "EXA_API_KEY", "PARALLEL_API_KEY", "KEENABLE_API_KEY",
    "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "BRAVE_SEARCH_API_KEY", "SEARXNG_URL",
)


def _env(monkeypatch, raw_web=None, selection=None, **env):
    from tools import tool_backend_helpers as helpers
    from tools import web_tools

    raw_web = dict(raw_web or {})
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: dict(raw_web))
    monkeypatch.setattr(helpers, "_raw_section", lambda section: dict(raw_web) if section == "web" else None)
    monkeypatch.setattr(web_tools, "read_selection", lambda name: selection)
    for var in _KEYS:
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
    monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: True)  # entitled Nous account
    return web_tools


@pytest.mark.parametrize("env,expected", [
    ({"BRAVE_SEARCH_API_KEY": "k"}, "brave-free"),
    ({"SEARXNG_URL": "https://s.test"}, "searxng"),
    ({"SEARXNG_URL": "https://s.test", "BRAVE_SEARCH_API_KEY": "k"}, "searxng"),  # existing ladder order
])
def test_search_only_key_beats_gateway_for_search_but_extract_stays_on_gateway(monkeypatch, env, expected):
    wt = _env(monkeypatch, **env)

    assert wt._managed_web_search() is False
    assert wt._get_search_backend() == expected
    assert wt._get_extract_backend() == "firecrawl"


def test_stored_nous_selection_still_routes_search_to_managed(monkeypatch):
    wt = _env(monkeypatch, {"backend": "nous"}, selection="nous", BRAVE_SEARCH_API_KEY="k")

    assert wt._managed_web_search() is True
    assert wt._get_search_backend() == "perplexity"


def test_stored_search_pin_and_stored_backend_keep_winning(monkeypatch):
    wt = _env(monkeypatch, {"search_backend": "exa"}, selection="exa", BRAVE_SEARCH_API_KEY="k")
    assert wt._get_search_backend() == "exa"

    wt = _env(monkeypatch, {"backend": "firecrawl"}, selection="firecrawl", BRAVE_SEARCH_API_KEY="k")
    assert wt._get_search_backend() == "firecrawl"


def test_gateway_alone_and_paid_key_behaviour_unchanged(monkeypatch):
    wt = _env(monkeypatch)
    assert wt._managed_web_search() is True
    assert wt._get_search_backend() == "perplexity"

    wt = _env(monkeypatch, TAVILY_API_KEY="t", BRAVE_SEARCH_API_KEY="k")
    assert wt._managed_web_search() is False
    assert wt._get_search_backend() == "tavily"
    assert wt._get_extract_backend() == "tavily"
