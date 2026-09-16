"""A per-capability web backend key must not reroute the other capability.

Regression for #113017: ``web.extract_backend: keenable`` (with no
``web.backend``) made ``_get_search_backend()`` resolve to ``firecrawl``
through the shared ``_get_backend()`` fallback, because
``selection_exists("web")`` counts per-capability keys as a stored selection.
"""
from __future__ import annotations


def _ladder_env(monkeypatch, raw_web):
    from tools import web_tools
    from tools import tool_backend_helpers as helpers

    monkeypatch.setattr(web_tools, "_load_web_config", lambda: dict(raw_web))
    monkeypatch.setattr(
        helpers, "_raw_section",
        lambda section: dict(raw_web) if section == "web" else None,
    )
    # No keyed backends, no gateway: the ladder can only reach ddgs.
    for var in (
        "TAVILY_API_KEY", "PERPLEXITY_API_KEY", "EXA_API_KEY",
        "PARALLEL_API_KEY", "KEENABLE_API_KEY",
        "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL",
        "BRAVE_SEARCH_API_KEY", "SEARXNG_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
    monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)


def test_extract_only_config_does_not_reroute_search(monkeypatch):
    from tools import web_tools

    _ladder_env(monkeypatch, {"extract_backend": "keenable"})
    # Broken before the fix: search resolved to "firecrawl" through the shared fallback.
    assert web_tools._get_search_backend() == "ddgs"
    assert web_tools._get_extract_backend() == "keenable"


def test_managed_use_gateway_still_resolves_firecrawl(monkeypatch):
    from tools import web_tools

    _ladder_env(monkeypatch, {"use_gateway": True})
    # Ladder would find ddgs if it ran — the managed selection must not reach it.
    assert web_tools._get_backend() == "firecrawl"
