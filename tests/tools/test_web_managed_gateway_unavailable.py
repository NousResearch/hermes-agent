"""Unavailable explicitly-selected Nous web gateway behavior."""

import json
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def unavailable_managed_web(tmp_path, monkeypatch, web_registry_populated):
    """Persist a managed selection while making every managed call impossible."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("web:\n  backend: nous\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli import nous_account
    from hermes_cli.nous_account import NousPortalAccountInfo
    from plugins.web.firecrawl import provider as firecrawl_provider
    from tools import managed_tool_gateway, web_tools
    from tools.registry import invalidate_check_fn_cache

    monkeypatch.setattr(
        nous_account,
        "get_nous_portal_account_info",
        lambda force_fresh=False: NousPortalAccountInfo(
            logged_in=False,
            source="none",
            fresh=False,
            portal_base_url="https://portal.example.test",
        ),
    )
    monkeypatch.setattr(managed_tool_gateway, "resolve_managed_tool_gateway", lambda *args, **kwargs: None)
    monkeypatch.setattr(web_tools, "_is_backend_available", lambda backend: False)
    monkeypatch.setattr("agent.web_search_registry.get_active_search_provider", lambda: None)
    monkeypatch.setattr("agent.web_search_registry.get_active_extract_provider", lambda: None)
    monkeypatch.setattr(
        firecrawl_provider,
        "Firecrawl",
        lambda **kwargs: pytest.fail("unavailable managed route attempted a Firecrawl SDK call"),
    )
    invalidate_check_fn_cache()
    yield
    invalidate_check_fn_cache()


def test_explicit_unavailable_managed_gateway_keeps_web_tools_visible(unavailable_managed_web):
    import tools.web_tools  # noqa: F401 - registers the tools
    from tools.registry import registry

    definitions = registry.get_definitions({"web_search", "web_extract"})

    assert {item["function"]["name"] for item in definitions} == {"web_search", "web_extract"}


def test_explicit_unavailable_managed_gateway_is_actionable_for_search_and_extract(
    unavailable_managed_web, monkeypatch
):
    from tools import web_tools
    from tools.registry import registry

    monkeypatch.setattr(web_tools, "async_is_safe_url", AsyncMock(return_value=True))

    search = json.loads(registry.dispatch("web_search", {"query": "offline fixture"}))
    extract = json.loads(registry.dispatch("web_extract", {"urls": ["https://example.test/page"]}))

    search_error = search["error"]
    extract_error = extract["results"][0]["error"]
    assert search_error == f"Error searching web: {extract_error}"
    for error in (search_error, extract_error):
        assert "explicitly selected Nous Tool Gateway" in error
        assert "not entitled or unreachable" not in error
        assert "run `hermes model`" in error
        assert "https://portal.example.test/billing" in error
        assert "Run `hermes tools` to choose a different web provider." in error
