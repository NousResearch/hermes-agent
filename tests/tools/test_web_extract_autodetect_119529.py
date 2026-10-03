"""#119529: extract autodetect must skip search-only backends.

Fresh install + importable ddgs + no keys: ``_get_extract_backend()`` returned
``'ddgs'`` (capability-blind shared ladder), so ``web_extract`` died with a
terminal search-only typed error before the keyless extract ring was consulted.
"""
from __future__ import annotations

import json

import pytest

from tests.tools.conftest import register_all_web_providers


@pytest.fixture
def _clean_web_env(monkeypatch):
    from tools import web_tools

    register_all_web_providers()
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
    monkeypatch.setattr(web_tools, "_has_env", lambda _name: False)
    monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
    monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
    monkeypatch.setattr(web_tools, "selection_exists", lambda _section: False)
    yield web_tools
    from agent.web_search_registry import _reset_for_tests

    _reset_for_tests()


class TestExtractAutodetect119529:
    def test_extract_autodetect_skips_search_only_ddgs(self, _clean_web_env):
        web_tools = _clean_web_env
        backend = web_tools._get_extract_backend()
        assert backend != "ddgs", f"extract autodetect picked search-only {backend!r}"
        provider = web_tools._registered_web_provider(backend)
        assert provider is not None, f"no registered provider for {backend!r}"
        assert provider.supports_extract(), f"{backend!r} cannot extract"

    def test_explicit_ddgs_selection_keeps_strict_error(self, monkeypatch):
        from tools import web_tools
        from tools.web_tools_extract import _resolve_extract_provider

        register_all_web_providers()
        try:
            monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "ddgs"})
            assert web_tools._get_extract_backend() == "ddgs"
            _provider, error_json = _resolve_extract_provider("ddgs")
            assert error_json is not None
            assert "search-only" in json.loads(error_json)["error"].lower()
        finally:
            from agent.web_search_registry import _reset_for_tests

            _reset_for_tests()
