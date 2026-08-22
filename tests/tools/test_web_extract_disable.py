"""Regression tests for explicitly disabling web_extract.

An explicit ``web.extract_backend: disabled`` must be a hard off-switch for
web_extract. It should not silently fall through to managed Firecrawl or any
other available extract backend.
"""
from __future__ import annotations

import asyncio
import json


class TestWebExtractDisabled:
    def test_extract_backend_disabled_resolves_to_disabled_even_when_fallback_available(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"backend": "", "extract_backend": "disabled"},
        )
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: True)
        monkeypatch.setattr(web_tools, "check_firecrawl_api_key", lambda: True)

        assert web_tools._get_extract_backend() == "disabled"

    def test_none_is_not_treated_as_disabled(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"backend": "", "extract_backend": "none"},
        )

        assert web_tools._get_extract_backend() == "none"
        assert web_tools._is_web_extract_disabled("none") is False

    def test_web_extract_tool_returns_disabled_error_without_loading_providers(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"backend": "", "extract_backend": "disabled"},
        )
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: True)
        monkeypatch.setattr(web_tools, "check_firecrawl_api_key", lambda: True)

        def fail_if_called():  # pragma: no cover - only runs on regression
            raise AssertionError("disabled web_extract must not load provider plugins")

        monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", fail_if_called)

        result = json.loads(asyncio.run(web_tools.web_extract_tool(["https://example.com"])))

        assert result["success"] is False
        assert "web_extract disabled" in result["error"]

    def test_check_web_extract_api_key_false_when_disabled_even_if_firecrawl_available(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"backend": "", "extract_backend": "disabled"},
        )
        monkeypatch.setattr(web_tools, "_is_backend_available", lambda backend: backend == "firecrawl")

        assert web_tools.check_web_extract_api_key() is False
