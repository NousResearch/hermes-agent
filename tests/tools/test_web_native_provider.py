"""Regression tests for the native local-fetch extract provider.

Covers:
- Capability flags (extract-only, no search)
- ``web.native`` config loading / default merge
- SSRF pre-check before any request
- Per-hop redirect re-validation (a public URL must not redirect into a
  private/internal address)
- Too-many-redirects handling
- The happy-path extraction pipeline (skipped when optional deps absent)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from plugins.web.native import provider as native


# ---------------------------------------------------------------------------
# Fake httpx plumbing
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        text: str = "",
        is_redirect: bool = False,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self.is_redirect = is_redirect
        self.reason_phrase = "OK"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("err", request=None, response=self)  # type: ignore[arg-type]


class _FakeClient:
    """Async-context-manager stand-in that replays queued responses."""

    def __init__(self, responses: List[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.requested_urls: List[str] = []

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def get(self, url: str, headers: Optional[Dict[str, str]] = None) -> _FakeResponse:
        self.requested_urls.append(url)
        if not self._responses:
            raise AssertionError(f"unexpected extra request to {url}")
        return self._responses.pop(0)


def _install_fake_client(monkeypatch, responses: List[_FakeResponse]) -> _FakeClient:
    client = _FakeClient(responses)

    def _factory(*args: Any, **kwargs: Any) -> _FakeClient:
        return client

    monkeypatch.setattr(native.httpx, "AsyncClient", _factory)
    return client


def _patch_safety(monkeypatch, verdicts: List[bool]) -> None:
    """Patch async_is_safe_url to return successive verdicts from ``verdicts``."""
    seq = list(verdicts)

    async def _fake_safe(url: str) -> bool:
        return seq.pop(0) if seq else True

    import tools.url_safety as url_safety

    monkeypatch.setattr(url_safety, "async_is_safe_url", _fake_safe)


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_extract_only(self):
        p = native.WebFetchWebSearchProvider()
        assert p.name == "native"
        assert p.supports_search() is False
        assert p.supports_extract() is True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_when_config_missing(self, monkeypatch):
        def _boom():
            raise RuntimeError("no config")

        monkeypatch.setattr("hermes_cli.config.load_config", _boom)
        cfg = native._load_native_web_config()
        assert cfg["timeout"] == native._NATIVE_DEFAULTS["timeout"]
        assert cfg["max_redirects"] == 5
        assert cfg["readability"] is True

    def test_user_overrides_merge_over_defaults(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"web": {"native": {"timeout": 7, "max_chars": 123}}},
        )
        cfg = native._load_native_web_config()
        assert cfg["timeout"] == 7
        assert cfg["max_chars"] == 123
        # untouched keys keep their defaults
        assert cfg["max_redirects"] == native._NATIVE_DEFAULTS["max_redirects"]


# ---------------------------------------------------------------------------
# SSRF
# ---------------------------------------------------------------------------


class TestSSRF:
    @pytest.mark.asyncio
    async def test_precheck_blocks_before_request(self, monkeypatch):
        _patch_safety(monkeypatch, [False])
        client = _install_fake_client(monkeypatch, [])

        result = await native._fetch_single_url(
            "http://169.254.169.254/latest/meta-data/",
            cfg=dict(native._NATIVE_DEFAULTS),
        )

        assert "private or internal" in result["error"]
        assert client.requested_urls == []  # never fetched

    @pytest.mark.asyncio
    async def test_redirect_target_is_revalidated(self, monkeypatch):
        # Initial URL is safe, but it redirects to an internal address that
        # must be rejected before the second request is issued.
        _patch_safety(monkeypatch, [True, False])
        redirect = _FakeResponse(
            status_code=302,
            headers={"location": "http://169.254.169.254/"},
            is_redirect=True,
        )
        client = _install_fake_client(monkeypatch, [redirect])

        result = await native._fetch_single_url(
            "https://example.com/redirect",
            cfg=dict(native._NATIVE_DEFAULTS),
        )

        assert "private or internal" in result["error"]
        # Only the initial request happened; the unsafe hop was not followed.
        assert len(client.requested_urls) == 1

    @pytest.mark.asyncio
    async def test_too_many_redirects(self, monkeypatch):
        _patch_safety(monkeypatch, [True] * 20)
        cfg = dict(native._NATIVE_DEFAULTS)
        cfg["max_redirects"] = 2
        loops = [
            _FakeResponse(
                status_code=302,
                headers={"location": "https://example.com/next"},
                is_redirect=True,
            )
            for _ in range(5)
        ]
        _install_fake_client(monkeypatch, loops)

        result = await native._fetch_single_url("https://example.com/", cfg=cfg)
        assert "Too many redirects" in result["error"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestBackendSelection:
    """The extract-only native provider must not hijack search selection."""

    def _register_native_only(self):
        from agent.web_search_registry import register_provider, _reset_for_tests

        _reset_for_tests()
        register_provider(native.WebFetchWebSearchProvider())

    def test_native_not_auto_selected_as_shared_backend(self, monkeypatch):
        from tools import web_tools

        self._register_native_only()
        try:
            # No creds, no configured backend, native available and registered.
            monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
            monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
            monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
            monkeypatch.setattr(native.WebFetchWebSearchProvider, "is_available", lambda self: True)

            # Shared fallback must skip the extract-only provider and keep the
            # search-capable default rather than returning "native".
            assert web_tools._get_backend() != "native"
            assert web_tools._get_search_backend() != "native"
        finally:
            from agent.web_search_registry import _reset_for_tests
            _reset_for_tests()

    def test_native_selected_when_extract_backend_configured(self, monkeypatch):
        from tools import web_tools

        self._register_native_only()
        try:
            monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"extract_backend": "native"})
            monkeypatch.setattr(native.WebFetchWebSearchProvider, "is_available", lambda self: True)

            assert web_tools._get_extract_backend() == "native"
        finally:
            from agent.web_search_registry import _reset_for_tests
            _reset_for_tests()


class TestZeroConfigAutoFallback:
    """A fully unconfigured install with only free plugins should still work:
    ddgs auto-selected for search, native auto-selected for extract.
    """

    def _register_all_plus_native(self):
        from tests.tools.conftest import register_all_web_providers
        from agent.web_search_registry import register_provider

        register_all_web_providers()  # resets + registers the 8 built-ins
        register_provider(native.WebFetchWebSearchProvider())

    def _clear(self, monkeypatch):
        for k in (
            "BRAVE_SEARCH_API_KEY", "SEARXNG_URL", "TAVILY_API_KEY", "EXA_API_KEY",
            "PARALLEL_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL",
            "FIRECRAWL_GATEWAY_URL", "TOOL_GATEWAY_DOMAIN",
        ):
            monkeypatch.delenv(k, raising=False)
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(native.WebFetchWebSearchProvider, "is_available", lambda self: True)

    def test_no_keys_ddgs_installed_auto_selects_ddgs_and_native(self, monkeypatch):
        from tools import web_tools

        self._register_all_plus_native()
        try:
            self._clear(monkeypatch)
            monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
            monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)

            assert web_tools._get_search_backend() == "ddgs"
            assert web_tools._get_extract_backend() == "native"
        finally:
            from agent.web_search_registry import _reset_for_tests
            _reset_for_tests()

    def test_extract_falls_back_to_native_even_without_ddgs(self, monkeypatch):
        from tools import web_tools

        self._register_all_plus_native()
        try:
            self._clear(monkeypatch)
            monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
            monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)

            # Shared fallback resolves to the unavailable "firecrawl" default;
            # extract must still be rescued by the available native provider.
            assert web_tools._get_extract_backend() == "native"
        finally:
            from agent.web_search_registry import _reset_for_tests
            _reset_for_tests()

    def test_explicit_search_only_backend_is_respected_not_rescued(self, monkeypatch):
        from tools import web_tools

        self._register_all_plus_native()
        try:
            self._clear(monkeypatch)
            # web.backend EXPLICITLY set to a search-only backend. Auto-rescue
            # must NOT kick in — extract stays on searxng so the dispatcher can
            # surface the clear "search-only" error (existing contract).
            monkeypatch.setenv("SEARXNG_URL", "http://searx.example")
            monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "searxng"})

            assert web_tools._get_search_backend() == "searxng"
            assert web_tools._get_extract_backend() == "searxng"
        finally:
            from agent.web_search_registry import _reset_for_tests
            _reset_for_tests()

    def test_paid_extract_backend_still_wins_over_native(self, monkeypatch):
        from tools import web_tools

        self._register_all_plus_native()
        try:
            self._clear(monkeypatch)
            monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key")
            monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
            monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)

            # firecrawl is available and extract-capable → it wins; native is
            # NOT force-substituted.
            assert web_tools._get_extract_backend() == "firecrawl"
        finally:
            from agent.web_search_registry import _reset_for_tests
            _reset_for_tests()


class TestExtraction:
    @pytest.mark.asyncio
    async def test_extracts_markdown_from_html(self, monkeypatch):
        pytest.importorskip("readability")
        pytest.importorskip("html2text")

        _patch_safety(monkeypatch, [True])
        html = (
            "<html><head><title>Hello Title</title></head>"
            "<body><article><h1>Heading</h1>"
            "<p>Some readable paragraph content here.</p></article></body></html>"
        )
        ok = _FakeResponse(
            status_code=200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=html,
        )
        _install_fake_client(monkeypatch, [ok])

        result = await native._fetch_single_url(
            "https://example.com/article",
            cfg=dict(native._NATIVE_DEFAULTS),
        )

        assert result.get("error") in (None, "")
        assert "readable paragraph" in result["content"].lower()
        assert result["title"] == "Hello Title"

    @pytest.mark.asyncio
    async def test_max_chars_cap_is_enforced(self, monkeypatch):
        _patch_safety(monkeypatch, [True])
        # Non-text content-type returns the body verbatim (truncated to cap).
        big = "x" * 5000
        resp = _FakeResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            text=big,
        )
        _install_fake_client(monkeypatch, [resp])

        cfg = dict(native._NATIVE_DEFAULTS)
        cfg["max_chars"] = 100
        cfg["max_chars_cap"] = 100
        result = await native._fetch_single_url("https://example.com/data", cfg=cfg)

        assert result["content"].startswith("x" * 100)
        assert "truncated" in result["content"]
