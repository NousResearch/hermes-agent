"""Tests for the local (httpx + trafilatura) web extract provider.

Covers:
- LocalWebSearchProvider capability flags — extract-only, keyless
- is_available() — trafilatura importable / lazy-install-allowed fallback
- extract() happy path — result shape (url, title, content, raw_content,
  metadata) and markdown extraction via a stubbed trafilatura
- Manual redirect loop — final URL reported, per-hop SSRF re-check
  (redirect to a cloud metadata address is blocked), redirect cap
- Website-policy gate — blocked_by_policy shape, checked per hop
- Fetch hygiene — HTTP error status, unsupported content type, size cap,
  text/plain passthrough, per-URL error isolation
- html2txt fallback when readability extraction returns nothing
- Interrupt handling
- Registry integration — _is_backend_available("local") and
  _get_extract_backend() resolution
"""
from __future__ import annotations

import asyncio
import sys
import types

import httpx
import pytest

from tests.tools.conftest import register_all_web_providers

import plugins.web.local.provider as local_mod
from plugins.web.local.provider import LocalWebSearchProvider, _extract_one


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_fake_trafilatura(
    monkeypatch,
    *,
    extract_returns="Readable body text.",
    extract_raises=None,
    html2txt_returns="",
    title="Fake Title",
):
    """Install a stub ``trafilatura`` module in sys.modules for one test.

    The provider imports trafilatura lazily inside ``_extract_one``, so a
    ``sys.modules`` entry is all it takes — no network, no real package
    required for these unit tests (the venv does ship the real one).
    """
    fake = types.ModuleType("trafilatura")

    def _extract(content, url=None, output_format="markdown", **kwargs):
        if extract_raises is not None:
            raise extract_raises
        return extract_returns

    def _html2txt(content, **kwargs):
        return html2txt_returns

    class _Meta:
        pass

    def _extract_metadata(content, default_url=None, **kwargs):
        meta = _Meta()
        meta.title = title
        meta.author = None
        meta.description = None
        meta.sitename = None
        meta.date = None
        return meta

    fake.extract = _extract
    fake.html2txt = _html2txt
    fake.extract_metadata = _extract_metadata
    monkeypatch.setitem(sys.modules, "trafilatura", fake)
    return fake


def _use_transport(monkeypatch, handler):
    """Route the provider's httpx client through a MockTransport."""
    monkeypatch.setattr(
        local_mod, "_transport_for_tests", httpx.MockTransport(handler)
    )


def _allow_all_policy(monkeypatch):
    """Neutralize the website-access policy gate for a test."""
    monkeypatch.setattr(local_mod, "check_website_access", lambda url: None)


def _allow_all_ssrf(monkeypatch):
    """Neutralize the SSRF gate for tests that use fake public hosts."""
    monkeypatch.setattr(local_mod, "is_safe_url", lambda url: True)


HTML_PAGE = b"<html><head><title>T</title></head><body><p>Hello</p></body></html>"


# ---------------------------------------------------------------------------
# Capability flags
# ---------------------------------------------------------------------------


class TestFlags:
    def test_identity_and_capabilities(self):
        prov = LocalWebSearchProvider()
        assert prov.name == "local"
        assert prov.supports_search() is False
        assert prov.supports_extract() is True

    def test_search_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            LocalWebSearchProvider().search("query")

    def test_keyless_setup_schema(self):
        schema = LocalWebSearchProvider().get_setup_schema()
        assert schema["env_vars"] == []
        assert schema["post_setup"] == "local_extract"

    def test_is_available_when_trafilatura_importable(self, monkeypatch):
        _install_fake_trafilatura(monkeypatch)
        assert LocalWebSearchProvider().is_available() is True


# ---------------------------------------------------------------------------
# extract() — happy path and fetch hygiene
# ---------------------------------------------------------------------------


class TestExtract:
    def test_happy_path_shape(self, monkeypatch):
        _install_fake_trafilatura(monkeypatch, extract_returns="# T\n\nHello")
        _allow_all_policy(monkeypatch)
        _allow_all_ssrf(monkeypatch)
        _use_transport(
            monkeypatch,
            lambda request: httpx.Response(
                200, headers={"content-type": "text/html; charset=utf-8"},
                content=HTML_PAGE,
            ),
        )

        results = asyncio.run(
            LocalWebSearchProvider().extract(["https://example.com/a"])
        )

        assert len(results) == 1
        item = results[0]
        assert item["url"] == "https://example.com/a"
        assert item["title"] == "Fake Title"
        assert item["content"] == "# T\n\nHello"
        assert item["raw_content"] == "# T\n\nHello"
        assert item["metadata"]["sourceURL"] == "https://example.com/a"
        assert "error" not in item

    def test_redirect_followed_and_final_url_reported(self, monkeypatch):
        _install_fake_trafilatura(monkeypatch)
        _allow_all_policy(monkeypatch)
        _allow_all_ssrf(monkeypatch)

        def handler(request):
            if request.url.path == "/old":
                return httpx.Response(
                    301, headers={"location": "https://example.com/new"}
                )
            return httpx.Response(
                200, headers={"content-type": "text/html"}, content=HTML_PAGE
            )

        _use_transport(monkeypatch, handler)
        results = asyncio.run(
            LocalWebSearchProvider().extract(["https://example.com/old"])
        )
        assert results[0]["url"] == "https://example.com/new"
        assert "error" not in results[0]

    def test_redirect_to_metadata_endpoint_blocked(self, monkeypatch):
        """SSRF is re-checked per hop — a redirect into the cloud metadata
        address must be blocked BEFORE any request is sent to it. Uses the
        real is_safe_url: 169.254.169.254 is blocked unconditionally."""
        _install_fake_trafilatura(monkeypatch)
        _allow_all_policy(monkeypatch)

        def handler(request):
            assert request.url.host != "169.254.169.254", (
                "request must never reach the metadata endpoint"
            )
            return httpx.Response(
                302, headers={"location": "http://169.254.169.254/latest/"}
            )

        _use_transport(monkeypatch, handler)
        results = asyncio.run(
            LocalWebSearchProvider().extract(["https://example.com/"])
        )
        assert "private or internal" in results[0]["error"]

    def test_redirect_loop_capped(self, monkeypatch):
        _install_fake_trafilatura(monkeypatch)
        _allow_all_policy(monkeypatch)
        _allow_all_ssrf(monkeypatch)
        _use_transport(
            monkeypatch,
            lambda request: httpx.Response(
                302, headers={"location": "https://example.com/again"}
            ),
        )
        results = asyncio.run(
            LocalWebSearchProvider().extract(["https://example.com/loop"])
        )
        assert "redirects" in results[0]["error"].lower()

    def test_policy_block_shape(self, monkeypatch):
        _install_fake_trafilatura(monkeypatch)
        _allow_all_ssrf(monkeypatch)
        blocked = {
            "host": "example.com",
            "rule": "deny:example.com",
            "source": "config",
            "message": "Access to example.com is blocked by policy",
        }
        monkeypatch.setattr(local_mod, "check_website_access", lambda url: blocked)

        results = asyncio.run(
            LocalWebSearchProvider().extract(["https://example.com/"])
        )
        item = results[0]
        assert item["error"] == blocked["message"]
        assert item["blocked_by_policy"] == {
            "host": "example.com",
            "rule": "deny:example.com",
            "source": "config",
        }

    def test_http_error_status(self, monkeypatch):
        _install_fake_trafilatura(monkeypatch)
        _allow_all_policy(monkeypatch)
        _allow_all_ssrf(monkeypatch)
        _use_transport(monkeypatch, lambda request: httpx.Response(404))
        results = asyncio.run(
            LocalWebSearchProvider().extract(["https://example.com/gone"])
        )
        assert "404" in results[0]["error"]

    def test_unsupported_content_type(self, monkeypatch):
        _install_fake_trafilatura(monkeypatch)
        _allow_all_policy(monkeypatch)
        _allow_all_ssrf(monkeypatch)
        _use_transport(
            monkeypatch,
            lambda request: httpx.Response(
                200, headers={"content-type": "application/pdf"}, content=b"%PDF"
            ),
        )
        results = asyncio.run(
            LocalWebSearchProvider().extract(["https://example.com/doc.pdf"])
        )
        assert "content type" in results[0]["error"].lower()

    def test_size_cap(self, monkeypatch):
        _install_fake_trafilatura(monkeypatch)
        _allow_all_policy(monkeypatch)
        _allow_all_ssrf(monkeypatch)
        monkeypatch.setattr(local_mod, "_MAX_CONTENT_BYTES", 64)
        _use_transport(
            monkeypatch,
            lambda request: httpx.Response(
                200, headers={"content-type": "text/html"}, content=b"x" * 1024
            ),
        )
        results = asyncio.run(
            LocalWebSearchProvider().extract(["https://example.com/huge"])
        )
        assert "limit" in results[0]["error"]

    def test_text_plain_passthrough(self, monkeypatch):
        # No trafilatura stub on purpose — plain text must not need it.
        _allow_all_policy(monkeypatch)
        _allow_all_ssrf(monkeypatch)
        _use_transport(
            monkeypatch,
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/plain; charset=utf-8"},
                content="ligne un\nligne deux".encode(),
            ),
        )
        results = asyncio.run(
            LocalWebSearchProvider().extract(["https://example.com/readme.txt"])
        )
        assert results[0]["content"] == "ligne un\nligne deux"
        assert "error" not in results[0]

    def test_per_url_error_isolation(self, monkeypatch):
        _install_fake_trafilatura(monkeypatch)
        _allow_all_policy(monkeypatch)
        _allow_all_ssrf(monkeypatch)

        def handler(request):
            if request.url.path == "/bad":
                raise httpx.ConnectError("connection refused")
            return httpx.Response(
                200, headers={"content-type": "text/html"}, content=HTML_PAGE
            )

        _use_transport(monkeypatch, handler)
        results = asyncio.run(
            LocalWebSearchProvider().extract(
                ["https://example.com/bad", "https://example.com/good"]
            )
        )
        assert len(results) == 2
        assert "error" in results[0]
        assert "error" not in results[1]
        assert results[1]["url"] == "https://example.com/good"

    def test_html2txt_fallback_when_extract_empty(self, monkeypatch):
        _install_fake_trafilatura(
            monkeypatch, extract_returns=None, html2txt_returns="fallback text"
        )
        _allow_all_policy(monkeypatch)
        _allow_all_ssrf(monkeypatch)
        _use_transport(
            monkeypatch,
            lambda request: httpx.Response(
                200, headers={"content-type": "text/html"}, content=HTML_PAGE
            ),
        )
        results = asyncio.run(
            LocalWebSearchProvider().extract(["https://example.com/nav"])
        )
        assert results[0]["content"] == "fallback text"

    def test_no_readable_content_error(self, monkeypatch):
        _install_fake_trafilatura(
            monkeypatch, extract_returns=None, html2txt_returns=""
        )
        _allow_all_policy(monkeypatch)
        _allow_all_ssrf(monkeypatch)
        _use_transport(
            monkeypatch,
            lambda request: httpx.Response(
                200, headers={"content-type": "text/html"}, content=HTML_PAGE
            ),
        )
        results = asyncio.run(
            LocalWebSearchProvider().extract(["https://example.com/empty"])
        )
        assert "No readable content" in results[0]["error"]

    def test_interrupted(self, monkeypatch):
        import tools.interrupt as interrupt_mod

        monkeypatch.setattr(interrupt_mod, "is_interrupted", lambda: True)
        results = asyncio.run(
            LocalWebSearchProvider().extract(["https://example.com/a"])
        )
        assert results[0]["error"] == "Interrupted"


# ---------------------------------------------------------------------------
# Registry / dispatch integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_backend_available_via_registry(self, monkeypatch):
        _install_fake_trafilatura(monkeypatch)
        register_all_web_providers()
        try:
            from tools.web_tools import _is_backend_available

            assert _is_backend_available("local") is True
        finally:
            from agent.web_search_registry import _reset_for_tests

            _reset_for_tests()

    def test_extract_backend_resolves_local(self, monkeypatch):
        from unittest.mock import patch

        _install_fake_trafilatura(monkeypatch)
        register_all_web_providers()
        try:
            with patch(
                "tools.web_tools._load_web_config",
                return_value={"extract_backend": "local"},
            ):
                from tools.web_tools import _get_extract_backend

                assert _get_extract_backend() == "local"
        finally:
            from agent.web_search_registry import _reset_for_tests

            _reset_for_tests()

    def test_extract_only_provider_never_wins_search_autodetect(self, monkeypatch):
        """The keyless local provider is always available, so without a
        capability filter it would win ``_get_backend()``'s plugin walk for
        SEARCH too and reroute unconfigured-search dispatch (regression
        caught by test_unconfigured_search_emits_top_level_error). The walk
        must skip providers that cannot service the requested capability."""
        from unittest.mock import patch

        _install_fake_trafilatura(monkeypatch)
        register_all_web_providers()
        try:
            import tools.web_tools as web_tools

            monkeypatch.setattr(
                web_tools, "_ddgs_package_importable", lambda: False
            )
            monkeypatch.setattr(
                web_tools, "_peek_nous_access_token", lambda: None
            )
            with patch("tools.web_tools._load_web_config", return_value={}):
                # Search auto-detect must NOT pick the extract-only local
                # provider; with nothing configured it keeps the legacy
                # firecrawl default. Extract auto-detect may pick it.
                assert web_tools._get_search_backend() != "local"
                assert web_tools._get_backend("extract") == "local"
        finally:
            from agent.web_search_registry import _reset_for_tests

            _reset_for_tests()
