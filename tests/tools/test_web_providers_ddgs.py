"""Tests for the DuckDuckGo (ddgs) web search provider.

Covers:
- DDGSWebSearchProvider.is_available() — reflects package importability
- DDGSWebSearchProvider.search() — happy path, missing package, runtime error
- Result normalization (title, url, description, position)
- DDGSWebSearchProvider.extract() — happy path, missing package, timeout, errors
- _is_backend_available("ddgs") / _get_backend() integration
- web_extract dispatches to ddgs when configured (ddgs now supports_extract())
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from tests.tools.conftest import register_all_web_providers


def _install_fake_ddgs(
    monkeypatch, *, text_results=None, text_raises=None, text_sleep=None,
    extract_result=None, extract_raises=None, extract_sleep=None,
    extract_fn=None,
):
    """Install a stub ``ddgs`` module in sys.modules for the duration of a test.

    ``text_results``: iterable of dicts to yield from DDGS().text(...).
    ``text_raises``: if set, DDGS().text raises this exception instead.
    ``text_sleep``: if set, DDGS().text blocks for this many seconds before
        yielding — simulates a hung/slow search for the timeout test.
    ``extract_result``: dict returned from DDGS().extract(...) (defaults to
        ``{"url": <url>, "content": "<content>"}``).
    ``extract_raises``: if set, DDGS().extract raises this exception instead.
    ``extract_sleep``: if set, DDGS().extract blocks for this many seconds
        before returning — simulates a hung/slow fetch for the timeout test.
    ``extract_fn``: if set, a ``(url, fmt) -> dict`` callable that fully
        controls DDGS().extract(...) — takes priority over the other
        extract_* kwargs. Use when different URLs need different behavior.
    """
    import time as _time

    fake = types.ModuleType("ddgs")

    class _FakeDDGS:
        def __init__(self, **kwargs):
            # Accept timeout= (and any other constructor kwargs) — the provider
            # now passes DDGS(timeout=10).
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False
        def text(self, query, max_results=5):
            if text_sleep is not None:
                _time.sleep(text_sleep)
            if text_raises is not None:
                raise text_raises
            for hit in (text_results or []):
                yield hit
        def extract(self, url, fmt="text_markdown"):
            if extract_fn is not None:
                return extract_fn(url, fmt)
            if extract_sleep is not None:
                _time.sleep(extract_sleep)
            if extract_raises is not None:
                raise extract_raises
            if extract_result is not None:
                return extract_result
            return {"url": url, "content": "<content>"}

    fake.DDGS = _FakeDDGS  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ddgs", fake)
    return fake


# ---------------------------------------------------------------------------
# DDGSWebSearchProvider unit tests
# ---------------------------------------------------------------------------


class TestDDGSProviderIsConfigured:
    def test_configured_when_package_importable(self, monkeypatch):
        _install_fake_ddgs(monkeypatch)
        # Drop any cached ``plugins.web.ddgs.provider`` so is_configured re-imports ddgs fresh
        monkeypatch.delitem(sys.modules, "plugins.web.ddgs.provider", raising=False)
        from plugins.web.ddgs.provider import DDGSWebSearchProvider
        assert DDGSWebSearchProvider().is_available() is True

    def test_not_configured_when_package_missing(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "ddgs", raising=False)
        monkeypatch.delitem(sys.modules, "plugins.web.ddgs.provider", raising=False)
        # Block the import so ``import ddgs`` raises ImportError even if the package is actually installed
        import builtins
        orig_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "ddgs":
                raise ImportError("blocked for test")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        from plugins.web.ddgs.provider import DDGSWebSearchProvider
        assert DDGSWebSearchProvider().is_available() is False

    def test_provider_name(self):
        from plugins.web.ddgs.provider import DDGSWebSearchProvider
        assert DDGSWebSearchProvider().name == "ddgs"

    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.ddgs.provider import DDGSWebSearchProvider
        assert issubclass(DDGSWebSearchProvider, WebSearchProvider)


class TestDDGSProviderSearch:
    def test_happy_path_normalizes_results(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_results=[
            {"title": "A", "href": "https://a.example.com", "body": "desc A"},
            {"title": "B", "href": "https://b.example.com", "body": "desc B"},
            {"title": "C", "href": "https://c.example.com", "body": "desc C"},
        ])
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        result = DDGSWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0] == {"title": "A", "url": "https://a.example.com", "description": "desc A", "position": 1}
        assert web[2]["position"] == 3

    def test_accepts_url_key_as_fallback_for_href(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_results=[
            {"title": "A", "url": "https://a.example.com", "body": "desc A"},
        ])
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        result = DDGSWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        assert result["data"]["web"][0]["url"] == "https://a.example.com"

    def test_limit_is_respected(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_results=[
            {"title": f"R{i}", "href": f"https://r{i}.example.com", "body": ""}
            for i in range(10)
        ])
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        result = DDGSWebSearchProvider().search("q", limit=3)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 3

    def test_missing_package_returns_failure(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "ddgs", raising=False)
        monkeypatch.delitem(sys.modules, "plugins.web.ddgs.provider", raising=False)
        import builtins
        orig_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "ddgs":
                raise ImportError("blocked for test")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        result = DDGSWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "ddgs" in result["error"].lower()

    def test_runtime_error_returns_failure(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_raises=RuntimeError("rate limited 202"))
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        result = DDGSWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "rate limited" in result["error"] or "failed" in result["error"].lower()

    def test_empty_results(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_results=[])
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        result = DDGSWebSearchProvider().search("nothing", limit=5)
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_hung_search_times_out_and_returns_failure(self, monkeypatch):
        """#36776: a ddgs call that never returns must be bounded by the
        wall-clock timeout and surface a failure instead of hanging the
        shared agent loop. We patch the blocking helper to wait on an Event
        (released in finally so no worker thread leaks past the test) and
        shrink the timeout; search() must return success=False promptly."""
        import threading
        import time

        # ddgs must import-probe True for search() to proceed.
        _install_fake_ddgs(monkeypatch)
        monkeypatch.delitem(sys.modules, "plugins.web.ddgs.provider", raising=False)
        import plugins.web.ddgs.provider as _prov

        release = threading.Event()

        def _blocking_search(query, safe_limit):
            release.wait(timeout=10)  # bounded so the worker can never truly leak
            return []

        monkeypatch.setattr(_prov, "_run_ddgs_search", _blocking_search, raising=True)
        monkeypatch.setattr(_prov, "_SEARCH_TIMEOUT_SECS", 0.3, raising=True)

        try:
            start = time.monotonic()
            result = _prov.DDGSWebSearchProvider().search("hangs forever", limit=5)
            elapsed = time.monotonic() - start

            assert result["success"] is False
            assert "timed out" in result["error"].lower()
            # Returned well before the worker's 10s wait — proves the cap fired.
            assert elapsed < 3.0, f"search did not return promptly ({elapsed:.1f}s)"
        finally:
            release.set()  # let the orphaned worker finish immediately

    def test_fast_search_not_affected_by_timeout_wrapper(self, monkeypatch):
        """Happy-path guard: the timeout wrapper must not break a normal,
        fast search — results flow through unchanged."""
        _install_fake_ddgs(
            monkeypatch,
            text_results=[{"title": "T", "href": "https://e.com", "body": "B"}],
        )
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        result = DDGSWebSearchProvider().search("q", limit=5)
        assert result["success"] is True
        assert result["data"]["web"][0]["url"] == "https://e.com"
        assert result["data"]["web"][0]["title"] == "T"


# ---------------------------------------------------------------------------
# Integration: _is_backend_available / _get_backend / check_web_api_key
# ---------------------------------------------------------------------------


class TestDDGSBackendWiring:
    def test_is_backend_available_true_when_package_importable(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        assert web_tools._is_backend_available("ddgs") is True

    def test_is_backend_available_false_when_package_missing(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._is_backend_available("ddgs") is False

    def test_configured_backend_accepted(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "ddgs"})
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        assert web_tools._get_backend() == "ddgs"

    def test_ddgs_trails_paid_providers_in_auto_detect(self, monkeypatch):
        """Exa (priority) should win over ddgs in auto-detect."""
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
                    "TAVILY_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("EXA_API_KEY", "exa-key")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        assert web_tools._get_backend() == "exa"

    def test_auto_detect_picks_ddgs_as_last_resort(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
                    "TAVILY_API_KEY", "EXA_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        assert web_tools._get_backend() == "ddgs"

    def test_check_web_api_key_true_when_ddgs_configured(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "ddgs"})
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        assert web_tools.check_web_api_key() is True


# ---------------------------------------------------------------------------
# DDGSWebSearchProvider.extract() unit tests
# ---------------------------------------------------------------------------


class TestDDGSProviderExtract:
    def test_supports_extract_is_true(self):
        from plugins.web.ddgs.provider import DDGSWebSearchProvider
        assert DDGSWebSearchProvider().supports_extract() is True

    def test_happy_path_returns_content(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, extract_result={"url": "https://a.example.com", "content": "# Hello"})
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        docs = DDGSWebSearchProvider().extract(["https://a.example.com"])

        assert len(docs) == 1
        assert docs[0]["url"] == "https://a.example.com"
        assert docs[0]["content"] == "# Hello"
        assert docs[0]["raw_content"] == "# Hello"
        assert "error" not in docs[0]

    def test_multiple_urls_each_extracted(self, monkeypatch):
        _install_fake_ddgs(monkeypatch)
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        docs = DDGSWebSearchProvider().extract(
            ["https://a.example.com", "https://b.example.com"]
        )

        assert len(docs) == 2
        assert {d["url"] for d in docs} == {"https://a.example.com", "https://b.example.com"}

    def test_format_kwarg_maps_to_ddgs_fmt(self, monkeypatch):
        _install_fake_ddgs(monkeypatch)
        monkeypatch.delitem(sys.modules, "plugins.web.ddgs.provider", raising=False)
        import plugins.web.ddgs.provider as _prov

        captured = {}
        orig = _prov._run_ddgs_extract

        def _spy(url, fmt):
            captured["fmt"] = fmt
            return orig(url, fmt)

        monkeypatch.setattr(_prov, "_run_ddgs_extract", _spy)
        _prov.DDGSWebSearchProvider().extract(["https://a.example.com"], format="html")
        assert captured["fmt"] == "text"

    def test_missing_package_returns_error_per_url(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "ddgs", raising=False)
        monkeypatch.delitem(sys.modules, "plugins.web.ddgs.provider", raising=False)
        import builtins
        orig_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "ddgs":
                raise ImportError("blocked for test")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        docs = DDGSWebSearchProvider().extract(["https://a.example.com", "https://b.example.com"])
        assert len(docs) == 2
        assert all("ddgs" in d["error"].lower() for d in docs)
        assert all(d["content"] == "" for d in docs)

    def test_runtime_error_returns_error_entry(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, extract_raises=RuntimeError("HTTP 404"))
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        docs = DDGSWebSearchProvider().extract(["https://a.example.com"])
        assert len(docs) == 1
        assert docs[0]["content"] == ""
        assert "404" in docs[0]["error"] or "failed" in docs[0]["error"].lower()

    def test_one_bad_url_does_not_abort_the_batch(self, monkeypatch):
        """A failing URL must not prevent other URLs in the same batch from
        being extracted — each URL is fetched independently."""
        calls = {"n": 0}

        def _extract_fn(url, fmt):
            calls["n"] += 1
            if url == "https://bad.example.com":
                raise RuntimeError("boom")
            return {"url": url, "content": "ok"}

        _install_fake_ddgs(monkeypatch, extract_fn=_extract_fn)
        monkeypatch.delitem(sys.modules, "plugins.web.ddgs.provider", raising=False)
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        docs = DDGSWebSearchProvider().extract(
            ["https://bad.example.com", "https://good.example.com"]
        )
        by_url = {d["url"]: d for d in docs}
        assert by_url["https://bad.example.com"]["error"]
        assert by_url["https://good.example.com"]["content"] == "ok"
        assert calls["n"] == 2

    def test_hung_extract_times_out_and_returns_error_entry(self, monkeypatch):
        """Mirrors #36776's search timeout fix: a hung ddgs.extract() call
        must be bounded and must not hang the shared agent loop."""
        import threading
        import time

        _install_fake_ddgs(monkeypatch)
        monkeypatch.delitem(sys.modules, "plugins.web.ddgs.provider", raising=False)
        import plugins.web.ddgs.provider as _prov

        release = threading.Event()

        def _blocking_extract(url, fmt):
            release.wait(timeout=10)
            return {"url": url, "content": "too late"}

        monkeypatch.setattr(_prov, "_run_ddgs_extract", _blocking_extract, raising=True)
        monkeypatch.setattr(_prov, "_EXTRACT_TIMEOUT_SECS", 0.3, raising=True)

        try:
            start = time.monotonic()
            docs = _prov.DDGSWebSearchProvider().extract(["https://hangs.example.com"])
            elapsed = time.monotonic() - start

            assert len(docs) == 1
            assert "timed out" in docs[0]["error"].lower()
            assert elapsed < 3.0, f"extract did not return promptly ({elapsed:.1f}s)"
        finally:
            release.set()


# ---------------------------------------------------------------------------
# web_extract dispatches to ddgs when configured (ddgs supports_extract())
# ---------------------------------------------------------------------------


class TestDDGSWebExtractDispatch:
    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_web_extract_dispatches_to_ddgs_and_returns_content(self, monkeypatch):
        import asyncio
        from tools import web_tools

        _install_fake_ddgs(monkeypatch, extract_result={"url": "https://example.com", "content": "# Hi"})
        monkeypatch.delitem(sys.modules, "plugins.web.ddgs.provider", raising=False)

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "ddgs"})
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)

        async def _allow_ssrf(_url: str) -> bool:
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_ssrf)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        result_str = asyncio.get_event_loop().run_until_complete(
            web_tools.web_extract_tool(["https://example.com"])
        )
        result = json.loads(result_str)
        assert "results" in result
        assert result["results"][0]["content"] == "# Hi"
