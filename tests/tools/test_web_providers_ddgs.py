"""Tests for the DuckDuckGo (ddgs) web search provider.

Covers:
- DDGSWebSearchProvider.is_available() — reflects package importability
- DDGSWebSearchProvider.search() — happy path, missing package, runtime error
- Result normalization (title, url, description, position)
- Process-isolated timeout / interrupt / GIL-hold / reap (#68096)
- _is_backend_available("ddgs") / _get_backend() integration
- web_extract returns a search-only error when ddgs is active
"""
from __future__ import annotations

import json
import sys
import time
import types

import pytest

from tests.tools.conftest import register_all_web_providers


def _install_fake_ddgs(monkeypatch, *, text_results=None, text_raises=None, text_sleep=None):
    """Install a stub ``ddgs`` module in sys.modules for the duration of a test.

    ``text_results``: iterable of dicts to yield from DDGS().text(...).
    ``text_raises``: if set, DDGS().text raises this exception instead.
    ``text_sleep``: if set, DDGS().text blocks for this many seconds before
        yielding — simulates a hung/slow search for the timeout test.
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

    fake.DDGS = _FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake)
    return fake


def _force_inprocess_search(monkeypatch, prov):
    """Route bounded search through the in-process helper.

    Happy-path unit tests install a fake ``ddgs`` in the parent interpreter;
    spawn workers would not see that fake. Isolation behavior is covered by
    dedicated process tests below.
    """
    monkeypatch.setattr(
        prov,
        "_run_ddgs_search_bounded",
        lambda query, safe_limit: prov._run_ddgs_search(query, safe_limit),
        raising=True,
    )


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
        import plugins.web.ddgs.provider as prov
        _force_inprocess_search(monkeypatch, prov)

        result = prov.DDGSWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0] == {"title": "A", "url": "https://a.example.com", "description": "desc A", "position": 1}
        assert web[2]["position"] == 3

    def test_accepts_url_key_as_fallback_for_href(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_results=[
            {"title": "A", "url": "https://a.example.com", "body": "desc A"},
        ])
        import plugins.web.ddgs.provider as prov
        _force_inprocess_search(monkeypatch, prov)

        result = prov.DDGSWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        assert result["data"]["web"][0]["url"] == "https://a.example.com"

    def test_limit_is_respected(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_results=[
            {"title": f"R{i}", "href": f"https://r{i}.example.com", "body": ""}
            for i in range(10)
        ])
        import plugins.web.ddgs.provider as prov
        _force_inprocess_search(monkeypatch, prov)

        result = prov.DDGSWebSearchProvider().search("q", limit=3)

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
        import plugins.web.ddgs.provider as prov
        _force_inprocess_search(monkeypatch, prov)

        result = prov.DDGSWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "rate limited" in result["error"] or "failed" in result["error"].lower()

    def test_empty_results(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_results=[])
        import plugins.web.ddgs.provider as prov
        _force_inprocess_search(monkeypatch, prov)

        result = prov.DDGSWebSearchProvider().search("nothing", limit=5)
        assert result["success"] is True
        assert result["data"]["web"] == []

    @pytest.mark.live_system_guard_bypass
    def test_hung_search_times_out_and_returns_failure(self, monkeypatch):
        """#36776 / #68096: a hung worker must be bounded by the wall-clock
        timeout and reaped — even when the child never returns to Python."""
        _install_fake_ddgs(monkeypatch)
        import plugins.web.ddgs.provider as prov

        monkeypatch.setattr(prov, "_test_hook", "sleep", raising=True)
        monkeypatch.setattr(prov, "_SEARCH_TIMEOUT_SECS", 0.4, raising=True)
        monkeypatch.setattr(prov, "_TERMINATE_GRACE_SECS", 0.5, raising=True)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        start = time.monotonic()
        result = prov.DDGSWebSearchProvider().search("hangs forever", limit=5)
        elapsed = time.monotonic() - start

        assert result["success"] is False
        assert "timed out" in result["error"].lower()
        assert elapsed < 5.0, f"search did not return promptly ({elapsed:.1f}s)"
        _assert_worker_reaped(prov)

    def test_fast_search_not_affected_by_timeout_wrapper(self, monkeypatch):
        """Happy-path guard: the timeout wrapper must not break a normal,
        fast search — results flow through unchanged."""
        _install_fake_ddgs(
            monkeypatch,
            text_results=[{"title": "T", "href": "https://e.com", "body": "B"}],
        )
        import plugins.web.ddgs.provider as prov
        _force_inprocess_search(monkeypatch, prov)

        result = prov.DDGSWebSearchProvider().search("q", limit=5)
        assert result["success"] is True
        assert result["data"]["web"][0]["url"] == "https://e.com"
        assert result["data"]["web"][0]["title"] == "T"


# ---------------------------------------------------------------------------
# Process isolation (#68096)
# ---------------------------------------------------------------------------


def _assert_worker_reaped(prov) -> None:
    """Assert the last DDGS worker process has exited."""
    proc = prov._last_worker_proc
    assert proc is not None, "expected a DDGS worker process to have been started"
    assert proc.poll() is not None, (
        f"DDGS worker still alive (pid={proc.pid}, returncode={proc.returncode})"
    )


@pytest.mark.live_system_guard_bypass
class TestDDGSProcessIsolation:
    def test_gil_holding_worker_times_out_and_is_reaped(self, monkeypatch):
        """#68096: parent deadline still fires when the child holds its GIL."""
        _install_fake_ddgs(monkeypatch)
        import plugins.web.ddgs.provider as prov

        monkeypatch.setattr(prov, "_test_hook", "gil", raising=True)
        monkeypatch.setattr(prov, "_SEARCH_TIMEOUT_SECS", 0.5, raising=True)
        monkeypatch.setattr(prov, "_TERMINATE_GRACE_SECS", 0.5, raising=True)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        start = time.monotonic()
        result = prov.DDGSWebSearchProvider().search("gil hold", limit=5)
        elapsed = time.monotonic() - start

        assert result["success"] is False
        assert "timed out" in result["error"].lower()
        assert elapsed < 5.0, f"GIL-hold search did not time out promptly ({elapsed:.1f}s)"
        _assert_worker_reaped(prov)

    def test_interrupt_terminates_worker_promptly(self, monkeypatch):
        """TUI/gateway interrupt must kill the DDGS child before the deadline."""
        _install_fake_ddgs(monkeypatch)
        import plugins.web.ddgs.provider as prov

        # Flip interrupt after the first poll so the wait loop observes it.
        calls = {"n": 0}

        def _interrupt_after_poll():
            calls["n"] += 1
            return calls["n"] >= 2

        monkeypatch.setattr(prov, "_test_hook", "sleep", raising=True)
        monkeypatch.setattr(prov, "_SEARCH_TIMEOUT_SECS", 30, raising=True)
        monkeypatch.setattr(prov, "_TERMINATE_GRACE_SECS", 0.5, raising=True)
        monkeypatch.setattr("tools.interrupt.is_interrupted", _interrupt_after_poll)

        start = time.monotonic()
        result = prov.DDGSWebSearchProvider().search("interrupt me", limit=5)
        elapsed = time.monotonic() - start

        assert result["success"] is False
        assert "interrupted" in result["error"].lower()
        assert elapsed < 5.0, f"interrupt did not return promptly ({elapsed:.1f}s)"
        _assert_worker_reaped(prov)

    def test_spawned_worker_success_envelope(self, monkeypatch):
        """Real spawn path: success envelope round-trips through the pipe."""
        _install_fake_ddgs(monkeypatch)
        import plugins.web.ddgs.provider as prov

        monkeypatch.setattr(prov, "_test_hook", "success", raising=True)
        monkeypatch.setattr(prov, "_SEARCH_TIMEOUT_SECS", 5, raising=True)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        result = prov.DDGSWebSearchProvider().search("q", limit=5)
        assert result["success"] is True
        assert result["data"]["web"][0]["url"] == "https://example.com"
        _assert_worker_reaped(prov)

    def test_spawned_worker_error_envelope(self, monkeypatch):
        """Real spawn path: error envelope becomes success=False."""
        _install_fake_ddgs(monkeypatch)
        import plugins.web.ddgs.provider as prov

        monkeypatch.setattr(prov, "_test_hook", "error", raising=True)
        monkeypatch.setattr(prov, "_SEARCH_TIMEOUT_SECS", 5, raising=True)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        result = prov.DDGSWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "boom" in result["error"]
        _assert_worker_reaped(prov)

    def test_no_orphan_after_successful_search(self, monkeypatch):
        _install_fake_ddgs(monkeypatch)
        import plugins.web.ddgs.provider as prov

        monkeypatch.setattr(prov, "_test_hook", "empty", raising=True)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        result = prov.DDGSWebSearchProvider().search("q", limit=5)
        assert result["success"] is True
        _assert_worker_reaped(prov)

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
# ddgs is search-only: web_extract returns a clear error
# ---------------------------------------------------------------------------


class TestDDGSSearchOnlyErrors:
    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_web_extract_returns_search_only_error(self, monkeypatch):
        import asyncio
        from tools import web_tools

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
        assert result["success"] is False
        assert "search-only" in result["error"].lower()
        assert "duckduckgo" in result["error"].lower() or "ddgs" in result["error"].lower()
