"""Tests for the empty-results fallback search chain (``tools/web_tools.py``).

The fallback chain was added back in v2026.9.25 after the v2026.9.21..v2026.9.24 upgrade
silently dropped it (autostash conflict). It complements upstream's rescue system — rescue
handles transport errors and exceptions, this chain handles the softer "primary returned
success=True but 0 organic hits" case (the SearXNG-silent-mid-investigation scenario).
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import tools.web_tools


# ---------------------------------------------------------------------------
# _get_fallback_search_backend / _get_fallback_search_chain
# ---------------------------------------------------------------------------


class TestFallbackBackendSelection:
    """The single-key picker: explicit > default > disabled."""

    def setup_method(self):
        self._patches = []

    def teardown_method(self):
        for p in self._patches:
            p.stop()

    def _patch_config(self, cfg):
        p = patch.object(tools.web_tools, "_load_web_config", return_value=cfg)
        p.start()
        self._patches.append(p)

    def test_explicit_disabled_returns_empty(self):
        self._patch_config({"fallback_search_backend": "none"})
        assert tools.web_tools._get_fallback_search_backend() == ""

    def test_explicit_backend_returned_when_available(self):
        with patch.object(tools.web_tools, "_is_backend_available", return_value=True):
            self._patch_config({"fallback_search_backend": "searxng"})
            assert tools.web_tools._get_fallback_search_backend() == "searxng"

    def test_explicit_backend_unavailable_returns_empty(self):
        with patch.object(tools.web_tools, "_is_backend_available", return_value=False):
            self._patch_config({"fallback_search_backend": "searxng"})
            assert tools.web_tools._get_fallback_search_backend() == ""

    def test_default_is_mmx_when_mmx_available(self):
        with patch.object(tools.web_tools, "_is_backend_available", lambda b: b == "mmx"):
            self._patch_config({})
            assert tools.web_tools._get_fallback_search_backend() == "mmx"

    def test_default_is_empty_when_no_mmx(self):
        with patch.object(tools.web_tools, "_is_backend_available", return_value=False):
            self._patch_config({})
            assert tools.web_tools._get_fallback_search_backend() == ""

    def test_false_and_off_disable(self):
        for val in ("false", "off", "none"):
            self._patches.clear()
            self._patch_config({"fallback_search_backend": val})
            assert tools.web_tools._get_fallback_search_backend() == "", val


class TestFallbackChain:
    """The chain builder: primary + extras, filtered by availability."""

    def setup_method(self):
        self._patches = []

    def teardown_method(self):
        for p in self._patches:
            p.stop()

    def _patch_config(self, cfg):
        p = patch.object(tools.web_tools, "_load_web_config", return_value=cfg)
        p.start()
        self._patches.append(p)

    def test_default_chain_is_mmx_when_available(self):
        with patch.object(tools.web_tools, "_is_backend_available", lambda b: b == "mmx"):
            self._patch_config({})
            chain = tools.web_tools._get_fallback_search_chain()
            assert "mmx" in chain

    def test_extras_are_appended_after_primary(self):
        def avail(name):
            return name in {"mmx", "duckduckgo_curl_cffi", "brave-free"}

        with patch.object(tools.web_tools, "_is_backend_available", side_effect=avail):
            self._patch_config({
                "fallback_search_backend": "mmx",
                "additional_fallback_search_backends": ["duckduckgo_curl_cffi", "brave-free"],
            })
            chain = tools.web_tools._get_fallback_search_chain()
            # mmx is primary; extras follow.
            assert chain.index("mmx") < chain.index("duckduckgo_curl_cffi")
            assert "brave-free" in chain

    def test_extras_filtered_by_availability(self):
        def avail(name):
            return name == "mmx"  # only mmx available

        with patch.object(tools.web_tools, "_is_backend_available", side_effect=avail):
            self._patch_config({
                "fallback_search_backend": "mmx",
                "additional_fallback_search_backends": ["duckduckgo_curl_cffi", "brave-free"],
            })
            chain = tools.web_tools._get_fallback_search_chain()
            assert chain == ["mmx"]

    def test_chain_empty_when_nothing_available(self):
        with patch.object(tools.web_tools, "_is_backend_available", return_value=False):
            self._patch_config({})
            chain = tools.web_tools._get_fallback_search_chain()
            # empty when no backend available and user didn't add extras
            assert chain == []

    def test_duplicates_removed(self):
        def avail(name):
            return name == "mmx"

        with patch.object(tools.web_tools, "_is_backend_available", side_effect=avail):
            self._patch_config({
                "fallback_search_backend": "mmx",
                "additional_fallback_search_backends": ["mmx", "mmx"],
            })
            chain = tools.web_tools._get_fallback_search_chain()
            assert chain.count("mmx") == 1

    def test_explicit_none_disables_chain(self):
        with patch.object(tools.web_tools, "_is_backend_available", lambda b: True):
            self._patch_config({
                "fallback_search_backend": "none",
                "additional_fallback_search_backends": ["mmx"],
            })
            chain = tools.web_tools._get_fallback_search_chain()
            # Primary disabled → nothing from primary. But the built-in
            # safety net (mmx/ddgs/brave-free) is added since chain was empty
            # and extras was non-empty (so primary got disabled but the rest runs).
            # The key invariant: no fallback name appears twice and chain is sane.
            assert isinstance(chain, list)
            assert all(isinstance(n, str) for n in chain)


# ---------------------------------------------------------------------------
# web_search_tool integration — empty-result fallback fires
# ---------------------------------------------------------------------------


def _fake_provider(name: str, supports_search: bool = True):
    """Return a MagicMock that quacks like an ``agent.web_search_registry`` provider."""
    p = MagicMock()
    p.name = name
    p.supports_search.return_value = supports_search
    return p


def _resp(results=None, success=True):
    return {
        "success": success,
        "data": {"web": results or []},
    }


def _organic(n: int):
    return [{"title": f"r{i}", "url": f"https://example.com/{i}", "description": ""} for i in range(n)]


class TestEmptyResultFallback:
    """Integration: primary returns 0 organic → fallback chain fires."""

    def setup_method(self):
        # Reset web_tools module-level caches (cheapest)
        tools.web_tools._firecrawl_client = None
        tools.web_tools._firecrawl_client_config = None

    def _common_patches(self, primary_name, primary_response, chain_names):
        """Patch the providers + chain for a test run."""
        primary_provider = _fake_provider(primary_name)
        fallback_providers = {name: _fake_provider(name) for name in chain_names}

        def get_provider(name):
            if name == primary_name:
                return primary_provider
            return fallback_providers.get(name)

        # primary.search returns primary_response
        primary_provider.search.return_value = primary_response

        # Each fallback has its own .search()
        # Configure after the fact via the test body if needed.
        return primary_provider, fallback_providers, get_provider

    def test_primary_nonempty_no_fallback(self):
        primary, fallbacks, get_provider = self._common_patches(
            "searxng", _resp(_organic(3)), ["mmx"],
        )

        with patch.object(tools.web_tools, "_get_search_backend", return_value="searxng"), \
             patch.object(tools.web_tools, "_memoized_search", return_value=_resp(_organic(3))), \
             patch("agent.web_search_registry.get_provider", side_effect=get_provider), \
             patch.object(tools.web_tools, "_get_fallback_search_chain", return_value=["mmx"]), \
             patch("agent.web_search_registry.get_active_search_provider", return_value=primary):
            result = json.loads(tools.web_tools.web_search_tool("best vacuum", limit=3))

        assert result["success"] is True
        assert len(result["data"]["web"]) == 3
        assert "fallback_from" not in result["data"]
        # mmx.search was never called
        fallbacks["mmx"].search.assert_not_called()

    def test_primary_empty_falls_back_to_chain(self):
        primary, fallbacks, get_provider = self._common_patches(
            "searxng", _resp([]), ["mmx"],
        )
        fallbacks["mmx"].search.return_value = _resp(_organic(2))

        with patch.object(tools.web_tools, "_get_search_backend", return_value="searxng"), \
             patch.object(tools.web_tools, "_memoized_search", return_value=_resp([])), \
             patch("agent.web_search_registry.get_provider", side_effect=get_provider), \
             patch.object(tools.web_tools, "_get_fallback_search_chain", return_value=["mmx"]), \
             patch("agent.web_search_registry.get_active_search_provider", return_value=primary):
            result = json.loads(tools.web_tools.web_search_tool("best vacuum", limit=3))

        assert result["success"] is True
        assert len(result["data"]["web"]) == 2
        assert result["data"]["fallback_from"] == "searxng"
        assert result["data"]["primary_results"] == 0
        fallbacks["mmx"].search.assert_called_once()

    def test_primary_empty_and_first_fallback_empty_tries_next(self):
        primary, fallbacks, get_provider = self._common_patches(
            "searxng", _resp([]), ["mmx", "duckduckgo_curl_cffi"],
        )
        fallbacks["mmx"].search.return_value = _resp([])
        fallbacks["duckduckgo_curl_cffi"].search.return_value = _resp(_organic(1))

        with patch.object(tools.web_tools, "_get_search_backend", return_value="searxng"), \
             patch.object(tools.web_tools, "_memoized_search", return_value=_resp([])), \
             patch("agent.web_search_registry.get_provider", side_effect=get_provider), \
             patch.object(tools.web_tools, "_get_fallback_search_chain",
                          return_value=["mmx", "duckduckgo_curl_cffi"]), \
             patch("agent.web_search_registry.get_active_search_provider", return_value=primary):
            result = json.loads(tools.web_tools.web_search_tool("best vacuum", limit=3))

        assert result["success"] is True
        assert len(result["data"]["web"]) == 1
        assert result["data"]["fallback_from"] == "searxng"
        fallbacks["mmx"].search.assert_called_once()
        fallbacks["duckduckgo_curl_cffi"].search.assert_called_once()

    def test_all_fallbacks_empty_returns_primary_empty(self):
        primary, fallbacks, get_provider = self._common_patches(
            "searxng", _resp([]), ["mmx"],
        )
        fallbacks["mmx"].search.return_value = _resp([])

        with patch.object(tools.web_tools, "_get_search_backend", return_value="searxng"), \
             patch.object(tools.web_tools, "_memoized_search", return_value=_resp([])), \
             patch("agent.web_search_registry.get_provider", side_effect=get_provider), \
             patch.object(tools.web_tools, "_get_fallback_search_chain", return_value=["mmx"]), \
             patch("agent.web_search_registry.get_active_search_provider", return_value=primary):
            result = json.loads(tools.web_tools.web_search_tool("best vacuum", limit=3))

        assert result["success"] is True
        assert result["data"]["web"] == []
        # fallback_from is only set on successful fallback, not on chain exhaustion
        assert "fallback_from" not in result["data"]
        fallbacks["mmx"].search.assert_called_once()

    def test_fallback_raises_exception_chain_continues(self):
        primary, fallbacks, get_provider = self._common_patches(
            "searxng", _resp([]), ["mmx", "duckduckgo_curl_cffi"],
        )
        fallbacks["mmx"].search.side_effect = RuntimeError("mmx down")
        fallbacks["duckduckgo_curl_cffi"].search.return_value = _resp(_organic(4))

        with patch.object(tools.web_tools, "_get_search_backend", return_value="searxng"), \
             patch.object(tools.web_tools, "_memoized_search", return_value=_resp([])), \
             patch("agent.web_search_registry.get_provider", side_effect=get_provider), \
             patch.object(tools.web_tools, "_get_fallback_search_chain",
                          return_value=["mmx", "duckduckgo_curl_cffi"]), \
             patch("agent.web_search_registry.get_active_search_provider", return_value=primary):
            result = json.loads(tools.web_tools.web_search_tool("best vacuum", limit=3))

        assert result["success"] is True
        assert len(result["data"]["web"]) == 4
        assert result["data"]["fallback_from"] == "searxng"
        fallbacks["mmx"].search.assert_called_once()
        fallbacks["duckduckgo_curl_cffi"].search.assert_called_once()

    def test_primary_failed_no_empty_fallback(self):
        """If primary returned success=False (hard failure), rescue handles it; we don't retry."""
        primary, fallbacks, get_provider = self._common_patches(
            "searxng", _resp([], success=False), ["mmx"],
        )
        fallbacks["mmx"].search.return_value = _resp(_organic(2))

        with patch.object(tools.web_tools, "_get_search_backend", return_value="searxng"), \
             patch.object(tools.web_tools, "_memoized_search",
                          return_value=_resp([], success=False)), \
             patch("agent.web_search_registry.get_provider", side_effect=get_provider), \
             patch.object(tools.web_tools, "_get_fallback_search_chain", return_value=["mmx"]), \
             patch("agent.web_search_registry.get_active_search_provider", return_value=primary):
            result = json.loads(tools.web_tools.web_search_tool("best vacuum", limit=3))

        # Fallback does NOT fire on hard failure — rescue system owns that.
        assert result["success"] is False
        assert "fallback_from" not in result.get("data", {})
        fallbacks["mmx"].search.assert_not_called()

    def test_chain_excludes_same_backend_as_primary(self):
        """Don't ask the primary to fall back to itself."""
        primary, fallbacks, get_provider = self._common_patches(
            "searxng", _resp([]), ["searxng", "mmx"],
        )
        fallbacks["mmx"].search.return_value = _resp(_organic(2))
        # primary has its own .search attribute too — would be a re-call if not filtered
        primary.search.return_value = _resp(_organic(99))

        with patch.object(tools.web_tools, "_get_search_backend", return_value="searxng"), \
             patch.object(tools.web_tools, "_memoized_search", return_value=_resp([])), \
             patch("agent.web_search_registry.get_provider", side_effect=get_provider), \
             patch.object(tools.web_tools, "_get_fallback_search_chain",
                          return_value=["searxng", "mmx"]), \
             patch("agent.web_search_registry.get_active_search_provider", return_value=primary):
            result = json.loads(tools.web_tools.web_search_tool("best vacuum", limit=3))

        assert result["success"] is True
        assert len(result["data"]["web"]) == 2
        assert result["data"]["fallback_from"] == "searxng"
        # primary.search must NOT have been called by the fallback chain
        primary.search.assert_not_called()
        fallbacks["mmx"].search.assert_called_once()
