"""Tests for multi-source fallback chain and search_engine parameter."""
from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest


class TestGetFallbackChain:
    """_get_fallback_chain() produces ordered, deduplicated chain."""

    def test_empty_config_returns_all_providers(self):
        from tools.web_tools import _get_fallback_chain
        with (
            patch("tools.web_tools._load_web_config") as mock_cfg,
            patch("tools.web_tools._list_registered_web_providers") as mock_list,
        ):
            mock_cfg.return_value = {}
            mock_p1 = MagicMock()
            mock_p1.name = "ddgs"
            mock_p2 = MagicMock()
            mock_p2.name = "firecrawl"
            mock_list.return_value = [mock_p1, mock_p2]
            chain = _get_fallback_chain()
            assert len(chain) >= 2
            assert len(chain) == len(set(chain))

    def test_fallback_backends_respected(self):
        from tools.web_tools import _get_fallback_chain
        with (
            patch("tools.web_tools._load_web_config") as mock_cfg,
            patch("tools.web_tools._list_registered_web_providers") as mock_list,
        ):
            mock_cfg.return_value = {
                "fallback_backends": ["serper", "baidu", "ddgs"],
            }
            mock_p1 = MagicMock()
            mock_p1.name = "firecrawl"
            mock_list.return_value = [mock_p1]
            chain = _get_fallback_chain()
            # serper, baidu, ddgs must appear in that order
            positions = {
                name: chain.index(name)
                for name in ["serper", "baidu", "ddgs"]
                if name in chain
            }
            assert positions["serper"] < positions["baidu"]
            assert positions["baidu"] < positions["ddgs"]

    def test_string_fallback_parsed(self):
        from tools.web_tools import _get_fallback_chain
        with (
            patch("tools.web_tools._load_web_config") as mock_cfg,
            patch("tools.web_tools._list_registered_web_providers") as mock_list,
        ):
            mock_cfg.return_value = {"fallback_backends": "brave-free, ddgs"}
            mock_list.return_value = []
            chain = _get_fallback_chain()
            assert chain[0] == "brave-free"
            assert chain[1] == "ddgs"

    def test_primary_from_search_backend(self):
        """search_backend takes precedence over web.backend."""
        from tools.web_tools import _get_fallback_chain
        with (
            patch("tools.web_tools._load_web_config") as mock_cfg,
            patch("tools.web_tools._get_search_backend") as mock_sb,
            patch("tools.web_tools._list_registered_web_providers") as mock_list,
        ):
            mock_cfg.return_value = {}
            mock_sb.return_value = "searxng"
            mock_list.return_value = []
            chain = _get_fallback_chain()
            assert chain[0] == "searxng"

    def test_divergent_search_vs_web_backend(self):
        """When web.search_backend differs from web.backend, search_backend wins."""
        from tools.web_tools import _get_fallback_chain
        with (
            patch("tools.web_tools._load_web_config") as mock_cfg,
            patch("tools.web_tools._get_search_backend") as mock_sb,
            patch("tools.web_tools._list_registered_web_providers") as mock_list,
        ):
            mock_cfg.return_value = {
                "backend": "firecrawl",
                "search_backend": "ddgs",
            }
            mock_sb.return_value = "ddgs"
            mock_list.return_value = []
            chain = _get_fallback_chain()
            assert chain[0] == "ddgs"
            assert "firecrawl" not in chain[:1]


class TestSearchWithFallback:
    """_search_with_fallback() stops at first success, skips failures."""

    def test_first_success_returns_immediately(self):
        from tools.web_tools import _search_with_fallback

        chain = ["backend-a", "backend-b"]
        results_stack = [
            {"success": True, "data": {"web": [{"title": "hit"}]}},
        ]
        with patch("tools.web_tools.get_provider") as mock_gp:
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = True
            mock_provider.supports_search.return_value = True
            mock_provider.search.side_effect = (
                lambda *a, **kw: results_stack.pop(0) if results_stack else None
            )
            mock_gp.return_value = mock_provider
            result, errors = _search_with_fallback("test", 5, chain)
            assert result is not None
            assert result["success"] is True
            assert mock_provider.search.call_count == 1

    def test_skip_unavailable_provider(self):
        from tools.web_tools import _search_with_fallback

        chain = ["unavailable", "good"]
        with patch("tools.web_tools.get_provider") as mock_gp:
            mock_unavail = MagicMock()
            mock_unavail.is_available.return_value = False
            mock_unavail.supports_search.return_value = True

            mock_good = MagicMock()
            mock_good.is_available.return_value = True
            mock_good.supports_search.return_value = True
            mock_good.search.return_value = {
                "success": True, "data": {"web": [{"title": "x"}]},
            }

            mock_gp.side_effect = lambda name: (
                mock_unavail if name == "unavailable" else mock_good
            )
            result, errors = _search_with_fallback("test", 5, chain)
            assert result is not None
            assert result["success"] is True
            assert "unavailable: not available" in errors[0]

    def test_skip_zero_results(self):
        from tools.web_tools import _search_with_fallback

        chain = ["empty", "good"]
        with patch("tools.web_tools.get_provider") as mock_gp:
            mock_empty = MagicMock()
            mock_empty.is_available.return_value = True
            mock_empty.supports_search.return_value = True
            mock_empty.search.return_value = {
                "success": True, "data": {"web": []},
            }
            mock_good = MagicMock()
            mock_good.is_available.return_value = True
            mock_good.supports_search.return_value = True
            mock_good.search.return_value = {
                "success": True, "data": {"web": [{"title": "x"}]},
            }
            mock_gp.side_effect = lambda name: (
                mock_empty if name == "empty" else mock_good
            )
            result, errors = _search_with_fallback("test", 5, chain)
            assert result is not None
            assert result["success"] is True
            assert "empty: returned 0 results" in errors[0]

    def test_all_fail_returns_none(self):
        from tools.web_tools import _search_with_fallback

        chain = ["fail-a", "fail-b"]
        with patch("tools.web_tools.get_provider") as mock_gp:
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = True
            mock_provider.supports_search.return_value = True
            mock_provider.search.return_value = {
                "success": False, "error": "test-error",
            }
            mock_gp.return_value = mock_provider
            result, errors = _search_with_fallback("test", 5, chain)
            assert result is None
            assert len(errors) == 2

    def test_provider_crash_skipped(self):
        from tools.web_tools import _search_with_fallback

        chain = ["crasher", "good"]
        with patch("tools.web_tools.get_provider") as mock_gp:
            mock_bad = MagicMock()
            mock_bad.is_available.return_value = True
            mock_bad.supports_search.return_value = True
            mock_bad.search.side_effect = RuntimeError("boom")
            mock_good = MagicMock()
            mock_good.is_available.return_value = True
            mock_good.supports_search.return_value = True
            mock_good.search.return_value = {
                "success": True, "data": {"web": [{"title": "x"}]},
            }
            mock_gp.side_effect = lambda name: (
                mock_bad if name == "crasher" else mock_good
            )
            result, errors = _search_with_fallback("test", 5, chain)
            assert result is not None
            assert "crasher: boom" in errors[0]

    def test_provider_not_found_skipped(self):
        from tools.web_tools import _search_with_fallback

        chain = ["nonexistent", "good"]
        with patch("tools.web_tools.get_provider") as mock_gp:
            mock_good = MagicMock()
            mock_good.is_available.return_value = True
            mock_good.supports_search.return_value = True
            mock_good.search.return_value = {
                "success": True, "data": {"web": [{"title": "x"}]},
            }
            mock_gp.side_effect = lambda name: (
                None if name == "nonexistent" else mock_good
            )
            result, errors = _search_with_fallback("test", 5, chain)
            assert result is not None
            assert "nonexistent: not found" in errors[0]


class TestGetValidEngineNames:
    """_get_valid_engine_names() returns dynamically updated set."""

    def test_returns_set_with_auto(self):
        from tools.web_tools import _get_valid_engine_names
        with patch("tools.web_tools._list_registered_web_providers") as mock_list:
            mock_p = MagicMock()
            mock_p.name = "ddgs"
            mock_p.is_available.return_value = True
            mock_list.return_value = [mock_p]
            names = _get_valid_engine_names()
            assert isinstance(names, set)
            assert "auto" in names
            assert "ddgs" in names

    def test_unavailable_providers_excluded(self):
        from tools.web_tools import _get_valid_engine_names
        with patch("tools.web_tools._list_registered_web_providers") as mock_list:
            mock_ok = MagicMock()
            mock_ok.name = "ddgs"
            mock_ok.is_available.return_value = True
            mock_bad = MagicMock()
            mock_bad.name = "no-key"
            mock_bad.is_available.return_value = False
            mock_list.return_value = [mock_ok, mock_bad]
            names = _get_valid_engine_names()
            assert "ddgs" in names
            assert "no-key" not in names
