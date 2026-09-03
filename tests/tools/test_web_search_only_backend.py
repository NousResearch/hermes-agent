"""Tests for search-only backend check_fn split (issue #89912).

When web.backend is a search-only provider (ddgs, searxng, brave-free)
and no per-capability extract_backend override is configured:
- web_search must be available (check_fn returns True).
- web_extract must NOT be available (check_fn returns False) so the model
  is never shown a tool that always fails with "X is a search-only backend".

When web.extract_backend is set explicitly, web_extract is available even if
the shared backend is search-only.
"""

import pytest
from unittest.mock import patch


def _make_web_config(backend: str, extract_backend: str = "") -> dict:
    return {"backend": backend, "extract_backend": extract_backend}


@pytest.mark.parametrize("backend", ["ddgs", "searxng", "brave-free"])
def test_search_only_backend_search_available(backend):
    """web_search check_fn returns True for search-only backends."""
    from tools.web_tools import check_web_search_available
    with (
        patch("tools.web_tools._load_web_config", return_value=_make_web_config(backend)),
        patch("tools.web_tools._is_backend_available", return_value=True),
    ):
        assert check_web_search_available() is True


@pytest.mark.parametrize("backend", ["ddgs", "searxng", "brave-free"])
def test_search_only_backend_extract_not_available(backend):
    """web_extract check_fn returns False when only a search-only backend is configured."""
    from tools.web_tools import check_web_extract_available
    with (
        patch("tools.web_tools._load_web_config", return_value=_make_web_config(backend)),
        patch("tools.web_tools._is_backend_available", return_value=True),
    ):
        assert check_web_extract_available() is False


@pytest.mark.parametrize("backend", ["ddgs", "searxng", "brave-free"])
def test_search_only_backend_with_extract_override_extract_available(backend):
    """web_extract is available when extract_backend override is set even with search-only shared backend."""
    from tools.web_tools import check_web_extract_available
    with (
        patch("tools.web_tools._load_web_config",
              return_value=_make_web_config(backend, extract_backend="firecrawl")),
        patch("tools.web_tools._is_backend_available", return_value=True),
    ):
        assert check_web_extract_available() is True


def test_full_backend_extract_available():
    """web_extract is available when a full (non-search-only) backend is configured."""
    from tools.web_tools import check_web_extract_available
    with (
        patch("tools.web_tools._load_web_config", return_value=_make_web_config("firecrawl")),
        patch("tools.web_tools._is_backend_available", return_value=True),
    ):
        assert check_web_extract_available() is True


def test_search_only_constant_contents():
    """_SEARCH_ONLY_BACKENDS contains exactly the known search-only built-in backends."""
    from tools.web_tools import _SEARCH_ONLY_BACKENDS
    assert "ddgs" in _SEARCH_ONLY_BACKENDS
    assert "searxng" in _SEARCH_ONLY_BACKENDS
    assert "brave-free" in _SEARCH_ONLY_BACKENDS
    # Full backends must not be in the set
    assert "firecrawl" not in _SEARCH_ONLY_BACKENDS
    assert "exa" not in _SEARCH_ONLY_BACKENDS
    assert "tavily" not in _SEARCH_ONLY_BACKENDS
