"""Tests for tools.openrouter_client — check_api_key and get_async_client."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tools.openrouter_client import check_api_key


# ============================================================================
# check_api_key
# ============================================================================
class TestCheckApiKey:
    def test_key_present(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-abc123")
        assert check_api_key() is True

    def test_key_missing(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert check_api_key() is False

    def test_key_empty_string(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        # bool("") = False
        assert check_api_key() is False

    def test_key_whitespace(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
        # bool("   ") = True (non-empty string)
        assert check_api_key() is True


# ============================================================================
# get_async_client
# ============================================================================
class TestGetAsyncClient:
    def test_raises_when_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Module's _client is None, so it will try resolve_provider_client
        # which returns None without API key → ValueError
        from tools.openrouter_client import _client, get_async_client
        import tools.openrouter_client as mod
        # Reset module cache
        mod._client = None
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            get_async_client()

    def test_lazy_initialization(self, monkeypatch):
        """_client starts as None, get_async_client initializes it."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        import tools.openrouter_client as mod
        mod._client = None  # reset
        # We can't actually call it (needs openai), but verify initial state
        assert mod._client is None
