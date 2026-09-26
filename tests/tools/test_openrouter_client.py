"""Tests for tools/openrouter_client.py - lazy singleton client + API key check."""

import pytest
from unittest.mock import MagicMock, patch

import tools.openrouter_client as mod


@pytest.fixture(autouse=True)
def _reset_singleton():
    mod._client = None
    yield
    mod._client = None


class TestGetAsyncClient:
    @patch("agent.auxiliary_client.resolve_provider_client")
    def test_happy_path(self, mock_resolve):
        mock_client = MagicMock()
        mock_resolve.return_value = (mock_client, {"provider": "openrouter"})

        result = mod.get_async_client()

        assert result is mock_client
        mock_resolve.assert_called_once_with("openrouter", async_mode=True)

    @patch("agent.auxiliary_client.resolve_provider_client")
    def test_singleton_caching(self, mock_resolve):
        mock_client = MagicMock()
        mock_resolve.return_value = (mock_client, {"provider": "openrouter"})

        first = mod.get_async_client()
        second = mod.get_async_client()

        assert first is second
        mock_resolve.assert_called_once()

    @patch("agent.auxiliary_client.resolve_provider_client")
    def test_raises_on_none_client(self, mock_resolve):
        mock_resolve.return_value = (None, None)

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            mod.get_async_client()


class TestCheckApiKey:
    def test_returns_true_when_set(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
        assert mod.check_api_key() is True

    def test_returns_false_when_missing(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert mod.check_api_key() is False

    def test_returns_false_when_empty(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        assert mod.check_api_key() is False
