"""Tests for trajectory_compressor AsyncOpenAI event loop binding.

The AsyncOpenAI client was created once at __init__ time and stored as an
instance attribute. When process_directory() calls asyncio.run() — which
creates and closes a fresh event loop — the client's internal httpx
transport remains bound to the now-closed loop. A second call to
process_directory() would fail with "Event loop is closed".

The fix creates the AsyncOpenAI client lazily via _get_async_client() so
each asyncio.run() gets a client bound to the current loop.
"""

import types
from unittest.mock import MagicMock, patch

import pytest


class TestAsyncClientLazyCreation:
    """trajectory_compressor.py — _get_async_client()"""

    def test_init_leaves_async_client_uninitialized(self, monkeypatch):
        """__init__ should not create AsyncOpenAI eagerly for custom endpoints."""
        from trajectory_compressor import CompressionConfig, TrajectoryCompressor

        config = CompressionConfig()
        config.base_url = "https://api.example.com/v1"
        config.api_key_env = "TEST_API_KEY"
        monkeypatch.setenv("TEST_API_KEY", "test-key")

        with (
            patch.object(TrajectoryCompressor, "_init_tokenizer", return_value=None),
            patch("openai.OpenAI") as mock_openai,
            patch("openai.AsyncOpenAI") as mock_async_openai,
        ):
            comp = TrajectoryCompressor(config)

        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.example.com/v1",
        )
        mock_async_openai.assert_not_called()
        assert comp.async_client is None

    def test_get_async_client_creates_new_client(self):
        """_get_async_client() should create a fresh AsyncOpenAI instance."""
        from trajectory_compressor import TrajectoryCompressor

        comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
        comp.config = MagicMock()
        comp.config.base_url = "https://api.example.com/v1"
        comp._async_client_api_key = "test-key"
        comp.async_client = None

        mock_async_openai = MagicMock()
        with patch("openai.AsyncOpenAI", mock_async_openai):
            client = comp._get_async_client()

        mock_async_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.example.com/v1",
        )
        assert comp.async_client is not None

    def test_get_async_client_creates_fresh_each_call(self):
        """Each call to _get_async_client() creates a NEW client instance,
        so it binds to the current event loop."""
        from trajectory_compressor import TrajectoryCompressor

        comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
        comp.config = MagicMock()
        comp.config.base_url = "https://api.example.com/v1"
        comp._async_client_api_key = "test-key"
        comp.async_client = None

        call_count = 0
        instances = []

        def mock_constructor(**kwargs):
            nonlocal call_count
            call_count += 1
            instance = MagicMock()
            instances.append(instance)
            return instance

        with patch("openai.AsyncOpenAI", side_effect=mock_constructor):
            client1 = comp._get_async_client()
            client2 = comp._get_async_client()

        # Should have created two separate instances
        assert call_count == 2
        assert instances[0] is not instances[1]
