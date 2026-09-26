"""Tests for trajectory_compressor AsyncOpenAI event loop binding.

The AsyncOpenAI client was created once at __init__ time and stored as an
instance attribute. When process_directory() calls asyncio.run() — which
creates and closes a fresh event loop — the client's internal httpx
transport remains bound to the now-closed loop. A second call to
process_directory() would fail with "Event loop is closed".

The fix creates the AsyncOpenAI client lazily via _get_async_client() so
each asyncio.run() gets a client bound to the current loop.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestAsyncClientLazyCreation:
    """trajectory_compressor.py — _get_async_client()"""



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
            comp._get_async_client()
            comp._get_async_client()

        # Should have created two separate instances
        assert call_count == 2
        assert instances[0] is not instances[1]




@pytest.mark.asyncio
async def test_generate_summary_async_kimi_omits_temperature():
    """Kimi models should have temperature omitted — server manages it."""
    from trajectory_compressor import CompressionConfig, TrajectoryCompressor, TrajectoryMetrics

    config = CompressionConfig(
        summarization_model="kimi-for-coding",
        temperature=0.3,
        summary_target_tokens=100,
        max_retries=1,
    )
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = config
    compressor.logger = MagicMock()
    compressor._use_call_llm = False
    async_client = MagicMock()
    async_client.chat.completions.create = MagicMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="[CONTEXT SUMMARY]: summary"))]
    ))
    compressor._get_async_client = MagicMock(return_value=async_client)

    metrics = TrajectoryMetrics()
    result = await compressor._generate_summary_async("tool output", metrics)

    assert result.startswith("[CONTEXT SUMMARY]:")
    assert "temperature" not in async_client.chat.completions.create.call_args.kwargs






@pytest.mark.asyncio
async def test_process_entry_async_passes_non_dict_through():
    """A scalar JSONL line used to crash on '"conversations" not in entry';
    unknown shapes pass through byte-faithful."""
    from trajectory_compressor import TrajectoryCompressor

    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    entry, metrics = await compressor.process_entry_async(42)
    assert entry == 42
