"""Tests for custom-endpoint async client lifecycle in trajectory_compressor."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_custom_compressor(monkeypatch):
    from trajectory_compressor import CompressionConfig, TrajectoryCompressor

    monkeypatch.setenv("TRAJECTORY_COMPRESSOR_TEST_API_KEY", "test-key")
    config = CompressionConfig(
        base_url="https://custom.example/v1",
        api_key_env="TRAJECTORY_COMPRESSOR_TEST_API_KEY",
        max_retries=2,
        retry_delay=0,
    )
    with patch.object(TrajectoryCompressor, "_init_tokenizer"), patch("openai.OpenAI"):
        return TrajectoryCompressor(config)


def test_directory_runs_share_one_async_client_and_close_between_event_loops(monkeypatch, tmp_path):
    from trajectory_compressor import TrajectoryMetrics

    compressor = _make_custom_compressor(monkeypatch)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="[CONTEXT SUMMARY]: summary"))]
    )
    clients = []

    def make_client(**_kwargs):
        client = MagicMock()
        if not clients:
            client.chat.completions.create = AsyncMock(
                side_effect=[RuntimeError("transient"), response, response, response]
            )
        else:
            client.chat.completions.create = AsyncMock(return_value=response)
        client.close = AsyncMock()
        clients.append(client)
        return client

    async def process_entries(_input_dir, _output_dir):
        for _ in range(3):
            summary = await compressor._generate_summary_async("turns", TrajectoryMetrics())
            assert summary == "[CONTEXT SUMMARY]: summary"

    with patch("openai.AsyncOpenAI", side_effect=make_client) as constructor, \
            patch.object(compressor, "_process_directory_async", new=process_entries), \
            patch("trajectory_compressor.jittered_backoff", return_value=0):
        compressor.process_directory(tmp_path / "input", tmp_path / "output")
        compressor.process_directory(tmp_path / "input", tmp_path / "output")

    assert len(clients) == 2
    assert constructor.call_count == 2
    for call in constructor.call_args_list:
        assert call.kwargs == {"api_key": "test-key", "base_url": "https://custom.example/v1"}
    assert clients[0].chat.completions.create.await_count == 4
    assert clients[1].chat.completions.create.await_count == 3
    for client in clients:
        client.close.assert_awaited_once()
    assert compressor.async_client is None


def test_directory_error_closes_async_client_exactly_once(monkeypatch, tmp_path):
    compressor = _make_custom_compressor(monkeypatch)
    client = MagicMock()
    client.close = AsyncMock()

    async def fail_directory(_input_dir, _output_dir):
        compressor._get_async_client()
        raise RuntimeError("directory failed")

    with patch("openai.AsyncOpenAI", return_value=client), \
            patch.object(compressor, "_process_directory_async", new=fail_directory):
        with pytest.raises(RuntimeError, match="directory failed"):
            compressor.process_directory(tmp_path / "input", tmp_path / "output")

    client.close.assert_awaited_once()
    assert compressor.async_client is None


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
    entry, _metrics = await compressor.process_entry_async(42)
    assert entry == 42
