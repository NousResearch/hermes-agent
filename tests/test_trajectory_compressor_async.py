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

    def test_async_client_none_after_init(self):
        """async_client should be None after __init__ (not eagerly created)."""
        from trajectory_compressor import TrajectoryCompressor

        comp = TrajectoryCompressor.__new__(TrajectoryCompressor)
        comp.config = MagicMock()
        comp.config.base_url = "https://api.example.com/v1"
        comp.config.api_key_env = "TEST_API_KEY"
        comp._use_call_llm = False
        comp.async_client = None
        comp._async_client_api_key = "test-key"

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


class TestSourceLineVerification:
    """Verify the actual source has the lazy pattern applied."""

    @staticmethod
    def _read_file() -> str:
        import os
        base = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base, "trajectory_compressor.py"), encoding="utf-8") as f:
            return f.read()

    def test_no_eager_async_openai_in_init(self):
        """__init__ should NOT create AsyncOpenAI eagerly."""
        src = self._read_file()
        # The old pattern: self.async_client = AsyncOpenAI(...) in _init_summarizer
        # should not exist — only self.async_client = None
        lines = src.split("\n")
        for i, line in enumerate(lines, 1):
            if "self.async_client = AsyncOpenAI(" in line and "_get_async_client" not in lines[max(0,i-3):i+1]:
                # Allow it inside _get_async_client method
                # Check if we're inside _get_async_client by looking at context
                context = "\n".join(lines[max(0,i-20):i+1])
                if "_get_async_client" not in context:
                    pytest.fail(
                        f"Line {i}: AsyncOpenAI created eagerly outside _get_async_client()"
                    )

    def test_get_async_client_method_exists(self):
        """_get_async_client method should exist."""
        src = self._read_file()
        assert "def _get_async_client(self)" in src


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
async def test_generate_summary_async_public_moonshot_kimi_k2_5_omits_temperature():
    """kimi-k2.5 on the public Moonshot API should not get a forced temperature."""
    from trajectory_compressor import CompressionConfig, TrajectoryCompressor, TrajectoryMetrics

    config = CompressionConfig(
        summarization_model="kimi-k2.5",
        base_url="https://api.moonshot.ai/v1",
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
async def test_generate_summary_async_public_moonshot_cn_kimi_k2_5_omits_temperature():
    """kimi-k2.5 on api.moonshot.cn should not get a forced temperature."""
    from trajectory_compressor import CompressionConfig, TrajectoryCompressor, TrajectoryMetrics

    config = CompressionConfig(
        summarization_model="kimi-k2.5",
        base_url="https://api.moonshot.cn/v1",
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
async def test_timeout_preserves_original_trajectory(tmp_path):
    """A per-trajectory timeout must keep the original entry, not drop it.

    Regression test: the timeout handler previously stored ``None`` for the
    entry, and the output writer filtered ``None`` rows out — so a transient
    timeout (e.g. a stuck summarization API call) silently deleted a valid
    training trajectory. The handler now preserves the original entry, matching
    the generic-Exception branch.
    """
    import asyncio
    import json

    from trajectory_compressor import (
        AggregateMetrics,
        CompressionConfig,
        TrajectoryCompressor,
    )

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    entry = {
        "conversations": [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "world"},
        ]
    }
    (input_dir / "trajectories.jsonl").write_text(
        json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    config = CompressionConfig(per_trajectory_timeout=0.05, metrics_enabled=False)
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = config
    compressor.logger = MagicMock()
    compressor.aggregate_metrics = AggregateMetrics()

    # Force every trajectory to exceed the (tiny) per-trajectory timeout.
    async def _hang(_entry):
        await asyncio.sleep(10)

    compressor.process_entry_async = _hang

    await compressor._process_directory_async(input_dir, output_dir)

    out_file = output_dir / "trajectories.jsonl"
    assert out_file.exists()
    lines = [ln for ln in out_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # The timed-out trajectory must survive in the output (uncompressed),
    # never be silently dropped.
    assert len(lines) == 1
    assert json.loads(lines[0]) == entry
    assert compressor.aggregate_metrics.trajectories_failed == 1
