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
    entry, _metrics = await compressor.process_entry_async(42)
    assert entry == 42


@pytest.mark.asyncio
async def test_process_directory_does_not_gather_one_task_per_entry(tmp_path):
    """Directory processing must not create one coroutine per row then gather(*all).

    The semaphore only caps in-flight API calls. Materializing every JSONL
    row and wrapping each in a Task still OOMs on large offline dumps (#84703).
    """
    import asyncio
    import json

    from trajectory_compressor import (
        AggregateMetrics,
        CompressionConfig,
        TrajectoryCompressor,
    )

    n_entries = 12
    max_concurrent = 3
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    src = in_dir / "traj.jsonl"
    with src.open("w", encoding="utf-8") as handle:
        for idx in range(n_entries):
            handle.write(json.dumps({"id": idx, "conversations": []}) + "\n")

    gather_sizes: list[int] = []
    real_gather = asyncio.gather

    async def spy_gather(*aws, **kwargs):
        gather_sizes.append(len(aws))
        return await real_gather(*aws, **kwargs)

    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = CompressionConfig(
        max_concurrent_requests=max_concurrent,
        metrics_enabled=False,
    )
    compressor.aggregate_metrics = AggregateMetrics()
    compressor.logger = MagicMock()

    with patch("trajectory_compressor.asyncio.gather", spy_gather):
        await compressor._process_directory_async(in_dir, out_dir)

    assert gather_sizes, "directory processing should still use asyncio.gather"
    assert max(gather_sizes) <= max_concurrent, (
        f"gather submitted {max(gather_sizes)} awaitables; "
        f"must be <= max_concurrent_requests ({max_concurrent})"
    )

    written = (out_dir / "traj.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in written] == list(range(n_entries))


@pytest.mark.asyncio
async def test_process_directory_timeout_skips_and_error_keeps_original(tmp_path):
    """Timeouts are omitted; unexpected errors keep the original row."""
    import asyncio
    import json

    from trajectory_compressor import (
        AggregateMetrics,
        CompressionConfig,
        TrajectoryCompressor,
        TrajectoryMetrics,
    )

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    src = in_dir / "traj.jsonl"
    with src.open("w", encoding="utf-8") as handle:
        for idx, tag in enumerate(("ok", "timeout", "error")):
            handle.write(json.dumps({"id": idx, "tag": tag, "conversations": []}) + "\n")

    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = CompressionConfig(max_concurrent_requests=2, metrics_enabled=False)
    compressor.aggregate_metrics = AggregateMetrics()
    compressor.logger = MagicMock()

    async def fake_process(entry):
        if entry["tag"] == "timeout":
            raise asyncio.TimeoutError
        if entry["tag"] == "error":
            raise RuntimeError("boom")
        return {**entry, "ok": True}, TrajectoryMetrics()

    with patch.object(compressor, "process_entry_async", fake_process):
        await compressor._process_directory_async(in_dir, out_dir)

    rows = [
        json.loads(line)
        for line in (out_dir / "traj.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["tag"] for row in rows] == ["ok", "error"]
    assert rows[0]["ok"] is True
    assert "ok" not in rows[1]
    assert compressor.aggregate_metrics.trajectories_failed == 2
