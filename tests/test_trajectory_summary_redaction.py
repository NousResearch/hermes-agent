"""The trajectory summariser must not ship raw secrets to a third party."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from trajectory_compressor import CompressionConfig, TrajectoryCompressor, TrajectoryMetrics

SECRET = "sk-" + "V" * 40
TURNS = f"$ cat .env\nANTHROPIC_API_KEY={SECRET}\nrequest succeeded\n"


def _compressor():
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = CompressionConfig(
        summarization_model="test-model",
        temperature=0.3,
        summary_target_tokens=100,
        max_retries=1,
    )
    compressor.logger = MagicMock()
    compressor._use_call_llm = False
    compressor.client = MagicMock()
    compressor.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="[CONTEXT SUMMARY]: ok"))]
    )
    return compressor


def _sent_prompt(compressor) -> str:
    kwargs = compressor.client.chat.completions.create.call_args.kwargs
    return kwargs["messages"][0]["content"]


def test_sync_summary_does_not_send_the_secret():
    compressor = _compressor()
    compressor._generate_summary(TURNS, TrajectoryMetrics())
    assert SECRET not in _sent_prompt(compressor)


def test_async_summary_does_not_send_the_secret():
    compressor = _compressor()
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[CONTEXT SUMMARY]: ok"))]
        )
    )
    client = MagicMock()
    client.chat.completions.create = create
    compressor._get_async_client = MagicMock(return_value=client)
    asyncio.run(compressor._generate_summary_async(TURNS, TrajectoryMetrics()))
    assert SECRET not in create.call_args.kwargs["messages"][0]["content"]


def test_the_surrounding_turn_text_still_reaches_the_model():
    compressor = _compressor()
    compressor._generate_summary(TURNS, TrajectoryMetrics())
    prompt = _sent_prompt(compressor)
    assert "request succeeded" in prompt
    assert "ANTHROPIC_API_KEY" in prompt


@pytest.mark.parametrize("secret", ["sk-" + "V" * 40, "ghp_" + "b" * 36, "xoxb-" + "c" * 24])
def test_common_credential_shapes_are_scrubbed(secret):
    compressor = _compressor()
    compressor._generate_summary(f"leaked: {secret}\n", TrajectoryMetrics())
    assert secret not in _sent_prompt(compressor)


def test_summary_is_still_returned_normally():
    compressor = _compressor()
    result = compressor._generate_summary(TURNS, TrajectoryMetrics())
    assert result.startswith("[CONTEXT SUMMARY]:")
