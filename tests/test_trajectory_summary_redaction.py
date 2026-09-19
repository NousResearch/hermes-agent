"""The trajectory summariser must not ship raw secrets to a third party.

``_generate_summary`` sends the turns being compressed to OpenRouter. Those
turns are raw tool output: an API key printed by a terminal command or read out
of a file is still verbatim in the text, and OpenRouter is a third party.

These tests capture what the summariser actually puts on the wire — the
``messages`` payload handed to the client — and assert the secret is not in it.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from trajectory_compressor import (
    CompressionConfig,
    TrajectoryCompressor,
    TrajectoryMetrics,
)

# Shaped like a real credential so the redactor treats it as one.
SECRET = "sk-ant-api03-" + "V" * 40
TURNS = f"$ cat .env\nANTHROPIC_API_KEY={SECRET}\nrequest succeeded\n"


def _compressor():
    """A compressor with the network stubbed, built like the module's own tests."""
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
    """The async path builds its own prompt; fixing only the sync one still leaks."""
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
    """Redaction must remove the credential, not gut the content being summarised."""
    compressor = _compressor()
    compressor._generate_summary(TURNS, TrajectoryMetrics())
    prompt = _sent_prompt(compressor)
    assert "request succeeded" in prompt
    assert "ANTHROPIC_API_KEY" in prompt


@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-api03-" + "V" * 40,
        "ghp_" + "b" * 36,
        "xoxb-123456789012-123456789012-" + "c" * 24,
    ],
)
def test_common_credential_shapes_are_scrubbed(secret):
    compressor = _compressor()
    compressor._generate_summary(f"leaked: {secret}\n", TrajectoryMetrics())
    assert secret not in _sent_prompt(compressor)


def test_summary_is_still_returned_normally():
    """The fix must not disturb the summariser's contract."""
    compressor = _compressor()
    result = compressor._generate_summary(TURNS, TrajectoryMetrics())
    assert result.startswith("[CONTEXT SUMMARY]:")
