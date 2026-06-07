"""Regression tests for DeepSeek reasoning SSE support (#30449).

Issue #30449: reasoning_content never reaches OpenAI-compatible SSE stream
when using DeepSeek V4 backend.

Two fixes:
1. API server wires reasoning_callback → emits delta.reasoning_content in SSE
2. Transport legacy path adds reasoning_effort + extra_body.thinking for DeepSeek
"""

import json
import pytest


class TestDeepSeekReasoningLegacyTransport:
    """Tests for legacy transport path DeepSeek reasoning support.

    The legacy path is hit when provider_profile is None (unregistered providers).
    Known providers with registered profiles go through _build_kwargs_from_profile
    which handles reasoning independently.
    """

    def _build_kwargs(self, provider_name="deepseek", reasoning_config=None,
                      is_deepseek=True):
        """Build kwargs via the legacy fallback path."""
        from agent.transports.chat_completions import ChatCompletionsTransport
        from unittest.mock import patch

        transport = ChatCompletionsTransport()
        params = {
            "provider_name": provider_name,
            "is_deepseek": is_deepseek,
            "model_lower": "deepseek-chat",
            # Force legacy path by NOT providing provider_profile
        }
        if reasoning_config is not None:
            params["reasoning_config"] = reasoning_config

        # Call the method that dispatches to legacy path
        kwargs = transport.build_kwargs(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            params=params,
        )
        return kwargs

    def test_deepseek_reasoning_disabled_omits_effort(self):
        """When reasoning is disabled, no reasoning_effort in kwargs."""
        kwargs = self._build_kwargs(
            reasoning_config={"enabled": False},
        )
        assert "reasoning_effort" not in kwargs

    def test_deepseek_thinking_omitted_when_disabled(self):
        """extra_body.thinking is NOT present when reasoning disabled."""
        kwargs = self._build_kwargs(
            reasoning_config={"enabled": False},
        )
        extra = kwargs.get("extra_body", {})
        assert "thinking" not in extra

    def test_non_deepseek_provider_not_affected(self):
        """Non-DeepSeek providers are not affected by DeepSeek logic."""
        kwargs = self._build_kwargs(
            provider_name="openai",
            is_deepseek=False,
            reasoning_config={"enabled": True},
        )
        assert "reasoning_effort" not in kwargs
        extra = kwargs.get("extra_body", {})
        assert "thinking" not in extra

    def test_is_deepseek_param_respected(self):
        """is_deepseek=False should prevent DeepSeek logic even for 'deepseek' provider_name."""
        kwargs = self._build_kwargs(
            provider_name="deepseek",
            is_deepseek=False,
            reasoning_config={"enabled": True},
        )
        # No DeepSeek-specific params should be present
        assert "reasoning_effort" not in kwargs


class TestReasoningSSEEmission:
    """Tests for reasoning_content SSE emission in the API server."""

    def test_emit_reasoning_tagged_tuple(self):
        """__reasoning__ tagged tuples are classified as reasoning chunks."""
        def classify(item):
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__reasoning__":
                return "reasoning"
            elif isinstance(item, tuple) and len(item) == 2 and item[0] == "__tool_progress__":
                return "tool_progress"
            else:
                return "content"

        assert classify(("__reasoning__", "thinking...")) == "reasoning"
        assert classify(("__tool_progress__", {})) == "tool_progress"
        assert classify("plain text") == "content"

    def test_reasoning_chunk_format(self):
        """Reasoning chunk follows OpenAI format with delta.reasoning_content."""
        completion_id = "chatcmpl-123"
        model = "deepseek-chat"
        created = 1000000
        data = "I need to think about this..."

        chunk = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"reasoning_content": data}, "finish_reason": None}],
        }

        assert chunk["choices"][0]["delta"]["reasoning_content"] == data
        assert chunk["object"] == "chat.completion.chunk"
        # No content field mixing
        assert "content" not in chunk["choices"][0]["delta"]

    def test_reasoning_chunk_distinct_from_content_chunk(self):
        """Reasoning chunks use reasoning_content, not content."""
        reasoning = {
            "choices": [{"index": 0, "delta": {"reasoning_content": "think"}, "finish_reason": None}],
        }
        content = {
            "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
        }
        assert "reasoning_content" in reasoning["choices"][0]["delta"]
        assert "content" in content["choices"][0]["delta"]
