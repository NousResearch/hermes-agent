"""Tests for None guard on response.choices[0].message.content.strip().

OpenAI-compatible APIs return ``message.content = None`` when the model
responds with tool calls only or reasoning-only output (e.g. DeepSeek-R1,
Qwen-QwQ via OpenRouter with ``reasoning.enabled = True``).  Calling
``.strip()`` on ``None`` raises ``AttributeError``.

These tests verify that ``extract_content_or_reasoning()`` handles
``content is None`` safely and falls back to structured reasoning fields
when content is empty.
"""

import types

from agent.auxiliary_client import extract_content_or_reasoning


# ── helpers ────────────────────────────────────────────────────────────────

def _make_response(content, **msg_attrs):
    """Build a minimal OpenAI-compatible ChatCompletion response stub.

    Extra keyword args are set as attributes on the message object
    (e.g. reasoning="...", reasoning_content="...", reasoning_details=[...]).
    """
    message = types.SimpleNamespace(content=content, tool_calls=None, **msg_attrs)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


# ── extract_content_or_reasoning() ────────────────────────────────────────

class TestExtractContentOrReasoning:
    """agent/auxiliary_client.py — extract_content_or_reasoning()"""

    def test_normal_content_returned(self):
        response = _make_response("  Hello world  ")
        assert extract_content_or_reasoning(response) == "Hello world"

    def test_none_content_returns_empty(self):
        response = _make_response(None)
        assert extract_content_or_reasoning(response) == ""


    def test_think_blocks_stripped_with_remaining_content(self):
        response = _make_response("<think>internal reasoning</think>The answer is 42.")
        assert extract_content_or_reasoning(response) == "The answer is 42."

    def test_think_only_content_falls_back_to_reasoning_field(self):
        """When content is only think blocks, fall back to structured reasoning."""
        response = _make_response(
            "<think>some reasoning</think>",
            reasoning="The actual reasoning output",
        )
        assert extract_content_or_reasoning(response) == "The actual reasoning output"


    def test_content_preferred_over_reasoning(self):
        """When both content and reasoning exist, content wins."""
        response = _make_response("Actual answer", reasoning="Internal reasoning")
        assert extract_content_or_reasoning(response) == "Actual answer"

    def test_dict_message_and_whitespace_fall_back(self):
        assert extract_content_or_reasoning(
            {"content": " ", "reasoning_content": "dict reasoning"}
        ) == "dict reasoning"
        assert extract_content_or_reasoning(
            {"choices": [{"message": {"content": "", "reasoning": "from response"}}]}
        ) == "from response"

    def test_string_message_passthrough(self):
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message="plain summary text")]
        )
        assert extract_content_or_reasoning(response) == "plain summary text"

    def test_reasoning_fallback_respects_max_chars(self):
        huge = "t" * 20_000
        text = extract_content_or_reasoning(
            {"content": "", "reasoning_content": huge},
            max_reasoning_chars=8000,
        )
        assert text == huge[:8000]
        assert extract_content_or_reasoning(
            {"content": "", "reasoning_content": "short"},
            max_reasoning_chars=8000,
        ) == "short"
