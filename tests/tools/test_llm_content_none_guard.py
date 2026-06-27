"""Tests for None guard on response.choices[0].message.content.strip().

OpenAI-compatible APIs return ``message.content = None`` when the model
responds with tool calls only or reasoning-only output (e.g. DeepSeek-R1,
Qwen-QwQ via OpenRouter with ``reasoning.enabled = True``).  Calling
``.strip()`` on ``None`` raises ``AttributeError``.

These tests verify that every call site handles ``content is None`` safely,
and that ``extract_content_or_reasoning()`` falls back to structured
reasoning fields when content is empty.
"""

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── mixture_of_agents_tool — reference model (line 146) ───────────────────

class TestMoAReferenceModelContentNone:
    """tools/mixture_of_agents_tool.py — _query_model()"""

    def test_none_content_raises_before_fix(self):
        """Demonstrate that None content from a reasoning model crashes."""
        response = _make_response(None)

        # Simulate the exact line: response.choices[0].message.content.strip()
        with pytest.raises(AttributeError):
            response.choices[0].message.content.strip()

    def test_none_content_safe_with_or_guard(self):
        """The ``or ""`` guard should convert None to empty string."""
        response = _make_response(None)

        content = (response.choices[0].message.content or "").strip()
        assert content == ""

    def test_normal_content_unaffected(self):
        """Regular string content should pass through unchanged."""
        response = _make_response("  Hello world  ")

        content = (response.choices[0].message.content or "").strip()
        assert content == "Hello world"


# ── mixture_of_agents_tool — aggregator (line 214) ────────────────────────

class TestMoAAggregatorContentNone:
    """tools/mixture_of_agents_tool.py — _run_aggregator()"""

    def test_none_content_raises_before_fix(self):
        response = _make_response(None)

        with pytest.raises(AttributeError):
            response.choices[0].message.content.strip()

    def test_none_content_safe_with_or_guard(self):
        response = _make_response(None)

        content = (response.choices[0].message.content or "").strip()
        assert content == ""


# ── web_tools — LLM content processor (line 419) ─────────────────────────

class TestWebToolsProcessorContentNone:
    """tools/web_tools.py — _process_with_llm() return line"""

    def test_none_content_raises_before_fix(self):
        response = _make_response(None)

        with pytest.raises(AttributeError):
            response.choices[0].message.content.strip()

    def test_none_content_safe_with_or_guard(self):
        response = _make_response(None)

        content = (response.choices[0].message.content or "").strip()
        assert content == ""


# ── web_tools — synthesis/summarization (line 538) ────────────────────────

class TestWebToolsSynthesisContentNone:
    """tools/web_tools.py — synthesize_content() final_summary line"""

    def test_none_content_raises_before_fix(self):
        response = _make_response(None)

        with pytest.raises(AttributeError):
            response.choices[0].message.content.strip()

    def test_none_content_safe_with_or_guard(self):
        response = _make_response(None)

        content = (response.choices[0].message.content or "").strip()
        assert content == ""


# ── vision_tools (line 350) ───────────────────────────────────────────────

class TestVisionToolsContentNone:
    """tools/vision_tools.py — analyze_image() analysis extraction"""

    def test_none_content_raises_before_fix(self):
        response = _make_response(None)

        with pytest.raises(AttributeError):
            response.choices[0].message.content.strip()

    def test_none_content_safe_with_or_guard(self):
        response = _make_response(None)

        content = (response.choices[0].message.content or "").strip()
        assert content == ""


# ── skills_guard (line 963) ───────────────────────────────────────────────

class TestSkillsGuardContentNone:
    """tools/skills_guard.py — _llm_audit_skill() llm_text extraction"""

    def test_none_content_raises_before_fix(self):
        response = _make_response(None)

        with pytest.raises(AttributeError):
            response.choices[0].message.content.strip()

    def test_none_content_safe_with_or_guard(self):
        response = _make_response(None)

        content = (response.choices[0].message.content or "").strip()
        assert content == ""


# ── session_search_tool (line 164) ────────────────────────────────────────

class TestSessionSearchContentNone:
    """tools/session_search_tool.py — _summarize_session() return line"""

    def test_none_content_raises_before_fix(self):
        response = _make_response(None)

        with pytest.raises(AttributeError):
            response.choices[0].message.content.strip()

    def test_none_content_safe_with_or_guard(self):
        response = _make_response(None)

        content = (response.choices[0].message.content or "").strip()
        assert content == ""


# ── integration: verify the actual source lines are guarded ───────────────

class TestSourceLinesAreGuarded:
    """Read the actual source files and verify the fix is applied.

    These tests will FAIL before the fix (bare .content.strip()) and
    PASS after ((.content or "").strip()).
    """

    @staticmethod
    def _read_file(rel_path: str) -> str:
        import os
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(base, rel_path)) as f:
            return f.read()

    def test_mixture_of_agents_reference_model_guarded(self):
        src = self._read_file("tools/mixture_of_agents_tool.py")
        # The unguarded pattern should NOT exist
        assert ".message.content.strip()" not in src, (
            "tools/mixture_of_agents_tool.py still has unguarded "
            ".content.strip() — apply `(... or \"\").strip()` guard"
        )

    def test_web_tools_guarded(self):
        src = self._read_file("tools/web_tools.py")
        assert ".message.content.strip()" not in src, (
            "tools/web_tools.py still has unguarded "
            ".content.strip() — apply `(... or \"\").strip()` guard"
        )

    def test_vision_tools_guarded(self):
        src = self._read_file("tools/vision_tools.py")
        assert ".message.content.strip()" not in src, (
            "tools/vision_tools.py still has unguarded "
            ".content.strip() — apply `(... or \"\").strip()` guard"
        )

    def test_skills_guard_guarded(self):
        src = self._read_file("tools/skills_guard.py")
        assert ".message.content.strip()" not in src, (
            "tools/skills_guard.py still has unguarded "
            ".content.strip() — apply `(... or \"\").strip()` guard"
        )

    def test_session_search_tool_guarded(self):
        src = self._read_file("tools/session_search_tool.py")
        assert ".message.content.strip()" not in src, (
            "tools/session_search_tool.py still has unguarded "
            ".content.strip() — apply `(... or \"\").strip()` guard"
        )


# ── extract_content_or_reasoning() ────────────────────────────────────────

class TestExtractContentOrReasoning:
    """agent/auxiliary_client.py — extract_content_or_reasoning()"""

    def test_normal_content_returned(self):
        response = _make_response("  Hello world  ")
        assert extract_content_or_reasoning(response) == "Hello world"

    def test_none_content_returns_empty(self):
        response = _make_response(None)
        assert extract_content_or_reasoning(response) == ""

    def test_empty_string_returns_empty(self):
        response = _make_response("")
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

    def test_none_content_with_reasoning_field(self):
        """DeepSeek-R1 pattern: content=None, reasoning='...'"""
        response = _make_response(None, reasoning="Step 1: analyze the problem...")
        assert extract_content_or_reasoning(response) == "Step 1: analyze the problem..."

    def test_none_content_with_reasoning_content_field(self):
        """Moonshot/Novita pattern: content=None, reasoning_content='...'"""
        response = _make_response(None, reasoning_content="Let me think about this...")
        assert extract_content_or_reasoning(response) == "Let me think about this..."

    def test_none_content_with_reasoning_details(self):
        """OpenRouter unified format: reasoning_details=[{summary: ...}]"""
        response = _make_response(None, reasoning_details=[
            {"type": "reasoning.summary", "summary": "The key insight is..."},
        ])
        assert extract_content_or_reasoning(response) == "The key insight is..."

    def test_reasoning_fields_not_duplicated(self):
        """When reasoning and reasoning_content have the same value, don't duplicate."""
        response = _make_response(None, reasoning="same text", reasoning_content="same text")
        assert extract_content_or_reasoning(response) == "same text"

    def test_multiple_reasoning_sources_combined(self):
        """Different reasoning sources are joined with double newline."""
        response = _make_response(
            None,
            reasoning="First part",
            reasoning_content="Second part",
        )
        result = extract_content_or_reasoning(response)
        assert "First part" in result
        assert "Second part" in result

    def test_content_preferred_over_reasoning(self):
        """When both content and reasoning exist, content wins."""
        response = _make_response("Actual answer", reasoning="Internal reasoning")
        assert extract_content_or_reasoning(response) == "Actual answer"
