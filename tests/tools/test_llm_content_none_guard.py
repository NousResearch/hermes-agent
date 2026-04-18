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
import json
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

    def test_reference_model_handles_none_content_without_attribute_error(self):
        from tools.mixture_of_agents_tool import _run_reference_model_safe

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=AsyncMock(return_value=_make_response(None))
                )
            )
        )

        with patch("tools.mixture_of_agents_tool._get_openrouter_client", return_value=client):
            model, content, ok = _run(
                _run_reference_model_safe("test-model", "hello", max_retries=1)
            )

        assert model == "test-model"
        assert content == ""
        assert ok is True


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

    def test_aggregator_retries_and_returns_empty_string(self):
        from tools.mixture_of_agents_tool import _run_aggregator_model

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=AsyncMock(side_effect=[_make_response(None), _make_response(None)])
                )
            )
        )

        with patch("tools.mixture_of_agents_tool._get_openrouter_client", return_value=client):
            result = _run(_run_aggregator_model("system", "hello"))

        assert result == ""


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

    def test_call_summarizer_llm_returns_empty_string_after_none_content_retry(self):
        from tools.web_tools import _call_summarizer_llm

        with (
            patch("tools.web_tools._resolve_web_extract_auxiliary", return_value=(object(), "test-model", None)),
            patch("tools.web_tools.async_call_llm", new=AsyncMock(side_effect=[_make_response(None), _make_response(None)])),
            patch("tools.web_tools.asyncio.sleep", new=AsyncMock()),
        ):
            result = _run(
                _call_summarizer_llm(
                    "x" * 100,
                    "Context: example\n\n",
                    model="test-model",
                    max_tokens=100,
                )
            )

        assert result == ""


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

    def test_large_content_synthesis_falls_back_when_final_summary_is_empty(self):
        from tools.web_tools import _process_large_content_chunked

        with (
            patch(
                "tools.web_tools._call_summarizer_llm",
                new=AsyncMock(side_effect=["sum-1", "sum-2", "sum-3"]),
            ),
            patch("tools.web_tools._resolve_web_extract_auxiliary", return_value=(object(), "test-model", None)),
            patch("tools.web_tools.async_call_llm", new=AsyncMock(side_effect=[_make_response(None), _make_response(None)])),
        ):
            result = _run(
                _process_large_content_chunked(
                    "A" * 12,
                    "",
                    model="test-model",
                    chunk_size=5,
                    max_output_size=1000,
                )
            )

        assert result == "## Section 1\nsum-1\n\n## Section 2\nsum-2\n\n## Section 3\nsum-3"


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

    def test_legacy_vision_tool_returns_fallback_error_after_empty_retry(self, tmp_path):
        from tools.vision_tools import _legacy_vision_analyze_tool_impl

        image_path = tmp_path / "image.png"
        image_path.write_bytes(b"not-a-real-png")

        with (
            patch("tools.vision_tools.resolve_vision_request_target", return_value=("openrouter", "https://openrouter.ai/api/v1")),
            patch("tools.vision_tools._get_recent_vision_failure", return_value=None),
            patch("tools.vision_tools._get_provider_unhealthy_reason", return_value=(0.0, "")),
            patch("tools.vision_tools._detect_image_mime_type", return_value="image/png"),
            patch("tools.vision_tools._image_to_base64_data_url", return_value="data:image/png;base64,AAAA"),
            patch("tools.vision_tools.async_call_llm", new=AsyncMock(side_effect=[_make_response(None), _make_response(None)])),
            patch("tools.vision_tools._store_recent_vision_failure"),
            patch("tools.vision_tools._debug.log_call"),
            patch("tools.vision_tools._debug.save"),
            patch("hermes_cli.config.load_config", return_value={}),
        ):
            result = json.loads(
                _run(
                    _legacy_vision_analyze_tool_impl(
                        str(image_path),
                        "What is in the image?",
                        model="test-vision-model",
                    )
                )
            )

        assert result["success"] is False
        assert "empty response twice" in result["analysis"]


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

    def test_llm_audit_skill_returns_static_result_when_llm_content_is_empty(self, tmp_path):
        from tools.skills_guard import ScanResult, llm_audit_skill

        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# Demo skill\n", encoding="utf-8")
        static_result = ScanResult(
            skill_name="demo-skill",
            source="community",
            trust_level="community",
            verdict="safe",
            findings=[],
            summary="safe",
        )

        with patch("agent.auxiliary_client.call_llm", return_value=_make_response(None)):
            result = llm_audit_skill(skill_file, static_result, model="gpt-5.4")

        assert result is static_result


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

    def test_summarize_session_returns_empty_string_after_empty_retries(self):
        from tools.session_search_tool import _summarize_session

        with (
            patch("tools.session_search_tool.async_call_llm", new=AsyncMock(side_effect=[_make_response(None), _make_response(None), _make_response(None)])),
            patch("tools.session_search_tool.asyncio.sleep", new=AsyncMock()),
        ):
            result = _run(
                _summarize_session(
                    "user: hi\nassistant: hello",
                    "find greeting",
                    {"source": "cli", "started_at": "2025-01-01T00:00:00Z"},
                )
            )

        assert result == ""


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

    def test_sse_string_content_is_collapsed_into_final_text(self):
        response = (
            'data: {"choices":[{"delta":{"reasoning_content":"先想一下"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"图里"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"有一只猫"}}]}\n\n'
            "data: [DONE]\n\n"
        )

        assert extract_content_or_reasoning(response) == "图里有一只猫"
