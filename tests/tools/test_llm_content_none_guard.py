"""Tests for None guard on response.choices[0].message.content.strip().

OpenAI-compatible APIs return ``message.content = None`` when the model
responds with tool calls only or reasoning-only output (e.g. DeepSeek-R1,
Qwen-QwQ via OpenRouter with ``reasoning.enabled = True``).  Calling
``.strip()`` on ``None`` raises ``AttributeError``.

These tests verify that every call site handles ``content is None`` safely.
"""

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── helpers ────────────────────────────────────────────────────────────────

def _make_response(content):
    """Build a minimal OpenAI-compatible ChatCompletion response stub."""
    message = types.SimpleNamespace(content=content, tool_calls=None)
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
