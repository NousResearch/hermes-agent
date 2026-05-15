"""Tests for None guard on browser_tool LLM response content.

browser_tool.py has two call sites that access response.choices[0].message.content
without checking for None — _extract_relevant_content (line 996) and
browser_vision (line 1626). When reasoning-only models (DeepSeek-R1, QwQ)
return content=None, these produce null snapshots or null analysis.

These tests verify both sites are guarded.
"""

import types
from unittest.mock import MagicMock, patch

import pytest


# ── helpers ────────────────────────────────────────────────────────────────

def _make_response(content):
    """Build a minimal OpenAI-compatible ChatCompletion response stub."""
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


# ── _extract_relevant_content (line 996) ──────────────────────────────────

class TestExtractRelevantContentNoneGuard:
    """tools/browser_tool.py — _extract_relevant_content()"""

    def test_none_content_falls_back_to_truncated(self):
        """When LLM returns None content, should fall back to truncated snapshot."""
        with patch("tools.browser_tool.call_llm", return_value=_make_response(None)), \
             patch("tools.browser_tool._get_extraction_model", return_value="test-model"):
            from tools.browser_tool import _extract_relevant_content
            result = _extract_relevant_content("This is a long snapshot text", "find the button")

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_normal_content_returned(self):
        """Normal string content should pass through."""
        with patch("tools.browser_tool.call_llm", return_value=_make_response("Extracted content here")), \
             patch("tools.browser_tool._get_extraction_model", return_value="test-model"):
            from tools.browser_tool import _extract_relevant_content
            result = _extract_relevant_content("snapshot text", "task")

        assert result == "Extracted content here"

    def test_empty_string_content_falls_back(self):
        """Empty string content should also fall back to truncated."""
        with patch("tools.browser_tool.call_llm", return_value=_make_response("   ")), \
             patch("tools.browser_tool._get_extraction_model", return_value="test-model"):
            from tools.browser_tool import _extract_relevant_content
            result = _extract_relevant_content("This is a long snapshot text", "task")

        assert result is not None
        assert len(result) > 0


# ── browser_vision (line 1626) ────────────────────────────────────────────

class TestBrowserVisionNoneGuard:
    """tools/browser_tool.py — browser_vision() analysis extraction"""

    def test_none_content_produces_fallback_message(self):
        """When LLM returns None content, analysis should have a fallback message."""
        response = _make_response(None)
        analysis = (response.choices[0].message.content or "").strip()
        fallback = analysis or "Vision analysis returned no content."

        assert fallback == "Vision analysis returned no content."

    def test_normal_content_passes_through(self):
        """Normal analysis content should pass through unchanged."""
        response = _make_response("  The page shows a login form.  ")
        analysis = (response.choices[0].message.content or "").strip()
        fallback = analysis or "Vision analysis returned no content."

        assert fallback == "The page shows a login form."


# ── source line verification ──────────────────────────────────────────────

class TestBrowserSourceLinesAreGuarded:
    """Verify the actual source file has the fix applied."""

    @staticmethod
    def _read_file() -> str:
        import os
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(base, "tools", "browser_tool.py"), encoding="utf-8") as f:
            return f.read()

    def test_extract_relevant_content_guarded(self):
        src = self._read_file()
        # The old unguarded pattern should NOT exist
        assert "return response.choices[0].message.content\n" not in src, (
            "browser_tool.py _extract_relevant_content still has unguarded "
            ".content return — apply None guard"
        )

    def test_browser_vision_guarded(self):
        src = self._read_file()
        assert "analysis = response.choices[0].message.content\n" not in src, (
            "browser_tool.py browser_vision still has unguarded "
            ".content assignment — apply None guard"
        )
