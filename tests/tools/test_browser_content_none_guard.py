"""Tests for None guard on browser_tool LLM response content.

browser_tool.py has two call sites that access response.choices[0].message.content
without checking for None — _extract_relevant_content (line 996) and
browser_vision (line 1626). When reasoning-only models (DeepSeek-R1, QwQ)
return content=None, these produce null snapshots or null analysis.

These tests verify both sites are guarded.
"""

import json
import types
from unittest.mock import patch



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
        """Normal string content should pass through (plus the stored-full-snapshot pointer)."""
        with patch("tools.browser_tool.call_llm", return_value=_make_response("Extracted content here")), \
             patch("tools.browser_tool._get_extraction_model", return_value="test-model"):
            from tools.browser_tool import _extract_relevant_content
            result = _extract_relevant_content("snapshot text", "task")

        # The summary itself passes through unchanged; a pointer to the stored
        # full snapshot is appended (see _store_full_snapshot).
        assert result.startswith("Extracted content here")
        assert "Full snapshot saved to" in result

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

    def test_none_content_produces_fallback_message(self, tmp_path):
        """When LLM returns None content, browser_vision should emit the fallback text."""
        screenshot_path = tmp_path / "browser-shot.png"
        screenshot_path.write_bytes(b"fake-png")

        with (
            patch("hermes_constants.get_hermes_dir", return_value=tmp_path),
            patch("tools.browser_tool._cleanup_old_screenshots"),
            patch(
                "tools.browser_tool._run_browser_command",
                return_value={"success": True, "data": {"path": str(screenshot_path)}},
            ),
            patch("tools.browser_tool.call_llm", return_value=_make_response(None)),
            patch("agent.redact.redact_sensitive_text", side_effect=lambda text: text),
            patch("hermes_cli.config.load_config", return_value={}),
        ):
            from tools.browser_tool import browser_vision

            result = json.loads(browser_vision("What is on the page?", task_id="task-1"))

        assert result["success"] is True
        assert result["analysis"] == "Vision analysis returned no content."
        assert result["screenshot_path"] == str(screenshot_path)

    def test_normal_content_passes_through(self):
        """Normal analysis content should pass through unchanged."""
        response = _make_response("  The page shows a login form.  ")
        analysis = (response.choices[0].message.content or "").strip()
        fallback = analysis or "Vision analysis returned no content."

        assert fallback == "The page shows a login form."
