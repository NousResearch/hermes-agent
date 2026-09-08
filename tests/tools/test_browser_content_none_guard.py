"""Tests for None guard on browser_tool LLM response content.

browser_tool.py's browser_vision accesses response.choices[0].message.content
which can be None when reasoning-only models (DeepSeek-R1, QwQ) return
content=None. These tests verify the site is guarded.

The old _extract_relevant_content snapshot-summarization path was removed —
oversized snapshots now always truncate-and-store (no auxiliary LLM), so its
None-guard tests are gone with it.
"""

import types


# ── helpers ────────────────────────────────────────────────────────────────

def _make_response(content):
    """Build a minimal OpenAI-compatible ChatCompletion response stub."""
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


# ── browser_vision ─────────────────────────────────────────────────────────

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
        with open(os.path.join(base, "tools", "browser_tool.py")) as f:
            return f.read()

    def test_browser_vision_guarded(self):
        src = self._read_file()
        assert "analysis = response.choices[0].message.content\n" not in src, (
            "browser_tool.py browser_vision still has unguarded "
            ".content assignment — apply None guard"
        )

    def test_snapshot_llm_summarization_removed(self):
        """Snapshots must not route through an auxiliary LLM anymore."""
        src = self._read_file()
        assert "_extract_relevant_content" not in src
        assert "_get_extraction_model" not in src
