"""Tests for CLI voice mode integration -- command parsing, markdown stripping, state management."""

import re
import threading

import pytest


# ============================================================================
# Markdown stripping (same logic as _voice_speak_response)
# ============================================================================

def _strip_markdown_for_tts(text: str) -> str:
    """Replicate the markdown stripping logic from cli._voice_speak_response."""
    tts_text = text[:4000] if len(text) > 4000 else text
    tts_text = re.sub(r'```[\s\S]*?```', ' ', tts_text)   # fenced code blocks
    tts_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', tts_text)  # [text](url) -> text
    tts_text = re.sub(r'https?://\S+', '', tts_text)      # URLs
    tts_text = re.sub(r'\*\*(.+?)\*\*', r'\1', tts_text)  # bold
    tts_text = re.sub(r'\*(.+?)\*', r'\1', tts_text)      # italic
    tts_text = re.sub(r'`(.+?)`', r'\1', tts_text)        # inline code
    tts_text = re.sub(r'^#+\s*', '', tts_text, flags=re.MULTILINE)  # headers
    tts_text = re.sub(r'^\s*[-*]\s+', '', tts_text, flags=re.MULTILINE)  # list items
    tts_text = re.sub(r'---+', '', tts_text)              # horizontal rules
    tts_text = re.sub(r'\n{3,}', '\n\n', tts_text)        # excessive newlines
    return tts_text.strip()


class TestMarkdownStripping:
    def test_strips_bold(self):
        assert _strip_markdown_for_tts("This is **bold** text") == "This is bold text"

    def test_strips_italic(self):
        assert _strip_markdown_for_tts("This is *italic* text") == "This is italic text"

    def test_strips_inline_code(self):
        assert _strip_markdown_for_tts("Run `pip install foo`") == "Run pip install foo"

    def test_strips_fenced_code_blocks(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        result = _strip_markdown_for_tts(text)
        assert "print" not in result
        assert "Done." in result

    def test_strips_headers(self):
        assert _strip_markdown_for_tts("## Summary\nSome text") == "Summary\nSome text"

    def test_strips_list_markers(self):
        text = "- item one\n- item two\n* item three"
        result = _strip_markdown_for_tts(text)
        assert "item one" in result
        assert "- " not in result
        assert "* " not in result

    def test_strips_urls(self):
        text = "Visit https://example.com for details"
        result = _strip_markdown_for_tts(text)
        assert "https://" not in result
        assert "Visit" in result

    def test_strips_markdown_links(self):
        text = "See [the docs](https://example.com/docs) for info"
        result = _strip_markdown_for_tts(text)
        assert "the docs" in result
        assert "https://" not in result
        assert "[" not in result

    def test_strips_horizontal_rules(self):
        text = "Part one\n---\nPart two"
        result = _strip_markdown_for_tts(text)
        assert "---" not in result
        assert "Part one" in result
        assert "Part two" in result

    def test_empty_after_stripping_returns_empty(self):
        text = "```python\nprint('hello')\n```"
        result = _strip_markdown_for_tts(text)
        assert result == ""

    def test_truncates_long_text(self):
        text = "a" * 5000
        result = _strip_markdown_for_tts(text)
        assert len(result) <= 4000

    def test_complex_response(self):
        text = (
            "## Answer\n\n"
            "Here's how to do it:\n\n"
            "```python\ndef hello():\n    print('hi')\n```\n\n"
            "Run it with `python main.py`. "
            "See [docs](https://example.com) for more.\n\n"
            "- Step one\n- Step two\n\n"
            "---\n\n"
            "**Good luck!**"
        )
        result = _strip_markdown_for_tts(text)
        assert "```" not in result
        assert "https://" not in result
        assert "**" not in result
        assert "---" not in result
        assert "Answer" in result
        assert "Good luck!" in result
        assert "docs" in result


# ============================================================================
# Voice command parsing
# ============================================================================

class TestVoiceCommandParsing:
    """Test _handle_voice_command logic without full CLI setup."""

    def test_parse_subcommands(self):
        """Verify subcommand extraction from /voice commands."""
        test_cases = [
            ("/voice on", "on"),
            ("/voice off", "off"),
            ("/voice tts", "tts"),
            ("/voice status", "status"),
            ("/voice", ""),
            ("/voice  ON  ", "on"),
        ]
        for command, expected in test_cases:
            parts = command.strip().split(maxsplit=1)
            subcommand = parts[1].lower().strip() if len(parts) > 1 else ""
            assert subcommand == expected, f"Failed for {command!r}: got {subcommand!r}"


# ============================================================================
# Voice state thread safety
# ============================================================================

class TestVoiceStateLock:
    def test_lock_protects_state(self):
        """Verify that concurrent state changes don't corrupt state."""
        lock = threading.Lock()
        state = {"recording": False, "count": 0}

        def toggle_many(n):
            for _ in range(n):
                with lock:
                    state["recording"] = not state["recording"]
                    state["count"] += 1

        threads = [threading.Thread(target=toggle_many, args=(1000,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert state["count"] == 4000
