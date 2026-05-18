"""Tests for browser tool schema routing guidance."""

from pathlib import Path


def test_browser_navigate_plain_text_guidance_uses_safe_extract_path():
    """Plain-text URL guidance must not route around URL safety checks."""
    source = Path("tools/browser_tool.py").read_text()
    description_line = next(
        line for line in source.splitlines()
        if '"name": "browser_navigate"' in line
    )
    # The schema description is adjacent to the name in the same dict; inspect
    # a small stable window without importing optional browser dependencies.
    start = source.index(description_line)
    schema_window = source[start:start + 2500]

    assert "plain-text endpoints" in schema_window
    assert "web_extract so URL safety checks are applied" in schema_window
    assert "curl via the terminal tool" not in schema_window
