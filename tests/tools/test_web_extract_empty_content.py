"""Tests for web_extract_tool empty-content handling (#46756).

When extraction results have empty ``content`` fields (e.g., the backend
timed out or a URL was policy-blocked), ``web_extract_tool`` must populate
the content with a meaningful placeholder so the tool result never has
empty content.  Some providers (e.g., Xiaomi MiMo) reject empty-content
tool results with a non-retryable ``400 text is not set`` error.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── helpers ──────────────────────────────────────────────────────────────

def _make_mock_provider(results):
    """Return a mock provider whose ``extract`` yields ``results``."""
    provider = MagicMock()
    provider.name = "mock"
    provider.display_name = "Mock"
    provider.supports_extract.return_value = True
    provider.extract = AsyncMock(return_value=results)
    return provider


@pytest.fixture
def _patched_extract_env():
    """Patch everything *around* the trim/return logic so we exercise the
    real ``web_extract_tool`` production path."""
    with \
        patch("tools.web_tools._ensure_web_plugins_loaded"), \
        patch("tools.web_tools.async_is_safe_url", new=AsyncMock(return_value=True)), \
        patch("tools.web_tools._get_extract_backend", return_value="parallel"):
        yield


# ── RED→GREEN: empty content gets populated ─────────────────────────────

@pytest.mark.asyncio
async def test_empty_content_with_error_gets_populated(_patched_extract_env):
    """When a result has empty content but an error, the content field
    must be populated with the error detail (#46756)."""
    empty_results = [
        {
            "url": "https://example.com",
            "title": "",
            "content": "",
            "error": "Parallel extract failed: The read operation timed out",
        }
    ]
    mock_provider = _make_mock_provider(empty_results)

    with \
        patch("agent.web_search_registry.get_provider", return_value=mock_provider), \
        patch("agent.web_search_registry.get_active_extract_provider", return_value=mock_provider):
        from tools.web_tools import web_extract_tool

        result = await web_extract_tool(
            ["https://example.com"],
        )

    parsed = json.loads(result)
    assert "results" in parsed
    entry = parsed["results"][0]
    # Content must be non-empty (the fix).
    assert entry["content"], (
        "Content field must be populated when extraction fails, "
        "not left empty (#46756)"
    )
    # Error detail should be surfaced in the content.
    assert "timed out" in entry["content"].lower(), (
        f"Content should include error detail, got: {entry['content']}"
    )
    # Original error field must remain intact.
    assert "timed out" in entry["error"]


@pytest.mark.asyncio
async def test_empty_content_no_error_gets_placeholder(_patched_extract_env):
    """When a result has empty content AND no error field, the content
    must still be populated with a generic placeholder."""
    empty_results = [
        {"url": "https://example.com", "title": "", "content": ""},
    ]
    mock_provider = _make_mock_provider(empty_results)

    with \
        patch("agent.web_search_registry.get_provider", return_value=mock_provider), \
        patch("agent.web_search_registry.get_active_extract_provider", return_value=mock_provider):
        from tools.web_tools import web_extract_tool

        result = await web_extract_tool(
            ["https://example.com"],
        )

    parsed = json.loads(result)
    entry = parsed["results"][0]
    assert entry["content"], (
        "Content field must be populated even without an error message"
    )


# ── GREEN guard: normal results unaffected ──────────────────────────────

@pytest.mark.asyncio
async def test_normal_content_preserved(_patched_extract_env):
    """Results with real content must be returned unchanged."""
    good_results = [
        {"url": "https://example.com", "title": "Example", "content": "Full content"},
    ]
    mock_provider = _make_mock_provider(good_results)

    with \
        patch("agent.web_search_registry.get_provider", return_value=mock_provider), \
        patch("agent.web_search_registry.get_active_extract_provider", return_value=mock_provider):
        from tools.web_tools import web_extract_tool

        result = await web_extract_tool(
            ["https://example.com"],
        )

    parsed = json.loads(result)
    assert "results" in parsed
    assert parsed["results"][0]["content"] == "Full content"


@pytest.mark.asyncio
async def test_mixed_empty_and_nonempty_content(_patched_extract_env):
    """When some results have content and some don't, only the empty ones
    get the placeholder; non-empty ones pass through unchanged."""
    mixed_results = [
        {"url": "https://example.com", "title": "Example", "content": "Real content here"},
        {"url": "https://other.com", "title": "", "content": "", "error": "timeout"},
    ]
    mock_provider = _make_mock_provider(mixed_results)

    with \
        patch("agent.web_search_registry.get_provider", return_value=mock_provider), \
        patch("agent.web_search_registry.get_active_extract_provider", return_value=mock_provider):
        from tools.web_tools import web_extract_tool

        result = await web_extract_tool(
            ["https://example.com", "https://other.com"],
        )

    parsed = json.loads(result)
    assert len(parsed["results"]) == 2
    # Non-empty content preserved
    assert parsed["results"][0]["content"] == "Real content here"
    # Empty content populated
    assert parsed["results"][1]["content"], (
        "Empty content must be populated even when other results have content"
    )
