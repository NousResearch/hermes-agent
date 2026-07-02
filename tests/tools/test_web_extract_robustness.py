"""Tests for web_extract truncate-store robustness (findings from #54843 review).

Covers two robustness gaps left unaddressed when #54843 merged:
  1. _store_full_text bounded by MAX_STORED_TEXT_CHARS (no unbounded disk write).
  2. _truncate_with_footer emits a CONCRETE read_file offset for the omitted
     middle (was a literal `offset=<line>` placeholder the model had to guess).
"""
from __future__ import annotations

import re

import pytest

import tools.web_tools as wt


def test_store_full_text_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Force the cache dir under the temp home.
    from hermes_constants import get_hermes_dir  # noqa: F401
    huge = "x\n" * (wt.MAX_STORED_TEXT_CHARS)  # > MAX_STORED_TEXT_CHARS chars
    assert len(huge) > wt.MAX_STORED_TEXT_CHARS
    path = wt._store_full_text("https://example.com/big", huge)
    assert path is not None
    stored = open(path, encoding="utf-8").read()
    # Stored copy capped (+ short marker), not the full unbounded blob.
    assert len(stored) <= wt.MAX_STORED_TEXT_CHARS + 200
    assert "stored copy truncated" in stored


def test_truncate_footer_gives_concrete_offset(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Build content well over the limit with many lines so head has a known count.
    content = "\n".join(f"line {i}" for i in range(5000))
    model_text, truncated = wt._truncate_with_footer(
        content, "https://example.com/page", char_limit=4000
    )
    assert truncated
    # Footer must contain a real integer offset, NOT the <line> placeholder.
    assert "offset=<line>" not in model_text
    m = re.search(r"offset=(\d+) limit=\d+", model_text)
    assert m, f"no concrete offset in footer: {model_text[-400:]}"
    offset = int(m.group(1))
    # Offset should point past the head we showed (head is ~75% of 4000 chars).
    assert offset > 1


def test_small_page_not_truncated_no_footer(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    content = "short page\nwith a few lines\n"
    model_text, truncated = wt._truncate_with_footer(
        content, "https://example.com/s", char_limit=15000
    )
    assert not truncated
    assert model_text == content
    assert "[TRUNCATED]" not in model_text


@pytest.mark.asyncio
async def test_extract_timeout_returns_error_results(monkeypatch, tmp_path):
    """When a provider hangs past _DEFAULT_EXTRACT_TIMEOUT_S, the wrapper
    should cancel the coroutine and return per-URL error results instead of
    blocking the event loop indefinitely (#57155)."""
    import asyncio
    import json
    import sys
    import types

    import tools.web_tools as wt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    async def _slow_extract(self, urls, **_kw):
        """Simulates a provider that hangs forever."""
        await asyncio.sleep(9999)
        return [{"url": u, "content": "never"} for u in urls]

    # Patch the timeout to 0.3s so the test is fast.
    monkeypatch.setattr(wt, "_DEFAULT_EXTRACT_TIMEOUT_S", 0.3)

    # Patch the SSRF check and backend resolution so only the dispatch runs.
    async def _safe(url):
        return True
    monkeypatch.setattr(wt, "async_is_safe_url", _safe)

    class _FakeProvider:
        name = "fake"
        def supports_extract(self):
            return True
        extract = _slow_extract  # async

    monkeypatch.setattr(wt, "_get_extract_backend", lambda: "fake")
    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)

    # The import-mock for the registry inside web_extract_tool
    fake_registry = types.ModuleType("agent.web_search_registry")
    fake_registry.get_active_extract_provider = lambda: _FakeProvider()
    fake_registry.get_provider = lambda _: _FakeProvider()
    monkeypatch.setitem(sys.modules, "agent.web_search_registry", fake_registry)

    result_json = await wt.web_extract_tool(["https://example.com/slow"])
    result = json.loads(result_json)
    assert "results" in result
    assert len(result["results"]) == 1
    entry = result["results"][0]
    assert "timed out" in entry.get("error", "").lower()
    assert entry["url"] == "https://example.com/slow"
