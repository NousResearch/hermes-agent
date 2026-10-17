"""web_extract must SIGNAL an empty extraction instead of returning a success-shaped empty result.

An entry ``{"content": "", "error": null}`` reads as a successfully fetched empty page and models fill
the gap with invented page content (arXiv:2609.14758; #52120, #99533). Only a falsy ``error`` is filled,
so provider-level errors (Firecrawl status >= 400) survive verbatim.
"""
from __future__ import annotations

import asyncio
import json

import tools.web_tools as wt
from tools.web_tools_extract import EMPTY_CONTENT_ERROR


class _Provider:
    name = "stub"
    display_name = "Stub"

    def __init__(self, results):
        self._results = results

    async def extract(self, urls, format=None):
        return list(self._results)


def _run_extract(monkeypatch, tmp_path, results):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wt, "_get_extract_backend", lambda: "stub")
    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_resolve_extract_provider", lambda backend: (_Provider(results), None))
    monkeypatch.setattr(wt, "async_is_safe_url", _true)
    monkeypatch.setattr("tools.web_tools_extract.extract_cache_get", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("tools.web_result_cache.extract_cache_get", lambda *a, **k: None)
    monkeypatch.setattr("tools.web_result_cache.extract_cache_put", lambda *a, **k: None)
    monkeypatch.setattr("tools.web_tools_extract._rescue_eligible", lambda provider: False)
    out = asyncio.run(wt.web_extract_tool(["https://example.com/a", "https://example.com/b"]))
    return json.loads(out)["results"]


async def _true(url):
    return True


def test_empty_body_without_error_is_signalled(monkeypatch, tmp_path):
    results = _run_extract(monkeypatch, tmp_path, [
        {"url": "https://example.com/a", "title": "", "content": "", "raw_content": "", "error": None},
        {"url": "https://example.com/b", "title": "Real page", "content": "# Hello\n\nbody text", "error": None},
    ])
    assert results[0]["error"] == EMPTY_CONTENT_ERROR
    assert results[0]["content"] == ""
    # A page with content is untouched.
    assert results[1]["error"] is None
    assert results[1]["content"].startswith("# Hello")


def test_provider_error_is_kept_verbatim(monkeypatch, tmp_path):
    results = _run_extract(monkeypatch, tmp_path, [
        {"url": "https://example.com/a", "title": "", "content": "", "error": "Target responded 403 Forbidden"},
        {"url": "https://example.com/b", "title": "", "content": "   \n\t", "error": None},  # whitespace-only body
    ])
    assert results[0]["error"] == "Target responded 403 Forbidden"
    assert results[1]["error"] == EMPTY_CONTENT_ERROR
